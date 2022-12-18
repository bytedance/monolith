// Copyright 2022 ByteDance and/or its affiliates.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "monolith/native_training/optimizers/cc/kernels/training_ops.h"

namespace tensorflow {
namespace monolith_tf {
typedef Eigen::GpuDevice GPUDevice;

template <>
struct ApplyRmsprop<GPUDevice> {
  void operator()(const GPUDevice& d, typename TTypes<float>::Flat var,
                  typename TTypes<float>::Flat m,
                  typename TTypes<float>::Flat v,
                  typename TTypes<float>::ConstScalar lr,
                  typename TTypes<float>::ConstScalar beta1,
                  typename TTypes<float>::ConstScalar beta2,
                  typename TTypes<float>::ConstScalar epsilon,
                  typename TTypes<float>::ConstScalar weight_decay,
                  typename TTypes<float>::ConstFlat grad, bool update_slots) {
    Eigen::array<typename TTypes<float>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;
    auto grad_after_decay =
        weight_decay.reshape(single).broadcast(bcast) * var + grad;
    if (update_slots) {
      v.device(d) +=
          (grad_after_decay.square() - v) *
          (beta2.constant(1.0f) - beta2).reshape(single).broadcast(bcast);
      m.device(d) = beta1.reshape(single).broadcast(bcast) * m +
                    (grad_after_decay * lr.reshape(single).broadcast(bcast)) *
                        (v + epsilon.reshape(single).broadcast(bcast)).rsqrt();
      var.device(d) -= m;
    }
  }
};

template <>
struct ApplyRmspropV2<GPUDevice> {
  void operator()(const GPUDevice& d, typename TTypes<float>::Flat var,
                  typename TTypes<float>::Flat m,
                  typename TTypes<float>::Flat v,
                  typename TTypes<float>::ConstScalar lr,
                  typename TTypes<float>::ConstScalar beta1,
                  typename TTypes<float>::ConstScalar beta2,
                  typename TTypes<float>::ConstScalar epsilon,
                  typename TTypes<float>::ConstScalar weight_decay,
                  typename TTypes<float>::ConstFlat grad, bool update_slots) {
    Eigen::array<typename TTypes<float>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;
    auto grad_after_decay =
        weight_decay.reshape(single).broadcast(bcast) * var + grad;
    if (update_slots) {
      v.device(d) = beta2.reshape(single).broadcast(bcast) * v +
                    grad_after_decay.square();
      //      m.device(d) = beta1() * m + (grad_after_decay * lr()) * (v +
      //      epsilon()).rsqrt();
      m.device(d) = beta1.reshape(single).broadcast(bcast) * m +
                    (grad_after_decay * lr.reshape(single).broadcast(bcast)) /
                        (v.sqrt() + epsilon.reshape(single).broadcast(bcast));
      var.device(d) -= m;
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("ResourceApplyRmsprop").Device(DEVICE_GPU),
                        ApplyRmspropOp<GPUDevice>);

}  // namespace monolith_tf
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
