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

#define EIGEN_USE_THREADS
#include "monolith/native_training/optimizers/cc/kernels/training_ops.h"

namespace tensorflow {
namespace monolith_tf {

template <>
struct ApplyRmsprop<CPUDevice> {
  void operator()(const CPUDevice& d, typename TTypes<float>::Flat var,
                  typename TTypes<float>::Flat m,
                  typename TTypes<float>::Flat v,
                  typename TTypes<float>::ConstScalar lr,
                  typename TTypes<float>::ConstScalar beta1,
                  typename TTypes<float>::ConstScalar beta2,
                  typename TTypes<float>::ConstScalar epsilon,
                  typename TTypes<float>::ConstScalar weight_decay,
                  typename TTypes<float>::ConstFlat grad, bool update_slots) {
    auto grad_after_decay = weight_decay() * var + grad;
    if (update_slots) {
      v.device(d) += (grad_after_decay.square() - v) * (1.0f - beta2());
      m.device(d) =
          beta1() * m + (grad_after_decay * lr()) * (v + epsilon()).rsqrt();
      var.device(d) -= m;
    }
  }
};

template <>
struct ApplyRmspropV2<CPUDevice> {
  void operator()(const CPUDevice& d, typename TTypes<float>::Flat var,
                  typename TTypes<float>::Flat m,
                  typename TTypes<float>::Flat v,
                  typename TTypes<float>::ConstScalar lr,
                  typename TTypes<float>::ConstScalar beta1,
                  typename TTypes<float>::ConstScalar beta2,
                  typename TTypes<float>::ConstScalar epsilon,
                  typename TTypes<float>::ConstScalar weight_decay,
                  typename TTypes<float>::ConstFlat grad, bool update_slots) {
    auto grad_after_decay = weight_decay() * var + grad;
    if (update_slots) {
      v.device(d) = beta2() * v + grad_after_decay.square();
      //      m.device(d) = beta1() * m + (grad_after_decay * lr()) * (v +
      //      epsilon()).rsqrt();
      m.device(d) =
          beta1() * m + (grad_after_decay * lr()) / (v.sqrt() + epsilon());
      var.device(d) -= m;
    }
  }
};

template <>
struct ApplyAdamom<CPUDevice> {
  void operator()(const CPUDevice& d, typename TTypes<float>::Flat var,
                  typename TTypes<float>::Flat m,
                  typename TTypes<float>::Flat v,
                  typename TTypes<float>::Flat c,
                  typename TTypes<float>::ConstScalar lr,
                  typename TTypes<float>::ConstScalar ada_decay,
                  typename TTypes<float>::ConstScalar mom_decay,
                  typename TTypes<float>::ConstScalar epsilon,
                  typename TTypes<float>::ConstScalar weight_decay,
                  typename TTypes<float>::ConstFlat grad, bool update_slots) {
    auto grad_after_decay = weight_decay() * var + grad;

    if (update_slots) {
      m.device(d) = mom_decay() * m + (1.0f - mom_decay()) * grad_after_decay;
      v.device(d) = ada_decay() * v + grad_after_decay * grad_after_decay;
      c.device(d) = ada_decay() * c + 1.0f;
    }
    var.device(d) -= m * lr() * (v / c + epsilon()).rsqrt();
  }
};

template <>
struct ApplyAdamomV2<CPUDevice> {
  void operator()(const CPUDevice& d, typename TTypes<float>::Flat var,
                  typename TTypes<float>::Flat m,
                  typename TTypes<float>::Flat v,
                  typename TTypes<float>::Flat c,
                  typename TTypes<float>::ConstScalar lr,
                  typename TTypes<float>::ConstScalar ada_decay,
                  typename TTypes<float>::ConstScalar mom_decay,
                  typename TTypes<float>::ConstScalar epsilon,
                  typename TTypes<float>::ConstScalar weight_decay,
                  typename TTypes<float>::ConstFlat grad, bool update_slots) {
    auto grad_after_decay = weight_decay() * var + grad;

    if (update_slots) {
      m.device(d) = mom_decay() * m + (1.0f - mom_decay()) * grad_after_decay;
      v.device(d) = ada_decay() * v + grad_after_decay * grad_after_decay;
      c.device(d) = ada_decay() * c + 1.0f;
    }
    var.device(d) -= m * lr() / ((v / c).sqrt() + epsilon());
  }
};

REGISTER_KERNEL_BUILDER(Name("ResourceApplyAdamom")
                            // .HostMemory("var")
                            // .HostMemory("m")
                            // .HostMemory("v")
                            // .HostMemory("c")
                            .Device(DEVICE_CPU),
                        ApplyAdamomOp<CPUDevice>);

REGISTER_KERNEL_BUILDER(Name("ResourceApplyRmsprop").Device(DEVICE_CPU),
                        ApplyRmspropOp<CPUDevice>);

}  // namespace monolith_tf
}  // namespace tensorflow
