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

#ifndef MONOLITH_MONOLITH_NATIVE_TRAINING_OPTIMIZERS_CC_TRAINING_OPS_H_
#define MONOLITH_MONOLITH_NATIVE_TRAINING_OPTIMIZERS_CC_TRAINING_OPS_H_

#include "monolith/native_training/optimizers/cc/kernels/training_op_helpers.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace monolith_tf {

template <typename Device>
struct ApplyRmsprop {
  void operator()(const Device& d, typename TTypes<float>::Flat var,
                  typename TTypes<float>::Flat m,
                  typename TTypes<float>::Flat v,
                  typename TTypes<float>::ConstScalar lr,
                  typename TTypes<float>::ConstScalar beta1,
                  typename TTypes<float>::ConstScalar beta2,
                  typename TTypes<float>::ConstScalar epsilon,
                  typename TTypes<float>::ConstScalar weight_decay,
                  typename TTypes<float>::ConstFlat grad, bool update_slots);
};

template <typename Device>
struct ApplyRmspropV2 {
  void operator()(const Device& d, typename TTypes<float>::Flat var,
                  typename TTypes<float>::Flat m,
                  typename TTypes<float>::Flat v,
                  typename TTypes<float>::ConstScalar lr,
                  typename TTypes<float>::ConstScalar beta1,
                  typename TTypes<float>::ConstScalar beta2,
                  typename TTypes<float>::ConstScalar epsilon,
                  typename TTypes<float>::ConstScalar weight_decay,
                  typename TTypes<float>::ConstFlat grad, bool update_slots);
};

template <typename Device>
struct ApplyAdamom {
  void operator()(const Device& d, typename TTypes<float>::Flat var,
                  typename TTypes<float>::Flat m,
                  typename TTypes<float>::Flat v,
                  typename TTypes<float>::Flat c,
                  typename TTypes<float>::ConstScalar lr,
                  typename TTypes<float>::ConstScalar ada_decay,
                  typename TTypes<float>::ConstScalar mom_decay,
                  typename TTypes<float>::ConstScalar epsilon,
                  typename TTypes<float>::ConstScalar weight_decay,
                  typename TTypes<float>::ConstFlat grad, bool update_slots);
};

template <typename Device>
struct ApplyAdamomV2 {
  void operator()(const Device& d, typename TTypes<float>::Flat var,
                  typename TTypes<float>::Flat m,
                  typename TTypes<float>::Flat v,
                  typename TTypes<float>::Flat c,
                  typename TTypes<float>::ConstScalar lr,
                  typename TTypes<float>::ConstScalar ada_decay,
                  typename TTypes<float>::ConstScalar mom_decay,
                  typename TTypes<float>::ConstScalar epsilon,
                  typename TTypes<float>::ConstScalar weight_decay,
                  typename TTypes<float>::ConstFlat grad, bool update_slots);
};

template <typename Device>
class ApplyRmspropOp : public OpKernel {
 public:
  explicit ApplyRmspropOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("update_slots", &update_slots_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_v2", &use_v2_));
  }

  void Compute(OpKernelContext* ctx) override {
    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, float>(
        ctx, use_exclusive_lock_, sparse, {0, 1});
    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, float>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor m;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, float>(
                            ctx, 1, use_exclusive_lock_, sparse, &m));
    Tensor v;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, float>(
                            ctx, 2, use_exclusive_lock_, sparse, &v));
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, m.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));
    OP_REQUIRES(
        ctx, v.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(2)));
    const Tensor& lr = ctx->input(3);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& beta1 = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()),
                errors::InvalidArgument("beta1 is not a scalar: ",
                                        beta1.shape().DebugString()));
    const Tensor& beta2 = ctx->input(5);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2.shape().DebugString()));
    const Tensor& epsilon = ctx->input(6);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));
    const Tensor& weight_decay = ctx->input(7);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(weight_decay.shape()),
                errors::InvalidArgument("weight_decay is not a scalar: ",
                                        weight_decay.shape().DebugString()));
    const Tensor& grad = ctx->input(8);
    OP_REQUIRES(ctx, var.shape().IsSameSize(m.shape()),
                errors::InvalidArgument("var and m do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        m.shape().DebugString()));
    OP_REQUIRES(ctx, var.shape().IsSameSize(v.shape()),
                errors::InvalidArgument("var and v do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        v.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));

    const Device& device = ctx->eigen_device<Device>();
    if (!use_v2_) {
      ApplyRmsprop<Device>()(
          device, var.flat<float>(), m.flat<float>(), v.flat<float>(),
          lr.scalar<float>(), beta1.scalar<float>(), beta2.scalar<float>(),
          epsilon.scalar<float>(), weight_decay.scalar<float>(),
          grad.flat<float>(), update_slots_);
    } else {
      ApplyRmspropV2<Device>()(
          device, var.flat<float>(), m.flat<float>(), v.flat<float>(),
          lr.scalar<float>(), beta1.scalar<float>(), beta2.scalar<float>(),
          epsilon.scalar<float>(), weight_decay.scalar<float>(),
          grad.flat<float>(), update_slots_);
    }

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
  bool update_slots_;
  bool use_v2_;
};

template <typename Device>
class ApplyAdamomOp : public OpKernel {
 public:
  explicit ApplyAdamomOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("update_slots", &update_slots_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_v2", &use_v2_));
  }

  void Compute(OpKernelContext* ctx) override {
    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder<CPUDevice, float>(
        ctx, use_exclusive_lock_, sparse, {0, 1});
    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, float>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor m;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, float>(
                            ctx, 1, use_exclusive_lock_, sparse, &m));
    Tensor v;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, float>(
                            ctx, 2, use_exclusive_lock_, sparse, &v));
    Tensor c;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, float>(
                            ctx, 3, use_exclusive_lock_, sparse, &c));
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, m.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));
    OP_REQUIRES(
        ctx, v.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(2)));
    OP_REQUIRES(
        ctx, c.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(3)));
    const Tensor& lr = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& ada_decay = ctx->input(5);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(ada_decay.shape()),
                errors::InvalidArgument("ada_decay is not a scalar: ",
                                        ada_decay.shape().DebugString()));
    const Tensor& mom_decay = ctx->input(6);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(mom_decay.shape()),
                errors::InvalidArgument("mom_decay is not a scalar: ",
                                        mom_decay.shape().DebugString()));
    const Tensor& epsilon = ctx->input(7);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));
    const Tensor& weight_decay = ctx->input(8);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(weight_decay.shape()),
                errors::InvalidArgument("weight_decay is not a scalar: ",
                                        weight_decay.shape().DebugString()));
    const Tensor& grad = ctx->input(9);
    OP_REQUIRES(ctx, var.shape().IsSameSize(m.shape()),
                errors::InvalidArgument(
                    "var and accum do not have the same shape",
                    var.shape().DebugString(), " ", m.shape().DebugString()));
    OP_REQUIRES(ctx, var.shape().IsSameSize(v.shape()),
                errors::InvalidArgument(
                    "var and accum do not have the same shape",
                    var.shape().DebugString(), " ", v.shape().DebugString()));
    OP_REQUIRES(ctx, var.shape().IsSameSize(c.shape()),
                errors::InvalidArgument(
                    "var and accum do not have the same shape",
                    var.shape().DebugString(), " ", c.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));

    const Device& device = ctx->eigen_device<Device>();
    if (!use_v2_) {
      ApplyAdamom<Device>()(
          device, var.flat<float>(), m.flat<float>(), v.flat<float>(),
          c.flat<float>(), lr.scalar<float>(), ada_decay.scalar<float>(),
          mom_decay.scalar<float>(), epsilon.scalar<float>(),
          weight_decay.scalar<float>(), grad.flat<float>(), update_slots_);
    } else {
      ApplyAdamomV2<Device>()(
          device, var.flat<float>(), m.flat<float>(), v.flat<float>(),
          c.flat<float>(), lr.scalar<float>(), ada_decay.scalar<float>(),
          mom_decay.scalar<float>(), epsilon.scalar<float>(),
          weight_decay.scalar<float>(), grad.flat<float>(), update_slots_);
    }

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
  bool update_slots_;
  bool use_v2_;
};

}  // namespace monolith_tf
}  // namespace tensorflow

#endif  // MONOLITH_MONOLITH_NATIVE_TRAINING_OPTIMIZERS_CC_TRAINING_OPS_H_
