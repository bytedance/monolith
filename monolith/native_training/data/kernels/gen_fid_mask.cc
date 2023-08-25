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

#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace monolith_tf {

template <typename T>
class MonolithGenFidMaskOp : public OpKernel {
 public:
  using OpKernel::OpKernel;

  explicit MonolithGenFidMaskOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fid", &fid_));
  }

  void Compute(OpKernelContext *context) override {
    // Grab the input tensor
    const Tensor *splits, *values;
    OP_REQUIRES_OK(context, context->input("splits", &splits));
    auto splits_flat = splits->flat<T>();
    OP_REQUIRES_OK(context, context->input("values", &values));
    auto values_flat = values->flat<int64>();

    // Create an output tensor
    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, {splits->NumElements() - 1},
                                            &output_tensor));
    auto output_flat = output_tensor->flat<float>();
    output_flat.setZero();

    for (int i = 1; i < splits->NumElements(); ++i) {
      int32 start = splits_flat(i - 1);
      int32 end = splits_flat(i);
      for (int j = start; j < end; ++j) {
        if (values_flat(j) == fid_) {
          output_flat(i - 1) = 1.0;
          break;
        }
      }
    }
  }

 private:
  int64 fid_;
};

namespace {
REGISTER_KERNEL_BUILDER(
    Name("MonolithGenFidMask").Device(DEVICE_CPU).TypeConstraint<int32>("T"),
    MonolithGenFidMaskOp<int32>);

REGISTER_KERNEL_BUILDER(
    Name("MonolithGenFidMask").Device(DEVICE_CPU).TypeConstraint<int64>("T"),
    MonolithGenFidMaskOp<int64>);
}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow
