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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace monolith_tf {

class MonolithFidCounterOp : public OpKernel {
 public:
  explicit MonolithFidCounterOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("step", &step_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("counter_threshold", &counter_threshold_));
  }

  void Compute(OpKernelContext *ctx) override {
    ctx->set_output(0, ctx->input(0));
  }

 private:
  float step_;
  int counter_threshold_;
};

namespace {

REGISTER_KERNEL_BUILDER(Name("MonolithFidCounter").Device(DEVICE_CPU), MonolithFidCounterOp)

}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow
