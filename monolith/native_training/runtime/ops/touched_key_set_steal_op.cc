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

#include <memory>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_op_kernel.h"

#include "monolith/native_training/runtime/ops/touched_key_set_tf_bridge.h"

namespace tensorflow {
namespace monolith_tf {

class TouchedKeySetStealOp : public OpKernel {
 public:
  explicit TouchedKeySetStealOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    TouchedKeySetTfBridge* touched_key_set = nullptr;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &touched_key_set));
    core::ScopedUnref unref(touched_key_set);

    auto ids = touched_key_set->Steal();
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {static_cast<int64>(ids.size())}, &output));
    auto output_vec = output->vec<int64>();
    for (size_t i = 0; i < ids.size(); ++i) {
      output_vec(i) = ids[i];
    }
  }
};

REGISTER_OP("MonolithTouchedKeySetSteal")
        .Input("handle: resource")
        .Output("ids: int64")
        .SetIsStateful()
        .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(Name("MonolithTouchedKeySetSteal").Device(DEVICE_CPU),
                        TouchedKeySetStealOp);

}  // namespace monolith_tf
}  // namespace tensorflow
