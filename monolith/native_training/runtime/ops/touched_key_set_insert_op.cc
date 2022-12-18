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

class TouchedKeySetInsertOp : public OpKernel {
 public:
  explicit TouchedKeySetInsertOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    TouchedKeySetTfBridge* touched_key_set = nullptr;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &touched_key_set));
    core::ScopedUnref unref(touched_key_set);
    const Tensor& tensor = ctx->input(1);
    const int64 num_elements = tensor.NumElements();
    auto ids = tensor.vec<int64>();
    int64 total_dropped_num = 0;
    for (int64 i = 0; i < num_elements; ++i) {
      total_dropped_num += touched_key_set->Insert(ids(i));
    }

    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {1}, &output));
    auto output_vec = output->vec<int64>();
    output_vec(0) = total_dropped_num;
  }
};

REGISTER_OP("MonolithTouchedKeySetInsert")
        .Input("handle: resource")
        .Input("ids: int64")
        .Output("size: int64")
        .SetIsStateful()
        .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(Name("MonolithTouchedKeySetInsert").Device(DEVICE_CPU),
                        TouchedKeySetInsertOp);

}  // namespace monolith_tf
}  // namespace tensorflow
