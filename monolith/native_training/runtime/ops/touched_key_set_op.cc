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

class TouchedKeySetOp : public ResourceOpKernel<TouchedKeySetTfBridge> {
 public:
  explicit TouchedKeySetOp(OpKernelConstruction* ctx) : ResourceOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("capacity", &capacity_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("concurrency_level", &concurrency_level_));
  }

  ~TouchedKeySetOp() override = default;

 private:
  Status CreateResource(TouchedKeySetTfBridge** touched_key_set_bridge)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    auto touched_key_set =
        std::make_unique<monolith::hopscotch::HopscotchHashSet<int64_t>>(
            capacity_,
            concurrency_level_);
    *touched_key_set_bridge =
        new TouchedKeySetTfBridge(std::move(touched_key_set));
    return Status::OK();
  };

  int64 capacity_;

  int concurrency_level_;
};

REGISTER_OP("MonolithTouchedKeySet")
        .Output("handle: resource")
        .Attr("capacity: int = 2097152")
        .Attr("concurrency_level: int = 1024")
        .Attr("container: string = ''")
        .Attr("shared_name: string = ''")
        .SetIsStateful()
        .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(Name("MonolithTouchedKeySet").Device(DEVICE_CPU),
                        TouchedKeySetOp);

}  // namespace monolith_tf
}  // namespace tensorflow
