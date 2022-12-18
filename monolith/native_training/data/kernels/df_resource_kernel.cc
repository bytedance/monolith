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

#include "monolith/native_training/data/kernels/df_resource_kernel.h"

namespace tensorflow {
namespace monolith_tf {

using Queue = ::monolith::concurrency::Queue<Item>;

Status RegisterCancellationCallback(CancellationManager* cancellation_manager,
                                    CancelCallback callback,
                                    std::function<void()>* deregister_fn) {
  if (cancellation_manager) {
    CancellationToken token = cancellation_manager->get_cancellation_token();
    if (!cancellation_manager->RegisterCallback(token, std::move(callback))) {
      return errors::Cancelled("Operation was cancelled");
    }
    *deregister_fn = [cancellation_manager, token]() {
      cancellation_manager->DeregisterCallback(token);
    };
  } else {
    VLOG(1) << "Cancellation manager is not set. Cancellation callback will "
               "not be registered.";
    *deregister_fn = []() {};
  }
  return Status::OK();
}

class CreateQueueOp : public ResourceOpKernel<QueueResource> {
 public:
  explicit CreateQueueOp(OpKernelConstruction* c) : ResourceOpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("max_size", &max_size_));
  }

  ~CreateQueueOp() override {}

 private:
  Status CreateResource(QueueResource** queue)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *queue = new QueueResource(max_size_);
    return Status::OK();
  }

  int max_size_;
};

REGISTER_OP("CreateQueue")
    .Output("handle: resource")
    .Attr("max_size: int")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(Name("CreateQueue").Device(DEVICE_CPU), CreateQueueOp);

}  // namespace monolith_tf
}  // namespace tensorflow
