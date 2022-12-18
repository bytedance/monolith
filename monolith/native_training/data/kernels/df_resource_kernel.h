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

#ifndef MONOLITH_NATIVE_TRAINING_DATA_KERNELS_DF_RESOURCE_KERNEL_H_
#define MONOLITH_NATIVE_TRAINING_DATA_KERNELS_DF_RESOURCE_KERNEL_H_

#include <chrono>
#include <thread>
#include "absl/synchronization/mutex.h"
#include "monolith/native_training/runtime/concurrency/queue.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"

namespace tensorflow {
namespace monolith_tf {

enum class VariantType { PBInstance, PBExample };

typedef struct {
  std::vector<Tensor> out_tensors;
  bool end_of_sequence;
} Item;

// It is a thin wrapper of GFile. Make it compatible with ResourceKernelOp
// and thread safe.
class QueueResource : public ResourceBase {
 public:
  explicit QueueResource(size_t max_size = 100) {
    queue_ = std::make_unique<::monolith::concurrency::Queue<Item>>(max_size);
  }

  ~QueueResource() = default;

  std::string DebugString() const override { return "QueueResource"; }

  void Push(const Item &item) {
    bool pushed = false;
    do {
      pushed = queue_->try_push(item, std::chrono::milliseconds(100));
      if (!pushed) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    } while (!pushed);
  }

  bool TryPush(const Item &item, int64_t timeout = 100) {
    return queue_->try_push(item, std::chrono::milliseconds(timeout));
  }

  Item Pop() const {
    bool poped = false;
    Item item;
    do {
      poped = queue_->try_pop(item, std::chrono::milliseconds(10));
      if (!poped) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    } while (!poped);

    return item;
  }

  bool TryPop(Item &item, int64_t timeout = 100) const {
    return queue_->try_pop(item, std::chrono::milliseconds(timeout));
  }

  bool Empty() const { return queue_->empty(); }

 private:
  mutable std::unique_ptr<::monolith::concurrency::Queue<Item>> queue_;
};

Status RegisterCancellationCallback(CancellationManager *cancellation_manager,
                                    CancelCallback callback,
                                    std::function<void()> *deregister_fn);
}  // namespace monolith_tf
}  // namespace tensorflow
#endif  // MONOLITH_NATIVE_TRAINING_DATA_KERNELS_DF_RESOURCE_KERNEL_H_
