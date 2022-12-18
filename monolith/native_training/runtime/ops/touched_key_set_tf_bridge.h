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

#ifndef MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_OPS_TOUCHED_KEY_SET_TF_BRIDGE_H_
#define MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_OPS_TOUCHED_KEY_SET_TF_BRIDGE_H_

#include <memory>

#include "absl/strings/str_format.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"

#include "monolith/native_training/runtime/hopscotch/hopscotch_hash_set.h"

namespace tensorflow {
namespace monolith_tf {

class TouchedKeySetTfBridge : public ResourceBase {
 public:
  explicit TouchedKeySetTfBridge(std::unique_ptr<monolith::hopscotch::HopscotchHashSet<
      int64_t>> touched_key_set)
      : touched_key_set_(std::move(touched_key_set)) {}

  size_t Insert(int64_t key) {
    return touched_key_set_->insert(key);
  }

  std::vector<int64_t> Steal() {
    return touched_key_set_->GetAndClear();
  }

  size_t Size() const {
    return touched_key_set_->size();
  }

  std::string DebugString() const override {
    return absl::StrFormat("TouchedKeySet with capacity: %d",
                           touched_key_set_->capacity());
  }

 private:
  std::unique_ptr<monolith::hopscotch::HopscotchHashSet<int64_t>>
      touched_key_set_;
};

}  // namespace monolith_tf
}  // namespace tensorflow

#endif  // MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_OPS_TOUCHED_KEY_SET_TF_BRIDGE_H_
