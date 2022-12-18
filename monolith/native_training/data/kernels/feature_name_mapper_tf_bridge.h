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

#ifndef MONOLITH_MONOLITH_NATIVE_TRAINING_DATA_KERNELS_FEATURE_NAME_MAPPER_TF_BRIDGE_H_
#define MONOLITH_MONOLITH_NATIVE_TRAINING_DATA_KERNELS_FEATURE_NAME_MAPPER_TF_BRIDGE_H_

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#include "monolith/native_training/data/training_instance/cc/reader_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace monolith_tf {

// A feature name mapper which can be used in TF runtime.
// It captures all potential exceptions and convert them into error.
class FeatureNameMapperTfBridge : public ResourceBase {
 public:
  static constexpr const char* const kName = "FeatureNameMapper";

  ~FeatureNameMapperTfBridge() override = default;

  static Status New(FeatureNameMapperTfBridge** new_bridge);

  Status RegisterValidIds(
      const std::vector<std::pair<int, int>>& valid_ids) const;

  std::string DebugString() const override { return mapper_->DebugString(); }

  FeatureNameMapper* GetFeatureNameMapper() const { return mapper_.get(); }

 private:
  FeatureNameMapperTfBridge() = default;

  std::unique_ptr<FeatureNameMapper> mapper_;
};

}  // namespace monolith_tf
}  // namespace tensorflow

#endif  // MONOLITH_MONOLITH_NATIVE_TRAINING_DATA_KERNELS_FEATURE_NAME_MAPPER_TF_BRIDGE_H_
