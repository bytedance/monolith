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

#include "monolith/native_training/data/kernels/feature_name_mapper_tf_bridge.h"

namespace tensorflow {
namespace monolith_tf {

Status FeatureNameMapperTfBridge::New(FeatureNameMapperTfBridge** new_bridge) {
  auto bridge = core::RefCountPtr<FeatureNameMapperTfBridge>(
      new FeatureNameMapperTfBridge());
  bridge->mapper_ = std::make_unique<FeatureNameMapper>();
  *new_bridge = bridge.release();
  return Status::OK();
}

Status FeatureNameMapperTfBridge::RegisterValidIds(
    const std::vector<std::pair<int, int>>& valid_ids) const {
  try {
    if (mapper_->RegisterValidIds(valid_ids)) {
      return Status::OK();
    } else {
      return errors::InvalidArgument("RegisterValidIds failed!");
    }
  } catch (const std::exception& e) {
    return errors::InvalidArgument(e.what());
  }
}

}  // namespace monolith_tf
}  // namespace tensorflow
