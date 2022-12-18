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

#ifndef MONOLITH_NATIVE_TRAINING_RUNTIME_OPS_DEEP_INSIGHT_CLIENT_TF_BRIDGE
#define MONOLITH_NATIVE_TRAINING_RUNTIME_OPS_DEEP_INSIGHT_CLIENT_TF_BRIDGE

#include <string>

#include "absl/strings/str_format.h"
#include "monolith/native_training/runtime/deep_insight/deep_insight.h"
#include "tensorflow/core/framework/resource_mgr.h"

using monolith::deep_insight::ExtraField;

namespace tensorflow {
namespace monolith_tf {

class DeepInsightClientTfBridge : public ResourceBase {
 public:
  explicit DeepInsightClientTfBridge(
      std::unique_ptr<monolith::deep_insight::DeepInsight> deep_insight_client)
      : deep_insight_client_(std::move(deep_insight_client)) {}

  std::string SendV2(
      const std::string& model_name, const std::vector<std::string>& targets,
      uint64_t uid, int64_t req_time, int64_t train_time,
      const std::vector<float>& labels, const std::vector<float>& preds,
      const std::vector<float>& sample_rates, float sample_ratio,
      const std::vector<std::shared_ptr<ExtraField>>& extra_fields,
      bool return_msgs) {
    return deep_insight_client_->SendV2(
        model_name, targets, uid, req_time, train_time, labels, preds,
        sample_rates, sample_ratio, extra_fields, return_msgs);
  }

  int64_t GenerateTrainingTime() {
    return deep_insight_client_->GenerateTrainingTime();
  }

  uint64_t GetTotalSendCounter() {
    return deep_insight_client_->GetTotalSendCounter();
  }

  std::string DebugString() const override {
    return absl::StrFormat("Total send counter: %d",
                           deep_insight_client_->GetTotalSendCounter());
  }

 private:
  std::unique_ptr<monolith::deep_insight::DeepInsight> deep_insight_client_;
};

}  // namespace monolith_tf
}  // namespace tensorflow
#endif  // MONOLITH_NATIVE_TRAINING_RUNTIME_OPS_DEEP_INSIGHT_CLIENT_TF_BRIDGE
