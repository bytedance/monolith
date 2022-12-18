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

#ifndef MONOLITH_NATIVE_TRAINING_DATA_KERNELS_NEGATIVE_GEN_DATASET_KERNEL_H_
#define MONOLITH_NATIVE_TRAINING_DATA_KERNELS_NEGATIVE_GEN_DATASET_KERNEL_H_

#include "monolith/native_training/data/kernels/feature_name_mapper_tf_bridge.h"
#include "tensorflow/core/framework/dataset.h"

namespace tensorflow {
namespace data {
namespace monolith_tf {

enum class VariantType { PBInstance, PBExample };

class InstanceNegativeGenDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit InstanceNegativeGenDatasetOp(OpKernelConstruction* ctx);

  ~InstanceNegativeGenDatasetOp() override { mapper_->Unref(); }

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override;

 private:
  class Dataset;
  int32 neg_num_;
  bool per_channel_;
  std::string channel_feature_;
  std::vector<std::string> item_features_;
  int32 label_index_;
  int32 positive_label_;
  int32 negative_label_;
  int32 negative_action_;
  std::string action_priority_;
  std::vector<int32> positive_actions_;
  std::string index_feature_;
  bool throw_origin_;
  bool throw_origin_neg_;
  bool cache_only_pos_;
  float real_neg_instance_weight_ = 1.0;
  float sampled_neg_instance_weight_ = -1;
  bool unbias_sampled_neg_;
  float origin_neg_in_pool_proba_;
  float neg_sample_declay_factor_;
  float hard_easy_ratio_;
  std::string variant_type_;
  tensorflow::monolith_tf::FeatureNameMapperTfBridge* mapper_ = nullptr;
};

}  // namespace monolith_tf
}  // namespace data
}  // namespace tensorflow
#endif  // MONOLITH_NATIVE_TRAINING_DATA_KERNELS_NEGATIVE_GEN_DATASET_KERNEL_H_
