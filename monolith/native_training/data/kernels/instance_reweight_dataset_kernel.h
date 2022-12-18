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

#ifndef MONOLITH_NATIVE_TRAINING_DATA_KERNELS_INSTANCE_REWEIGHT_DATASET_KERNEL_H_
#define MONOLITH_NATIVE_TRAINING_DATA_KERNELS_INSTANCE_REWEIGHT_DATASET_KERNEL_H_

#include "tensorflow/core/framework/dataset.h"

namespace tensorflow {
namespace data {
namespace monolith_tf {

class InstanceReweightDatasetOp : public UnaryDatasetOpKernel {
 public:
  static constexpr const char* const kDatasetType = "instance_reweight";
  static constexpr const char* const kInputDataset = "input_dataset";
  static constexpr const char* const kMethod = "method";
  static constexpr const char* const kActions = "actions";
  static constexpr const char* const kWeights = "weights";
  static constexpr const char* const kLabels = "labels";
  static constexpr const char* const kPriority = "priorities";
  static constexpr const char* const kVariantType = "variant_type";

  explicit InstanceReweightDatasetOp(OpKernelConstruction* ctx);

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override;

 private:
  class Dataset;

  int instance_reweight_method_;
  std::string variant_type_;
  std::vector<int32> actions_;
  std::vector<int32> weights_;
  std::vector<int32> labels_;
  std::vector<int32> priorities_;
};

}  // namespace monolith_tf
}  // namespace data
}  // namespace tensorflow
#endif MONOLITH_NATIVE_TRAINING_DATA_KERNELS_INSTANCE_REWEIGHT_DATASET_KERNEL_H_
