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


#include "monolith/native_training/runtime/hash_table/initializer/initializer_combination.h"

namespace monolith {
namespace hash_table {
namespace {

class CombinedInitializer : public InitializerInterface {
 public:
  CombinedInitializer(std::unique_ptr<InitializerInterface> init1,
                      std::unique_ptr<InitializerInterface> init2)
      : init1_(std::move(init1)), init2_(std::move(init2)) {}

  int DimSize() const override { return init1_->DimSize() + init2_->DimSize(); }

  void Initialize(absl::Span<float> nums) const override {
    init1_->Initialize(nums);
    init2_->Initialize(nums.subspan(init1_->DimSize()));
  }

 private:
  std::unique_ptr<InitializerInterface> init1_;
  std::unique_ptr<InitializerInterface> init2_;
};

}  // namespace

std::unique_ptr<InitializerInterface> CombineInitializers(
    std::unique_ptr<InitializerInterface> init1,
    std::unique_ptr<InitializerInterface> init2) {
  return std::make_unique<CombinedInitializer>(std::move(init1),
                                               std::move(init2));
}

}  // namespace hash_table
}  // namespace monolith
