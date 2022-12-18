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

#include <cstring>

#include "monolith/native_training/runtime/hash_table/initializer/constants_initializer.h"

namespace monolith {
namespace hash_table {
namespace {

class ConstantsInitializer : public InitializerInterface {
 public:
  explicit ConstantsInitializer(int dim_size, float constant)
      : dim_size_(dim_size), constant_(constant) {}

  int DimSize() const override { return dim_size_; }

  void Initialize(absl::Span<float> nums) const override {
    for (int i = 0; i < dim_size_; ++i) {
      nums[i] = constant_;
    }
  }

 private:
  int dim_size_;
  float constant_;
};

}  // namespace

std::unique_ptr<InitializerInterface> NewZerosInitializer(
    ZerosInitializerConfig config) {
  return std::make_unique<ConstantsInitializer>(config.dim_size(), 0);
}

std::unique_ptr<InitializerInterface> NewZerosInitializer(int dim_size) {
  return std::make_unique<ConstantsInitializer>(dim_size, 0);
}

std::unique_ptr<InitializerInterface> NewOnesInitializer(
    OnesInitializerConfig config) {
  return std::make_unique<ConstantsInitializer>(config.dim_size(), 1);
}

std::unique_ptr<InitializerInterface> NewConstantsInitializer(
    ConstantsInitializerConfig config) {
  return std::make_unique<ConstantsInitializer>(config.dim_size(),
                                                config.constant());
}

std::unique_ptr<InitializerInterface> NewConstantsInitializer(int dim_size,
                                                              float constant) {
  return std::make_unique<ConstantsInitializer>(dim_size, constant);
}

}  // namespace hash_table
}  // namespace monolith
