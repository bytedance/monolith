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

#include "monolith/native_training/runtime/hash_table/initializer/random_uniform_initializer.h"

#include <random>

namespace monolith {
namespace hash_table {
namespace {

class RandomUniformInitializer : public InitializerInterface {
 public:
  explicit RandomUniformInitializer(RandomUniformInitializerConfig conf)
      : conf_(std::move(conf)) {}

  int DimSize() const override { return conf_.dim_size(); }

  void Initialize(absl::Span<float> nums) const override {
    thread_local std::mt19937 generator;
    std::uniform_real_distribution<float> distribution(conf_.minval(),
                                                       conf_.maxval());
    for (int i = 0; i < conf_.dim_size(); ++i) {
      nums[i] = distribution(generator);
    }
  }

 private:
  RandomUniformInitializerConfig conf_;
};

}  // namespace

std::unique_ptr<InitializerInterface> NewRandomUniformInitializer(
    RandomUniformInitializerConfig config) {
  return std::make_unique<RandomUniformInitializer>(std::move(config));
}

}  // namespace hash_table
}  // namespace monolith
