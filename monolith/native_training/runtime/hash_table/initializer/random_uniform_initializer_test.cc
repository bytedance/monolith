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

#include <algorithm>

#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "monolith/native_training/runtime/hash_table/initializer/initializer_config.pb.h"

namespace monolith {
namespace hash_table {
namespace {

using ::testing::Gt;
using ::testing::Lt;

TEST(RandomUniformInitializer, Basic) {
  const int kDimSize = 1000;
  std::vector<float> num(kDimSize, 0);
  RandomUniformInitializerConfig config;
  config.set_dim_size(kDimSize);
  config.set_minval(-1);
  config.set_maxval(1);
  auto initializer = NewRandomUniformInitializer(config);
  initializer->Initialize(absl::Span<float>(num));
  EXPECT_THAT(*std::max_element(num.begin(), num.end()), Gt(0.9));
  EXPECT_THAT(*std::min_element(num.begin(), num.end()), Lt(0.9));
}

}  // namespace
}  // namespace hash_table
}  // namespace monolith
