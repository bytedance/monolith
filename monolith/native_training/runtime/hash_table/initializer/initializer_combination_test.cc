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

#include <algorithm>

#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "monolith/native_training/runtime/hash_table/initializer/constants_initializer.h"

namespace monolith {
namespace hash_table {
namespace {

using ::testing::ElementsAre;

TEST(RandomUniformInitializer, Basic) {
  std::vector<float> num(3, 1);
  auto init1 = NewConstantsInitializer(1, 3);
  auto init2 = NewConstantsInitializer(2, 4);
  auto combined_init = CombineInitializers(std::move(init1), std::move(init2));
  EXPECT_THAT(combined_init->DimSize(), 3);
  combined_init->Initialize(absl::Span<float>(num));
  EXPECT_THAT(num, ElementsAre(3, 4, 4));
}

}  // namespace
}  // namespace hash_table
}  // namespace monolith
