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

#include "monolith/native_training/runtime/common/linalg_utils.h"

#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace monolith {
namespace common {

TEST(LinalgUtils, IsAlmostEqual) {
  EXPECT_TRUE(IsAlmostEqual(0.f, 0.f));
  EXPECT_FALSE(IsAlmostEqual(0.f, 1e-6f));
}

TEST(LinalgUtils, L2NormSquare) {
  std::vector<float> vec1 = {};
  EXPECT_TRUE(IsAlmostEqual(L2NormSquare(vec1.data(), vec1.size()), 0.f));

  std::vector<float> vec2 = {1};
  EXPECT_TRUE(IsAlmostEqual(L2NormSquare(vec2.data(), vec2.size()), 1.f));

  std::vector<float> vec3 = {1, 2, 3, 4};
  EXPECT_TRUE(IsAlmostEqual(L2NormSquare(vec3.data(), vec3.size()), 30.f));
}

}  // namespace common
}  // namespace monolith
