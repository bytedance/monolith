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

#include "monolith/native_training/data/kernels/internal/relational_utils.h"

#include <cstring>
#include <memory>
#include <thread>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace tensorflow {
namespace monolith_tf {
namespace internal {
namespace {

TEST(RelationalUtils, GT) {
  EXPECT_TRUE(compare(GT, 1, {0}));
  EXPECT_TRUE(!compare(GT, 1, {1}));
  EXPECT_TRUE(!compare(GT, -1, {0LL}));
  EXPECT_TRUE(compare(GT, std::string("1"), {"0"}));
  EXPECT_TRUE(!compare(GT, std::string("1"), {"1"}));
}

TEST(RelationalUtils, GE) {
  EXPECT_TRUE(compare(GE, 1, {0}));
  EXPECT_TRUE(compare(GE, 1, {1}));
  EXPECT_TRUE(!compare(GE, -1, {0LL}));
  EXPECT_TRUE(compare(GE, std::string("1"), {"0"}));
  EXPECT_TRUE(compare(GE, std::string("1"), {"1"}));
}

TEST(RelationalUtils, EQ) {
  EXPECT_TRUE(!compare(EQ, 1, {0}));
  EXPECT_TRUE(compare(EQ, 1, {1}));
  EXPECT_TRUE(!compare(EQ, -1, {0LL}));
  EXPECT_TRUE(!compare(EQ, std::string("1"), {"0"}));
  EXPECT_TRUE(compare(EQ, std::string("1"), {"1"}));
}

TEST(RelationalUtils, LT) {
  EXPECT_TRUE(!compare(LT, 1, {0}));
  EXPECT_TRUE(!compare(LT, 1, {1}));
  EXPECT_TRUE(compare(LT, -1, {0LL}));
  EXPECT_TRUE(!compare(LT, std::string("1"), {"0"}));
  EXPECT_TRUE(!compare(LT, std::string("1"), {"1"}));
}

TEST(RelationalUtils, LE) {
  EXPECT_TRUE(!compare(LE, 1, {0}));
  EXPECT_TRUE(compare(LE, 1, {1}));
  EXPECT_TRUE(compare(LE, -1, {0LL}));
  EXPECT_TRUE(!compare(LE, std::string("1"), {"0"}));
  EXPECT_TRUE(compare(LE, std::string("1"), {"1"}));
}

TEST(RelationalUtils, NEQ) {
  EXPECT_TRUE(compare(NEQ, 1, {0}));
  EXPECT_TRUE(!compare(NEQ, 1, {1}));
  EXPECT_TRUE(compare(NEQ, -1, {0LL}));
  EXPECT_TRUE(compare(NEQ, std::string("1"), {"0"}));
  EXPECT_TRUE(!compare(NEQ, std::string("1"), {"1"}));
}

TEST(RelationalUtils, BETWEEN) {
  EXPECT_TRUE(!compare(BETWEEN, 1, {0, 1}));
  EXPECT_TRUE(compare(BETWEEN, 1, {1, 2}));
  EXPECT_TRUE(!compare(BETWEEN, -1, {0LL, 1LL}));
  EXPECT_TRUE(!compare(BETWEEN, std::string("1"), {"0", "1"}));
  EXPECT_TRUE(compare(BETWEEN, std::string("1"), {"1", "2"}));
}

TEST(RelationalUtils, IN) {
  EXPECT_TRUE(!contains(IN, 1, {0}));
  EXPECT_TRUE(contains(IN, 1, {1, 2}));
  EXPECT_TRUE(!contains(IN, -1, {0LL, 1LL}));
  EXPECT_TRUE(!contains(IN, std::string("1"), {"0"}));
  EXPECT_TRUE(contains(IN, std::string("1"), {"1", "2"}));
}

TEST(RelationalUtils, NOT_IN) {
  EXPECT_TRUE(contains(NOT_IN, 1, {0}));
  EXPECT_TRUE(!contains(NOT_IN, 1, {1, 2}));
  EXPECT_TRUE(contains(NOT_IN, -1, {0LL, 1LL}));
  EXPECT_TRUE(contains(NOT_IN, std::string("1"), {"0"}));
  EXPECT_TRUE(!contains(NOT_IN, std::string("1"), {"1", "2"}));
}

}  // namespace
}  // namespace internal
}  // namespace monolith_tf
}  // namespace tensorflow
