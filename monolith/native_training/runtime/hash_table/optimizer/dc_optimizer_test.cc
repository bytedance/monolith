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

#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "monolith/native_training/runtime/hash_table/optimizer/optimizer.pb.h"
#include "monolith/native_training/runtime/hash_table/optimizer/test_utils.h"

#include "monolith/native_training/runtime/hash_table/optimizer/dc_optimizer.h"
#include "monolith/native_training/runtime/hash_table/optimizer/adadelta_optimizer.h"

namespace monolith {
namespace hash_table {
namespace {

using ::testing::Pointwise;
using ::testing::FloatNear;
using ::testing::ElementsAreArray;

TEST(DcOptimizer, Basic) {
  AdadeltaOptimizerConfig config1;
  config1.set_dim_size(1);
  auto opt1 = NewAdadeltaOptimizer(config1);
  DcOptimizerConfig config2;
  config2.set_dim_size(1);
  config2.set_lambda_(0.1f);
  auto opt2 = NewDcOptimizer(config2, std::move(opt1));
  TestOptimizerEntry mem(opt2.get());
  opt2->Init(mem.mutable_ctx());
  float arr[] = {0.1f};
  opt2->OptimizeWithLatestValue(mem.mutable_ctx(), mem.mutable_num_span(), {10.0f}, {0.01f}, arr);
  auto expected = {-0.0031603f};
  ASSERT_THAT(mem.num(), Pointwise(FloatNear(1e-6), expected));

  // Test dump & restore
  OptimizerDump dump = opt2->Save(mem.ctx());
  EXPECT_NEAR(dump.dump(0).adadelta().accum(0), 8.1, 1e-4);
  TestOptimizerEntry mem2(opt2.get());
  opt2->Restore(mem2.mutable_ctx(), dump);
  *mem2.mutable_num() = mem.num();
  arr[0] = 0.0f;
  opt2->OptimizeWithLatestValue(mem.mutable_ctx(), mem.mutable_num_span(), {10.0f}, {0.01f}, arr);
  auto expected2 = {-0.0065548f};
  ASSERT_THAT(mem.num(), Pointwise(FloatNear(1e-6), expected2));
}

TEST(DcOptimizer, ListUpdate) {
  AdadeltaOptimizerConfig config1;
  config1.set_dim_size(2);
  auto opt1 = NewAdadeltaOptimizer(config1);
  DcOptimizerConfig config2;
  config2.set_dim_size(2);
  config2.set_lambda_(0.1f);
  auto opt2 = NewDcOptimizer(config2, std::move(opt1));
  TestOptimizerEntry mem(opt2.get());
  opt2->Init(mem.mutable_ctx());
  float arr[] = {0.1f, 0.1f};
  opt2->OptimizeWithLatestValue(mem.mutable_ctx(), mem.mutable_num_span(), {10.0f, 1.0f}, {0.01f}, arr);
  auto expected = {-0.0031603f, -0.00301233f};
  ASSERT_THAT(mem.num(), Pointwise(FloatNear(1e-6), expected));

  // Test dump & restore
  OptimizerDump dump = opt2->Save(mem.ctx());
  EXPECT_NEAR(dump.dump(0).adadelta().accum(0), 8.1, 1e-4);
  EXPECT_NEAR(dump.dump(0).adadelta().accum(1), .09801, 1e-4);
  TestOptimizerEntry mem2(opt2.get());
  opt2->Restore(mem2.mutable_ctx(), dump);
  *mem2.mutable_num() = mem.num();
  arr[0] = 0.0f;
  arr[1] = 0.0f;
  opt2->OptimizeWithLatestValue(mem.mutable_ctx(), mem.mutable_num_span(), {10.0f, 1.0f}, {0.01f}, arr);
  auto expected2 = {-0.0065548f, -0.00611400f};
  ASSERT_THAT(mem.num(), Pointwise(FloatNear(1e-6), expected2));
}

}  // namespace
}  // namespace hash_table
}  // namespace monolith
