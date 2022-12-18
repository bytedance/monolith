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

#include "monolith/native_training/runtime/hash_table/optimizer/amsgrad_optimizer.h"

#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "monolith/native_training/runtime/hash_table/optimizer/optimizer.pb.h"
#include "monolith/native_training/runtime/hash_table/optimizer/test_utils.h"

namespace monolith {
namespace hash_table {
namespace {

using ::testing::Pointwise;
using ::testing::FloatNear;
using ::testing::ElementsAreArray;

TEST(AmsgradOptimizer, Basic) {
  AmsgradOptimizerConfig config;
  config.set_dim_size(1);
  auto opt = NewAmsgradOptimizer(config);
  TestOptimizerEntry mem(opt.get());
  opt->Init(mem.mutable_ctx());
  opt->Optimize(mem.mutable_ctx(), mem.mutable_num_span(), {10.0f}, {0.01f});
  auto expected = {-0.00990099f};
  ASSERT_THAT(mem.num(), Pointwise(FloatNear(1e-6), expected));

  // Test dump & restore
  OptimizerDump dump = opt->Save(mem.ctx());
  EXPECT_NEAR(dump.dump(0).amsgrad().m(0), 1, 1e-4);
  TestOptimizerEntry mem2(opt.get());
  opt->Restore(mem2.mutable_ctx(), dump);
  *mem2.mutable_num() = mem.num();
  opt->Optimize(mem.mutable_ctx(), mem.mutable_num_span(), {10.0f}, {0.01f});
  auto expected2 = {-0.01983060f};
  ASSERT_THAT(mem.num(), Pointwise(FloatNear(1e-6), expected2));
}

TEST(AmsgradOptimizer, ListUpdate) {
  AmsgradOptimizerConfig config;
  config.set_dim_size(2);
  auto opt = NewAmsgradOptimizer(config);
  TestOptimizerEntry mem(opt.get());
  opt->Init(mem.mutable_ctx());
  opt->Optimize(mem.mutable_ctx(), mem.mutable_num_span(), {10.0f, 1.0f}, {0.01f});
  auto expected = {-0.00990099f, -0.00909091f};
  ASSERT_THAT(mem.num(), Pointwise(FloatNear(1e-6), expected));

  // Test dump & restore
  OptimizerDump dump = opt->Save(mem.ctx());
  EXPECT_NEAR(dump.dump(0).amsgrad().m(0), 1, 1e-4);
  EXPECT_NEAR(dump.dump(0).amsgrad().m(1), .1, 1e-4);
  TestOptimizerEntry mem2(opt.get());
  opt->Restore(mem2.mutable_ctx(), dump);
  *mem2.mutable_num() = mem.num();
  opt->Optimize(mem.mutable_ctx(), mem.mutable_num_span(), {10.0f, 1.0f}, {0.01f});
  auto expected2 = {-0.01983060f, -0.01842895f};
  ASSERT_THAT(mem.num(), Pointwise(FloatNear(1e-6), expected2));
}

}  // namespace
}  // namespace hash_table
}  // namespace monolith
