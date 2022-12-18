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

#include "monolith/native_training/runtime/hash_table/optimizer/adagrad_optimizer.h"

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

TEST(AdagradOptimizer, Basic) {
  AdagradOptimizerConfig config;
  config.set_dim_size(2);
  config.set_initial_accumulator_value(1.0f);
  auto opt = NewAdagradOptimizer(config);
  TestOptimizerEntry mem(opt.get());
  opt->Init(mem.mutable_ctx());
  opt->Optimize(mem.mutable_ctx(), mem.mutable_num_span(), {1.0f, 2.0f},
                {0.1f});
  auto expected = {-0.07071067, -0.08944272};
  ASSERT_THAT(mem.num(), Pointwise(FloatNear(1e-6), expected));

  // Test dump & restore
  OptimizerDump dump = opt->Save(mem.ctx());
  TestOptimizerEntry mem2(opt.get());
  opt->Restore(mem2.mutable_ctx(), dump);
  *mem2.mutable_num() = mem.num();
  opt->Optimize(mem.mutable_ctx(), mem.mutable_num_span(), {1.0f, 1.0f},
                {0.1f});
  opt->Optimize(mem2.mutable_ctx(), mem2.mutable_num_span(), {1.0f, 1.0f},
                {0.1f});
  EXPECT_THAT(mem.num(), ElementsAreArray(mem2.num()));
}

TEST(AdagradOptimizer, OptimizeWithWeightDecay) {
  AdagradOptimizerConfig config;
  config.set_dim_size(2);
  config.set_initial_accumulator_value(1.0f);
  config.set_weight_decay_factor(0.1f);

  auto opt = NewAdagradOptimizer(config);
  TestOptimizerEntry mem(opt.get());
  opt->Init(mem.mutable_ctx());
  opt->Optimize(mem.mutable_ctx(), mem.mutable_num_span(), {1.0f, 2.0f},
                {0.1f});
  auto expected = {-0.07071067, -0.08944272};
  ASSERT_THAT(mem.num(), Pointwise(FloatNear(1e-6), expected));

  opt->Optimize(mem.mutable_ctx(), mem.mutable_num_span(), {1.0f, 2.0f},
                {0.1f});
  auto expected2 = {-0.128173, -0.155943};
  ASSERT_THAT(mem.num(), Pointwise(FloatNear(1e-6), expected2));

  // Test dump & restore
  OptimizerDump dump = opt->Save(mem.ctx());
  TestOptimizerEntry mem2(opt.get());
  opt->Restore(mem2.mutable_ctx(), dump);
  *mem2.mutable_num() = mem.num();
  opt->Optimize(mem.mutable_ctx(), mem.mutable_num_span(), {1.0f, 1.0f},
                {0.1f});
  opt->Optimize(mem2.mutable_ctx(), mem2.mutable_num_span(), {1.0f, 1.0f},
                {0.1f});
  EXPECT_THAT(mem.num(), ElementsAreArray(mem2.num()));
}


}  // namespace
}  // namespace hash_table
}  // namespace monolith
