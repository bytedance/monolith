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

#include "monolith/native_training/runtime/hash_table/optimizer/group_adagrad_optimizer.h"

#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "monolith/native_training/runtime/hash_table/optimizer/optimizer.pb.h"
#include "monolith/native_training/runtime/hash_table/optimizer/test_utils.h"

namespace monolith {
namespace hash_table {
namespace {

using ::testing::ElementsAreArray;
using ::testing::FloatNear;
using ::testing::Pointwise;

TEST(GroupAdaGradOptimizer, Basic) {
  GroupAdaGradOptimizerConfig config;
  config.set_dim_size(1);
  config.set_l2_regularization_strength(1.0);
  config.set_beta(1.0);
  config.set_initial_accumulator_value(0.0);
  config.set_weight_decay_factor(0.0);
  auto opt = NewGroupAdaGradOptimizer(config);
  TestOptimizerEntry mem(opt.get());
  opt->Init(mem.mutable_ctx());
  opt->Optimize(mem.mutable_ctx(), mem.mutable_num_span(), {10.0f}, {0.01f});
  auto expected = {-0.008182f};
  ASSERT_THAT(mem.num(), Pointwise(FloatNear(1e-6), expected));

  // Test dump & restore
  OptimizerDump dump = opt->Save(mem.ctx());
  EXPECT_NEAR(dump.dump(0).group_adagrad().grad_square_sum(), 1e-6, 100.0);
  TestOptimizerEntry mem2(opt.get());
  opt->Restore(mem2.mutable_ctx(), dump);
  *mem2.mutable_num() = mem.num();
  opt->Optimize(mem2.mutable_ctx(), mem2.mutable_num_span(), {10.0f}, {0.01f});
  auto expected2 = {-0.014125f};
  ASSERT_THAT(mem2.num(), Pointwise(FloatNear(1e-6), expected2));
}

TEST(GroupAdaGradOptimizer, ListUpdate) {
  GroupAdaGradOptimizerConfig config;
  config.set_dim_size(2);
  config.set_l2_regularization_strength(0.5);
  config.set_beta(1.0);
  config.set_initial_accumulator_value(0.0);
  config.set_weight_decay_factor(0.0);
  auto opt = NewGroupAdaGradOptimizer(config);
  TestOptimizerEntry mem(opt.get());
  opt->Init(mem.mutable_ctx());
  opt->Optimize(mem.mutable_ctx(), mem.mutable_num_span(), {10.0f, 1.0f},
                {0.01f});
  auto expected = {-0.008639f, -0.000864f};
  ASSERT_THAT(mem.num(), Pointwise(FloatNear(1e-6), expected));

  // Test dump & restore
  OptimizerDump dump = opt->Save(mem.ctx());
  EXPECT_NEAR(dump.dump(0).group_adagrad().grad_square_sum(), 1e-6, 100.0);
  TestOptimizerEntry mem2(opt.get());
  opt->Restore(mem2.mutable_ctx(), dump);
  *mem2.mutable_num() = mem.num();
  opt->Optimize(mem2.mutable_ctx(), mem2.mutable_num_span(), {1.0f, 5.0f},
                {0.01f});
  auto expected2 = {-0.009096f, -0.004778f};
  ASSERT_THAT(mem2.num(), Pointwise(FloatNear(1e-6), expected2));
}

TEST(GroupAdaGradOptimizer, ZeroLambda) {
  GroupAdaGradOptimizerConfig config;
  config.set_dim_size(2);
  config.set_l2_regularization_strength(0);
  config.set_beta(1.0);
  config.set_initial_accumulator_value(0.0);
  config.set_weight_decay_factor(0.0);
  auto opt = NewGroupAdaGradOptimizer(config);
  TestOptimizerEntry mem(opt.get());
  opt->Init(mem.mutable_ctx());
  opt->Optimize(mem.mutable_ctx(), mem.mutable_num_span(), {10.0f, 1.0f},
                {0.01f});
  auto expected = {-0.009091f, -0.000909f};
  ASSERT_THAT(mem.num(), Pointwise(FloatNear(1e-6), expected));
}

TEST(GroupAdaGradOptimizer, SetZero) {
  GroupAdaGradOptimizerConfig config;
  config.set_dim_size(2);
  config.set_l2_regularization_strength(1000);
  config.set_beta(1.0);
  config.set_initial_accumulator_value(0.0);
  config.set_weight_decay_factor(0.0);
  auto opt = NewGroupAdaGradOptimizer(config);
  TestOptimizerEntry mem(opt.get());
  opt->Init(mem.mutable_ctx());
  opt->Optimize(mem.mutable_ctx(), mem.mutable_num_span(), {10.0f, 1.0f},
                {0.01f});
  auto expected = {0.0f, 0.0f};
  ASSERT_THAT(mem.num(), Pointwise(FloatNear(1e-6), expected));
}

}  // namespace
}  // namespace hash_table
}  // namespace monolith
