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

#include "monolith/native_training/runtime/hash_table/optimizer/optimizer_combination.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "monolith/native_training/runtime/hash_table/optimizer/adagrad_optimizer.h"
#include "monolith/native_training/runtime/hash_table/optimizer/optimizer.pb.h"
#include "monolith/native_training/runtime/hash_table/optimizer/test_utils.h"

namespace monolith {
namespace hash_table {
namespace {

using ::testing::Pointwise;
using ::testing::FloatNear;

TEST(CombineOptimizers, Basic) {
  AdagradOptimizerConfig config1;
  config1.set_dim_size(1);
  config1.set_initial_accumulator_value(1);
  auto opt1 = NewAdagradOptimizer(config1);

  AdagradOptimizerConfig config2;
  config2.set_dim_size(2);
  config2.set_initial_accumulator_value(2);
  auto opt2 = NewAdagradOptimizer(config2);

  auto combined_opt = CombineOptimizers(std::move(opt1), std::move(opt2));
  OptimizerDump dump;
  {
    TestOptimizerEntry mem(combined_opt.get());
    combined_opt->Init(mem.mutable_ctx());
    combined_opt->Optimize(mem.mutable_ctx(), mem.mutable_num_span(),
                           {1.0f, 2.0f, 3.0f}, {1.0f, 2.0f});
    auto expected = {-0.70710677, -1.6329931, -1.8090681};
    EXPECT_THAT(mem.num(), Pointwise(FloatNear(1e-6), expected));
    dump = combined_opt->Save(mem.ctx());
  }
  TestOptimizerEntry mem(combined_opt.get());
  combined_opt->Restore(mem.mutable_ctx(), dump);
  combined_opt->Optimize(mem.mutable_ctx(), mem.mutable_num_span(),
                         {1.0f, 2.0f, 3.0f}, {1.0f, 2.0f});
  auto expected = {-0.57735026, -1.264911, -1.3416407};
  EXPECT_THAT(mem.num(), Pointwise(FloatNear(1e-6), expected));
}

}  // namespace
}  // namespace hash_table
}  // namespace monolith
