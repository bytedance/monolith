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

#include "monolith/native_training/runtime/hash_table/optimizer/batch_softmax_optimizer.h"

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

TEST(BatchSoftmaxOptimizer, Basic) {
  BatchSoftmaxOptimizerConfig config;
  config.set_dim_size(1);
  auto opt = NewBatchSoftmaxOptimizer(config);
  TestOptimizerEntry mem(opt.get());
  int64_t global_step = 1;
  opt->Init(mem.mutable_ctx());
  opt->Optimize(mem.mutable_ctx(), mem.mutable_num_span(), {2.0f}, {0.1f},
                global_step);
  EXPECT_FLOAT_EQ(mem.num().front(), 0.1f);

  // Test dump & restore
  OptimizerDump dump = opt->Save(mem.ctx());
  TestOptimizerEntry mem2(opt.get());
  opt->Restore(mem2.mutable_ctx(), dump);
  *mem2.mutable_num() = mem.num();
  opt->Optimize(mem.mutable_ctx(), mem.mutable_num_span(), {2.0f}, {0.1f});
  opt->Optimize(mem2.mutable_ctx(), mem2.mutable_num_span(), {2.0f}, {0.1f});
  EXPECT_THAT(mem.num(), ElementsAreArray(mem2.num()));
}

}  // namespace
}  // namespace hash_table
}  // namespace monolith
