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
#include "monolith/native_training/runtime/hash_table/optimizer/optimizer_factory.h"
#include "monolith/native_training/runtime/hash_table/optimizer/stochastic_rounding.h"
#include "monolith/native_training/runtime/hash_table/optimizer/test_utils.h"

namespace monolith {
namespace hash_table {
namespace {

TEST(StochasticRoundingFloat16OptimizerDecorator, Basic) {
  OptimizerConfig config;
  config.mutable_sgd()->set_dim_size(1);

  // Float32 optimizer.
  // By default, config.stochastic_rounding_float16() == false
  auto opt = NewOptimizerFromConfig(config);

  // Float16 optimizer.
  config.set_stochastic_rounding_float16(true);
  auto opt_float16 = NewOptimizerFromConfig(config);

  EXPECT_NE(typeid(*(opt.get())), typeid(*(opt_float16.get())));
}

}  // namespace
}  // namespace hash_table
}  // namespace monolith
