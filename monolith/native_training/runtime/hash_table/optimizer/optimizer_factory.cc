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

#include <exception>

#include "monolith/native_training/runtime/hash_table/optimizer/optimizer_factory.h"

#include "absl/strings/str_format.h"
#include "monolith/native_training/runtime/hash_table/optimizer/adadelta_optimizer.h"
#include "monolith/native_training/runtime/hash_table/optimizer/adagrad_optimizer.h"
#include "monolith/native_training/runtime/hash_table/optimizer/adam_optimizer.h"
#include "monolith/native_training/runtime/hash_table/optimizer/amsgrad_optimizer.h"
#include "monolith/native_training/runtime/hash_table/optimizer/batch_softmax_optimizer.h"
#include "monolith/native_training/runtime/hash_table/optimizer/dynamic_wd_adagrad_optimizer.h"
#include "monolith/native_training/runtime/hash_table/optimizer/ftrl_optimizer.h"
#include "monolith/native_training/runtime/hash_table/optimizer/group_ftrl_optimizer.h"
#include "monolith/native_training/runtime/hash_table/optimizer/momentum_optimizer.h"
#include "monolith/native_training/runtime/hash_table/optimizer/moving_average_optimizer.h"
#include "monolith/native_training/runtime/hash_table/optimizer/rmsprop_optimizer.h"
#include "monolith/native_training/runtime/hash_table/optimizer/sgd_optimizer.h"
#include "monolith/native_training/runtime/hash_table/optimizer/stochastic_rounding.h"

namespace monolith {
namespace hash_table {

std::unique_ptr<OptimizerInterface> NewOptimizerFromConfig(
    OptimizerConfig config) {
  std::unique_ptr<OptimizerInterface> opt = nullptr;
  switch (config.type_case()) {
    case OptimizerConfig::kAdagrad:
      opt = NewAdagradOptimizer(std::move(*config.mutable_adagrad()));
      break;
    case OptimizerConfig::kSgd:
      opt = NewSgdOptimizer(std::move(*config.mutable_sgd()));
      break;
    case OptimizerConfig::kFtrl:
      opt = NewFtrlOptimizer(std::move(*config.mutable_ftrl()));
      break;
    case OptimizerConfig::kDynamicWdAdagrad:
      opt = NewDynamicWdAdagradOptimizer(
          std::move(*config.mutable_dynamic_wd_adagrad()));
      break;
    case OptimizerConfig::kAdadelta:
      opt = NewAdadeltaOptimizer(std::move(*config.mutable_adadelta()));
      break;
    case OptimizerConfig::kAdam:
      opt = NewAdamOptimizer(std::move(*config.mutable_adam()));
      break;
    case OptimizerConfig::kAmsgrad:
      opt = NewAmsgradOptimizer(std::move(*config.mutable_amsgrad()));
      break;
    case OptimizerConfig::kMomentum:
      opt = NewMomentumOptimizer(std::move(*config.mutable_momentum()));
      break;
    case OptimizerConfig::kMovingAverage:
      opt = NewMovingAverageOptimizer(
          std::move(*config.mutable_moving_average()));
      break;
    case OptimizerConfig::kRmsprop:
      opt = NewRmspropOptimizer(std::move(*config.mutable_rmsprop()));
      break;
    case OptimizerConfig::kRmspropv2:
      opt = NewRmspropV2Optimizer(std::move(*config.mutable_rmspropv2()));
      break;
    case OptimizerConfig::kGroupFtrl:
      opt = NewGroupFtrlOptimizer(std::move(*config.mutable_group_ftrl()));
      break;
    case OptimizerConfig::kBatchSoftmax:
      opt =
          NewBatchSoftmaxOptimizer(std::move(*config.mutable_batch_softmax()));
      break;
    default:
      throw std::invalid_argument(absl::StrFormat(
          "optimizer is not implemented yet. %s", config.ShortDebugString()));
  }
  if (config.stochastic_rounding_float16()) {
    opt = std::make_unique<StochasticRoundingFloat16OptimizerDecorator>(
        std::move(opt));
  }
  return std::move(opt);
}

}  // namespace hash_table
}  // namespace monolith
