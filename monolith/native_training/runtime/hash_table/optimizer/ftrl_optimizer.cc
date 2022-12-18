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

#include <cmath>
#include <memory>

#include "monolith/native_training/runtime/hash_table/optimizer/ftrl_optimizer.h"

namespace monolith {
namespace hash_table {
namespace {

class FtrlOptimizer : public OptimizerInterface {
 public:
  explicit FtrlOptimizer(FtrlOptimizerConfig config)
      : conf_(std::move(config)) {}

  // We need both Zero and Norm in the opt state.
  int64_t SizeBytes() const override {
    return 2 * conf_.dim_size() * sizeof(float);
  }

  int DimSize() const override { return conf_.dim_size(); }

  int SliceSize() const override { return 1; }

  void Init(void* ctx) const override {
    float* norm = static_cast<float*>(ctx);
    float* zero = norm + conf_.dim_size();
    for (int i = 0; i < conf_.dim_size(); ++i) {
      norm[i] = conf_.initial_accumulator_value();
      zero[i] = 0;
    }
  }

  // Please refer to this link for the algorithm:
  // https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
  void Optimize(void* ctx, absl::Span<float> num,
                absl::Span<const float> grad,
                absl::Span<const float> learning_rates,
                const int64_t global_step) const override {
    float* norm = static_cast<float*>(ctx);
    float* zero = norm + conf_.dim_size();
    float effective_lr = learning_rates[0];
    for (int i = 0; i < conf_.dim_size(); ++i) {
      auto norm_new = norm[i] + grad[i] * grad[i];
      auto sigma =
          (std::sqrt(norm_new) - std::sqrt(norm[i])) / effective_lr;
      zero[i] += (grad[i] - sigma * num[i]);
      norm[i] = norm_new;
      num[i] =
          (std::abs(zero[i]) > conf_.l1_regularization_strength())
              ? effective_lr *
                    (std::signbit(zero[i]) *
                         conf_.l1_regularization_strength() -
                     zero[i]) /
                    (std::sqrt(norm[i]) + conf_.beta() +
                     conf_.l2_regularization_strength() * effective_lr)
              : 0.0;
    }
  }

  OptimizerDump Save(const void* ctx) const override {
    OptimizerDump dump;
    FtrlOptimizerDump* ftrl_dump = dump.add_dump()->mutable_ftrl();
    const float* norm = static_cast<const float*>(ctx);
    const float* zero = norm + conf_.dim_size();
    for (int i = 0; i < conf_.dim_size(); ++i) {
      ftrl_dump->add_norm(norm[i]);
      ftrl_dump->add_zero(zero[i]);
    }
    return dump;
  }

  void Restore(void* ctx, OptimizerDump dump) const override {
    const FtrlOptimizerDump& ftrl_dump = dump.dump(0).ftrl();
    float* norm = static_cast<float*>(ctx);
    float* zero = norm + conf_.dim_size();
    for (int i = 0; i < conf_.dim_size(); ++i) {
      norm[i] = ftrl_dump.norm(i);
      zero[i] = ftrl_dump.zero(i);
    }
  }

 private:
  FtrlOptimizerConfig conf_;
};

}  // namespace

std::unique_ptr<OptimizerInterface> NewFtrlOptimizer(
    FtrlOptimizerConfig config) {
  return std::make_unique<FtrlOptimizer>(std::move(config));
}

}  // namespace hash_table
}  // namespace monolith
