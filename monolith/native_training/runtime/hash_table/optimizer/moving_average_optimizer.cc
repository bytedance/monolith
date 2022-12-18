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

#include "monolith/native_training/runtime/hash_table/optimizer/moving_average_optimizer.h"

namespace monolith {
namespace hash_table {
namespace {

class MovingAverageOptimizer : public OptimizerInterface {
 public:
  explicit MovingAverageOptimizer(MovingAverageOptimizerConfig config)
      : conf_(std::move(config)) {}

  int64_t SizeBytes() const override {
    return 0;
  }

  int DimSize() const override { return conf_.dim_size(); }

  int SliceSize() const override { return 1; }

  void Init(void* ctx) const override { }

  void Optimize(void* ctx, absl::Span<float> num,
                absl::Span<const float> grad,
                absl::Span<const float> learning_rates,
                const int64_t global_step) const override {
    for (int i = 0; i < conf_.dim_size(); ++i) {
      float new_w = conf_.momentum() * num[i] + (1 - conf_.momentum()) * grad[i];
      num[i] = new_w;
    }
  }

  OptimizerDump Save(const void* ctx) const override {
    OptimizerDump dump;
    return dump;
  }

  void Restore(void* ctx, OptimizerDump dump) const override { }

 private:
  MovingAverageOptimizerConfig conf_;
};

}  // namespace

std::unique_ptr<OptimizerInterface> NewMovingAverageOptimizer(
    MovingAverageOptimizerConfig config) {
  return std::make_unique<MovingAverageOptimizer>(std::move(config));
}

}  // namespace hash_table
}  // namespace monolith
