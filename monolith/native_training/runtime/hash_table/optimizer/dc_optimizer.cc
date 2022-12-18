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

#include "monolith/native_training/runtime/hash_table/optimizer/dc_optimizer.h"

namespace monolith {
namespace hash_table {
namespace {
class DcOptimizer : public OptimizerDecorator {
 public:
  explicit DcOptimizer(DcOptimizerConfig config,
                       std::unique_ptr<OptimizerInterface> base_opt)
      : OptimizerDecorator(std::move(base_opt)), conf_(std::move(config)) { }

  void OptimizeWithLatestValue(void* ctx, absl::Span<float> num,
                               absl::Span<const float> grad,
                               absl::Span<const float> learning_rates,
                               absl::Span<float> latest_value,
                               const int64_t global_step) const override {
    std::vector<float> compensated_g(conf_.dim_size());
    // add in Float16 stuff later?
    for (int i = 0; i < conf_.dim_size(); ++i) {
      float new_grad = grad[i] + conf_.lambda_() * grad[i] * grad[i] *
                        (num[i] - latest_value[i]);
      compensated_g[i] = new_grad;
    }
    base_opt_.get()->Optimize(ctx, num, compensated_g, learning_rates, global_step);
  }

 private:
  DcOptimizerConfig conf_;
};

}  // namespace

std::unique_ptr<OptimizerDecorator> NewDcOptimizer(
    DcOptimizerConfig config, std::unique_ptr<OptimizerInterface> base_opt) {
  return std::make_unique<DcOptimizer>(std::move(config), std::move(base_opt));
}

}  // namespace hash_table
}  // namespace monolith
