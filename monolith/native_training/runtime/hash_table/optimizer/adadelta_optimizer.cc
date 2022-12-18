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

#include "monolith/native_training/runtime/hash_table/optimizer/adadelta_optimizer.h"

namespace monolith {
namespace hash_table {
namespace {

class AdadeltaOptimizer : public OptimizerInterface {
 public:
  explicit AdadeltaOptimizer(AdadeltaOptimizerConfig config)
      : conf_(std::move(config)) {}

  int64_t SizeBytes() const override {
    return 2 * conf_.dim_size() * sizeof(float);
  }

  int DimSize() const override { return conf_.dim_size(); }

  int SliceSize() const override { return 1; }

  void Init(void* ctx) const override {
    float* accum = static_cast<float*>(ctx);
    float* accum_update = accum + conf_.dim_size();
    for (int i = 0; i < conf_.dim_size(); ++i) {
      accum[i] = accum_update[i] = 0;
    }
  }

  void Optimize(void* ctx, absl::Span<float> num,
                absl::Span<const float> grad,
                absl::Span<const float> learning_rates,
                const int64_t global_step) const override {
    float* accum = static_cast<float*>(ctx);
    float* accum_update = accum + conf_.dim_size();
    float effective_lr = learning_rates[0];
    for (int i = 0; i < conf_.dim_size(); ++i) {
      float cur_grad = grad[i] + conf_.weight_decay_factor() * num[i];
      float new_accum = accum[i] * conf_.averaging_ratio() + cur_grad * cur_grad * (1 - conf_.averaging_ratio());
      float update = std::sqrt(accum_update[i] + conf_.epsilon()) /
                     std::sqrt(new_accum + conf_.epsilon()) * cur_grad;
      float new_w = num[i] - update * effective_lr;
      float new_accum_update =
          accum_update[i] * conf_.averaging_ratio() + update * update * (1 - conf_.averaging_ratio());
      // printf("%d: %f %f %f %f %f\n", i, cur_grad, new_accum, update, new_w, new_accum_update);
      num[i] = new_w;
      accum[i] = new_accum;
      accum_update[i] = new_accum_update;
    }
  }

  OptimizerDump Save(const void* ctx) const override {
    OptimizerDump dump;
    AdadeltaOptimizerDump* adadelta_dump = dump.add_dump()->mutable_adadelta();
    const float* accum = static_cast<const float*>(ctx);
    const float* accum_update = accum + conf_.dim_size();
    for (int i = 0; i < conf_.dim_size(); ++i) {
      adadelta_dump->add_accum(accum[i]);
      adadelta_dump->add_accum_update(accum_update[i]);
    }
    return dump;
  }

  void Restore(void* ctx, OptimizerDump dump) const override {
    const AdadeltaOptimizerDump& adadelta_dump = dump.dump(0).adadelta();
    float* accum = static_cast<float*>(ctx);
    float* accum_update = accum + conf_.dim_size();
    for (int i = 0; i < conf_.dim_size(); ++i) {
      accum[i] = adadelta_dump.accum(i);
      accum_update[i] = adadelta_dump.accum_update(i);
    }
  }

 private:
  AdadeltaOptimizerConfig conf_;
};

}  // namespace

std::unique_ptr<OptimizerInterface> NewAdadeltaOptimizer(
    AdadeltaOptimizerConfig config) {
  return std::make_unique<AdadeltaOptimizer>(std::move(config));
}

}  // namespace hash_table
}  // namespace monolith
