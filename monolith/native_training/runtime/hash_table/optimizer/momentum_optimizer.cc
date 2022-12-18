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

#include "monolith/native_training/runtime/hash_table/optimizer/momentum_optimizer.h"

namespace monolith {
namespace hash_table {
namespace {

class MomentumOptimizer : public OptimizerInterface {
 public:
  explicit MomentumOptimizer(MomentumOptimizerConfig config)
      : conf_(std::move(config)) {}

  int64_t SizeBytes() const override {
    return (conf_.dim_size()) * sizeof(float);
  }

  int DimSize() const override { return conf_.dim_size(); }

  int SliceSize() const override { return 1; }

  void Init(void* ctx) const override {
    float* n = static_cast<float*>(ctx);

    for (int i = 0; i < conf_.dim_size(); ++i) {
      n[i] = 0;
    }
  }

  void Optimize(void* ctx, absl::Span<float> num,
                absl::Span<const float> grad,
                absl::Span<const float> learning_rates,
                const int64_t global_step) const override {
    float* n = static_cast<float*>(ctx);
    float g_total = 0;
    for (int i = 0; i < conf_.dim_size(); ++i) {
      float dx = learning_rates[0] * (grad[i] +
                 conf_.weight_decay_factor() * num[i]);
      float new_n = n[i];
      float new_w = num[i];
      if (conf_.use_nesterov()) {
        float prev_n = new_n;
        new_n = conf_.momentum() * new_n - dx;
        new_w += -conf_.momentum() * prev_n + (1 + conf_.momentum()) * new_n;
      } else {
        new_n = conf_.momentum() * new_n - dx;
        new_w += new_n;
      }
      n[i] = new_n;
      num[i] = new_w;
      g_total += new_n;
    }
  }

  OptimizerDump Save(const void* ctx) const override {
    OptimizerDump dump;
    MomentumOptimizerDump* momentum_dump = dump.add_dump()->mutable_momentum();
    const float* n = static_cast<const float*>(ctx);

    for (int i = 0; i < conf_.dim_size(); ++i) {
      momentum_dump->add_n(n[i]);
    }
    return dump;
  }

  void Restore(void* ctx, OptimizerDump dump) const override {
    const MomentumOptimizerDump& momentum_dump = dump.dump(0).momentum();
    float* n = static_cast<float*>(ctx);

    for (int i = 0; i < conf_.dim_size(); ++i) {
      n[i] = momentum_dump.n(i);
    }
  }

 private:
  MomentumOptimizerConfig conf_;
};

}  // namespace

std::unique_ptr<OptimizerInterface> NewMomentumOptimizer(
    MomentumOptimizerConfig config) {
  return std::make_unique<MomentumOptimizer>(std::move(config));
}

}  // namespace hash_table
}  // namespace monolith
