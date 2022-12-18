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

#include "monolith/native_training/runtime/hash_table/optimizer/adagrad_optimizer.h"
#include "monolith/native_training/runtime/hash_table/optimizer/avx_utils.h"

namespace monolith {
namespace hash_table {
namespace {

class AdagradOptimizer : public OptimizerInterface {
 public:
  explicit AdagradOptimizer(AdagradOptimizerConfig config)
      : conf_(std::move(config)) {}

  int64_t SizeBytes() const override {
    return conf_.dim_size() * sizeof(float);
  }

  int DimSize() const override { return conf_.dim_size(); }

  int SliceSize() const override { return 1; }

  void Init(void* ctx) const override {
    float* norm = static_cast<float*>(ctx);
    for (int i = 0; i < conf_.dim_size(); ++i) {
      norm[i] = conf_.initial_accumulator_value();
    }
  }

  void Optimize(void* ctx, absl::Span<float> num, absl::Span<const float> grad,
                absl::Span<const float> learning_rates,
                const int64_t global_step) const override {
    float* norm = static_cast<float*>(ctx);
    AdagradOptimize(num.data(), norm, grad.data(), conf_.dim_size(),
                    learning_rates[0], conf_.weight_decay_factor());
  }

  OptimizerDump Save(const void* ctx) const override {
    OptimizerDump dump;
    AdagradOptimizerDump* adagrad_dump = dump.add_dump()->mutable_adagrad();
    const float* norm = static_cast<const float*>(ctx);
    for (int i = 0; i < conf_.dim_size(); ++i) {
      adagrad_dump->add_norm(norm[i]);
    }
    return dump;
  }

  void Restore(void* ctx, OptimizerDump dump) const override {
    const AdagradOptimizerDump& adagrad_dump = dump.dump(0).adagrad();
    float* norm = static_cast<float*>(ctx);
    for (int i = 0; i < conf_.dim_size(); ++i) {
      norm[i] = adagrad_dump.norm(i);
    }
  }

 private:
  AdagradOptimizerConfig conf_;
};

}  // namespace

std::unique_ptr<OptimizerInterface> NewAdagradOptimizer(
    AdagradOptimizerConfig config) {
  return std::make_unique<AdagradOptimizer>(std::move(config));
}

}  // namespace hash_table
}  // namespace monolith
