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

#include "monolith/native_training/runtime/hash_table/optimizer/amsgrad_optimizer.h"

namespace monolith {
namespace hash_table {
namespace {

class AmsgradOptimizer : public OptimizerInterface {
 public:
  explicit AmsgradOptimizer(AmsgradOptimizerConfig config)
      : conf_(std::move(config)) {}

  int64_t SizeBytes() const override {
    return (3 * conf_.dim_size() + 2) * sizeof(float);
  }

  int DimSize() const override { return conf_.dim_size(); }

  int SliceSize() const override { return 1; }

  void Init(void* ctx) const override {
    float* m = static_cast<float*>(ctx);
    float* v = m + conf_.dim_size();
    float* vhat = v + conf_.dim_size();
    float& beta1_power = vhat[conf_.dim_size()];
    float& beta2_power = vhat[conf_.dim_size()+1];

    for (int i = 0; i < conf_.dim_size(); ++i) {
      m[i] = v[i] = vhat[i] = 0;
    }
    beta1_power = conf_.beta1();
    beta2_power = conf_.beta2();
  }

  void Optimize(void* ctx, absl::Span<float> num,
                absl::Span<const float> grad,
                absl::Span<const float> learning_rates,
                const int64_t global_step) const override {
    float* m = static_cast<float*>(ctx);
    float* v = m + conf_.dim_size();
    float* vhat = v + conf_.dim_size();
    float& beta1_power = vhat[conf_.dim_size()];
    float& beta2_power = vhat[conf_.dim_size()+1];
    float lr = learning_rates[0] * sqrt(1 - beta2_power)
             / (1 - beta1_power);

    for (int i = 0; i < conf_.dim_size(); ++i) {
      float cur_grad = grad[i] + conf_.weight_decay_factor() * num[i];
      float new_m = m[i] + (cur_grad - m[i]) * (1 - conf_.beta1());
      float new_v = v[i] + (cur_grad * cur_grad -
                            v[i]) * (1 - conf_.beta2());
      float new_vhat = std::max(vhat[i], new_v);
      float new_w = num[i];
      if (conf_.use_nesterov()) {
        new_w -= ((cur_grad * (1 - conf_.beta1()) + conf_.beta1() * new_m)
                   * lr) / (sqrt(new_vhat) + conf_.epsilon());
      } else {
        new_w -= (new_m * lr) / (sqrt(new_vhat) + conf_.epsilon());
      }
      num[i] = new_w;
      m[i] = new_m;
      v[i] = new_v;
      vhat[i] = new_vhat;
    }
    beta1_power *= conf_.beta1();
    beta2_power *= conf_.beta2();
  }

  OptimizerDump Save(const void* ctx) const override {
    OptimizerDump dump;
    AmsgradOptimizerDump* amsgrad_dump = dump.add_dump()->mutable_amsgrad();
    const float* m = static_cast<const float*>(ctx);
    const float* v = m + conf_.dim_size();
    const float* vhat = v + conf_.dim_size();
    const float& beta1_power = vhat[conf_.dim_size()];
    const float& beta2_power = vhat[conf_.dim_size()+1];

    for (int i = 0; i < conf_.dim_size(); ++i) {
      amsgrad_dump->add_m(m[i]);
      amsgrad_dump->add_v(v[i]);
      amsgrad_dump->add_vhat(vhat[i]);
    }
    amsgrad_dump->set_beta1_power(beta1_power);
    amsgrad_dump->set_beta2_power(beta2_power);
    return dump;
  }

  void Restore(void* ctx, OptimizerDump dump) const override {
    const AmsgradOptimizerDump& amsgrad_dump = dump.dump(0).amsgrad();
    float* m = static_cast<float*>(ctx);
    float* v = m + conf_.dim_size();
    float* vhat = v + conf_.dim_size();
    float& beta1_power = vhat[conf_.dim_size()];
    float& beta2_power = vhat[conf_.dim_size()+1];

    for (int i = 0; i < conf_.dim_size(); ++i) {
      m[i] = amsgrad_dump.m(i);
      v[i] = amsgrad_dump.v(i);
      vhat[i] = amsgrad_dump.vhat(i);
    }
    beta1_power = amsgrad_dump.beta1_power();
    beta2_power = amsgrad_dump.beta2_power();
  }

 private:
  AmsgradOptimizerConfig conf_;
};

}  // namespace

std::unique_ptr<OptimizerInterface> NewAmsgradOptimizer(
    AmsgradOptimizerConfig config) {
  return std::make_unique<AmsgradOptimizer>(std::move(config));
}

}  // namespace hash_table
}  // namespace monolith
