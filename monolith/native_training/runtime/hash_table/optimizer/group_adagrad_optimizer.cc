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

#include "absl/strings/str_format.h"
#include "monolith/native_training/runtime/hash_table/optimizer/group_adagrad_optimizer.h"

namespace monolith {
namespace hash_table {
namespace {

class GroupAdaGradOptimizer : public OptimizerInterface {
 public:
  explicit GroupAdaGradOptimizer(GroupAdaGradOptimizerConfig config)
      : conf_(std::move(config)) {}

  // Only need 4 byte, grad_square_sum = sum(g_max^2)
  int64_t SizeBytes() const override { return sizeof(float); }

  int64_t UncompressedSizeBytes() const override {
    return conf_.dim_size() * sizeof(float);
  }

  std::string DebugString() const override {
    return absl::StrFormat("GroupAdaGrad(D=%d)", DimSize());
  }

  int DimSize() const override { return conf_.dim_size(); }

  int SliceSize() const override { return 1; }

  void Init(void* ctx) const override {
    float* grad_square_sum = static_cast<float*>(ctx);
    *grad_square_sum = conf_.initial_accumulator_value();
  }

  void Optimize(void* ctx, absl::Span<float> num, absl::Span<const float> grad,
                absl::Span<const float> learning_rates,
                const int64_t global_step) const override {
    float* grad_square_sum = static_cast<float*>(ctx);
    float effective_lr = learning_rates[0];

    float max_grad_square = 0.0;
    std::vector<float> g_decayed;
    g_decayed.reserve(conf_.dim_size());
    for (int i = 0; i < conf_.dim_size(); ++i) {
      // weight_decay
      float g = grad[i] + conf_.weight_decay_factor() * num[i];
      if (g * g > max_grad_square) {
        max_grad_square = g * g;
      }
      g_decayed.push_back(g);
    }

    *grad_square_sum = *grad_square_sum + max_grad_square;
    float lr = effective_lr / (conf_.beta() + std::sqrt(*grad_square_sum));

    float z_norm = 0.0;
    for (int i = 0; i < conf_.dim_size(); ++i) {
      num[i] = g_decayed[i] - num[i] / lr;
      z_norm += num[i] * num[i];
    }

    z_norm = std::sqrt(z_norm);
    if (z_norm < conf_.l2_regularization_strength()) {
      for (int i = 0; i < conf_.dim_size(); ++i) {
        num[i] = 0;
      }
    } else {
      float coeffi =
          -lr * (z_norm - conf_.l2_regularization_strength()) / z_norm;
      for (int i = 0; i < conf_.dim_size(); ++i) {
        num[i] = coeffi * num[i];
      }
    }
  }

  OptimizerDump Save(const void* ctx) const override {
    OptimizerDump dump;
    GroupAdaGradOptimizerDump* group_adagrad_dump =
        dump.add_dump()->mutable_group_adagrad();
    const float* grad_square_sum = static_cast<const float*>(ctx);
    group_adagrad_dump->set_grad_square_sum(*grad_square_sum);

    return dump;
  }

  void Restore(void* ctx, OptimizerDump dump) const override {
    const GroupAdaGradOptimizerDump& group_adagrad_dump =
        dump.dump(0).group_adagrad();
    float* grad_square_sum = static_cast<float*>(ctx);
    *grad_square_sum = group_adagrad_dump.grad_square_sum();
  }

 private:
  GroupAdaGradOptimizerConfig conf_;
};

}  // namespace

std::unique_ptr<OptimizerInterface> NewGroupAdaGradOptimizer(
    GroupAdaGradOptimizerConfig config) {
  return std::make_unique<GroupAdaGradOptimizer>(std::move(config));
}

}  // namespace hash_table
}  // namespace monolith
