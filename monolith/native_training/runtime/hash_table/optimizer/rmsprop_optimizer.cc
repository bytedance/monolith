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

#include "monolith/native_training/runtime/hash_table/optimizer/rmsprop_optimizer.h"

namespace monolith {
namespace hash_table {
namespace {

class RmspropOptimizer : public OptimizerInterface {
 public:
  explicit RmspropOptimizer(RmspropOptimizerConfig config)
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
    for (int i = 0; i < conf_.dim_size(); ++i) {
      const float& cur_grad = grad[i];
      float new_n = n[i];
      float new_w = num[i];
      double dx = cur_grad +
                  static_cast<double>(conf_.weight_decay_factor()) * new_w;
      new_n = static_cast<double>(conf_.momentum()) * new_n +
              (1 - static_cast<double>(conf_.momentum())) * dx * dx;
      double eta = static_cast<double>(conf_.learning_rate()) /
                   (std::sqrt(new_n) + 1);
      new_w -= eta * dx;
      n[i] = new_n;
      num[i] = new_w;
    }
  }

  OptimizerDump Save(const void* ctx) const override {
    OptimizerDump dump;
    RmspropOptimizerDump* rmsprop_dump = dump.add_dump()->mutable_rmsprop();
    const float* n = static_cast<const float*>(ctx);

    for (int i = 0; i < conf_.dim_size(); ++i) {
      rmsprop_dump->add_n(n[i]);
    }
    return dump;
  }

  void Restore(void* ctx, OptimizerDump dump) const override {
    const RmspropOptimizerDump& rmsprop_dump = dump.dump(0).rmsprop();
    float* n = static_cast<float*>(ctx);

    for (int i = 0; i < conf_.dim_size(); ++i) {
      n[i] = rmsprop_dump.n(i);
    }
  }

 private:
  RmspropOptimizerConfig conf_;
};


class RmspropV2Optimizer : public OptimizerInterface {
 public:
  explicit RmspropV2Optimizer(RmspropV2OptimizerConfig config)
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
    // TODO(eric.wei): implement RMSPropV2 using AVX
    OptimizeNormal(ctx, num, grad, learning_rates, global_step);
  }

  void OptimizeNormal(void* ctx, absl::Span<float> num,
                absl::Span<const float> grad,
                absl::Span<const float> learning_rates,
                const int64_t global_step) const {
    float* n = static_cast<float*>(ctx);
    for (int i = 0; i < conf_.dim_size(); ++i) {
      const float& cur_grad = grad[i];
      float new_n = n[i];
      float new_w = num[i];
      double dx = cur_grad +
                  static_cast<double>(conf_.weight_decay_factor()) * new_w;
      new_n = static_cast<double>(conf_.momentum()) * new_n +
              dx * dx;
      double eta = static_cast<double>(learning_rates[0]) /
                   (std::sqrt(new_n) + 1);
      new_w -= eta * dx;
      n[i] = new_n;
      num[i] = new_w;
    }
  }

  OptimizerDump Save(const void* ctx) const override {
    OptimizerDump dump;
    RmspropV2OptimizerDump* rmspropv2_dump = dump.add_dump()->mutable_rmspropv2();
    const float* n = static_cast<const float*>(ctx);

    for (int i = 0; i < conf_.dim_size(); ++i) {
      rmspropv2_dump->add_n(n[i]);
    }
    return dump;
  }

  void Restore(void* ctx, OptimizerDump dump) const override {
    const RmspropV2OptimizerDump& rmspropv2_dump = dump.dump(0).rmspropv2();
    float* n = static_cast<float*>(ctx);

    for (int i = 0; i < conf_.dim_size(); ++i) {
      n[i] = rmspropv2_dump.n(i);
    }
  }

 private:
  RmspropV2OptimizerConfig conf_;
};

}  // namespace


std::unique_ptr<OptimizerInterface> NewRmspropOptimizer(
    RmspropOptimizerConfig config) {
  return std::make_unique<RmspropOptimizer>(std::move(config));
}

std::unique_ptr<OptimizerInterface> NewRmspropV2Optimizer(
    RmspropV2OptimizerConfig config) {
  return std::make_unique<RmspropV2Optimizer>(std::move(config));
}

}  // namespace hash_table
}  // namespace monolith
