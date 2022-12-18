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

#ifndef MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_OPTIMIZER_STOCHASTIC_ROUNDING
#define MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_OPTIMIZER_STOCHASTIC_ROUNDING
#include <cstdint>

#include "absl/types/span.h"
#include "monolith/native_training/runtime/hash_table/optimizer/optimizer.pb.h"
#include "monolith/native_training/runtime/hash_table/optimizer/optimizer_interface.h"
#include "third_party/half_sourceforge_net/half.hpp"

namespace monolith {
namespace hash_table {

inline float stochastic_round(float vf, float p) {
  unsigned int half_up = half_float::detail::float2half<std::round_toward_infinity>(vf);
  unsigned int half_down = half_float::detail::float2half<std::round_toward_neg_infinity>(vf);
  float vf_up = half_float::detail::half2float<float>(half_up);
  float vf_down = half_float::detail::half2float<float>(half_down);
  if (p <= (vf - vf_down) / (vf_up - vf_down)) {
    return vf_up;
  } else {
    return vf_down;
  }
}

class StochasticRoundingFloat16OptimizerDecorator : public OptimizerInterface {
 public:
  explicit StochasticRoundingFloat16OptimizerDecorator(
    std::unique_ptr<OptimizerInterface> optimizer)
        : optimizer_(std::move(optimizer)) {}

  // optimize the num based on gradients and the optimizer's data.
  // |num|, |grad| are float arrays whose length is at least DimSize().
  // Result `num` will be stochastically rounded.
  void Optimize(void* ctx, absl::Span<float> num,
                absl::Span<const float> grad,
                absl::Span<const float> learning_rates,
                const int64_t global_step = 0) const override {
    optimizer_->Optimize(ctx, num, grad, learning_rates);
    for (size_t i = 0; i < num.size(); ++i) {
      num[i] = stochastic_round(num[i], rand());
    }
  }

  // Forward all other class methods.
  int64_t SizeBytes() const override {
    return optimizer_->SizeBytes();
  }

  // The dim that this optimizer can support.
  int DimSize() const override {
    return optimizer_->DimSize();
  }

  // The slice size that this optimizer holds.
  int SliceSize() const override {
    return optimizer_->SliceSize();
  }

  // Init optimizer ctx.
  // |num| is at least DimSize() long.
  void Init(void* ctx) const override {
    optimizer_->Init(ctx);
  }

  // Save and restore the entry.
  OptimizerDump Save(const void* ctx) const override {
    return optimizer_->Save(ctx);
  }
  void Restore(void* ctx, OptimizerDump dump) const override {
    optimizer_->Restore(ctx, dump);
  }

 private:
  std::unique_ptr<OptimizerInterface> optimizer_;

  static thread_local std::vector<uint64_t> rng_;

  static void update_rng() {
    rng_[0] = (36969 * (rng_[0] & 65535) + (rng_[0] >> 16)) & 4294967295;
    rng_[1] = (18000 * (rng_[1] & 65535) + (rng_[1] >> 16)) & 4294967295;
  }
  /*
   * multiply-with-carry generator to generate a float number in [0, 1],
   * for stochastic rounding from fp32 to fp16
   *
   * About Marsaglia's MWC generator, see
   * [http://www.cs.yorku.ca/~oz/marsaglia-rng.html]
   */
  static float rand() {
    update_rng();
    return static_cast<float>(
      (((rng_[0] & 65535) << 16) + rng_[1]) & 4294967295) / 4294967296;
  }
};

}  // namespace hash_table
}  // namespace monolith
#endif  // MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_OPTIMIZER_STOCHASTIC_ROUNDING
