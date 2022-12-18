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

#ifndef MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_OPTIMIZER_QUANTIZED_ENTRY_ACCESSOR_H_
#define MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_OPTIMIZER_QUANTIZED_ENTRY_ACCESSOR_H_

#include <vector>
#include "monolith/native_training/runtime/hash_table/compressor/fake_quantizer.h"
#include "monolith/native_training/runtime/hash_table/compressor/float_compressor.pb.h"
#include "monolith/native_training/runtime/hash_table/entry_accessor_decorator.h"

namespace monolith {
namespace hash_table {

struct SegmentQatConfig {
  explicit SegmentQatConfig(int dim_size, bool enable_qat = false,
                            float r = 1.0f)
      : dim_size(dim_size), enable_qat(enable_qat), r(r) {}

  int dim_size;
  bool enable_qat;
  float r;
};

// Makes the entry accessor support quantized aware training.
class QuantizedEntryAccessor : public EntryAccessorDecorator {
 public:
  QuantizedEntryAccessor(std::unique_ptr<EntryAccessorInterface> accessor,
                         std::vector<SegmentQatConfig> segment_qat_configs)
      : EntryAccessorDecorator(std::move(accessor)),
        configs_(std::move(segment_qat_configs)) {
    for (const auto &config : configs_) {
      if (config.enable_qat) {
        fake_quantizers_.emplace_back(
            std::make_unique<FakeQuantizer>(config.r));
      } else {
        fake_quantizers_.emplace_back(nullptr);
      }
    }
  }

  void Fill(const void *ctx, absl::Span<float> num) const override {
    int dim_size = entry_accessor_->DimSize();
    std::vector<float> quantized(dim_size);
    auto *ctx_float = static_cast<const float *>(ctx);

    // Simulate the quantization on weights
    int index = 0;
    for (size_t i = 0; i < fake_quantizers_.size(); ++i) {
      for (int j = 0; j < configs_[i].dim_size; ++j) {
        quantized[index] =
            fake_quantizers_[i] == nullptr
                ? ctx_float[index]
                : fake_quantizers_[i]->Quantize(ctx_float[index]);
        ++index;
      }
    }

    absl::c_copy(absl::MakeConstSpan(quantized), num.begin());
  }

  void Optimize(void *ctx, absl::Span<const float> grad,
                absl::Span<const float> learning_rates,
                const int64_t global_step) const override {
    // Apply gradients to real weights
    entry_accessor_->Optimize(ctx, grad, learning_rates, global_step);
  }

 private:
  std::vector<SegmentQatConfig> configs_;

  std::vector<std::unique_ptr<FakeQuantizer>> fake_quantizers_;
};

}  // namespace hash_table
}  // namespace monolith

#endif  // MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_OPTIMIZER_QUANTIZED_ENTRY_ACCESSOR_H_
