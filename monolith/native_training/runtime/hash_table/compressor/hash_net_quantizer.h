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

#ifndef MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_COMPRESSOR_HASH_NET_QUANTIZER_H_
#define MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_COMPRESSOR_HASH_NET_QUANTIZER_H_

#include <atomic>
#include <cmath>
#include <cstdint>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "glog/logging.h"
#include "tensorflow/core/platform/logging.h"

#include "monolith/native_training/runtime/hash_table/compressor/float_compressor.pb.h"

namespace monolith {
namespace hash_table {

class HashNetQuantizer {
 public:
  explicit HashNetQuantizer(FloatCompressorConfig_OneBit config)
      : config_(std::move(config)) {
    scale_ = config_.init_scale();
    LOG(INFO) << absl::StrFormat("HashNetQuantizer: %s, scale = %.6f",
                                 config_.ShortDebugString(), scale_.load());
  }

  float Forward(float f) const {
    return config_.amplitude() * std::tanh(scale_ * f);
  }

  void Backward(float num, float* grad, int64_t global_step) const {
    if (global_step % config_.step_size() == 0) {
      scale_ = config_.init_scale() *
               std::pow(1.f + kGamma * static_cast<float>(global_step), kPower);
      scale_ = std::min(scale_.load(), config_.max_scale());
      LOG_EVERY_N_SEC(INFO, 60) << absl::StrFormat(
          "HashNetQuantizer: %s, scale = %.6f, global_step = %ld",
          config_.ShortDebugString(), scale_, global_step);
    }

    float y = std::tanh(scale_ * num);
    *grad *= config_.amplitude() * scale_ * (1.f - y * y);
  }

  float GetScale() const { return scale_; }

  const FloatCompressorConfig_OneBit& GetConfig() const { return config_; }

 private:
  static constexpr float kGamma = 0.005;
  static constexpr float kPower = 0.5;

  mutable std::atomic<float> scale_;
  FloatCompressorConfig_OneBit config_;
};

}  // namespace hash_table
}  // namespace monolith

#endif  // MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_COMPRESSOR_HASH_NET_QUANTIZER_H_
