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

#ifndef MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_COMPRESSOR_FAKE_QUANTIZER_H_
#define MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_COMPRESSOR_FAKE_QUANTIZER_H_

#include <cmath>

namespace monolith {
namespace hash_table {

// Quantization Aware Training.
// This class quantize a float32 number into an int8_t.
// TODO(zhangbiao.david): support specifying min, max, num_bits etc.
class FakeQuantizer {
 public:
  explicit FakeQuantizer(float r)
      : r_(r), step_(r_ / kNegativeSlotNum), half_step_(step_ / 2) {}

  // Quantize a given floating-point number.
  float Quantize(float f) const { return IntegerToFloat(QuantizeToInteger(f)); }

  // Quantize a floating-point number into integer representation.
  int8_t QuantizeToInteger(float f) const {
    // Round f to nearest float slot. E.g.,
    // Assuming step = 1.0, and f = 3.6, we want nstep = 4.
    if (std::isnan(f)) {
      return 0;
    }

    if (f >= 0) {
      f += half_step_;
    } else {
      f -= half_step_;
    }
    int nstep = f / step_;

    if (nstep > kPositiveSlotNum) {
      nstep = kPositiveSlotNum;
    } else if (nstep < -kNegativeSlotNum) {
      nstep = -kNegativeSlotNum;
    }

    return nstep;
  }

  // Restores an integer representation to a floating-point number.
  float IntegerToFloat(int8_t x) const { return x * step_; }

 private:
  static constexpr int kNumBits = sizeof(int8_t) * 8;
  static constexpr int kSlotNum = 1 << kNumBits;
  static constexpr int kPositiveSlotNum = kSlotNum / 2 - 1;
  static constexpr int kNegativeSlotNum = kSlotNum / 2;

  const float r_;
  const float step_;
  const float half_step_;
};

}  // namespace hash_table
}  // namespace monolith

#endif  // MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_COMPRESSOR_FAKE_QUANTIZER_H_
