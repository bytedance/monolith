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

#include "monolith/native_training/runtime/hash_table/compressor/float_compressor.h"

#include <exception>

#include "absl/strings/str_format.h"
#include "monolith/native_training/runtime/hash_table/compressor/fake_quantizer.h"
#include "monolith/native_training/runtime/hash_table/compressor/hash_net_quantizer.h"
#include "third_party/half_sourceforge_net/half.hpp"

namespace monolith {
namespace hash_table {
namespace {

class FloatCompressorBase : public FloatCompressorInterface {
 public:
  FloatCompressorBase(int dim_size, int64_t size_bytes)
      : dim_size_(dim_size), size_bytes_(size_bytes) {}

  // Use final to inline this when possible.
  int64_t SizeBytes() const final { return size_bytes_; }
  int DimSize() const final { return dim_size_; }

 private:
  int dim_size_;
  int64_t size_bytes_;
};

class Fp32FloatCompressor final : public FloatCompressorBase {
 public:
  explicit Fp32FloatCompressor(const FloatCompressorConfig::Fp32& config)
      : FloatCompressorBase(config.dim_size(),
                            config.dim_size() * sizeof(float)) {}
  void Encode(absl::Span<const float> num, void* compressed) const override {
    auto* f = reinterpret_cast<float*>(compressed);
    for (int i = 0; i < DimSize(); ++i) {
      f[i] = num[i];
    }
  }

  void Decode(const void* compressed, absl::Span<float> num) const override {
    const auto* f = reinterpret_cast<const float*>(compressed);
    for (int i = 0; i < DimSize(); ++i) {
      num[i] = f[i];
    }
  }
};

// Converts a float to fp16
class Fp16FloatCompressor final : public FloatCompressorBase {
 public:
  explicit Fp16FloatCompressor(const FloatCompressorConfig::Fp16& config)
      : FloatCompressorBase(config.dim_size(),
                            config.dim_size() * sizeof(int16_t)) {}

  void Encode(absl::Span<const float> num, void* compressed) const override {
    auto* i16 = reinterpret_cast<int16_t*>(compressed);
    for (int i = 0; i < DimSize(); ++i) {
      half_float::half x(num[i]);
      i16[i] = *reinterpret_cast<int16_t*>(&x);
    }
  }

  void Decode(const void* compressed, absl::Span<float> num) const override {
    const auto* i16 = reinterpret_cast<const int16_t*>(compressed);
    for (int i = 0; i < DimSize(); ++i) {
      num[i] = *reinterpret_cast<const half_float::half*>(&i16[i]);
    }
  }
};

// Converts a float to fixed range int8.
class FixedR8FloatCompressor final : public FloatCompressorBase {
 public:
  explicit FixedR8FloatCompressor(const FloatCompressorConfig::FixedR8& config)
      : FloatCompressorBase(config.dim_size(),
                            config.dim_size() * sizeof(int8_t)),
        fake_quantizer_(config.r()) {}

  void Encode(absl::Span<const float> num, void* compressed) const override {
    auto* i8 = reinterpret_cast<int8_t*>(compressed);
    for (int i = 0; i < DimSize(); ++i) {
      i8[i] = fake_quantizer_.QuantizeToInteger(num[i]);
    }
  }

  void Decode(const void* compressed, absl::Span<float> num) const override {
    const auto* i8 = reinterpret_cast<const int8_t*>(compressed);
    for (int i = 0; i < DimSize(); ++i) {
      num[i] = fake_quantizer_.IntegerToFloat(i8[i]);
    }
  }

 private:
  FakeQuantizer fake_quantizer_;
};

// Converts a float to one bit.
// We use an int8_t for testing stage
class OneBitFloatCompressor final : public FloatCompressorBase {
 public:
  explicit OneBitFloatCompressor(const FloatCompressorConfig::OneBit& config)
      : FloatCompressorBase(config.dim_size(),
                            config.dim_size() * sizeof(int8_t)),
        hash_net_quantizer_(config) {}

  void Encode(absl::Span<const float> num, void* compressed) const override {
    auto* i8 = reinterpret_cast<int8_t*>(compressed);
    for (int i = 0; i < DimSize(); ++i) {
      i8[i] = hash_net_quantizer_.Forward(num[i]) > 0 ? 1 : -1;
    }
  }

  void Decode(const void* compressed, absl::Span<float> num) const override {
    float amplitude = hash_net_quantizer_.GetConfig().amplitude();
    const auto* i8 = reinterpret_cast<const int8_t*>(compressed);
    for (int i = 0; i < DimSize(); ++i) {
      num[i] = static_cast<float>(i8[i]) * amplitude;
    }
  }

 private:
  HashNetQuantizer hash_net_quantizer_;
};

class CombinedFloatCompressor final : public FloatCompressorBase {
 public:
  CombinedFloatCompressor(std::unique_ptr<FloatCompressorInterface> compressor1,
                          std::unique_ptr<FloatCompressorInterface> compressor2)
      : FloatCompressorBase(
            compressor1->DimSize() + compressor2->DimSize(),
            compressor1->SizeBytes() + compressor2->SizeBytes()),
        compressor1_(std::move(compressor1)),
        compressor2_(std::move(compressor2)),
        compressor1_dim_size_(compressor1_->DimSize()),
        compressor1_size_bytes_(compressor1_->SizeBytes()) {}

  void Encode(absl::Span<const float> num, void* compressed) const override {
    absl::Span<const float> num1 = num.subspan(0, compressor1_dim_size_);
    compressor1_->Encode(num1, compressed);
    absl::Span<const float> num2 = num.subspan(compressor1_dim_size_);
    void* compressed2 =
        reinterpret_cast<char*>(compressed) + compressor1_size_bytes_;
    compressor2_->Encode(num2, compressed2);
  }

  void Decode(const void* compressed, absl::Span<float> num) const override {
    absl::Span<float> num1 = num.subspan(0, compressor1_dim_size_);
    compressor1_->Decode(compressed, num1);
    absl::Span<float> num2 = num.subspan(compressor1_dim_size_);
    const void* compressed2 =
        reinterpret_cast<const char*>(compressed) + compressor1_size_bytes_;
    compressor2_->Decode(compressed2, num2);
  }

 private:
  std::unique_ptr<FloatCompressorInterface> compressor1_;

  std::unique_ptr<FloatCompressorInterface> compressor2_;
  const int compressor1_dim_size_;
  const int64_t compressor1_size_bytes_;
};

}  // namespace

std::unique_ptr<FloatCompressorInterface> NewFloatCompressor(
    FloatCompressorConfig config) {
  switch (config.type_case()) {
    case FloatCompressorConfig::kFp32:
      return std::make_unique<Fp32FloatCompressor>(
          std::move(*config.mutable_fp32()));
    case FloatCompressorConfig::kFp16:
      return std::make_unique<Fp16FloatCompressor>(
          std::move(*config.mutable_fp16()));
    case FloatCompressorConfig::kFixedR8:
      return std::make_unique<FixedR8FloatCompressor>(
          std::move(*config.mutable_fixed_r8()));
    case FloatCompressorConfig::kOneBit:
      return std::make_unique<OneBitFloatCompressor>(
          std::move(*config.mutable_one_bit()));
    default:
      throw std::invalid_argument(absl::StrFormat(
          "Unknown tpye of float compressor. %s", config.ShortDebugString()));
  }
}

std::unique_ptr<FloatCompressorInterface> CombineFloatCompressor(
    std::unique_ptr<FloatCompressorInterface> compressor1,
    std::unique_ptr<FloatCompressorInterface> compressor2) {
  return std::make_unique<CombinedFloatCompressor>(std::move(compressor1),
                                                   std::move(compressor2));
}

}  // namespace hash_table
}  // namespace monolith