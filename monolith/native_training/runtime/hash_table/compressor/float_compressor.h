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

#ifndef MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_COMPRESSOR_ENTRY_SERVING_COMPRESSOR
#define MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_COMPRESSOR_ENTRY_SERVING_COMPRESSOR
#include "absl/types/span.h"

#include "monolith/native_training/runtime/hash_table/compressor/float_compressor.pb.h"

namespace monolith {
namespace hash_table {

// Used to compress float number in online serving PS to save the memory.
class FloatCompressorInterface {
 public:
  virtual ~FloatCompressorInterface() = default;

  // How many bytes are required for the compressor.
  virtual int64_t SizeBytes() const = 0;

  // How many dimensions this compressor support.
  virtual int DimSize() const = 0;

  // Encodes a list of floats into compressed.
  virtual void Encode(absl::Span<const float> num, void* compressed) const = 0;

  // Decodes a list of Int into a list of float.
  virtual void Decode(const void* compressed, absl::Span<float> num) const = 0;
};

std::unique_ptr<FloatCompressorInterface> NewFloatCompressor(
    FloatCompressorConfig config);

std::unique_ptr<FloatCompressorInterface> CombineFloatCompressor(
    std::unique_ptr<FloatCompressorInterface> compressor1,
    std::unique_ptr<FloatCompressorInterface> compressor2);

}  // namespace hash_table
}  // namespace monolith
#endif  // MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_COMPRESSOR_ENTRY_SERVING_COMPRESSOR
