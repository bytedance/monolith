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

#include <string>

#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"

namespace monolith {
namespace hash_table {
namespace {

using ::testing::ElementsAre;
using ::testing::FloatNear;
using ::testing::Pointwise;

std::vector<float> EncodeDecode(const FloatCompressorInterface& compressor,
                                absl::Span<const float> num) {
  auto compressed = std::make_unique<char[]>(compressor.SizeBytes());
  compressor.Encode(num, compressed.get());
  std::vector<float> decoded(compressor.DimSize());
  compressor.Decode(compressed.get(), absl::MakeSpan(decoded));
  return decoded;
}

std::vector<float> EncodeDecode(const FloatCompressorConfig& config,
                                absl::Span<const float> num) {
  auto compressor = NewFloatCompressor(config);
  return EncodeDecode(*compressor, num);
}

FloatCompressorConfig ParseConfig(const std::string& text) {
  FloatCompressorConfig c;
  GOOGLE_CHECK(google::protobuf::TextFormat::ParseFromString(text, &c));
  return c;
}

TEST(Fp32FloatCompressorTest, Basic) {
  EXPECT_THAT(
      EncodeDecode(ParseConfig(R"(fp32 { dim_size: 3})"), {0.1, 0.2, 10000.0}),
      ElementsAre(0.1, 0.2, 10000.0));
}

TEST(Fp16FloatCompressorTest, Basic) {
  EXPECT_THAT(
      EncodeDecode(ParseConfig(R"(fp16 { dim_size: 3})"), {0.1, 0.2, 10000.0}),
      Pointwise(FloatNear(1e-4), {0.1, 0.2, 10000.0}));
}

TEST(FixedR8FloatCompressorTest, Basic) {
  const float kStep = 5.0f / 128;
  EXPECT_THAT(EncodeDecode(ParseConfig(R"(fixed_r8 { dim_size: 3 r : 5})"),
                           {100.0, 0.0, 3.5}),
              Pointwise(FloatNear(kStep), {5.0, 0.0, 3.5}));
  EXPECT_THAT(
      EncodeDecode(ParseConfig(R"(fixed_r8 { dim_size: 4 r : 5})"),
                   {kStep * 1.4f, kStep * 1.6f, -kStep * 1.4f, -kStep * 1.6f}),
      ElementsAre(kStep, kStep * 2, -kStep, -kStep * 2));
}

TEST(OneBitFloatCompressorTest, Basic) {
  EXPECT_THAT(
      EncodeDecode(
          ParseConfig(R"(one_bit { dim_size: 7 step_size : 5 amplitude: 1.0})"),
          {100.0, 0.1, 0.00001, 0.0, -0.00001, -0.1, -100.0}),
      Pointwise(FloatNear(0.1f), {1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0}));
}

TEST(CombinedFloatCompressorTest, Basic) {
  auto compressor1 = NewFloatCompressor(ParseConfig(R"(fp16 { dim_size: 1 })"));
  auto compressor2 = NewFloatCompressor(ParseConfig(R"(fp16 { dim_size: 2 })"));
  auto compressor =
      CombineFloatCompressor(std::move(compressor1), std::move(compressor2));
  EXPECT_THAT(EncodeDecode(*compressor, {1.0, 2.0, 3.0, 4.0}),
              Pointwise(FloatNear(1e-4), {1.0, 2.0, 3.0}));
}

}  // namespace
}  // namespace hash_table
}  // namespace monolith