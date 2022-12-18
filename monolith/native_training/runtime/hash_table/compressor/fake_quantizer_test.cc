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

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "monolith/native_training/runtime/hash_table/compressor/fake_quantizer.h"

namespace monolith {
namespace hash_table {
namespace {

using ::testing::Lt;
using ::testing::Le;
using ::testing::FloatEq;
using ::testing::Not;
using ::testing::Eq;

TEST(FakeQuantizer, Quantization) {
  FakeQuantizer model(5.0f);

  EXPECT_THAT(model.Quantize(100.0), Lt(5.0));

  // Symmetric
  EXPECT_THAT(model.Quantize(0.0), 0.0f);

  const float kStep = 5.0f / 128;
  // Make sure quantization result is small enough.
  EXPECT_THAT(std::abs(model.Quantize(3.5) - 3.5), Lt(kStep));

  // Make sure round works correctly.
  EXPECT_THAT(model.Quantize(kStep * 1.4), kStep);
  EXPECT_THAT(model.Quantize(kStep * 1.6), kStep * 2);
  EXPECT_THAT(model.Quantize(-kStep * 1.4), -kStep);
  EXPECT_THAT(model.Quantize(-kStep * 1.6), -kStep * 2);
  EXPECT_THAT(model.Quantize(std::nan("")), 0);
}

}  // namespace
}  // namespace hash_table
}  // namespace monolith
