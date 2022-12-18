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

#include <memory>
#include "absl/strings/str_format.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "monolith/native_training/runtime/hash_table/entry_accessor_decorator.h"
#include "monolith/native_training/runtime/hash_table/quantized_entry_accessor.h"

namespace monolith {
namespace hash_table {
namespace {

using ::testing::_;
using ::testing::ElementsAre;
using ::testing::ExplainMatchResult;
using ::testing::FloatEq;
using ::testing::Invoke;
using ::testing::Le;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::WithArgs;

class MockEntryAccessor : public EntryAccessorInterface {
 public:
  MOCK_CONST_METHOD0(SizeBytes, int64_t());
  MOCK_CONST_METHOD0(DimSize, int());
  MOCK_CONST_METHOD0(SliceSize, int());
  MOCK_CONST_METHOD1(Init, void(void* ctx));
  MOCK_CONST_METHOD2(Fill, void(const void* ctx, absl::Span<float>));
  MOCK_CONST_METHOD2(Assign, void(absl::Span<const float> num, void* ctx));
  MOCK_CONST_METHOD2(AssignAdd, void(absl::Span<const float> num, void* ctx));
  MOCK_CONST_METHOD2(Save, EntryDump(const void* ctx, uint32_t));
  MOCK_CONST_METHOD3(Restore, void(void* ctx, uint32_t*, EntryDump));
  MOCK_CONST_METHOD4(Optimize, void(void* ctx, absl::Span<const float>,
                                    absl::Span<const float>, const int64_t));
};

struct MockEntryAccessorFakeOption {
  float init_value = 1.0f;
  int dim = 10;
};

void MakeMockEntryAccessor(
    MockEntryAccessor* mock,
    MockEntryAccessorFakeOption option = MockEntryAccessorFakeOption()) {
  ON_CALL(*mock, DimSize()).WillByDefault(Return(option.dim));
  ON_CALL(*mock, SizeBytes())
      .WillByDefault(Return(option.dim * sizeof(float) * 2));
  ON_CALL(*mock, Init(_))
      .WillByDefault(Invoke([option](void* ctx) {
        // Initialize embedding
        auto* w = reinterpret_cast<float*>(ctx);
        for (int i = 0; i < option.dim; ++i) {
          w[i] = option.init_value;
        }

        // Initialize optimizer
        auto* norm = w + option.dim;
        for (int i = 0; i < option.dim; ++i) {
          norm[i] = 1.0f;
        }
      }));

  ON_CALL(*mock, Optimize(_, _, _, _))
      .WillByDefault(WithArgs<0, 1, 2, 3>(Invoke([option](
          void* ctx, absl::Span<const float> grad,
          absl::Span<const float> learning_rates, const int64_t global_step) {
        auto* embedding = reinterpret_cast<float*>(ctx);
        auto* norm = embedding + option.dim;
        for (int i = 0; i < option.dim; ++i) {
          norm[i] += grad[i] * grad[i];
          embedding[i] -= grad[i];
        }
      })));
}

TEST(QuantizedEntryAccessor, FixedRange) {
  auto accessor = std::make_unique<NiceMock<MockEntryAccessor>>();
  MakeMockEntryAccessor(accessor.get());
  int dim_size = accessor->DimSize();
  auto config1 = SegmentQatConfig(dim_size / 2, true, 1.0f);
  auto config2 = SegmentQatConfig(dim_size / 2, true, 0.5f);
  const float kStep1 = config1.r / 128, kStep2 = config2.r / 128;
  QuantizedEntryAccessor quantized_accessor(std::move(accessor),
                                            {config1, config2});
  auto ctx = std::make_unique<char[]>(quantized_accessor.SizeBytes());

  quantized_accessor.Init(ctx.get());
  std::vector<float> num(dim_size), grad(dim_size, 1.0f);
  quantized_accessor.Fill(ctx.get(), absl::MakeSpan(num));
  for (int i = 0; i < dim_size / 2; ++i) {
    EXPECT_THAT(std::abs(num[i] - 1.0f), Le(kStep1));
  }
  for (int i = dim_size / 2; i < dim_size; ++i) {
    EXPECT_THAT(std::abs(num[i] - 0.5f), Le(kStep2));
  }

  quantized_accessor.Optimize(ctx.get(), absl::MakeConstSpan(grad), {.0f}, 0);
  quantized_accessor.Fill(ctx.get(), absl::MakeSpan(num));
  for (int i = 0; i < dim_size; ++i) {
    EXPECT_FLOAT_EQ(num[i], 0.0f);
  }
}

}  // namespace
}  // namespace hash_table
}  // namespace monolith
