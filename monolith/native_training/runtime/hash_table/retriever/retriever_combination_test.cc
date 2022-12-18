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

#include "monolith/native_training/runtime/hash_table/retriever/retriever_combination.h"

#include <memory>

#include "absl/random/random.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "monolith/native_training/runtime/hash_table/retriever/raw_retriever.h"
#include "monolith/native_training/runtime/hash_table/retriever/fake_quant_retriever.h"

namespace monolith {
namespace hash_table {
namespace {

using ::testing::Le;

TEST(CombinedRetriever, Basic) {
  int dim_size1 = 10, dim_size2 = 20;
  int dim_size = dim_size1 + dim_size2;
  float r = 1.0f;
  const float kStep = r / 128;
  FakeQuantizer fake_quantizer(1.0f);
  auto retriever1 = NewRawRetriever(dim_size1);
  auto retriever2 = NewFakeQuantRetriever(dim_size2, fake_quantizer);
  auto retriever = CombineRetrievers(std::move(retriever1), std::move(retriever2));
  std::vector<float> entry(dim_size);
  absl::BitGen bit_gen;
  for (auto& val : entry) {
    val = absl::Uniform<float>(bit_gen, -1.f, 1.f);
  }

  std::vector<float> num(dim_size, 0);
  retriever->Retrieve(entry.data(), absl::MakeSpan(num));
  for (int i = 0; i < dim_size1; ++i) {
    EXPECT_EQ(entry[i], num[i]);
  }
  for (int i = dim_size1; i < dim_size; ++i) {
    EXPECT_THAT(std::abs(entry[i] - num[i]), Le(kStep));
  }
}

}  // namespace
}  // namespace hash_table
}  // namespace monolith
