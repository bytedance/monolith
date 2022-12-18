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

#include "monolith/native_training/runtime/hash_table/retriever/hash_net_retriever.h"

#include <cmath>

#include "absl/random/random.h"
#include "gtest/gtest.h"

namespace monolith {
namespace hash_table {
namespace {

TEST(HashNetRetriever, Basic) {
  FloatCompressorConfig_OneBit config;
  config.set_dim_size(10);
  config.set_step_size(100);
  float amplitude = config.amplitude();
  auto hash_net_quantizer = std::make_unique<HashNetQuantizer>(config);
  HashNetQuantizer* quantizer = hash_net_quantizer.get();
  auto retriever =
      NewHashNetRetriever(config.dim_size(), std::move(hash_net_quantizer));
  std::vector<float> entry(config.dim_size());
  absl::BitGen bit_gen;
  for (auto& val : entry) {
    val = absl::Uniform<float>(bit_gen, -1.f, 1.f);
  }

  std::vector<float> num(config.dim_size(), 0);
  retriever->Retrieve(entry.data(), absl::MakeSpan(num));
  for (int i = 0; i < config.dim_size(); ++i) {
    EXPECT_FLOAT_EQ(num[i], amplitude * std::tanh(entry[i]));
  }

  float grad = 1.0f;
  int64_t global_step = 100;
  quantizer->Backward(1.0f, &grad, global_step);
  EXPECT_FLOAT_EQ(grad, 0.35840667f * amplitude);

  float scale = quantizer->GetScale();
  retriever->Retrieve(entry.data(), absl::MakeSpan(num));
  for (int i = 0; i < config.dim_size(); ++i) {
    EXPECT_FLOAT_EQ(num[i], amplitude * std::tanh(scale * entry[i]));
  }
}

}  // namespace
}  // namespace hash_table
}  // namespace monolith
