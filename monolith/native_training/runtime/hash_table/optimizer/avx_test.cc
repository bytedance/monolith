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

//
// Created by david on 2020-11-27.
//

#include <vector>
#include "absl/random/random.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "monolith/native_training/runtime/hash_table/optimizer/avx_utils.h"

namespace monolith {
namespace hash_table {
namespace {


void TestAdagradOptimize(size_t dim = 32) {
  float lr = 0.01f;
  std::vector<float> norm(dim, 0), grad(dim, 0);
  absl::BitGen bit_gen;
  for (size_t i = 0; i < dim; ++i) {
    norm[i] = absl::Uniform<float>(bit_gen, .1f, 1.f);
    grad[i] = absl::Uniform<float>(bit_gen, -1.f, 1.f);
  }

  std::vector<float> norm2(norm.begin(), norm.end()), grad2(grad.begin(), grad.end());
  std::vector<float> result(dim, 0), result_avx(dim, 0);
  BaselineAdagradOptimize(result.data(), norm.data(), grad.data(), dim, lr, 0.1f);

#if defined(_ENABLE_AVX) && defined(__AVX__)
  Avx256AdagradOptimize(result_avx.data(), norm2.data(), grad2.data(), dim, lr, 0.1f);
#else
  static_assert(false, "AVX is not available, please check and recompile!");
#endif

  for (size_t i = 0; i < dim; ++i) {
    EXPECT_NEAR(result[i], result_avx[i], 1e-6);
  }
}

TEST(AVX, Basic) {
  TestAdagradOptimize(1);
  TestAdagradOptimize(7);
  TestAdagradOptimize(8);
  TestAdagradOptimize(16);
  TestAdagradOptimize(32);
  TestAdagradOptimize(39);
  TestAdagradOptimize(224);
}


}  // namespace
}  // namespace hash_table
}  // namespace monolith
