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

#include "absl/random/random.h"
#include "benchmark/benchmark.h"
#include "monolith/native_training/runtime/common/cpu_info.h"
#include "monolith/native_training/runtime/hash_table/optimizer/avx_utils.h"

namespace monolith {
namespace hash_table {
namespace {

void BM_AdagradOptimize(benchmark::State& state) {  // NOLINT
  size_t dim = state.range(0);
  float lr = 0.01f;
  std::vector<float> norm(dim, 0), grad(dim, 0);
  absl::BitGen bit_gen;
  for (size_t i = 0; i < dim; ++i) {
    norm[i] = absl::Uniform<float>(bit_gen, .1f, 1.f);
    grad[i] = absl::Uniform<float>(bit_gen, -1.f, 1.f);
  }

  std::vector<float> result(dim, 0);
  for (auto _ : state) {
    BaselineAdagradOptimize(result.data(), norm.data(), grad.data(), dim, lr,
                            0.01);
  }
}

void BM_AVXAdagradOptimize(benchmark::State& state) {  // NOLINT
  RunCPUGuard();
  size_t dim = state.range(0);
  float lr = 0.01f;
  std::vector<float> norm(dim, 0), grad(dim, 0);
  absl::BitGen bit_gen;
  for (size_t i = 0; i < dim; ++i) {
    norm[i] = absl::Uniform<float>(bit_gen, .1f, 1.f);
    grad[i] = absl::Uniform<float>(bit_gen, -1.f, 1.f);
  }

  std::vector<float> result(dim, 0);
  for (auto _ : state) {
    Avx256AdagradOptimize(result.data(), norm.data(), grad.data(), dim, lr,
                          0.01);
  }
}

BENCHMARK(BM_AdagradOptimize)->Arg(16)->Arg(64)->Arg(256);
BENCHMARK(BM_AVXAdagradOptimize)->Arg(16)->Arg(64)->Arg(256);

}  // namespace
}  // namespace hash_table
}  // namespace monolith

BENCHMARK_MAIN();
