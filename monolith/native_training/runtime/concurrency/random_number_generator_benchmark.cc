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

#include <random>
#include "absl/random/random.h"
#include "benchmark/benchmark.h"
#include "monolith/native_training/runtime/concurrency/xorshift.h"

namespace monolith {
namespace concurrency {
namespace {

const int NUM = 1000000;

void BM_STL(benchmark::State& state) {  // NOLINT
  std::random_device random_device;
  std::mt19937 engine{random_device()};
  std::uniform_int_distribution<uint32_t> dist(
      0, std::numeric_limits<uint32_t>::max());
  for (auto _ : state) {
    for (int i = 0; i < NUM; ++i) {
      dist(engine);
    }
  }
}

void BM_Absl(benchmark::State& state) {  // NOLINT
  absl::BitGen bit_gen;
  for (auto _ : state) {
    for (int i = 0; i < NUM; ++i) {
      absl::Uniform(bit_gen, 0u, std::numeric_limits<uint32_t>::max());
    }
  }
}

void BM_XorShift(benchmark::State& state) {  // NOLINT
  for (auto _ : state) {
    for (int i = 0; i < NUM; ++i) {
      XorShift::Rand32ThreadSafe();
    }
  }
}

// Run on (96 X 3900 MHz CPU s)
// CPU Caches:
//     L1 Data 32K (x48)
//     L1 Instruction 32K (x48)
//     L2 Unified 1024K (x48)
//     L3 Unified 36608K (x2)
// Load Average: 10.64, 12.83, 14.44
// ------------------------------------------------------
// Benchmark            Time             CPU   Iterations
// ------------------------------------------------------
// BM_STL         5849400 ns      5849341 ns          117
// BM_Absl        5574646 ns      5574647 ns          126
// BM_XorShift    3250932 ns      3244318 ns          216
BENCHMARK(BM_STL);
BENCHMARK(BM_Absl);
BENCHMARK(BM_XorShift);

}  // namespace
}  // namespace concurrency
}  // namespace monolith

BENCHMARK_MAIN();
