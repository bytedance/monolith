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

#include <unordered_map>
#include "absl/container/flat_hash_map.h"
#include "absl/random/random.h"
#include "absl/strings/str_format.h"
#include "benchmark/benchmark.h"
#include "google/protobuf/text_format.h"
#include "monolith/native_training/runtime/concurrency/thread_pool.h"
#include "monolith/native_training/runtime/hash_table/cuckoohash/cuckoo_embedding_hash_table.h"
#include "monolith/native_training/runtime/hash_table/embedding_hash_table_factory.h"

namespace monolith {
namespace hash_table {
namespace {

namespace proto2 = ::google::protobuf;

EmbeddingHashTableConfig SetupHashTableConfig(size_t dim) {
  EmbeddingHashTableConfig config;
  CHECK(proto2::TextFormat::ParseFromString(absl::StrFormat(R"(
    entry_config {
      segments {
        dim_size: %lu
        init_config { zeros {} }
        opt_config { sgd {} }
      }
    }
    cuckoo {}
  )",
                                                            dim),
                                            &config));
  return config;
}

void BM_Insert(benchmark::State& state) {  // NOLINT
  auto entry_num = state.range(0);
  auto thread_num = state.range(1);
  auto dim = state.range(2);
  monolith::concurrency::ThreadPool thread_pool(thread_num);
  auto config = SetupHashTableConfig(dim);
  auto table = NewEmbeddingHashTableFromConfig(config);

  std::vector<float> vector(dim);
  absl::BitGen bit_gen;
  for (auto& val : vector) {
    val = absl::Uniform<float>(bit_gen, -1.f, 1.f);
  }

  for (auto _ : state) {
    std::atomic_int join(thread_num);
    auto AssignAdd = [&]() {
      for (size_t i = 0; i < entry_num; ++i) {
        table->AssignAdd(i, absl::MakeSpan(vector), 0);
      }
      --join;
    };

    for (int64_t i = 0; i < thread_num; ++i) {
      thread_pool.Schedule(AssignAdd);
    }

    while (join) {
    }
  }
}

void BM_Find(benchmark::State& state) {  // NOLINT
  auto entry_num = state.range(0);
  auto thread_num = state.range(1);
  auto dim = state.range(2);
  monolith::concurrency::ThreadPool thread_pool(thread_num);
  auto config = SetupHashTableConfig(dim);
  auto table = NewEmbeddingHashTableFromConfig(config);

  std::vector<float> vector(dim);
  absl::BitGen bit_gen;
  for (auto& val : vector) {
    val = absl::Uniform<float>(bit_gen, -1.f, 1.f);
  }
  for (size_t i = 0; i < entry_num; ++i) {
    table->AssignAdd(i, absl::MakeSpan(vector), 0);
  }

  std::vector<int64_t> ids_to_find(entry_num / thread_num);
  for (size_t i = 0; i < ids_to_find.size(); ++i) {
    ids_to_find[i] = absl::Uniform<int64_t>(bit_gen, 0, 2 * entry_num);
  }

  for (auto _ : state) {
    std::atomic_int join(thread_num);
    auto Lookup = [&]() {
      std::vector<float> vector(dim);
      for (int64_t id : ids_to_find) {
        table->Lookup(id, absl::MakeSpan(vector));
      }
      --join;
    };

    for (int64_t i = 0; i < thread_num; ++i) {
      thread_pool.Schedule(Lookup);
    }

    while (join) {
    }
  }
}

/*
  Run on (12 X 2592 MHz CPU s)
  CPU Caches:
    L1 Data 32 KiB (x12)
    L1 Instruction 32 KiB (x12)
    L2 Unified 256 KiB (x12)
    L3 Unified 12288 KiB (x1)
  Load Average: 1.74, 1.35, 0.68
  ------------------------------------------------------------------
  Benchmark                        Time             CPU   Iterations
  ------------------------------------------------------------------
  BM_Insert/10000000/1/32 7834986861 ns   7833447327 ns            1
  BM_Insert/1000000/10/32 5021482710 ns   5019782037 ns            1
  BM_Find/10000000/1/32   6015355836 ns   6015255039 ns            1
  BM_Find/10000000/10/32  4677797500 ns   4677690867 ns            1
*/

// single thread, insert 10^7 entries
BENCHMARK(BM_Insert)->Args({1000 * 10000, 1, 32});

// 10 threads, insert 10^7 = 10^6 * 10 entries
BENCHMARK(BM_Insert)->Args({100 * 10000, 10, 32});

// single thread, find 10^7 times from 10^7 entries
BENCHMARK(BM_Find)->Args({1000 * 10000, 1, 32});

// 10 threads, find 10^7 = 10^6 * 10 times from 10^7 entries
BENCHMARK(BM_Find)->Args({1000 * 10000, 10, 32});

}  // namespace
}  // namespace hash_table
}  // namespace monolith

BENCHMARK_MAIN();
