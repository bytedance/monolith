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

#include <atomic>
#include "absl/random/random.h"
#include "absl/strings/str_format.h"
#include "benchmark/benchmark.h"
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "monolith/native_training/runtime/concurrency/thread_pool.h"
#include "monolith/native_training/runtime/hash_table/embedding_hash_table_factory.h"

namespace monolith {
namespace hash_table {
namespace {

namespace proto2 = ::google::protobuf;

constexpr int64_t kMaxId = 1 << 15;

std::unique_ptr<EmbeddingHashTableInterface> SetupHashTable() {
  EmbeddingHashTableConfig config;
  CHECK(proto2::TextFormat::ParseFromString(R"(
    entry_config {
      segments {
        dim_size: 1
        init_config { zeros {} }
        opt_config { ftrl {} }
      }
      segments {
        dim_size: 32
        init_config { zeros {} }
        opt_config { sgd {} }
      }
    }
    cuckoo {}
  )",
                                            &config));
  auto table = NewEmbeddingHashTableFromConfig(config);
  for (int64_t i = 0; i < kMaxId; ++i) {
    table->AssignAdd(i, std::vector<float>(33, 0.0f), 0);
  }
  return table;
}

std::vector<int64_t> SetupPickedIds(int num) {
  absl::BitGen bitgen;
  std::vector<int64_t> ids(num);
  for (int i = 0; i < num; ++i) {
    ids[i] = absl::Uniform(bitgen, 0u, kMaxId);
  }
  return ids;
}

std::vector<int64_t> ids = SetupPickedIds(1000 * 256);  // NOLINT
auto table = SetupHashTable();                          // NOLINT

void BM_LookUp(benchmark::State& state) {  // NOLINT
  int64_t thread_num = state.range(0);
  monolith::concurrency::ThreadPool thread_pool(thread_num);

  for (auto _ : state) {
    std::atomic_int join(thread_num);
    auto optimize = [&]() {
      // OPTIMIZE: remove memory allocation overhead
      std::vector<float> embeddings(33, 0);
      for (int64_t id : ids) {
        table->Lookup(id, absl::MakeSpan(embeddings));
      }
      --join;
    };

    // Simulate multi-workers lookup simultaneously
    for (int64_t i = 0; i < thread_num; ++i) {
      thread_pool.Schedule(optimize);
    }
    while (join) {
    }
  }
}

void BM_BatchLookUp(benchmark::State& state) {  // NOLINT
  int64_t thread_num = state.range(0);
  monolith::concurrency::ThreadPool thread_pool(thread_num);

  for (auto _ : state) {
    std::atomic_int join(thread_num);
    auto optimize = [&]() {
      // OPTIMIZE: remove memory allocation overhead
      std::vector<float> data(ids.size() * 33);
      std::vector<absl::Span<float>> embeddings;
      embeddings.reserve(ids.size());
      for (size_t i = 0; i < ids.size(); ++i) {
        embeddings.push_back(absl::MakeSpan(data.data() + i * 33, 33));
      }
      table->BatchLookup(absl::MakeSpan(ids), absl::MakeSpan(embeddings));
      --join;
    };

    // Simulate multi-workers lookup simultaneously
    for (int64_t i = 0; i < thread_num; ++i) {
      thread_pool.Schedule(optimize);
    }
    while (join) {
    }
  }
}

void BM_Optimize(benchmark::State& state) {  // NOLINT
  int64_t thread_num = state.range(0);
  monolith::concurrency::ThreadPool thread_pool(thread_num);
  std::vector<float> grad(33, 1.f);

  for (auto _ : state) {
    std::atomic_int join(thread_num);
    auto optimize = [&]() {
      for (int64_t id : ids) {
        table->Optimize(id, absl::MakeSpan(grad), {0.01f, 0.01f}, 0);
      }
      --join;
    };

    // Simulate multi-workers optimize simultaneously
    for (int64_t i = 0; i < thread_num; ++i) {
      thread_pool.Schedule(optimize);
    }
    while (join) {
    }
  }
}

void BM_BatchOptimize(benchmark::State& state) {  // NOLINT
  int64_t thread_num = state.range(0);
  monolith::concurrency::ThreadPool thread_pool(thread_num);
  std::vector<float> data(ids.size() * 33, 1.f);
  std::vector<absl::Span<const float>> grads;
  grads.reserve(ids.size());
  for (size_t i = 0; i < ids.size(); ++i) {
    grads.emplace_back(absl::MakeSpan(data.data() + i * 33, 33));
  }

  for (auto _ : state) {
    std::atomic_int join(thread_num);
    auto optimize = [&]() {
      table->BatchOptimize(absl::MakeSpan(ids), absl::MakeSpan(grads), {0.01f, 0.01f}, 0);
      --join;
    };

    // Simulate multi-workers optimize simultaneously
    for (int64_t i = 0; i < thread_num; ++i) {
      thread_pool.Schedule(optimize);
    }
    while (join) {
    }
  }
}

BENCHMARK(BM_LookUp)->Arg(1)->Arg(10);
BENCHMARK(BM_BatchLookUp)->Arg(1)->Arg(10);

BENCHMARK(BM_Optimize)->Arg(1)->Arg(10);
BENCHMARK(BM_BatchOptimize)->Arg(1)->Arg(10);

}  // namespace
}  // namespace hash_table
}  // namespace monolith

BENCHMARK_MAIN();
