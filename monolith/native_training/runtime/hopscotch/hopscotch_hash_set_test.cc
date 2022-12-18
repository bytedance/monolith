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

#include "monolith/native_training/runtime/hopscotch/hopscotch_hash_set.h"
#include <sys/time.h>
#include <algorithm>
#include <set>
#include <thread>
#include <unordered_set>
#include <vector>

#include "gtest/gtest.h"
#include "google/malloc_extension.h"


namespace monolith {
namespace hopscotch {
namespace {

using FID = int64_t;
constexpr int kMaxNumKeys = 2097152;
constexpr int kConcurrencyLevel = 200;
constexpr int kSeed = 2233333;

uint64_t GetTime() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000000ULL + tv.tv_usec;
}

static size_t MemoryUsage() {
  size_t result = 0;
  if (MallocExtension::instance()->GetNumericProperty(
          "generic.current_allocated_bytes", &result)) {
    return result;
  }
  return 0;
}

static size_t memory_last = 0;
static uint64_t time_last = 0;

void Reset() {
  memory_last = MemoryUsage();
  time_last = GetTime();
}

void Report() {
  printf("time:%6.1f ms, memory:%6.1f M\n", (GetTime() - time_last) / 1000.0,
         (MemoryUsage() - memory_last) / (1024.0 * 1024));
  Reset();
}

TEST(HOPSCOTCH_HASH_SET, simple_test) {
  HopscotchHashSet<FID> hash_set(1000, 1);
  std::vector<FID> keys;
  for (int i = 0; i < 1000; ++i) {
    keys.emplace_back(std::rand());
    hash_set.insert(keys.back());
  }
  auto all = hash_set.GetAndClear();
  ASSERT_EQ(all.size(), 1000);
  std::sort(all.begin(), all.end());
  std::sort(keys.begin(), keys.end());
  for (int i = 0; i < 1000; ++i) {
    ASSERT_EQ(keys[i], all[i]);
  }
}

template<class MapType>
void TestOneMap(MapType* map) {
  srand(kSeed);
  for (int i = 0; i < kMaxNumKeys; ++i) {
    map->insert(std::rand());
  }
}

// test google:dense_hash_set
// 2096080
// time:  94.3 ms, memory:  32.0 M
//
// test std::set
// 2096080
// time:1050.2 ms, memory:  96.0 M
//
// test std::unordered_set
// 2096080
// time: 333.8 ms, memory:  48.4 M
//
// test hopscotch_hash_set
// 2096080
// time: 188.3 ms, memory:  64.1 M
TEST(HOPSCOTCH_HASH_SET, compare_test) {
  Reset();

  std::cout << "test google:dense_hash_set" << std::endl;
  google::dense_hash_set<FID> dense_hash_set;
  dense_hash_set.set_empty_key(-1);
  TestOneMap(&dense_hash_set);
  std::cout << dense_hash_set.size() << std::endl;
  Report();

  std::cout << "test std::set" << std::endl;
  std::set<FID> std_set;
  TestOneMap(&std_set);
  std::cout << std_set.size() << std::endl;
  Report();

  std::cout << "test std::unordered_set" << std::endl;
  std::unordered_set<FID> std_unordered_set;
  TestOneMap(&std_unordered_set);
  std::cout << std_unordered_set.size() << std::endl;
  Report();

  std::cout << "test hopscotch_hash_set" << std::endl;
  HopscotchHashSet<FID> hash_set(kMaxNumKeys, 1);
  TestOneMap(&hash_set);
  std::cout << hash_set.size() << std::endl;
  Report();
}

TEST(HOPSCOTCH_HASH_SET, multithread_test) {
  HopscotchHashSet<FID> hash_set(kMaxNumKeys, 1000);
  srand(kSeed);
  for (int num_thread = 1; num_thread <= 10; ++num_thread) {
    std::cout << "test for " << num_thread << " threads" << std::endl;
    std::vector<FID> keys;
    for (int i = 0; i < kMaxNumKeys; ++i) {
      keys.emplace_back(std::rand());
    }
    std::vector<std::thread> writers(num_thread);
    for (int i = 0; i < num_thread; ++i) {
      writers[i] = std::thread(
          [&](int index) {
            for (int j = index; j < kMaxNumKeys; j += num_thread) {
              EXPECT_EQ(0, hash_set.insert(keys[j]));
            }
          },
          i);
    }
    for (int i = 0; i < num_thread; ++i) {
      writers[i].join();
    }
    auto all = hash_set.GetAndClear();
    std::sort(all.begin(), all.end());
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    std::cout << "insert finished. total insert keys: " << keys.size()
              << std::endl;
    ASSERT_EQ(all.size(), keys.size());
    for (int i = 0; i < keys.size(); ++i) {
      ASSERT_EQ(keys[i], all[i]);
    }
  }
}

TEST(HOPSCOTCH_HASH_SET, overflow_test) {
  HopscotchHashSet<FID> hash_set(kMaxNumKeys, 1000);
  const int num_thread = 10;
  const int num_keys = kMaxNumKeys * 20 + 10000;
  std::vector<FID> keys(num_keys);
  for (int i = 0; i < num_keys; ++i) {
    keys[i] = i;
  }
  std::vector<std::vector<int>> dropped_keys(num_thread);
  std::vector<std::thread> writers(num_thread);
  for (int i = 0; i < num_thread; ++i) {
    writers[i] = std::thread(
        [&](int index) {
          for (int j = index; j < num_keys; j += num_thread) {
            int result = hash_set.insert(keys[j]);
            if (result != 0) {
              dropped_keys[index].emplace_back(result);
            }
          }
        },
        i);
  }
  for (int i = 0; i < num_thread; ++i) {
    writers[i].join();
  }
  int clear_times = 0;
  for (int i = 0; i < num_thread; ++i) {
    clear_times += dropped_keys[i].size();
  }
  EXPECT_EQ(clear_times, 20);
}

}  // namespace
}  // namespace hopscotch
}  // namespace monolith
