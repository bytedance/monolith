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

#include "monolith/native_training/runtime/hash_filter/sliding_hash_filter.h"

#include <fstream>
#include <unordered_map>

#include "gtest/gtest.h"
#include "monolith/native_training/runtime/hash_filter/types.h"

namespace monolith {
namespace hash_filter {
namespace {

static void test_simple_bloom(size_t key_num) {
  SlidingHashFilter counter(key_num, 10);
  for (uint32_t i = 0; i <= HashFilter<uint16_t>::max_count(); ++i) {
    EXPECT_EQ(i, counter.add(1, 1));
  }
  ASSERT_EQ(HashFilter<uint16_t>::max_count(), counter.add(1, 1));
}

TEST(SlidingHashFilterTest, test_simple) {
  test_simple_bloom(1);
  test_simple_bloom(3);
  test_simple_bloom(100);
}

TEST(SlidingHashFilterTest, test_count) {
  SlidingHashFilter filter(1000000, 10);
  filter.add(std::rand(), 2);
  EXPECT_EQ(1llu, filter.estimated_total_element());
  size_t key_number = 10;
  for (size_t i = 0; i != key_number; ++i) {
    filter.add(std::rand(), 2);
  }
  EXPECT_EQ(key_number + 1, filter.estimated_total_element());

  SlidingHashFilter filter2(1000000, 10);
  filter2.add(10000002961562801052lu, 1);
  filter2.add(10000002961562801052lu, 20);
  filter2.add(10000002961562801052lu, 1);
  EXPECT_EQ(1llu, filter2.estimated_total_element());

  std::unique_ptr<SlidingHashFilter> filter3(filter2.clone());
  EXPECT_TRUE(filter2 == *filter3);
}

template <typename Container>
static void check_conflict_rate(SlidingHashFilter& filter,
                                const Container& map_counter,
                                double expected_rate) {
  int error_counter = 0;
  for (auto iter = map_counter.begin(); iter != map_counter.end(); ++iter) {
    if (filter.get(iter->first) != iter->second) error_counter += 1;
  }
  double error_rate = error_counter / double(map_counter.size());
  std::cout << "expect:" << expected_rate << " actual:" << error_rate
            << std::endl;
  EXPECT_NEAR(expected_rate, error_rate, expected_rate / 2);
  EXPECT_GT(map_counter.size() / 10000, filter.failure_count());
}

static void compare_to_unordered_map(int key_number, double expected_rate,
                                     size_t capacity) {
  std::srand(capacity);
  int split_num = 10;
  SlidingHashFilter filter(capacity, split_num);
  std::unordered_map<int, uint32_t> map_counter;
  for (int i = 0; i != key_number; ++i) {
    int num = std::rand();
    if (map_counter[num] < HashFilter<uint16_t>::max_count() - 1) {
      map_counter[num] += 2;
      filter.add(num, 2);
    }
  }
  check_conflict_rate(filter, map_counter, expected_rate);
}

TEST(SlidingHashFilterTest, compare_to_unordered_map) {
  compare_to_unordered_map(1000000, 0.00908, 1000000);
  compare_to_unordered_map(1000000, 0.50, 500000);
}

TEST(SlidingHashFilterTest, SkipZeroThresholdFeatures) {
  SlidingHashFilter filter(1000000, 10);
  for (uint32_t i = 0; i < 5; ++i) {
    int64_t fid_with_zero_threshold = i;
    EXPECT_FALSE(filter.ShouldBeFiltered(fid_with_zero_threshold, 1, 0,
                                         nullptr /* table */));
    int64_t normal_fid = i * 2;
    EXPECT_TRUE(filter.ShouldBeFiltered(normal_fid, 1, 1, nullptr /* table */));
  }
  // Only the normal_fids can be added to the filter.
  EXPECT_EQ(5llu, filter.estimated_total_element());
}

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */

}  // namespace
}  // namespace hash_filter
}  // namespace monolith
