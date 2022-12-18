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

#include <fstream>
#include <unordered_map>

#include <gtest/gtest.h>
#include "monolith/native_training/runtime/hash_filter/hash_filter.h"

namespace monolith {
namespace hash_filter {

template <typename DATA>
void test_simple_bloom(size_t key_num) {
  HashFilter<DATA> counter(key_num);
  for (uint32_t i = 0; i <= HashFilter<DATA>::max_count(); ++i) {
    uint32_t expect = std::min(i + 1, HashFilter<DATA>::max_count());
    EXPECT_EQ(i, counter.add(1, 1));
    EXPECT_EQ(expect, counter.get(1));
  }
  ASSERT_EQ(HashFilter<DATA>::max_count(), counter.add(1, 1));
  ASSERT_EQ(HashFilter<DATA>::max_count(), counter.get(1));
}

TEST(HashFilterTest, test_simple) {
  test_simple_bloom<uint8_t>(1);
  test_simple_bloom<uint8_t>(3);
  test_simple_bloom<uint8_t>(100);
  test_simple_bloom<uint16_t>(1);
  test_simple_bloom<uint16_t>(3);
  test_simple_bloom<uint16_t>(100);
}

template <typename DATA>
void test_count() {
  HashFilter<DATA> filter(1000000);
  std::srand(std::time(NULL));
  filter.add(std::rand(), 2);
  EXPECT_EQ(1llu, filter.estimated_total_element());
  int key_number = 10;
  for (int i = 0; i != key_number; ++i) {
    filter.add(std::rand(), 2);
  }
  EXPECT_EQ(size_t(key_number + 1), filter.estimated_total_element());

  HashFilter<DATA> filter2(1000000);
  filter2.add(10000002961562801052lu, 1);
  filter2.add(10000002961562801052lu, 20);
  filter2.add(10000002961562801052lu, 1);
  EXPECT_EQ(1llu, filter2.estimated_total_element());
  std::unique_ptr<HashFilter<DATA>> filter3(filter2.clone());
  EXPECT_TRUE(filter2 == *filter3);
}

TEST(HashFilterTest, test_count) {
  test_count<uint8_t>();
  test_count<uint16_t>();
}

template <typename DATA>
void compare_to_unordered_map(int key_number, double expected_rate,
                              double fill_rate) {
  HashFilter<DATA> filter(key_number, fill_rate);
  std::unordered_map<int, uint32_t> map_counter;
  std::srand(std::time(NULL));
  for (int i = 0; i != key_number; ++i) {
    int num = std::rand();
    if (map_counter[num] < HashFilter<DATA>::max_count() - 1) {
      map_counter[num] += 2;
      filter.add(num, 2);
    }
  }
  ASSERT_LE(filter.estimated_total_element(), map_counter.size());
  int error_counter = 0;
  for (auto iter = map_counter.begin(); iter != map_counter.end(); ++iter) {
    if (filter.get(iter->first) != iter->second) error_counter += 1;
  }
  double error_rate = error_counter / double(map_counter.size());
  double diff = error_rate / expected_rate;
  std::cout << "conflict rate diff " << diff << std::endl;
  EXPECT_GT(diff, 0.5);
  EXPECT_LT(diff, 2);
  EXPECT_EQ(0llu, filter.failure_count());
}

TEST(HashFilterTest, compare_to_unordered_map) {
  compare_to_unordered_map<uint8_t>(1000000, 0.0208, 4);
  compare_to_unordered_map<uint16_t>(1000000, 0.000242, 2);
}

template <typename DATA>
void TestSkipZeroThresholdFeatures() {
  HashFilter<DATA> filter(1000000, 10);
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

TEST(HashFilterTest, SkipZeroThresholdFeatures) {
  TestSkipZeroThresholdFeatures<uint8_t>();
  TestSkipZeroThresholdFeatures<uint16_t>();
}

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */

}  // namespace hash_filter
}  // namespace monolith
