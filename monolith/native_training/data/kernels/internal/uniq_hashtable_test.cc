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

#include "monolith/native_training/data/kernels/internal/uniq_hashtable.h"

#include <unistd.h>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"

using FID = uint64_t;

namespace tensorflow {
namespace monolith_tf {
namespace {

class UniqHashTableTest {
 public:
  UniqHashTable uniq_hashtable;
  std::vector<FID> fids;
  int fid_num_ = 0;
  int fid_range_ = 0;

  UniqHashTableTest(int fid_num, int fid_range) {
    fid_num_ = fid_num;
    fid_range_ = fid_range;
  }

  void Reset() {
    for (size_t fi = 0; fi < fid_num_; ++fi) {
      fids.resize(fid_num_);
      fids[fi] = (rand() % fid_range_) << 8;
    }
  }

  void Check() {
    Reset();
    std::unordered_map<FID, uint32_t> result;
    std::vector<FID> uniq_fids;
    for (const auto& fid : fids) {
      auto uniq_idx1 = result.size();
      auto uniq_idx2 = uniq_hashtable.UniqFid(fid, uniq_hashtable.Size());
      auto iter = result.find(fid);
      if (iter != result.end()) {
        EXPECT_EQ(iter->second, uniq_idx2) << "size: " << uniq_hashtable.Size();
      } else {
        result[fid] = uniq_idx1;
        EXPECT_EQ(uniq_idx1, uniq_idx2);
        EXPECT_EQ(uniq_idx2 + 1, uniq_hashtable.Size());
        uniq_fids.push_back(fid);
      }
    }
    EXPECT_EQ(uniq_fids.size(), result.size());
    EXPECT_EQ(result.size(), uniq_hashtable.Size());
    // check no repetition
    result.clear();
    for (const auto& fid : uniq_fids) {
      EXPECT_EQ(result.count(fid), 0);
      result.insert({fid, 0});
    }
  }
};

TEST(UniqHashTableTest, Small) {
  size_t fid_num = 1e3;
  size_t fid_range = 1e2;
  UniqHashTableTest test(fid_num, fid_range);
  test.Check();
}

TEST(UniqHashTableTest, Medium) {
  size_t fid_num = 1e5;
  size_t fid_range = 1e4;
  UniqHashTableTest test(fid_num, fid_range);
  test.Check();
}

TEST(UniqHashTableTest, Reset) {
  size_t fid_num = 1e5;
  size_t fid_range = 1e4;
  UniqHashTableTest test(fid_num, fid_range);
  test.Check();
  test.uniq_hashtable.Reset();
  test.Check();
}

TEST(UniqHashTableTest, ReqId) {
  size_t fid_num = 1e3;
  size_t fid_range = 1e2;
  UniqHashTableTest test(fid_num, fid_range);
  test.Check();
  for (uint64_t i = 0; i < static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1; i++) {
    test.uniq_hashtable.Reset();
  }
  test.Check();
}


}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow
