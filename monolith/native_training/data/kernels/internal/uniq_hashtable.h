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

#ifndef MONOLITH_NATIVE_TRAINING_DATA_KERNELS_INTERNAL_UNIQ_HASHTABLE_H_
#define MONOLITH_NATIVE_TRAINING_DATA_KERNELS_INTERNAL_UNIQ_HASHTABLE_H_

#include <sys/types.h>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/raw_coding.h"

#include "absl/base/macros.h"
#include "glog/logging.h"


#define MONOLITH_INLINE __attribute__((always_inline))

namespace tensorflow {
namespace monolith_tf {

class UniqHashTable {
  using FID = uint64_t;
  static constexpr uint32_t MIN_BUCKET_CAP = 1u << 10;
  static constexpr FID EMPTY_FID = static_cast<FID>(-1);
  static constexpr uint32_t ILLEGAL_BUCKET =
      std::numeric_limits<uint32_t>::max();
  static constexpr float LOAD_FACTOR = 0.75;

  struct HTItem {
    HTItem() = default;
    HTItem(const FID fid, uint32_t req_id, uint32_t uniq_idx) : fid(fid), req_id(req_id), uniq_idx(uniq_idx) {}
    HTItem(const HTItem& other) = default;
    HTItem& operator=(const HTItem& other) = default;

    bool operator==(const HTItem& other) const { return fid == other.fid && req_id == other.req_id; }

    bool IsEmpty(uint32_t cur_req_id) const { return req_id != cur_req_id || fid == EMPTY_FID;}

    FID fid = EMPTY_FID;
    uint32_t req_id = 0;
    uint32_t uniq_idx = 0;
  };

  struct HTIdx {
    HTIdx() = default;
    HTIdx(uint32_t item_pos, uint32_t insert_pos) : item_pos(item_pos), insert_pos(insert_pos) {}

    uint32_t item_pos = ILLEGAL_BUCKET;
    uint32_t insert_pos = ILLEGAL_BUCKET;
  };

  using HTItemPtr = HTItem*;

 public:
  UniqHashTable() : capacity_(MIN_BUCKET_CAP), num_elements_(0), cur_req_id_(1) {
    prob_table_ = CreateProbTable(MIN_BUCKET_CAP);
    bucket_idx_mask_ = MIN_BUCKET_CAP - 1;
    expand_threshold_ = static_cast<uint32_t>(capacity_ * LOAD_FACTOR);
  }

  ~UniqHashTable() {
    DeleteProbTable(prob_table_, capacity_);
  }

  uint32_t UniqFid(const FID fid, const uint32_t uniq_idx) {
    DCHECK_NE(fid, EMPTY_FID);
    auto idx = FindPosition(fid, prob_table_, bucket_idx_mask_, cur_req_id_);
    if (idx.item_pos != ILLEGAL_BUCKET) {
      DCHECK(idx.insert_pos == ILLEGAL_BUCKET);
      return prob_table_[idx.item_pos].uniq_idx;
    } else {
      DCHECK(idx.insert_pos != ILLEGAL_BUCKET);
      DCHECK_LE(idx.insert_pos, bucket_idx_mask_);
      prob_table_[idx.insert_pos] = HTItem(fid, cur_req_id_, uniq_idx);
      num_elements_++;
    }
    MaybeExpand();
    return uniq_idx;
  }

  void Reset() {
    num_elements_ = 0;
    if (++cur_req_id_ == 0) {
      FillTableWithEmptyItem(prob_table_, capacity_);
    }
    // std::cerr << "abcd " << cur_req_id_ << std::endl;
  }

  size_t Size() {
    return static_cast<size_t>(num_elements_);
  }

  size_t Capacity() {
    return static_cast<size_t>(capacity_);
  }

 private:
  static MONOLITH_INLINE HTIdx FindPosition(const FID fid, HTItemPtr prob_table, uint32_t bucket_idx_mask, uint32_t req_id) {
    uint32_t bucket_idx = FidHash(fid) & bucket_idx_mask;
    while (true) {
      const auto& item = prob_table[bucket_idx];
      if (item == HTItem(fid, req_id, 0)) {
        return HTIdx(bucket_idx, ILLEGAL_BUCKET);
      } else if (item.IsEmpty(req_id)) {
        return HTIdx(ILLEGAL_BUCKET, bucket_idx);
      }
      bucket_idx = (bucket_idx + 1) & bucket_idx_mask;
    }
  }

  static MONOLITH_INLINE bool TestEqual(const HTItem& item, const FID fid, uint32_t req_id) {
    return item.fid == fid && item.req_id == req_id;
  }

  static MONOLITH_INLINE HTItemPtr CreateProbTable(uint32_t capacity) {
    auto* prob_table = malloc(sizeof(HTItem) * capacity);
    FillTableWithEmptyItem(reinterpret_cast<HTItemPtr>(prob_table), capacity);
    return reinterpret_cast<HTItemPtr>(prob_table);
  }

  static MONOLITH_INLINE void FillTableWithEmptyItem(HTItemPtr prob_table, uint32_t capacity) {
    DCHECK(!!prob_table);
    static HTItem empty_item(EMPTY_FID, 0, 0);
    std::uninitialized_fill_n(prob_table, capacity, empty_item);
  }

  void DeleteProbTable(HTItemPtr prob_table, uint32_t capacity) {
    DCHECK(!!prob_table);
    if (!std::is_trivial<HTItem>::value) {
      for (uint32_t i = 0; i < capacity; ++i) {
        prob_table[i].~HTItem();
      }
    }
    free(prob_table);
  }

  void MaybeExpand() {
    if (GOOGLE_PREDICT_TRUE(num_elements_ <= expand_threshold_)) {
      return;
    }
    auto new_capacity = capacity_ << 1;
    auto new_bucket_idx_mask = new_capacity - 1;
    auto new_prob_table = CreateProbTable(new_capacity);
    for (uint32_t i = 0; i < capacity_; ++i) {
      if (prob_table_[i].IsEmpty(cur_req_id_)) {
        continue;
      }
      uint32_t bucket_idx = FidHash(prob_table_[i].fid) & new_bucket_idx_mask;
      while (!new_prob_table[bucket_idx].IsEmpty(cur_req_id_)) {
        bucket_idx = (bucket_idx + 1) & new_bucket_idx_mask;
      }
      new_prob_table[bucket_idx] = prob_table_[i];
    }
    DeleteProbTable(prob_table_, capacity_);
    prob_table_ = new_prob_table;
    capacity_ = new_capacity;
    expand_threshold_ = static_cast<uint32_t>(capacity_ * LOAD_FACTOR);
    bucket_idx_mask_ = capacity_ - 1;
  }

  static MONOLITH_INLINE uint32_t FidHash(const FID fid) {
    return Hash(reinterpret_cast<const char*>(&fid), sizeof(FID), 0);
  }

  // Copy from tensorflow/tsl/lib/io/cache.cc
  // Question: 这里怎么引用比较规范？
  static uint32_t Hash(const char* data, size_t n, uint32_t seed) {
    // Similar to murmur hash
    const uint32_t m = 0xc6a4a793;
    const uint32_t r = 24;
    const char* limit = data + n;
    uint32_t h = seed ^ (n * m);

    // Pick up four bytes at a time
    while (data + 4 <= limit) {
      uint32_t w = tensorflow::core::DecodeFixed32(data);
      data += 4;
      h += w;
      h *= m;
      h ^= (h >> 16);
    }

    // Pick up remaining bytes
    switch (limit - data) {
      case 3:
        h += static_cast<uint8_t>(data[2]) << 16;
        ABSL_FALLTHROUGH_INTENDED;
      case 2:
        h += static_cast<uint8_t>(data[1]) << 8;
        ABSL_FALLTHROUGH_INTENDED;
      case 1:
        h += static_cast<uint8_t>(data[0]);
        h *= m;
        h ^= (h >> r);
        break;
    }
    return h;
  }


  HTItemPtr prob_table_ = nullptr;
  uint32_t capacity_ = 0;  // must be 2 power;
  uint32_t expand_threshold_ = 0;
  uint32_t bucket_idx_mask_ = 0;

  uint32_t num_elements_ = 0;
  uint32_t cur_req_id_ = 0;

  TF_DISALLOW_COPY_AND_ASSIGN(UniqHashTable);
};


class MultiShardUniqHashTable {
  using FID = uint64_t;

 public:
  MultiShardUniqHashTable() = default;
  ~MultiShardUniqHashTable() = default;

  void init(UniqHashTable *uniq_hashtable) {
    uniq_hashtable_ = uniq_hashtable;
  }

  size_t uniq_fid(const FID fid, int shard) {
    DCHECK_LT(shard, fid_lists_.size());
    // store all shards' uniq indices in a single uniq_hashtable_
    auto uniq_idx = uniq_hashtable_->UniqFid(fid, fid_lists_[shard].size());
    if (uniq_idx == fid_lists_[shard].size()) {
      fid_lists_[shard].push_back(fid);
    }
    return uniq_idx;
  }

  int fid_num(size_t shard) const {
    return static_cast<int>(fid_lists_[shard].size());
  }

  std::vector<FID>& fid_list(int shard) {
    DCHECK_LT(shard, fid_lists_.size());
    return fid_lists_[shard];
  }

  void reset() {
    uniq_hashtable_->Reset();
  }

  void resize(size_t shard_num) {
    fid_lists_.resize(shard_num);
  }

  void reserve(size_t fid_num) {
    for (auto& fid_list : fid_lists_) {
      fid_list.reserve(fid_num);
    }
  }

 private:
  UniqHashTable* uniq_hashtable_;
  std::vector<std::vector<FID>> fid_lists_;
};

}  // namespace monolith_tf
}  // namespace tensorflow

#undef MONOLITH_INLINE

#endif  // MONOLITH_NATIVE_TRAINING_DATA_KERNELS_INTERNAL_UNIQ_HASHTABLE_H_
