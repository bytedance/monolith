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

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

namespace monolith {
namespace hopscotch {

using FID = int64_t;

inline static uint32_t NextPowerOfTwo(uint32_t n) {
  --n;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return n + 1;
}

inline static int FirstLsbBitIndex(uint32_t x) { return __builtin_ffs(x) - 1; }

template <typename Key>
HopscotchHashSet<Key>::HopscotchHashSet(uint32_t capacity,
                                        uint32_t concurrency_level)
    : capacity_(capacity) {
  init_ = false;
  lock_mask_ = NextPowerOfTwo(concurrency_level) - 1;
  bucket_mask_ = NextPowerOfTwo(capacity * 1.2) - 1;
  init_lock_.Init();
  num_elements_.store(0, std::memory_order_seq_cst);  // 可能还没init就会获取
}

template <typename Key>
void HopscotchHashSet<Key>::DoInit() {
  table_.resize(bucket_mask_ + kHopscotchHashInsertRange + 1);
  locks_.resize(lock_mask_ + 1);
  for (size_t i = 0; i <= lock_mask_; ++i) {
    locks_[i].Init();
  }
  extra_lock_.Init();
  clear_lock_.Init();
  num_elements_.store(0, std::memory_order_seq_cst);
  running_threads_.store(0, std::memory_order_seq_cst);
  DoClear();
}

template <typename Key>
void HopscotchHashSet<Key>::FindCloserFreeBucket(
    const concurrency::MicroOneBitSpinLock* lock, int* free_bucket,
    int* free_dist) {
  int move_bucket = *free_bucket - (kHopscotchHashHopRange - 1);
  int move_free_dist;
  for (move_free_dist = kHopscotchHashHopRange - 1; move_free_dist > 0;
       --move_free_dist) {
    auto new_lock = &locks_[move_bucket & lock_mask_];
    uint32_t start_hop_info = table_[move_bucket].hop_info;
    int move_new_free_dist = !start_hop_info ? kHopscotchHashHopRange
                                             : __builtin_ctz(start_hop_info);
    if (move_new_free_dist < move_free_dist) {
      if (new_lock != lock) {
        new_lock->Lock();
      }
      if (start_hop_info == table_[move_bucket].hop_info) {
        // new_free_bucket -> free_bucket and empty new_free_bucket
        int new_free_bucket = move_bucket + move_new_free_dist;
        table_[*free_bucket].key = table_[new_free_bucket].key;
        table_[*free_bucket].hash = table_[new_free_bucket].hash;
        table_[move_bucket].hop_info |= 1u << move_free_dist;
        table_[move_bucket].hop_info &= ~(1u << move_new_free_dist);

        *free_bucket = new_free_bucket;
        *free_dist -= move_free_dist - move_new_free_dist;
        if (new_lock != lock) {
          new_lock->Unlock();
        }
        return;
      }
      if (new_lock != lock) {
        new_lock->Unlock();
      }
    }
    ++move_bucket;
  }
  *free_bucket = -1;
  *free_dist = 0;
}

template <typename Key>
size_t HopscotchHashSet<Key>::insert(Key key) {
  // we do lazy init here to save memory
  if (!init_) {
    init_lock_.Lock();
    if (!init_) DoInit();
    init_ = true;
    init_lock_.Unlock();
  }
  size_t dropped_keys = 0;
  if (unlikely(size() > capacity_)) {
    clear_lock_.Lock();
    if (likely(size() > capacity_)) {
      for (int i = 0; i < locks_.size(); ++i) locks_[i].Lock();
      dropped_keys = size();
      this->DoClear();
      for (int i = 0; i < locks_.size(); ++i) locks_[i].Unlock();
    }
    clear_lock_.Unlock();
  }
  uint32_t hash = HashFunc(key);
  auto lock = &locks_[hash & lock_mask_];
  lock->Lock();
  int bucket = hash & bucket_mask_;
  uint32_t hop_info = table_[bucket].hop_info;
  // check if already exists
  while (0 != hop_info) {
    int i = FirstLsbBitIndex(hop_info);
    int current = bucket + i;
    if (key == table_[current].key) {
      lock->Unlock();
      return dropped_keys;
    }
    hop_info &= ~(1U << i);
  }
  // looking for free bucket
  int free_bucket = bucket, free_dist = 0;
  for (; free_dist < kHopscotchHashInsertRange; ++free_dist, ++free_bucket) {
    if (kHopscotchHashEmpty == table_[free_bucket].hash &&
        kHopscotchHashEmpty ==
            __sync_val_compare_and_swap(&table_[free_bucket].hash,
                                        kHopscotchHashEmpty, hash)) {
      break;
    }
  }

  // insert the new key
  num_elements_.fetch_add(1, std::memory_order_relaxed);
  if (free_dist < kHopscotchHashInsertRange) {
    do {
      if (free_dist < kHopscotchHashHopRange) {
        table_[free_bucket].key = key;
        table_[free_bucket].hash = hash;
        table_[bucket].hop_info |= 1u << free_dist;
        lock->Unlock();
        return dropped_keys;
      }
      FindCloserFreeBucket(lock, &free_bucket, &free_dist);
    } while (-1 != free_bucket);
  } else {
    // insert failed, insert into extra_ map
    extra_lock_.Lock();
    extra_.insert(key);
    extra_lock_.Unlock();
  }
  lock->Unlock();
  return dropped_keys;
}

template <typename Key>
std::vector<Key> HopscotchHashSet<Key>::GetAndClear() {
  if (!init_) return {};
  clear_lock_.Lock();
  for (int i = 0; i < locks_.size(); ++i) locks_[i].Lock();
  std::vector<Key> results(size());
  size_t index = 0;
  for (auto&& entry : table_) {
    if (entry.hash) {
      results[index++] = entry.key;
    }
    entry.hash = 0;
    entry.key = kEmptyKey;
    entry.hop_info = 0;
  }
  for (auto&& key : extra_) {
    results[index++] = key;
  }
  extra_.clear();
  num_elements_.store(0, std::memory_order_seq_cst);
  for (int i = 0; i < locks_.size(); ++i) locks_[i].Unlock();
  clear_lock_.Unlock();
  return results;
}

template <typename Key>
void HopscotchHashSet<Key>::DoClear() {
  for (size_t i = 0; i < table_.size(); ++i) {
    table_[i].hash = 0;
    table_[i].key = kEmptyKey;
    table_[i].hop_info = 0;
  }
  num_elements_.store(0, std::memory_order_seq_cst);
  extra_.clear();
}

template class HopscotchHashSet<FID>;
template class HopscotchHashSet<std::pair<int64_t, const void*>>;

template <>
FID GetEmptyValue<FID>() {
  return -1;
}

template <>
std::pair<int64_t, const void*>
GetEmptyValue<std::pair<int64_t, const void*>>() {
  return std::make_pair(-1, nullptr);
}

}  // namespace hopscotch
}  // namespace monolith
