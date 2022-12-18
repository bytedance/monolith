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

#ifndef MONOLITH_NATIVE_TRAINING_RUNTIME_ALLOCATOR_BLOCK_ALLOCATOR_H_
#define MONOLITH_NATIVE_TRAINING_RUNTIME_ALLOCATOR_BLOCK_ALLOCATOR_H_

#include <atomic>
#include <cassert>
#include <memory>
#include <vector>

#include <sys/param.h>
#include "absl/container/flat_hash_map.h"
#include "glog/logging.h"
#include "monolith/native_training/runtime/concurrency/xorshift.h"

namespace monolith {
namespace allocator {

//-------------------------------------------------------------------
// It is not thread safe!
//-------------------------------------------------------------------

class BlockAllocator {
 public:
  static const int kStartBlcokSize;

  BlockAllocator()
      : current_block_size_(kStartBlcokSize),
        allocated_size_(0),
        free_ptr_(nullptr),
        free_(0) {}

  BlockAllocator(const BlockAllocator &) = delete;
  BlockAllocator &operator=(const BlockAllocator &) = delete;

  ~BlockAllocator() {}

  // BlockAllocator owns the pointer.
  void *Allocate(size_t cl);

  void DeallocateAll();

  size_t AllocatedSize() { return allocated_size_; }

 private:
  size_t Align(size_t size) { return (size + (kAlign - 1)) & ~(kAlign - 1); }

  // This must be the power of 2.
  static const size_t kAlign = 8;

  std::vector<std::unique_ptr<char[]>> blocks_;
  size_t current_block_size_;
  size_t max_block_size_ = 1 * 1024 * 1024;
  size_t allocated_size_;
  char *free_ptr_;
  size_t free_;
};

// Thread safe version of BlockingAllocator by sharding.
class TSBlockAllocator {
 public:
  explicit TSBlockAllocator(int num_shards = 8) : num_shards_(num_shards) {
    for (int i = 0; i < num_shards_; ++i) {
      mus_.push_back(std::make_unique<absl::Mutex>());
      allocs_.push_back(std::make_unique<BlockAllocator>());
    }
  }

  // BlockAllocator owns the pointer.
  void *Allocate(size_t cl) {
    const int shard = concurrency::XorShift::Rand32ThreadSafe() % num_shards_;
    {
      absl::MutexLock l(mus_[shard].get());
      return allocs_[shard]->Allocate(cl);
    }
  }

  void DeallocateAll() {
    for (int shard = 0; shard < num_shards_; ++shard) {
      absl::MutexLock l(mus_[shard].get());
      allocs_[shard]->DeallocateAll();
    }
  }

  size_t AllocatedSize() {
    size_t allocated_size = 0;
    for (int shard = 0; shard < num_shards_; ++shard) {
      absl::MutexLock l(mus_[shard].get());
      allocated_size += allocs_[shard]->AllocatedSize();
    }
    return allocated_size;
  }

 private:
  int num_shards_;
  std::vector<std::unique_ptr<absl::Mutex>> mus_;
  std::vector<std::unique_ptr<BlockAllocator>> allocs_;
};

// This defines an address space for EmbeddingHashTable's RawEntry, it supports
// up to 2^32 entries.
struct EntryAddress {
  // 2^3 = 8 shards per thread-safe embedding block allocator
  uint32_t shard_id : 3;

  // No more than 2^17 = 131072 blocks per embedding allocator
  uint32_t block_id : 17;

  // 2^12 = 4096 entries per block
  uint32_t entry_id : 12;
};

// Thread compatible
class EmbeddingBlockAllocator {
 public:
  // This must be the power of 2.
  static const size_t kAlign = 8;
  static const size_t kMaxBlockNum = 1 << 17;
  static const size_t kMaxEntryNum = 1 << 12;

  explicit EmbeddingBlockAllocator(size_t entry_byte_size)
      : entry_byte_size_aligned_(Align(entry_byte_size)),
        block_size_(Align(entry_byte_size) * kMaxEntryNum) {
    Reset();
  }

  ~EmbeddingBlockAllocator() { FreeBlocks(); }

  EmbeddingBlockAllocator(const EmbeddingBlockAllocator &) = delete;
  EmbeddingBlockAllocator &operator=(const EmbeddingBlockAllocator &) = delete;

  void *GetEntryPointer(EntryAddress entry_address) const {
    return cur_block_head_.load(
               std::memory_order_relaxed)[entry_address.block_id] +
           entry_byte_size_aligned_ * entry_address.entry_id;
  }

  EntryAddress AllocateOne() {
    EntryAddress addr;
    if (entry_id_ < kMaxEntryNum) {
      addr.block_id = blocks_->size() - 1;
      addr.entry_id = entry_id_;
      entry_id_ += 1;
    } else {
      if (blocks_->size() == kMaxBlockNum) {
        throw std::bad_alloc();
      }
      if (blocks_->size() == blocks_->capacity()) {
        auto new_blocks = std::make_unique<std::vector<char *>>();
        new_blocks->reserve(blocks_->capacity() * 2);
        new_blocks->insert(new_blocks->begin(), blocks_->begin(),
                           blocks_->end());
        cur_block_head_.store(new_blocks->data());
        blocks_snapshots_.push_back(std::move(blocks_));
        blocks_ = std::move(new_blocks);
      }

      allocated_size_ += block_size_;
      blocks_->push_back(new char[block_size_]);
      addr.block_id = blocks_->size() - 1;
      addr.entry_id = 0;
      entry_id_ = 1;
    }
    return addr;
  }

  void DeallocateAll() { Reset(); }

  size_t AllocatedSize() { return allocated_size_; }

 private:
  size_t Align(size_t size) const {
    return (size + (kAlign - 1)) & ~(kAlign - 1);
  }

  void Reset() {
    FreeBlocks();
    blocks_snapshots_.clear();
    blocks_snapshots_.shrink_to_fit();
    blocks_ = std::make_unique<std::vector<char *>>();
    blocks_->reserve(1);
    allocated_size_ = 0;
    entry_id_ = kMaxEntryNum;
    cur_block_head_.store(blocks_->data());
  }

  void FreeBlocks() {
    if (blocks_) {
      for (char *block : *blocks_) {
        delete[] block;
      }
      blocks_ = nullptr;
    }
  }

  std::unique_ptr<std::vector<char *>> blocks_;
  // Stores blocks_.data(). Should be always valid.
  std::atomic<char **> cur_block_head_;
  // Used to save blocks snapshots, used for lock-free looking up
  std::vector<std::unique_ptr<std::vector<char *>>> blocks_snapshots_;
  size_t entry_byte_size_aligned_;
  size_t block_size_;
  size_t allocated_size_;
  size_t entry_id_;
};

// Thread safe version of EmbeddingBlockAllocator by sharding.
class TSEmbeddingBlockAllocator {
 public:
  explicit TSEmbeddingBlockAllocator(int64_t entry_byte_size) {
    for (int i = 0; i < kNumShards; ++i) {
      mus_.push_back(std::make_unique<absl::Mutex>());
      allocs_.push_back(
          std::make_unique<EmbeddingBlockAllocator>(entry_byte_size));
    }
  }

  void *GetEntryPointer(EntryAddress address) const {
    return allocs_[address.shard_id]->GetEntryPointer(address);
  }

  EntryAddress AllocateOne() {
    const int shard = concurrency::XorShift::Rand32ThreadSafe() % kNumShards;
    EntryAddress addr;
    {
      absl::WriterMutexLock l(mus_[shard].get());
      addr = allocs_[shard]->AllocateOne();
    }
    addr.shard_id = shard;
    return addr;
  }

  void DeallocateAll() {
    for (int shard = 0; shard < kNumShards; ++shard) {
      absl::MutexLock l(mus_[shard].get());
      allocs_[shard]->DeallocateAll();
    }
  }

  size_t AllocatedSize() {
    size_t allocated_size = 0;
    for (int shard = 0; shard < kNumShards; ++shard) {
      absl::MutexLock l(mus_[shard].get());
      allocated_size += allocs_[shard]->AllocatedSize();
    }
    return allocated_size;
  }

 private:
  static const size_t kNumShards = 1 << 3;

  std::vector<std::unique_ptr<absl::Mutex>> mus_;
  std::vector<std::unique_ptr<EmbeddingBlockAllocator>> allocs_;
};

}  // namespace allocator
}  // namespace monolith

#endif  // MONOLITH_NATIVE_TRAINING_RUNTIME_ALLOCATOR_BLOCK_ALLOCATOR_H_
