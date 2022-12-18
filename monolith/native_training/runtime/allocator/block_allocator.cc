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

#include <algorithm>

#include "monolith/native_training/runtime/allocator/block_allocator.h"

#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"

namespace monolith {
namespace allocator {

const int BlockAllocator::kStartBlcokSize = 1024;

void* BlockAllocator::Allocate(size_t cl) {
  size_t size = Align(cl);

  if (size <= free_) {
    void* ptr = reinterpret_cast<void*>(free_ptr_);
    free_ptr_ += size;
    free_ -= size;
    return ptr;
  } else {
    const size_t block_size = std::max(current_block_size_, size);
    if (current_block_size_ < max_block_size_) {
      current_block_size_ *= 2;
    }
    allocated_size_ += block_size;
    blocks_.push_back(std::make_unique<char[]>(block_size));
    char* block_ptr = blocks_.back().get();
    free_ptr_ = block_ptr + size;
    free_ = block_size - size;
    return reinterpret_cast<void*>(block_ptr);
  }
}

void BlockAllocator::DeallocateAll() {
  blocks_.clear();
  free_ = 0;
  allocated_size_ = 0;
}

BlockAllocator* GetThreadLocalAllocator(size_t key) {
  thread_local absl::flat_hash_map<size_t, std::unique_ptr<BlockAllocator>> m;
  auto it = m.find(key);
  if (it == m.end()) {
    auto it2 = m.insert({key, std::make_unique<BlockAllocator>()});
    return it2.first->second.get();
  } else {
    return it->second.get();
  }
}

}  // namespace allocator
}  // namespace monolith