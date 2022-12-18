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

#include "monolith/native_training/runtime/allocator/block_allocator.h"

namespace monolith {
namespace hash_table {

// A wrapper for raw pointer. This helps utilize try_emplace in map.
// TODO(leqi.zou): Essentailly we want to deprecate this. Will remove once
// we find this is not useful.

class PackedEntry {
 public:
  explicit PackedEntry(allocator::TSEmbeddingBlockAllocator* alloc)
      : p_(alloc->AllocateOne()), timestamp_(0) {}

  allocator::EntryAddress get_entry_addr() const { return p_; }

  uint32_t GetTimestamp() const { return timestamp_; }

  void SetTimestamp(uint32_t timestamp_sec) { timestamp_ = timestamp_sec; }

 private:
  allocator::EntryAddress p_;

  // Unix timestamp in seconds, UINT32_MAX means 2106-02-07 14:28:15+08:00
  uint32_t timestamp_;
};

class RawEntry {
 public:
  RawEntry(size_t entry_size) : p_(new char[entry_size]) {}

  void* get() const { return p_.get(); }

  uint32_t GetTimestamp() const { return timestamp_; }

  void SetTimestamp(uint32_t timestamp_sec) { timestamp_ = timestamp_sec; }

 private:
  std::unique_ptr<char[]> p_;
  // Unix timestamp in seconds, UINT32_MAX means 2106-02-07 14:28:15+08:00
  uint32_t timestamp_;
};

template <int64_t length>
class InlineEntry {
 public:
  static_assert(length % 8 == 0 && length > 0,
                "InlineEntry's should be divisible by 8.");

  InlineEntry() {
    static_assert(sizeof(InlineEntry<length>) == length,
                  "InlineEntry's implementation is wrong");
  }
  static int capacity() { return length - 4; }
  const void* get() const { return buffer_; }
  void* get() { return buffer_; }

  uint32_t GetTimestamp() const {
    return *reinterpret_cast<const uint32_t*>(buffer_ + length - 4);
  }

  void SetTimestamp(uint32_t timestamp_sec) {
    *reinterpret_cast<uint32_t*>(buffer_ + length - 4) = timestamp_sec;
  }

 private:
  char buffer_[length];
};

}  // namespace hash_table
}  // namespace monolith