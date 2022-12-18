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

#include <cstring>
#include <memory>
#include <thread>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace monolith {
namespace allocator {
namespace {

TEST(BlockAllocatorTest, Basic) {
  std::unique_ptr<BlockAllocator> allocator =
      std::make_unique<BlockAllocator>();

  size_t size1 = 10;
  char* ptr1 = reinterpret_cast<char*>(allocator->Allocate(size1));
  EXPECT_NE(ptr1, nullptr);

  ptr1[size1 - 1] = 'x';
  EXPECT_EQ(allocator->AllocatedSize(), BlockAllocator::kStartBlcokSize);

  size_t size2 = 12;
  char* ptr2 = reinterpret_cast<char*>(allocator->Allocate(size2));

  // Test Align.
  EXPECT_EQ(ptr2 - ptr1, 16);

  // Can't fit into current block anymore due to align.
  size_t size3 = BlockAllocator::kStartBlcokSize - size1 - size2;
  char* ptr3 = reinterpret_cast<char*>(allocator->Allocate(size3));
  EXPECT_NE(ptr3, nullptr);

  // Next block will be doubled.
  EXPECT_EQ(allocator->AllocatedSize(), BlockAllocator::kStartBlcokSize * 3);

  allocator->DeallocateAll();
  EXPECT_EQ(allocator->AllocatedSize(), 0);
}

TEST(BlockAllocatorTest, AllocateLarge) {
  size_t block_size = 1 << 20;
  auto allocator = std::make_unique<BlockAllocator>();
  char* p = reinterpret_cast<char*>(allocator->Allocate(block_size));
  std::memset(p, 0, block_size);
}

TEST(BlockAllocatorTest, TSBlockAllocator) {
  TSBlockAllocator alloc;
  auto func = [&alloc]() {
    for (int i = 0; i < 100; ++i) {
      char* p = reinterpret_cast<char*>(alloc.Allocate(16));
      std::memset(p, 0, 16);
    }
  };
  std::vector<std::thread> ths;
  for (int i = 0; i < 15; ++i) {
    ths.push_back(std::thread(func));
  }
  for (auto& th : ths) {
    th.join();
  }
  alloc.DeallocateAll();
  EXPECT_THAT(alloc.AllocatedSize(), 0);
}

TEST(EmbeddingBlockAllocatorTest, EmbeddingBlockAllocatorAllocateMany) {
  EmbeddingBlockAllocator alloc(8);
  for (int i = 0; i < EmbeddingBlockAllocator::kMaxEntryNum * 10; ++i) {
    auto addr = alloc.AllocateOne();
    void* real_addr = alloc.GetEntryPointer(addr);
    std::memset(real_addr, 0, 8);
  }
}

TEST(EmbeddingBlockAllocatorTest, TSEmbeddingBlockAllocator) {
  TSEmbeddingBlockAllocator alloc(16);
  auto func = [&alloc]() {
    for (int i = 0; i < 100; ++i) {
      EntryAddress p = alloc.AllocateOne();
      std::memset(static_cast<char*>(alloc.GetEntryPointer(p)), 0, 16);
    }
  };
  std::vector<std::thread> threads;
  for (int i = 0; i < 15; ++i) {
    threads.emplace_back(func);
  }
  for (auto& t : threads) {
    t.join();
  }
  alloc.DeallocateAll();
  EXPECT_THAT(alloc.AllocatedSize(), 0);
}

}  // namespace
}  // namespace allocator
}  // namespace monolith
