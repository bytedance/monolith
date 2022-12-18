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

#include "cached_mem_pool.h"

#include <memory>
#include <thread>

#include "gtest/gtest.h"

namespace tensorflow {
namespace monolith_tf {

TEST(CachedMemPoolTest, Basic) {
  CachedMemPool* mem_pool = CachedMemPool::init(1024 * 1024);
  auto buffer = mem_pool->allocate();
  EXPECT_EQ(mem_pool->get_buffer_size(), 0);
  mem_pool->deallocate(buffer);
  EXPECT_EQ(mem_pool->get_buffer_size(), 1);
}

TEST(CachedMemPoolTest, RecursiveAllocation) {
  CachedMemPool* mem_pool = CachedMemPool::init(1024 * 1024);
  std::vector<std::unique_ptr<char[]>> buffers;
  for (int i = 0; i < 30; i++) {
    auto buffer = mem_pool->allocate();
    buffers.emplace_back(std::move(buffer));
  }
  EXPECT_EQ(mem_pool->get_buffer_size(), 0);
  for (auto& buffer : buffers) {
    mem_pool->deallocate(buffer);
  }
  EXPECT_EQ(mem_pool->get_buffer_size(), 30);
  for (int i = 0; i < 30; i++) {
    auto buffer = mem_pool->allocate();
    buffers.emplace_back(std::move(buffer));
  }
  EXPECT_EQ(mem_pool->get_buffer_size(), 0);
}

}  // namespace monolith_tf
}  // namespace tensorflow
