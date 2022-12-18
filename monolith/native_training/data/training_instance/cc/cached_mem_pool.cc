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

#include "glog/logging.h"

namespace tensorflow {
namespace monolith_tf {

std::mutex session_mutex;
CachedMemPool* CachedMemPool::cached_mem_pool = nullptr;

CachedMemPool* CachedMemPool::init(size_t buffer_size) {
  std::unique_lock<std::mutex> lock(session_mutex);
  if (cached_mem_pool == nullptr) {
    cached_mem_pool = new CachedMemPool(buffer_size);
  }
  return cached_mem_pool;
}

std::unique_ptr<char[]> CachedMemPool::allocate() {
  std::unique_lock<std::mutex> lock(alloc_mtx_);
  if (cached_buffers_.empty()) {
    total_requested_++;
    return std::make_unique<char[]>(buffer_size_);
  } else {
    auto buffer = std::move(cached_buffers_.back());
    cached_buffers_.pop_back();
    return buffer;
  }
}

void CachedMemPool::deallocate(std::unique_ptr<char[]>& buffer) {
  std::unique_lock<std::mutex> lock(alloc_mtx_);
  cached_buffers_.emplace_back(std::move(buffer));
}

}  // namespace monolith_tf
}  // namespace tensorflow
