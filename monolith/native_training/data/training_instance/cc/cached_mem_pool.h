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

#ifndef MONOLITH_NATIVE_TRAINING_DATA_TRAINING_INSTANCE_CC_CACHED_MEM_POOL_H_
#define MONOLITH_NATIVE_TRAINING_DATA_TRAINING_INSTANCE_CC_CACHED_MEM_POOL_H_

#include <memory>
#include <mutex>
#include <vector>

namespace tensorflow {
namespace monolith_tf {

class CachedMemPool {
 public:
  static CachedMemPool* init(size_t buffer_size);

  std::unique_ptr<char[]> allocate();
  void deallocate(std::unique_ptr<char[]>& buffer);

  // Test Only method.
  size_t get_buffer_size() {
    std::unique_lock<std::mutex> lock(alloc_mtx_);
    return cached_buffers_.size();
  }

 private:
  explicit CachedMemPool(size_t buffer_size) : buffer_size_(buffer_size) {}
  ~CachedMemPool() { cached_buffers_.clear(); }

  size_t buffer_size_;

  std::mutex alloc_mtx_;
  size_t total_requested_ = 0;
  std::vector<std::unique_ptr<char[]>> cached_buffers_;
  static CachedMemPool* cached_mem_pool;
};

}  // namespace monolith_tf
}  // namespace tensorflow

#endif  // MONOLITH_NATIVE_TRAINING_DATA_TRAINING_INSTANCE_CC_CACHED_MEM_POOL_H_
