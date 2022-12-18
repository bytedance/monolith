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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_CONCURRENCY_THREAD_POOL_H_
#define MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_CONCURRENCY_THREAD_POOL_H_

#include <cassert>
#include <cstddef>
#include <functional>
#include <queue>
#include <thread>
#include <vector>
#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"

namespace monolith {
namespace concurrency {
/**
 * A simple ThreadPool implementation for tests/benchmarks.
 */
class ThreadPool {
 public:
  explicit ThreadPool(int num_threads) {
    for (int i = 0; i < num_threads; ++i) {
      threads_.emplace_back(&ThreadPool::WorkLoop, this);
    }
  }

  ThreadPool(const ThreadPool &) = delete;
  ThreadPool &operator=(const ThreadPool &) = delete;

  ~ThreadPool();

  // Schedule a function to be run on a ThreadPool thread immediately.
  void Schedule(std::function<void()> func);

 private:
  bool WorkAvailable() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return !queue_.empty();
  }

  void WorkLoop();

  absl::Mutex mu_;
  std::queue<std::function<void()>> queue_ ABSL_GUARDED_BY(mu_);
  std::vector<std::thread> threads_;
};

}  // namespace concurrency
}  // namespace monolith

#endif  // MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_CONCURRENCY_THREAD_POOL_H_
