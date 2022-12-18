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

#include "thread_pool.h"

namespace monolith {
namespace concurrency {

ThreadPool::~ThreadPool() {
  {
    absl::MutexLock l(&mu_);
    for (size_t i = 0; i < threads_.size(); i++) {
      queue_.push(nullptr);  // Shutdown signal.
    }
  }
  for (auto &t : threads_) {
    t.join();
  }
}

void ThreadPool::Schedule(std::function<void()> func) {
  assert(func != nullptr);
  absl::MutexLock l(&mu_);
  queue_.push(std::move(func));
}

void ThreadPool::WorkLoop() {
  while (true) {
    std::function<void()> func;
    {
      absl::MutexLock l(&mu_);
      mu_.Await(absl::Condition(this, &ThreadPool::WorkAvailable));
      func = std::move(queue_.front());
      queue_.pop();
    }
    if (func == nullptr) {  // Shutdown signal.
      break;
    }
    func();
  }
}

}  // namespace concurrency
}  // namespace monolith
