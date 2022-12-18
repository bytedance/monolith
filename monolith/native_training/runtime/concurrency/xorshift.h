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

#ifndef MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_CONCURRENCY_XORSHIFT_H_
#define MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_CONCURRENCY_XORSHIFT_H_

#include <cstdint>
#include <ctime>
#include <iostream>
#include <limits>
#include <vector>

namespace monolith {
namespace concurrency {

class XorShift {
 public:
  XorShift() : p(0) {
    srand(time(0));
    x = (uint64_t)std::rand() * RAND_MAX + std::rand();
    for (uint64_t& i : s) {
      i = XorShift64Star();
    }
  }

  uint32_t Rand32() { return (uint32_t)XorShift128Plus(); }
  static uint32_t Rand32ThreadSafe();

 private:
  uint64_t XorShift1024Star();
  uint64_t XorShift128Plus();
  uint64_t XorShift64Star();

 private:
  uint64_t s[16];
  int p;
  uint64_t x; /* The state must be seeded with a nonzero value. */
};

}  // namespace concurrency
}  // namespace monolith

#endif  // MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_CONCURRENCY_XORSHIFT_H_
