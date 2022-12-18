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

#include "monolith/native_training/runtime/concurrency/xorshift.h"

namespace monolith {
namespace concurrency {

uint64_t XorShift::XorShift1024Star() {
  uint64_t s0 = s[p];
  uint64_t s1 = s[p = (p + 1) & 15];
  s1 ^= s1 << 31;  // a
  s1 ^= s1 >> 11;  // b
  s0 ^= s0 >> 30;  // c
  return (s[p] = s0 ^ s1) * UINT64_C(1181783497276652981);
}

uint64_t XorShift::XorShift128Plus() {
  uint64_t x = s[0];
  uint64_t const y = s[1];
  s[0] = y;
  x ^= x << 23;        // a
  x ^= x >> 17;        // b
  x ^= y ^ (y >> 26);  // c
  s[1] = x;
  return x + y;
}

uint64_t XorShift::XorShift64Star() {
  x ^= x >> 12;  // a
  x ^= x << 25;  // b
  x ^= x >> 27;  // c
  return x * UINT64_C(2685821657736338717);
}

uint32_t XorShift::Rand32ThreadSafe() {
  static thread_local XorShift xor_shift;
  return xor_shift.Rand32();
}

}  // namespace concurrency
}  // namespace monolith
