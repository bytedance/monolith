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

#ifndef MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_COMMON_LINALG_UTILS_H_
#define MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_COMMON_LINALG_UTILS_H_

#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>

namespace monolith {
namespace common {

template <class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
IsAlmostEqual(T x, T y, int ulp = 2) {
  // the machine epsilon has to be scaled to the magnitude of the values used
  // and multiplied by the desired precision in ULPs (units in the last place)
  return std::abs(x - y) <=
             std::numeric_limits<T>::epsilon() * std::abs(x + y) * ulp
         // unless the result is subnormal
         || std::abs(x - y) < std::numeric_limits<T>::min();
}

template <class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, T>::type
L2NormSquare(const T* data, size_t length) {
  T sum = 0;
  for (size_t i = 0; i < length; ++i) {
    sum += data[i] * data[i];
  }

  return sum;
}

}  // namespace common
}  // namespace monolith

#endif  // MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_COMMON_LINALG_UTILS_H_
