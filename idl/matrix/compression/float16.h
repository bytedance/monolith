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

#ifndef FLOAT_16_H
#define FLOAT_16_H

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <limits>
#include "third_party/half_sourceforge_net/half.hpp"

namespace matrix {
namespace compression {

class Float16 {
 public:
  Float16() {}

  Float16(const Float16& other) : value(other.value) {}

  Float16(float vf) { set(vf); }

  void set(float vf) { value = vf; }

  float get() const {
    float val = value;
    return std::isinf(val) ? ((val < 0) ? -65504 : 65504) : val;
  }

  /*
   * get value with random rounding value
   *
   * explain:
   *     we want to store 1.23456 in a 16 bit unit, but because of truncation
   *     what we really store is 1.234
   *
   *     get_r() will return (1.234 + random(0, 1) * 0.001) to mitigate the
   *     truncation error
   */
  float get_r() const { return get() + random_rounding_value(); }

  /*
   * get value with median rounding value
   *
   * explain:
   *     we want to store 1.23456 in a 16 bit unit, but because of truncation
   *     what we really store is 1.234
   *
   *     get_m() will return (1.234 + 0.0005) to mitigate the truncation error
   */
  float get_m() const {
    if ((value.get_data() & 0x7FFF) == 0)
      return 0;
    else
      return get() + median_rounding_value();
  }

  unsigned short get_raw_data() const { return value.get_data(); }

 private:
  half_float::half value;

  /*
   * random make use of Marsaglia's xorshf generator to generator float
   * number in [0, 1]
   *
   * About Marsaglia's xorshf generator, see
   * [stackoverflow](http://stackoverflow.com/a/1640399)]
   */
  static float random(void) {
    static unsigned long x = 123456789, y = 362436069, z = 521288629;

    static unsigned long cnt = 0;

    if (!(cnt = (cnt + 1) & 0xFFFFFFFF)) {
      x = 123456789;
      y = 362436069;
      z = 521288629;
    }

    unsigned long t;
    x ^= x << 16;
    x ^= x >> 5;
    x ^= x << 1;

    t = x;
    x = y;
    y = z;
    z = t ^ x ^ y;

    return (double)(z & 0xFFFFFFFF) / (unsigned long)(0xFFFFFFFF);
  }

  float random_rounding_value() const {
    static constexpr float v[64] = {
        std::pow(2, -25),  std::pow(2, -24),  std::pow(2, -23),
        std::pow(2, -22),  std::pow(2, -21),  std::pow(2, -20),
        std::pow(2, -19),  std::pow(2, -18),  std::pow(2, -17),
        std::pow(2, -16),  std::pow(2, -15),  std::pow(2, -14),
        std::pow(2, -13),  std::pow(2, -12),  std::pow(2, -11),
        std::pow(2, -10),  std::pow(2, -9),   std::pow(2, -8),
        std::pow(2, -7),   std::pow(2, -6),   std::pow(2, -5),
        std::pow(2, -4),   std::pow(2, -3),   std::pow(2, -2),
        std::pow(2, -1),   std::pow(2, 0),    std::pow(2, 1),
        std::pow(2, 2),    std::pow(2, 3),    std::pow(2, 4),
        std::pow(2, 5),    std::pow(2, 6),    -std::pow(2, -25),
        -std::pow(2, -24), -std::pow(2, -23), -std::pow(2, -22),
        -std::pow(2, -21), -std::pow(2, -20), -std::pow(2, -19),
        -std::pow(2, -18), -std::pow(2, -17), -std::pow(2, -16),
        -std::pow(2, -15), -std::pow(2, -14), -std::pow(2, -13),
        -std::pow(2, -12), -std::pow(2, -11), -std::pow(2, -10),
        -std::pow(2, -9),  -std::pow(2, -8),  -std::pow(2, -7),
        -std::pow(2, -6),  -std::pow(2, -5),  -std::pow(2, -4),
        -std::pow(2, -3),  -std::pow(2, -2),  -std::pow(2, -1),
        -std::pow(2, 0),   -std::pow(2, 1),   -std::pow(2, 2),
        -std::pow(2, 3),   -std::pow(2, 4),   -std::pow(2, 5),
        -std::pow(2, 6),
    };
    return v[value.get_data() >> 10] * random();
  }

  float median_rounding_value() const {
    static constexpr float v[64] = {
        std::pow(2, -26),  std::pow(2, -25),  std::pow(2, -24),
        std::pow(2, -23),  std::pow(2, -22),  std::pow(2, -21),
        std::pow(2, -20),  std::pow(2, -19),  std::pow(2, -18),
        std::pow(2, -17),  std::pow(2, -16),  std::pow(2, -15),
        std::pow(2, -14),  std::pow(2, -13),  std::pow(2, -12),
        std::pow(2, -11),  std::pow(2, -10),  std::pow(2, -9),
        std::pow(2, -8),   std::pow(2, -7),   std::pow(2, -6),
        std::pow(2, -5),   std::pow(2, -4),   std::pow(2, -3),
        std::pow(2, -2),   std::pow(2, -1),   std::pow(2, 0),
        std::pow(2, 1),    std::pow(2, 2),    std::pow(2, 3),
        std::pow(2, 4),    std::pow(2, 5),    -std::pow(2, -26),
        -std::pow(2, -25), -std::pow(2, -24), -std::pow(2, -23),
        -std::pow(2, -22), -std::pow(2, -21), -std::pow(2, -20),
        -std::pow(2, -19), -std::pow(2, -18), -std::pow(2, -17),
        -std::pow(2, -16), -std::pow(2, -15), -std::pow(2, -14),
        -std::pow(2, -13), -std::pow(2, -12), -std::pow(2, -11),
        -std::pow(2, -10), -std::pow(2, -9),  -std::pow(2, -8),
        -std::pow(2, -7),  -std::pow(2, -6),  -std::pow(2, -5),
        -std::pow(2, -4),  -std::pow(2, -3),  -std::pow(2, -2),
        -std::pow(2, -1),  -std::pow(2, 0),   -std::pow(2, 1),
        -std::pow(2, 2),   -std::pow(2, 3),   -std::pow(2, 4),
        -std::pow(2, 5),
    };
    return v[value.get_data() >> 10];
  }
};

}  // end namespace compression
}  // end namespace matrix
#endif /* FLOAT_16_H */