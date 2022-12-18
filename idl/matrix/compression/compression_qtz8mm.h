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

#ifndef IDL_MATRIX_COMPRESSION_COMPRESSION_QTZ8MM_H_
#define IDL_MATRIX_COMPRESSION_COMPRESSION_QTZ8MM_H_

#include <string>
#include <vector>
#include "idl/matrix/compression/float16.h"

namespace matrix {
namespace compression {

inline void get_from_qtz8mm(float* dest, const int& dim,
                            const matrix::compression::Float16* w,
                            const uint8_t* v) {
  float min = w[0].get();
  float max = w[1].get();
  float step = (max - min) / 255;
  for (int i = 0; i < dim; ++i) {
    dest[i] = min + step * v[i];
  }
}

inline void set_to_qtz8mm(const float* src_list,
                          const int& data_size,  // data_size 可能<dim
                          const int& dim, matrix::compression::Float16* w,
                          uint8_t* v) {
  float min = src_list[0];
  float max = src_list[0];
  for (int i = 0; i < data_size && i < dim; ++i) {
    float tmp = src_list[i];
    min = std::min(min, tmp);
    max = std::max(max, tmp);
  }
  if (data_size < dim) {
    min = std::min(min, 0.0f);
    max = std::max(max, 0.0f);
  }
  float step = (max - min) / 255;
  w[0] = min;
  w[1] = max;
  for (int i = 0; i < data_size && i < dim; ++i) {
    float tmp = src_list[i];
    v[i] = int(0.5f + (tmp - min) / step);
  }
  for (int i = data_size; i < dim; ++i) {
    float tmp = 0.0f;
    v[i] = int(0.5f + (tmp - min) / step);
  }
}

}  // end namespace compression
}  // end namespace matrix

#endif  // IDL_MATRIX_COMPRESSION_COMPRESSION_QTZ8MM_H_
