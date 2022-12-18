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

#include <vector>

#include "glog/logging.h"
#include "idl/matrix/compression/compression.h"
#include "idl/matrix/compression/float16.h"

namespace matrix {
namespace compression {

using matrix::compression::Float16;

bool compress_float_list_f16(const char* raw_data, const size_t raw_size,
                             char* out_buffer, size_t* out_size) {
  if ((raw_size % sizeof(float)) != 0) {
    LOG(ERROR) << "compress_float_list_f16 got invalid input data";
    return false;
  }
  size_t num = raw_size / sizeof(float);
  if (sizeof(Float16) * num > *out_size) {
    LOG(ERROR) << "compress_float_list_f16 out_buffer size not enough";
    return false;
  }
  const float* raw_floats = reinterpret_cast<const float*>(raw_data);
  Float16* f16_buffer = reinterpret_cast<Float16*>(out_buffer);
  *out_size = 0;
  for (size_t i = 0; i < num; ++i) {
    f16_buffer[i].set(raw_floats[i]);
    (*out_size) += sizeof(Float16);
  }
  return true;
}

bool compress_float_list_f16(const char* raw_data, const size_t raw_size,
                             std::string* out) {
  if ((raw_size % sizeof(float)) != 0) {
    LOG(ERROR) << "compress_float_list_f16 got invalid input data";
    return false;
  }
  size_t num = raw_size / sizeof(float);
  size_t out_size = num * sizeof(Float16);
  out->resize(out_size);
  const float* raw_floats = reinterpret_cast<const float*>(raw_data);
  Float16* f16_buffer =
      reinterpret_cast<Float16*>(const_cast<char*>(out->data()));
  for (size_t i = 0; i < num; ++i) {
    f16_buffer[i].set(raw_floats[i]);
  }
  return true;
}

bool decompress_float_list_f16(const char* compressed_data,
                               size_t compressed_size, char* out_buffer,
                               size_t* out_size) {
  if ((compressed_size % sizeof(Float16)) != 0) {
    LOG(ERROR) << "decompress_float_list_f16 got invalid data";
    return false;
  }
  size_t num = compressed_size / sizeof(Float16);
  if (sizeof(float) * num > *out_size) {
    LOG(ERROR) << "decompress_float_list_f16 got no enough out_buffer";
    return false;
  }
  const Float16* f16_buffer = reinterpret_cast<const Float16*>(compressed_data);
  float* out_floats = reinterpret_cast<float*>(out_buffer);
  *out_size = 0;
  for (size_t i = 0; i < num; ++i) {
    out_floats[i] = f16_buffer[i].get_m();
    (*out_size) += sizeof(float);
  }
  return true;
}

bool decompress_float_list_f16(const char* compressed_data,
                               size_t compressed_size, std::string* out) {
  if ((compressed_size % sizeof(Float16)) != 0) {
    LOG(ERROR) << "decompress_float_list_f16 got invalid data";
    return false;
  }
  size_t num = compressed_size / sizeof(Float16);
  size_t out_size = num * sizeof(float);
  out->resize(out_size);
  const Float16* f16_buffer = reinterpret_cast<const Float16*>(compressed_data);
  float* out_floats = reinterpret_cast<float*>(const_cast<char*>(out->data()));
  for (size_t i = 0; i < num; ++i) {
    out_floats[i] = f16_buffer[i].get_m();
  }
  return true;
}

using bfloat16 = uint16_t;

bool compress_float_list_f16b(const char* raw_data, const size_t raw_size,
                              std::string* out) {
  if ((raw_size % sizeof(float)) != 0) {
    LOG(ERROR) << "compress_float_list_f16 got invalid input data";
    return false;
  }
  size_t num = raw_size / sizeof(float);
  size_t out_size = num * sizeof(bfloat16);
  out->resize(out_size);
  const uint16_t* p = reinterpret_cast<const uint16_t*>(raw_data);
  uint16_t* q = reinterpret_cast<uint16_t*>(const_cast<char*>(out->data()));
  for (; num != 0; p += 2, q++, num--) {
    *q = p[1];
  }
  return true;
}

bool decompress_float_list_f16b(const char* compressed_data,
                                size_t compressed_size, std::string* out) {
  if ((compressed_size % sizeof(bfloat16)) != 0) {
    LOG(ERROR) << "decompress_float_list_f16 got invalid data";
    return false;
  }
  size_t num = compressed_size / sizeof(bfloat16);
  size_t out_size = num * sizeof(float);
  out->resize(out_size);

  const uint16_t* p = reinterpret_cast<const uint16_t*>(compressed_data);
  uint16_t* q = reinterpret_cast<uint16_t*>(const_cast<char*>(out->data()));
  for (; num != 0; p++, q += 2, num--) {
    q[0] = 0;
    q[1] = *p;
  }
  return true;
}

}  // end namespace compression
}  // end namespace matrix
