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

#include "glog/logging.h"

#include "idl/matrix/compression/compression.h"
#include "idl/matrix/compression/compression_qtz8mm.h"

using matrix::compression::Float16;

namespace matrix {
namespace compression {

bool compress_float_list_qtz8mm(const char* raw_data, const size_t raw_size,
                                std::string* out) {
  if ((raw_size % sizeof(float)) != 0) {
    LOG(ERROR) << "compress_float_list_f16 got invalid input data";
    return false;
  }
  if (raw_size <= 4 * sizeof(float)) {
    // 如果长度 <=
    // 4时，由于min/max需要各占用2B，会导致qtz8mm相对于f16不节省内存，此时直接用f16
    return compress_float_list_f16(raw_data, raw_size, out);
  }

  size_t num = raw_size / sizeof(float);
  size_t out_size = 2 * sizeof(Float16) + num * sizeof(uint8_t);
  out->resize(out_size);
  char* qtz_buf_ptr = const_cast<char*>(out->data());

  const float* raw_floats = reinterpret_cast<const float*>(raw_data);
  Float16* w_ptr = reinterpret_cast<Float16*>(qtz_buf_ptr);
  uint8_t* v_ptr =
      reinterpret_cast<uint8_t*>(qtz_buf_ptr + 2 * sizeof(Float16));

  set_to_qtz8mm(raw_floats, num, num, w_ptr, v_ptr);
  return true;
}

bool decompress_float_list_qtz8mm(const char* compressed_data,
                                  size_t compressed_size, std::string* out) {
  if (compressed_size <= 4 * sizeof(Float16)) {
    // 长度<=8时，原始的数据长度<=4，直接用f16反解
    return decompress_float_list_f16(compressed_data, compressed_size, out);
  }

  size_t num = compressed_size - 2 * sizeof(Float16);
  size_t out_size = num * sizeof(float);
  out->resize(out_size);
  const char* qtz_buf_ptr = static_cast<const char*>(compressed_data);

  const Float16* w_ptr = reinterpret_cast<const Float16*>(qtz_buf_ptr);
  const uint8_t* v_ptr =
      reinterpret_cast<const uint8_t*>(qtz_buf_ptr + 2 * sizeof(Float16));
  float* out_floats = reinterpret_cast<float*>(const_cast<char*>(out->data()));

  get_from_qtz8mm(out_floats, num, w_ptr, v_ptr);
  return true;
}

}  // end namespace compression
}  // end namespace matrix
