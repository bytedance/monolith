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


#ifndef IDL_MATRIX_COMPRESSION_COMPRESSION_H_
#define IDL_MATRIX_COMPRESSION_COMPRESSION_H_

#include <string>

namespace matrix {
namespace compression {

bool compress_float_list_f16(const char* raw_data, const size_t raw_size,
                             char* out_buffer, size_t* out_size);
bool compress_float_list_f16(const char* raw_data, const size_t raw_size,
                             std::string* out);
bool decompress_float_list_f16(const char* compressed_data,
                               size_t compressed_size, char* out_buffer,
                               size_t* out_size);
bool decompress_float_list_f16(const char* compressed_data,
                               size_t compressed_size, std::string* out);
bool compress_float_list_f16b(const char* raw_data, const size_t raw_size,
                              std::string* out);
bool decompress_float_list_f16b(const char* compressed_data,
                                size_t compressed_size, std::string* out);

// qtz8mm, with min/max qtz8, by liuyizhou
// 注意raw_size不是float的数量，而是buffer长度，所以是float数量*4
// 前两个函数主要是给 wudi.yx 使用，对于一个vec的数据做操作
bool compress_float_list_qtz8mm(const char* raw_data, const size_t raw_size,
                                std::string* out);
bool decompress_float_list_qtz8mm(const char* compressed_data,
                                  size_t compressed_size, std::string* out);
}  // end namespace compression
}  // end namespace matrix

#include "idl/matrix/compression/compression_qtz8mm.h"

#endif  // IDL_MATRIX_COMPRESSION_COMPRESSION_H_
