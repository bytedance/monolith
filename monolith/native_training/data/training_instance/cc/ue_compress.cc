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

#include <string.h>

#include "monolith/native_training/data/training_instance/cc/ue_compress.h"

namespace tensorflow {
namespace monolith_tf {

using matrix::compression::Float16;

const char* UE_COMPRESS_FLAG = "UE_QTZ";

bool UECompress::compress_embeddings(
    ::idl::matrix::proto::Feature* feature_column,
    UECompressMethod compress_method) {
  if (compress_method == UECompressMethod::COMPRESS_QTZ8) {
    auto* bytes_value = feature_column->add_bytes_value();
    int embedding_size = feature_column->float_value_size();
    std::vector<float> compress_input;
    compress_input.reserve(embedding_size);
    for (auto value : feature_column->float_value()) {
      compress_input.push_back(value);
    }
    std::string compress_out;
    bool ret = matrix::compression::compress_float_list_qtz8mm(
        (const char*)compress_input.data(), embedding_size * sizeof(float),
        &compress_out);
    if (!ret) {
      LOG(ERROR) << "compress_embeddings failed, feature_column name=%s"
                 << feature_column->name().c_str();
      return false;
    }
    *bytes_value = UE_COMPRESS_FLAG + compress_out;
    return true;
  } else {
    LOG(ERROR) << "compress_embeddings invalid compress method:%d"
               << compress_method;
    return false;
  }
}

bool UECompress::decompress_embeddings(
    const idl::matrix::proto::Feature& feature_column,
    std::vector<float>* embedding, UECompressMethod compress_method) {
  if (compress_method == UECompressMethod::COMPRESS_QTZ8) {
    std::string compress_out;
    std::string bytes_value;
    for (auto& value : feature_column.bytes_value()) {
      if (value.find(UE_COMPRESS_FLAG) == 0) {
        bytes_value = value.substr(strlen(UE_COMPRESS_FLAG));
      }
    }
    if (bytes_value.empty()) {
      return false;
    }
    bool ret = matrix::compression::decompress_float_list_qtz8mm(
        (const char*)bytes_value.data(), bytes_value.size(), &compress_out);
    if (!ret) {
      LOG(ERROR) << "decompress_embeddings failed, feature_column name=%s"
                 << feature_column.name().c_str();
      return false;
    }
    size_t embedding_size = bytes_value.size() - 2 * sizeof(Float16);
    const float* output = reinterpret_cast<const float*>(compress_out.data());
    embedding->clear();
    for (size_t i = 0; i < embedding_size; ++i) {
      embedding->emplace_back(output[i]);
    }
    return true;
  } else {
    LOG(ERROR) << "decompress_embeddings invalid compress method:%d"
               << compress_method;
    return false;
  }
}

}  // namespace monolith_tf
}  // namespace tensorflow
