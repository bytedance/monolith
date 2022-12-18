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

#ifndef MONOLITH_NATIVE_TRAINING_DATA_TRAINING_INSTANCE_CC_UE_COMPRESS_H_
#define MONOLITH_NATIVE_TRAINING_DATA_TRAINING_INSTANCE_CC_UE_COMPRESS_H_

#include "glog/logging.h"
#include "idl/matrix/compression/compression.h"
#include "idl/matrix/compression/float16.h"
#include "idl/matrix/proto/proto_parser.pb.h"

namespace tensorflow {
namespace monolith_tf {

enum UECompressMethod {
  COMPRESS_QTZ8 = 0  // 8bit 量化
};

class UECompress {
 public:
  UECompress() = default;
  virtual ~UECompress() = default;

  bool compress_embeddings(::idl::matrix::proto::Feature* feature_column,
                           UECompressMethod compress_method);
  bool decompress_embeddings(
      const ::idl::matrix::proto::Feature& feature_column,
      std::vector<float>* embedding, UECompressMethod compress_method);
};

}  // namespace monolith_tf
}  // namespace tensorflow
#endif  // MONOLITH_NATIVE_TRAINING_DATA_TRAINING_INSTANCE_CC_UE_COMPRESS_H_
