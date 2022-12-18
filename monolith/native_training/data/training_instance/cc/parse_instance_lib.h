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

#ifndef MONOLITH_NATIVE_TRAINING_DATA_TRAINING_INSTANCE_CC_PARSE_INSTANCE_LIB_H_
#define MONOLITH_NATIVE_TRAINING_DATA_TRAINING_INSTANCE_CC_PARSE_INSTANCE_LIB_H_
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "idl/matrix/proto/proto_parser.pb.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace monolith_tf {

// The config to instantiate ParseInstanceSpec.
struct InstanceParserConfig {
  // Fid features.
  std::vector<int> fidv1_features;
  std::vector<std::string> fidv2_features;
  enum FidOutputType {
    // Each fid will have its own ragged tensor.
    REGULAR,
    // All fids will be outputted as a single ragged tensor.
    // Only available when collapse_batch_dim == True.
    CONCAT,
  };
  FidOutputType fid_output_type = REGULAR;

  // Float features.
  std::vector<std::string> float_features;
  std::vector<int> float_feature_dims;

  // Int64 features.
  std::vector<std::string> int64_features;
  std::vector<int> int64_feature_dims;

  // String features.
  std::vector<std::string> string_features;
  std::vector<int> string_feature_dims;

  // LineId related features, including labels and others.
  std::vector<std::string> misc_float_features;
  std::vector<int> misc_float_dims;
  std::vector<std::string> misc_int64_features;
  std::vector<int> misc_int64_dims;
  std::vector<std::string> misc_string_features;
  std::vector<int> misc_string_dims;

  bool collapse_batch_dim = false;
};

// A parser that is able to parse instance.
// Must call Init() before used.
class InstanceParser {
 public:
  explicit InstanceParser(const InstanceParserConfig &config);
  ~InstanceParser();

  Status Init();

  struct Output {
    std::vector<Tensor> tensors;
  };

  Status Parse(OpKernelContext *ctx,
               absl::Span<const parser::proto::Instance> instances,
               Output *tensors) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace monolith_tf
}  // namespace tensorflow
#endif  // MONOLITH_NATIVE_TRAINING_DATA_TRAINING_INSTANCE_CC_PARSE_INSTANCE_LIB_H_
