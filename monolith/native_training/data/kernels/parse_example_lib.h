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

#ifndef MONOLITH_NATIVE_TRAINING_DATA_KERNELS_PARSE_EXAMPLE_LIB_H_
#define MONOLITH_NATIVE_TRAINING_DATA_KERNELS_PARSE_EXAMPLE_LIB_H_

#include <unordered_map>
#include <unordered_set>

#include "google/protobuf/descriptor.h"
#include "idl/matrix/proto/example.pb.h"
#include "idl/matrix/proto/proto_parser.pb.h"

#include "monolith/native_training/data/kernels/internal/label_utils.h"
#include "monolith/native_training/data/training_instance/cc/reader_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace monolith_tf {

class BaseParser {
 public:
  explicit BaseParser(const std::vector<std::string> &names,
                      const std::vector<int> &shapes,
                      const std::vector<DataType> &dtypes,
                      const std::vector<std::string> extra_names,
                      DataType input_dtype);

 protected:
  void AllocateFeatures(OpKernelContext *ctx,
                        std::vector<Tensor *> *out_tensors,
                        OpOutputList *out_list, int batch_size);

  void AllocateRaggedValues(OpKernelContext *ctx,
                            std::vector<Tensor *> *out_tensors,
                            OpOutputList *out_list, int batch_size);

  void FillFeature(OpKernelContext *ctx,
                   const ::monolith::io::proto::Feature &feature,
                   Tensor *tensor, const std::string &name, int shape,
                   int offset);

  void FillFromLineId(OpKernelContext *ctx,
                      const ::idl::matrix::proto::LineId &line_id,
                      std::vector<Tensor *> *out_tensors, const int offset);

  Status FillFromLineIdByreflection(
      const ::idl::matrix::proto::LineId &line_id,
      const ::google::protobuf::FieldDescriptor *field, Tensor *tensor,
      int shape, int offset);

  std::unordered_map<std::string, std::tuple<int, int, DataType>> name2info_;
  std::unordered_map<int, std::tuple<std::string, int, DataType>> idx2info_;
  std::unordered_set<std::string> ragged_names_;
  std::vector<std::string> extra_names_;
  DataType input_dtype_;

  const ::google::protobuf::Descriptor *descriptor =
      ::idl::matrix::proto::LineId::GetDescriptor();
  const ::google::protobuf::Reflection *reflection =
      ::idl::matrix::proto::LineId::GetReflection();
};

class ExampleParser : public BaseParser {
 public:
  explicit ExampleParser(const std::vector<std::string> &names,
                         const std::vector<int> &shapes,
                         const std::vector<DataType> &dtypes,
                         const std::vector<std::string> extra_names,
                         DataType input_dtype, FeatureNameMapper *mapper);

  void Parse(
      OpKernelContext *ctx,
      const std::vector<const ::monolith::io::proto::Example *> &examples,
      OpOutputList *out_list);

 private:
  FeatureNameMapper *mapper_ = nullptr;
};

class ExampleBatchParser : public BaseParser {
 public:
  explicit ExampleBatchParser(const std::vector<std::string> &names,
                              const std::vector<int> &shapes,
                              const std::vector<DataType> &dtypes,
                              const std::vector<std::string> extra_names,
                              DataType input_dtype);

  void Parse(OpKernelContext *ctx,
             const ::monolith::io::proto::ExampleBatch &example_batch,
             OpOutputList *out_list);
};

class ExampleBatchListParser : public BaseParser {
 public:
  explicit ExampleBatchListParser(const std::vector<std::string> &names,
                                  const std::vector<int> &shapes,
                                  const std::vector<DataType> &dtypes,
                                  const std::vector<std::string> &extra_names,
                                  DataType input_dtype);

  void Parse(OpKernelContext *ctx,
             const ::monolith::io::proto::ExampleBatch &example_batchs,
             const std::vector<internal::TaskConfig> &label_config_,
             float positive_label, float negative_label,
             OpOutputList *out_list);

 private:
  uint64 mask_ = (1 << 48) - 1;

  void FillLabelFromLineId(
      OpKernelContext *ctx, const ::idl::matrix::proto::LineId &line_id,
      const std::vector<internal::TaskConfig> &label_config_,
      float positive_label, float negative_label, Tensor *out_tensor,
      const int offset);
};

}  // namespace monolith_tf
}  // namespace tensorflow
#endif MONOLITH_NATIVE_TRAINING_DATA_KERNELS_PARSE_EXAMPLE_LIB_H_
