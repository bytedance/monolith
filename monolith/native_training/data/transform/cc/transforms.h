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

#ifndef MONOLITH_NATIVE_TRAINING_DATA_TRANSFORM_CC_TRANSFORMS_H_
#define MONOLITH_NATIVE_TRAINING_DATA_TRANSFORM_CC_TRANSFORMS_H_

#include <memory>
#include <vector>

#include "idl/matrix/proto/example.pb.h"
#include "idl/matrix/proto/proto_parser.pb.h"
#include "monolith/native_training/data/transform/transform_config.pb.h"

namespace tensorflow {
namespace monolith_tf {

using monolith::native_training::data::AddLabelConfig;
using monolith::native_training::data::FilterByActionConfig;
using monolith::native_training::data::FilterByFidConfig;
using monolith::native_training::data::FilterByLabelConfig;
using monolith::native_training::data::TransformConfig;
using monolith::native_training::data::TransformConfig_OneTransformConfig;

class TransformInterface {
 public:
  virtual ~TransformInterface() = default;

  virtual void Transform(
      std::shared_ptr<::parser::proto::Instance>,
      std::vector<std::shared_ptr<::parser::proto::Instance>>*) = 0;

  virtual void Transform(
      std::shared_ptr<::monolith::io::proto::Example>,
      std::vector<std::shared_ptr<::monolith::io::proto::Example>>*) = 0;

  virtual void Transform(
      std::shared_ptr<::monolith::io::proto::ExampleBatch>,
      std::vector<std::shared_ptr<::monolith::io::proto::ExampleBatch>>*) = 0;
};

std::unique_ptr<TransformInterface> NewSampleCounter(
    std::unique_ptr<TransformInterface> transform,
    const std::string& transform_name = "end-2-end");

std::unique_ptr<TransformInterface> NewIdentity();

std::unique_ptr<TransformInterface> NewFilterByFid(FilterByFidConfig config);

std::unique_ptr<TransformInterface> NewFilterByAction(
    FilterByActionConfig config);

std::unique_ptr<TransformInterface> NewFilterByLabel(
    FilterByLabelConfig config);

std::unique_ptr<TransformInterface> NewAddLabel(AddLabelConfig config);

std::unique_ptr<TransformInterface> CombineTransforms(
    std::unique_ptr<TransformInterface> t1,
    std::unique_ptr<TransformInterface> t2);

std::unique_ptr<TransformInterface> NewTransformFromConfig(
    TransformConfig_OneTransformConfig config);

std::unique_ptr<TransformInterface> NewTransformFromConfig(
    const TransformConfig& config);

template <class T, class F>
void AssignOrCombine(T* t1, T t2, F combine_fn) {
  if (*t1 == nullptr) {
    *t1 = std::move(t2);
    return;
  }
  *t1 = combine_fn(std::move(*t1), std::move(t2));
}

}  // namespace monolith_tf
}  // namespace tensorflow

#endif  // MONOLITH_NATIVE_TRAINING_DATA_TRANSFORM_CC_TRANSFORMS_H_
