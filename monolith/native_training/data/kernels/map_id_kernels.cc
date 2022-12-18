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

#include <string>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace monolith_tf {
namespace {

template <typename T>
class MapIdOp : public OpKernel {
 public:
  explicit MapIdOp(OpKernelConstruction *context) : OpKernel(context) {
    std::vector<T> from, to;
    OP_REQUIRES_OK(context, context->GetAttr("from_value", &from));
    OP_REQUIRES_OK(context, context->GetAttr("to_value", &to));
    OP_REQUIRES_OK(context, context->GetAttr("default_value", &default_value_));

    CHECK_EQ(from.size(), to.size());
    for (size_t i = 0; i < from.size(); ++i) {
      map_.insert({from[i], to[i]});
    }
  }

  void Compute(OpKernelContext *context) override {
    const Tensor &input_tensor = context->input(0);
    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto input_flat = input_tensor.flat<T>();
    auto output_flat = output_tensor->flat<T>();
    for (size_t i = 0; i < input_flat.size(); ++i) {
      const T &value = input_flat(i);
      auto iter = map_.find(value);
      if (iter == map_.end()) {
        output_flat(i) = default_value_;
      } else {
        output_flat(i) = iter->second;
      }
    }
  }

 private:
  std::unordered_map<T, T> map_;
  T default_value_;
};

REGISTER_KERNEL_BUILDER(
    Name("MapId").Device(DEVICE_CPU).TypeConstraint<int32>("T"),
    MapIdOp<int32>);

REGISTER_KERNEL_BUILDER(
    Name("MapId").Device(DEVICE_CPU).TypeConstraint<int64>("T"),
    MapIdOp<int64>);

}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow
