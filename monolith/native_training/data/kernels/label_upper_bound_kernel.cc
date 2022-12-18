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

#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_format.h"
#include "monolith/native_training/data/kernels/internal/label_utils.h"
#include "monolith/native_training/data/training_instance/cc/instance_utils.h"
#include "monolith/native_training/data/training_instance/cc/pb_variant.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace monolith_tf {

using IFeature = ::idl::matrix::proto::Feature;
using Instance = ::parser::proto::Instance;
using Example = ::monolith::io::proto::Example;
using LineId = ::idl::matrix::proto::LineId;

// label_upper_bound:
class LabelUpperBoundOp : public OpKernel {
 public:
  explicit LabelUpperBoundOp(OpKernelConstruction *context)
      : OpKernel(context) {
    OP_REQUIRES_OK(
        context, context->GetAttr("label_upper_bounds", &label_upper_bounds_));
    OP_REQUIRES_OK(context, context->GetAttr("variant_type", &variant_type_));

    if (variant_type_ != "instance" && variant_type_ != "example") {
      LOG(FATAL) << "Invalid 'variant_type', please choose on from "
                    "['instance', 'example']!";
    }
  }

  void Compute(OpKernelContext *context) override {
    const Tensor &input_tensor = context->input(0);
    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    bool is_instance = variant_type_ == "instance";
    if (is_instance) {
      Instance instance;
      instance.CopyFrom(*input_tensor.scalar<Variant>()().get<Instance>());
      output_tensor->scalar<Variant>()() = std::move(instance);
    } else {
      Example example;
      example.CopyFrom(*input_tensor.scalar<Variant>()().get<Example>());
      output_tensor->scalar<Variant>()() = std::move(example);
    }
    auto labels = GetLabel(output_tensor, is_instance);
    if (labels->size() < label_upper_bounds_.size()) {
      LOG_EVERY_N_SEC(ERROR, 60) << absl::StrFormat(
          "Label size(=%ld) should be >= label_upper_bounds size(=%ld), please "
          "investigate!",
          labels->size(), label_upper_bounds_.size());
      return;
    } else {
      for (size_t i = 0; i < label_upper_bounds_.size(); ++i) {
        if (labels->Get(i) > label_upper_bounds_[i]) {
          labels->Set(i, label_upper_bounds_[i]);
        }
      }
    }
  }

 private:
  static ::google::protobuf::RepeatedField<float> *GetLabel(
      Tensor *output_tensor, bool is_instance) {
    if (is_instance) {
      return output_tensor->scalar<Variant>()()
          .get<Instance>()
          ->mutable_label();
    } else {
      return output_tensor->scalar<Variant>()().get<Example>()->mutable_label();
    }
  }

  std::vector<float> label_upper_bounds_;
  std::string variant_type_;
};

namespace {

REGISTER_KERNEL_BUILDER(Name("LabelUpperBound").Device(DEVICE_CPU),
                        LabelUpperBoundOp)

}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow
