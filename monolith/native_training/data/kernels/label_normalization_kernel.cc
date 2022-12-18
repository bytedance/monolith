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

// label_norm:
class LabelNormalizationOp : public OpKernel {
 public:
  explicit LabelNormalizationOp(OpKernelConstruction *context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("norm_methods", &norm_methods_));
    OP_REQUIRES_OK(context, context->GetAttr("norm_values", &norm_values_));
    OP_REQUIRES_OK(context, context->GetAttr("variant_type", &variant_type_));

    if (norm_methods_.size() != norm_values_.size()) {
      LOG(FATAL) << "Invalid 'norm_methods_', and 'norm_values', the size "
                    "should match each other.!";
    }

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
    for (int i = 0; i < labels->size(); ++i) {
      for (int j = 0; j < norm_methods_.size(); ++j) {
        float label = labels->Get(i);
        const auto &norm_method = norm_methods_[j];
        const auto &norm_value = norm_values_[j];
        if (norm_method == "log") {
          label = std::max(label + norm_value, 0.0f);
          label = log(label);
        } else if (norm_method == "scale") {
          label /= norm_value;
        } else if (norm_method == "scale2int") {
          label = int32_t(label / norm_value);
        } else if (norm_method == "pow") {
          label = std::pow(label + 1, norm_value);
        } else if (norm_method == "repow") {
          if (label > 0) {
            label = std::pow(label, norm_value);
          } else {
            label = 0;
          }
        } else if (norm_method == "scalelog") {
          label /= norm_value;
          label = log(std::max(label + 1, 0.0f));
        } else {
          assert(false && "illegal label norm params");
        }
        labels->Set(i, label);
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
  std::vector<std::string> norm_methods_;
  std::vector<float> norm_values_;
  std::string variant_type_;
};

namespace {

REGISTER_KERNEL_BUILDER(Name("LabelNormalization").Device(DEVICE_CPU),
                        LabelNormalizationOp)

}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow
