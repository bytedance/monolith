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
#include "third_party/nlohmann/json.hpp"

namespace tensorflow {
namespace monolith_tf {

using IFeature = ::idl::matrix::proto::Feature;
using Instance = ::parser::proto::Instance;
using Example = ::monolith::io::proto::Example;
using LineId = ::idl::matrix::proto::LineId;

// filter_invalid_conseq_time:
class FilterByLabelOp : public OpKernel {
 public:
  explicit FilterByLabelOp(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("label_threshold", &label_threshold_));
    OP_REQUIRES_OK(context, context->GetAttr("filter_equal", &filter_equal_));
    OP_REQUIRES_OK(context, context->GetAttr("variant_type", &variant_type_));

    if (variant_type_ != "instance" && variant_type_ != "example") {
      LOG(FATAL) << "Invalid 'variant_type', please choose on from "
                    "['instance', 'example']!";
    }

    nlohmann::json j;
    j["label_threshold"] = label_threshold_;
    LOG(INFO) << absl::StrFormat("Label threshold: %s", j.dump(2));
  }

  void Compute(OpKernelContext *context) override {
    const Tensor &input_tensor = context->input(0);
    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto valid = output_tensor->scalar<bool>();
    bool is_instance = variant_type_ == "instance";

    auto labels = GetLabels(&input_tensor, is_instance);
    if (labels.size() < label_threshold_.size()) {
      LOG_EVERY_N_SEC(ERROR, 60) << absl::StrFormat(
          "Label size(=%ld) should be >= label_threshold size(=%ld), please "
          "investigate!",
          labels.size(), label_threshold_.size());
      valid() = false;
    } else {
      bool has_valid_label = false;
      for (size_t i = 0; i < label_threshold_.size(); ++i) {
        if ((labels.Get(i) > label_threshold_[i]) ||
            (labels.Get(i) == label_threshold_[i] && !filter_equal_)) {
          has_valid_label = true;
          break;
        }
      }

      valid() = has_valid_label;
    }
  }

 private:
  static const ::google::protobuf::RepeatedField<float> &GetLabels(
      const Tensor *output_tensor, bool is_instance) {
    if (is_instance) {
      return output_tensor->scalar<Variant>()().get<Instance>()->label();
    } else {
      return output_tensor->scalar<Variant>()().get<Example>()->label();
    }
  }

  std::vector<float> label_threshold_;
  bool filter_equal_;
  std::string variant_type_;
};

namespace {

REGISTER_KERNEL_BUILDER(Name("FilterByLabel").Device(DEVICE_CPU),
                        FilterByLabelOp)

}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow
