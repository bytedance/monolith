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

class ScatterLabelOp : public OpKernel {
 public:
  explicit ScatterLabelOp(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("config", &config_));
    OP_REQUIRES_OK(context, context->GetAttr("variant_type", &variant_type_));

    if (variant_type_ != "instance" && variant_type_ != "example") {
      LOG(FATAL) << "Invalid 'variant_type', please choose on from "
                    "['instance', 'example']!";
    }

    std::vector<absl::string_view> splits = absl::StrSplit(config_, ",");
    CHECK_GT(splits.size(), 0);
    int max_label_index_ = 0;
    for (absl::string_view split : splits) {
      std::vector<absl::string_view> chnid_and_index =
          absl::StrSplit(split, ":");
      CHECK_EQ(chnid_and_index.size(), 2);
      int64_t chnid = 0;
      int index = 0;
      CHECK(absl::SimpleAtoi(chnid_and_index[0], &chnid));
      CHECK(absl::SimpleAtoi(chnid_and_index[1], &index));
      chnid_to_label_index_[chnid] = index;
      if (max_label_index_ < index) {
        max_label_index_ = index;
      }
    }
    multi_task_num_ = max_label_index_ + 1;

    nlohmann::json j;
    j["chnid_to_label_index"] = chnid_to_label_index_;
    j["multi_task_num"] = multi_task_num_;
    LOG(INFO) << j.dump();
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

    LineId *line_id = GetLineId(output_tensor, is_instance);
    auto label = GetLabel(output_tensor, is_instance);
    float label_value = internal::INVALID_LABEL;
    if (!label->empty()) {
      label_value = label->Get(0);
    } else {
      LOG_EVERY_N_SEC(ERROR, 60)
          << "Invalid data: label is empty, please investigate and retry!";
    }
    label->Clear();
    label->Resize(multi_task_num_, internal::INVALID_LABEL);

    int64_t chnid = line_id->chnid();
    if (chnid_to_label_index_.count(chnid)) {
      int idx = chnid_to_label_index_[chnid];
      label->Set(idx, label_value);
    }
  }

 private:
  static LineId *GetLineId(Tensor *output_tensor, bool is_instance) {
    if (is_instance) {
      return output_tensor->scalar<Variant>()()
          .get<Instance>()
          ->mutable_line_id();
    } else {
      return output_tensor->scalar<Variant>()()
          .get<Example>()
          ->mutable_line_id();
    }
  }

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

  std::string config_;
  std::string variant_type_;
  int multi_task_num_;
  std::map<int64_t, int> chnid_to_label_index_;
};

namespace {

REGISTER_KERNEL_BUILDER(Name("ScatterLabel").Device(DEVICE_CPU), ScatterLabelOp)

}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow
