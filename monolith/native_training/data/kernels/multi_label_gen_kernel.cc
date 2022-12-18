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

#include <sstream>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "monolith/native_training/data/kernels/internal/label_utils.h"
#include "monolith/native_training/data/training_instance/cc/pb_variant.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace monolith_tf {

using Instance = ::parser::proto::Instance;
using Example = ::monolith::io::proto::Example;
using LineId = ::idl::matrix::proto::LineId;
using Action = google::protobuf::RepeatedField<int>;
using Label = google::protobuf::RepeatedField<float>;

class MultiLabelGenOp : public OpKernel {
 public:
  explicit MultiLabelGenOp(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("task_num", &task_num_));
    OP_REQUIRES_OK(context, context->GetAttr("head_field", &head_field_));
    OP_REQUIRES_OK(context, context->GetAttr("pos_actions", &pos_actions_));
    OP_REQUIRES_OK(context, context->GetAttr("neg_actions", &neg_actions_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("use_origin_label", &use_origin_label_));
    OP_REQUIRES_OK(context, context->GetAttr("pos_label", &pos_label_));
    OP_REQUIRES_OK(context, context->GetAttr("neg_label", &neg_label_));

    std::string action_priority;
    OP_REQUIRES_OK(context,
                   context->GetAttr("action_priority", &action_priority));
    std::vector<absl::string_view> action_priority_items =
        absl::StrSplit(action_priority, ",");
    for (size_t i = 0; i < action_priority_items.size(); ++i) {
      int32 action;
      absl::SimpleAtoi(action_priority_items[i], &action);
      action_priority_.emplace(action, static_cast<int32>(i));
    }

    std::string head_to_index;
    OP_REQUIRES_OK(context, context->GetAttr("head_to_index", &head_to_index));
    for (absl::string_view split : absl::StrSplit(head_to_index, ",")) {
      std::pair<absl::string_view, absl::string_view> head_and_index =
          absl::StrSplit(split, ":");
      int index;
      absl::SimpleAtoi(head_and_index.second, &index);
      CHECK_LT(index, task_num_);
      head_to_index_.emplace(head_and_index.first, index);
    }

    OP_REQUIRES_OK(context, context->GetAttr("variant_type", &variant_type_));
    if (variant_type_ != "instance" && variant_type_ != "example") {
      LOG(FATAL) << "Invalid 'variant_type', please choose on from "
                    "['instance', 'example']!";
    }

    const ::google::protobuf::Descriptor *descriptor =
        ::idl::matrix::proto::LineId::GetDescriptor();
    field = descriptor->FindFieldByName(head_field_);
    CHECK_EQ(field->is_repeated(), false);
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
    if (use_origin_label_) {
      if (!label->empty()) {
        label_value = label->Get(0);
      } else {
        LOG_EVERY_N_SEC(ERROR, 60)
            << "Invalid data: label is empty, please investigate and retry!";
      }
    } else {
      int64_t action;
      if (FindMostPriorAction(line_id->actions(), &action)) {
        if (std::find(pos_actions_.begin(), pos_actions_.end(), action) !=
            pos_actions_.end()) {
          label_value = pos_label_;
        } else if (std::find(neg_actions_.begin(), neg_actions_.end(),
                             action) != neg_actions_.end()) {
          label_value = neg_label_;
        }
      }
    }

    label->Clear();
    label->Resize(task_num_, internal::INVALID_LABEL);

    std::string head_flag = GetHeadFlag(*line_id);
    if (head_to_index_.count(head_flag)) {
      int idx = head_to_index_[head_flag];
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

  bool FindMostPriorAction(const Action &actions, int64_t *action) {
    if (actions.size() != 0) {
      if (action_priority_.empty() || actions.size() == 1) {
        *action = actions[0];
      } else {
        int64_t priority = std::numeric_limits<int64_t>::max();
        for (auto &act : actions) {
          auto iter = action_priority_.find(act);
          if (iter != action_priority_.end() && iter->second < priority) {
            *action = iter->first;
            priority = iter->second;
          }
        }

        if (priority == std::numeric_limits<int64_t>::max())
          *action = actions[0];
      }
      return true;
    }

    return false;
  }

  std::string GetHeadFlag(const LineId &line_id) {
    std::stringstream ss;
    if (head_field_ == "chnid") {
      ss << line_id.chnid();
    } else if (head_field_ == "cid") {
      ss << line_id.cid();
    } else {
      switch (field->cpp_type()) {
        case ::google::protobuf::FieldDescriptor::CPPTYPE_INT32:
          ss << reflection->GetInt32(line_id, field);
        case ::google::protobuf::FieldDescriptor::CPPTYPE_UINT32:
          ss << reflection->GetUInt32(line_id, field);
        case ::google::protobuf::FieldDescriptor::CPPTYPE_INT64:
          ss << reflection->GetInt64(line_id, field);
        case ::google::protobuf::FieldDescriptor::CPPTYPE_UINT64:
          ss << reflection->GetUInt64(line_id, field);
        case ::google::protobuf::FieldDescriptor::CPPTYPE_STRING:
          ss << reflection->GetString(line_id, field);
        default:
          ss << "";
      }
    }

    return ss.str();
  }

  int task_num_;
  std::string head_field_;
  std::map<std::string, int> head_to_index_;
  std::unordered_map<int32, int32> action_priority_;
  std::vector<int> pos_actions_;
  std::vector<int> neg_actions_;
  bool use_origin_label_;
  float pos_label_;
  float neg_label_;
  std::string variant_type_;

  const ::google::protobuf::FieldDescriptor *field;
  const ::google::protobuf::Reflection *reflection =
      ::idl::matrix::proto::LineId::GetReflection();
};

namespace {

REGISTER_KERNEL_BUILDER(Name("MultiLabelGen").Device(DEVICE_CPU),
                        MultiLabelGenOp)

}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow
