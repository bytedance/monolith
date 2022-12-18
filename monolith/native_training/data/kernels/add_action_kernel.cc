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
#include <unordered_map>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_format.h"
#include "monolith/native_training/data/kernels/internal/relational_utils.h"
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

class AddActionOp : public OpKernel {
 public:
  explicit AddActionOp(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("field_name", &field_name_));
    OP_REQUIRES_OK(context, context->GetAttr("op", &op_));
    OP_REQUIRES_OK(context, context->GetAttr("float_operand", &float_operand_));
    OP_REQUIRES_OK(context, context->GetAttr("int_operand", &int_operand_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("string_operand", &string_operand_));
    OP_REQUIRES_OK(context, context->GetAttr("variant_type", &variant_type_));
    OP_REQUIRES_OK(context, context->GetAttr("actions", &actions_));

    uint_operand_.insert(uint_operand_.end(), int_operand_.begin(),
                         int_operand_.end());

    if (!internal::VALID_OPS.count(op_)) {
      LOG(FATAL) << absl::StrFormat(
          "Invalid op: %s, please choose one from [%s]", op_,
          absl::StrJoin(internal::VALID_OPS, ", "));
    }

    if (variant_type_ != "instance" && variant_type_ != "example") {
      LOG(FATAL) << "Invalid 'variant_type', please choose on from "
                    "['instance', 'example']!";
    }

    if (actions_.empty()) {
      LOG(FATAL) << "Please specify 'actions' to add!";
    }

    if (op_ == internal::IN || op_ == internal::NOT_IN) {
      float_operand_set_.insert(float_operand_.begin(), float_operand_.end());
      int_operand_set_.insert(int_operand_.begin(), int_operand_.end());
      uint_operand_set_.insert(uint_operand_.begin(), uint_operand_.end());
      string_operand_set_.insert(string_operand_.begin(),
                                 string_operand_.end());
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

    const google::protobuf::Descriptor *descriptor = LineId::GetDescriptor();
    const google::protobuf::Reflection *reflection = LineId::GetReflection();
    const google::protobuf::FieldDescriptor *field =
        descriptor->FindFieldByName(field_name_);

    if (field == nullptr || field->is_repeated()) {
      return;
    }

    LineId &line_id = *GetLineId(output_tensor, is_instance);

    bool to_add_action = false;
    switch (field->cpp_type()) {
      case google::protobuf::FieldDescriptor::CppType::CPPTYPE_FLOAT: {
        float value = reflection->GetFloat(line_id, field);
        to_add_action =
            internal::COMPARE_OPS.count(op_)
                ? internal::compare(op_, value, float_operand_)
                : internal::contains(op_, value, float_operand_set_);
        break;
      }
      case google::protobuf::FieldDescriptor::CppType::CPPTYPE_DOUBLE: {
        double value = reflection->GetDouble(line_id, field);
        to_add_action =
            internal::COMPARE_OPS.count(op_)
                ? internal::compare(op_, value, float_operand_)
                : internal::contains(op_, value, float_operand_set_);
        break;
      }
      case google::protobuf::FieldDescriptor::CppType::CPPTYPE_INT32: {
        int64 value = reflection->GetInt32(line_id, field);
        to_add_action = internal::COMPARE_OPS.count(op_)
                            ? internal::compare(op_, value, int_operand_)
                            : internal::contains(op_, value, int_operand_set_);
        break;
      }
      case google::protobuf::FieldDescriptor::CppType::CPPTYPE_INT64: {
        int64 value = reflection->GetInt64(line_id, field);
        to_add_action = internal::COMPARE_OPS.count(op_)
                            ? internal::compare(op_, value, int_operand_)
                            : internal::contains(op_, value, int_operand_set_);
        break;
      }
      case google::protobuf::FieldDescriptor::CppType::CPPTYPE_UINT32: {
        int64 value = reflection->GetUInt32(line_id, field);
        to_add_action = internal::COMPARE_OPS.count(op_)
                            ? internal::compare(op_, value, int_operand_)
                            : internal::contains(op_, value, int_operand_set_);
        break;
      }
      case google::protobuf::FieldDescriptor::CppType::CPPTYPE_UINT64: {
        uint64 value = reflection->GetUInt64(line_id, field);
        to_add_action = internal::COMPARE_OPS.count(op_)
                            ? internal::compare(op_, value, uint_operand_)
                            : internal::contains(op_, value, uint_operand_set_);
        break;
      }
      case google::protobuf::FieldDescriptor::CppType::CPPTYPE_STRING: {
        std::string value = reflection->GetString(line_id, field);
        to_add_action =
            internal::COMPARE_OPS.count(op_)
                ? internal::compare(op_, value, string_operand_)
                : internal::contains(op_, value, string_operand_set_);
        break;
      }
      default:
        to_add_action = false;
        LOG(INFO) << "dtype is " << field->cpp_type();
        break;
    }

    if (to_add_action) {
      for (int32 value : actions_) {
        line_id.mutable_actions()->Add(value);
      }
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

  std::string field_name_;
  std::string op_;

  std::vector<float> float_operand_;
  std::vector<int64> int_operand_;
  std::vector<uint64> uint_operand_;
  std::vector<std::string> string_operand_;

  std::unordered_set<float> float_operand_set_;
  std::unordered_set<int64> int_operand_set_;
  std::unordered_set<uint64> uint_operand_set_;
  std::unordered_set<std::string> string_operand_set_;

  std::string variant_type_;
  std::vector<int32> actions_;
};

namespace {

REGISTER_KERNEL_BUILDER(Name("AddAction").Device(DEVICE_CPU), AddActionOp)

}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow
