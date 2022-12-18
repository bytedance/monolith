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
#include "idl/matrix/proto/example.pb.h"
#include "monolith/native_training/data/kernels/feature_name_mapper_tf_bridge.h"
#include "monolith/native_training/data/kernels/internal/relational_utils.h"
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

class SetFilterOp : public OpKernel {
 public:
  explicit SetFilterOp(OpKernelConstruction *context) : OpKernel(context) {
    std::vector<int64> filter_fids;
    OP_REQUIRES_OK(context, context->GetAttr("filter_fids", &filter_fids));
    filter_fids_.insert(filter_fids.begin(), filter_fids.end());

    std::vector<int64> has_fids;
    OP_REQUIRES_OK(context, context->GetAttr("has_fids", &has_fids));
    has_fids_.insert(has_fids.begin(), has_fids.end());

    std::vector<int64> select_fids;
    OP_REQUIRES_OK(context, context->GetAttr("select_fids", &select_fids));
    select_fids_.insert(select_fids.begin(), select_fids.end());

    std::vector<int64> has_actions;
    OP_REQUIRES_OK(context, context->GetAttr("has_actions", &has_actions));
    has_actions_.insert(has_actions.begin(), has_actions.end());

    OP_REQUIRES_OK(context, context->GetAttr("variant_type", &variant_type_));

    OP_REQUIRES_OK(context, context->GetAttr("req_time_min", &req_time_min_));

    std::vector<int32_t> select_slots;
    OP_REQUIRES_OK(context, context->GetAttr("select_slots", &select_slots));
    for (int32_t slot : select_slots) {
      CHECK_GE(slot, 0);
    }
    select_slots_.insert(select_slots.begin(), select_slots.end());

    auto creator = [this](FeatureNameMapperTfBridge **out_mapper) {
      TF_RETURN_IF_ERROR(FeatureNameMapperTfBridge::New(out_mapper));
      return Status::OK();
    };
    ResourceMgr *resource_mgr = context->resource_manager();
    OP_REQUIRES_OK(context,
                   resource_mgr->LookupOrCreate<FeatureNameMapperTfBridge>(
                       resource_mgr->default_container(),
                       FeatureNameMapperTfBridge::kName, &mapper_, creator));
    if (variant_type_ == "example") {
      std::vector<std::pair<int, int>> valid_ids;
      for (uint32_t slot : select_slots_) {
        valid_ids.emplace_back(slot, slot);
      }
      for (uint64_t fid : filter_fids_) {
        valid_ids.emplace_back(slot_id_v1(fid), slot_id_v2(fid));
      }
      for (uint64_t fid : has_fids_) {
        valid_ids.emplace_back(slot_id_v1(fid), slot_id_v2(fid));
      }
      for (uint64_t fid : select_fids_) {
        valid_ids.emplace_back(slot_id_v1(fid), slot_id_v2(fid));
      }

      OP_REQUIRES_OK(context, mapper_->RegisterValidIds(valid_ids));
    }
  }

  ~SetFilterOp() override { mapper_->Unref(); }

  void Compute(OpKernelContext *context) override {
    const Tensor &input_tensor = context->input(0);
    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->scalar<bool>();
    output() = IsInstanceOfInterest(input_tensor);
  }

 private:
  bool IsInstanceOfInterest(const Tensor &input_tensor) {
    auto input = input_tensor.scalar<Variant>();
    if (variant_type_ == "instance") {
      const auto *instance = input().get<Instance>();
      return monolith_tf::IsInstanceOfInterest(
          *instance, filter_fids_, has_fids_, select_fids_, has_actions_,
          req_time_min_, select_slots_);
    } else {
      const auto *example = input().get<Example>();
      return monolith_tf::IsInstanceOfInterest(
          *example, filter_fids_, has_fids_, select_fids_, has_actions_,
          req_time_min_, select_slots_);
    }
  }

  std::set<uint64_t> filter_fids_;
  std::set<uint64_t> has_fids_;
  std::set<uint64_t> select_fids_;
  std::set<int32_t> has_actions_;
  std::string variant_type_ = "instance";
  int req_time_min_;
  std::set<uint32_t> select_slots_;
  FeatureNameMapperTfBridge *mapper_;
};

class ValueFilterOp : public OpKernel {
 public:
  explicit ValueFilterOp(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("field_name", &field_name_));
    OP_REQUIRES_OK(context, context->GetAttr("op", &op_));
    OP_REQUIRES_OK(context, context->GetAttr("float_operand", &float_operand_));
    OP_REQUIRES_OK(context, context->GetAttr("int_operand", &int_operand_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("string_operand", &string_operand_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("operand_filepath", &operand_filepath_));
    OP_REQUIRES_OK(context, context->GetAttr("keep_empty", &keep_empty_));
    OP_REQUIRES_OK(context, context->GetAttr("variant_type", &variant_type_));

    uint_operand_.insert(uint_operand_.end(), int_operand_.begin(),
                         int_operand_.end());

    std::unordered_set<std::string> valid_set_ops = {"any", "all", "diff",
                                                     "startswith", "endswith"};
    if (!internal::VALID_OPS.count(op_) && !valid_set_ops.count(op_)) {
      std::string valid_ops_str = absl::StrJoin(internal::VALID_OPS, ", ");
      std::string valid_set_ops_str = absl::StrJoin(valid_set_ops, ", ");
      LOG(FATAL) << absl::StrFormat(
          "Invalid op: %s, please choose one from [%s] or [%s]", op_,
          valid_ops_str, valid_set_ops_str);
    }

    nlohmann::json j;
    j["field_name"] = field_name_;
    j["op"] = op_;
    j["float_operand_count"] = float_operand_.size();
    j["int_operand_count"] = int_operand_.size();
    j["string_operand_count"] = string_operand_.size();
    j["operand_filepath"] = operand_filepath_;

    int64_t limit = 1000;
    if (float_operand_.size() <= limit) {
      j["float_operand"] = float_operand_;
    } else {
      std::vector<float> values(float_operand_.begin(),
                                float_operand_.begin() + limit);
      j["float_operand_first_1000"] = values;
    }

    if (int_operand_.size() <= limit) {
      j["int_operand"] = int_operand_;
    } else {
      std::vector<int> values(int_operand_.begin(),
                              int_operand_.begin() + limit);
      j["int_operand_first_1000"] = values;
    }

    if (string_operand_.size() <= limit) {
      j["string_operand"] = string_operand_;
    } else {
      std::vector<std::string> values(string_operand_.begin(),
                                      string_operand_.begin() + limit);
      j["string_operand_first_1000"] = values;
    }

    LOG(INFO) << j.dump(2);

    if ((op_ == internal::IN || op_ == internal::NOT_IN) &&
        operand_filepath_.empty()) {
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
    auto output = output_tensor->scalar<bool>();

    const google::protobuf::Descriptor *descriptor = nullptr;
    const google::protobuf::Reflection *reflection = nullptr;
    const google::protobuf::FieldDescriptor *field = nullptr;
    descriptor = ::idl::matrix::proto::LineId::GetDescriptor();
    reflection = ::idl::matrix::proto::LineId::GetReflection();
    field = descriptor->FindFieldByName(field_name_);

    // don't has field or the given field is not int like
    if (field == nullptr) {
      output() = false;
      return;
    }

    const LineId &line_id = GetLineId(input_tensor);

    if (!field->is_repeated()) {
      if ((op_ == internal::IN || op_ == internal::NOT_IN) &&
          !operand_filepath_.empty()) {
        OP_REQUIRES_OK(context, EnsureLoadFilterValues(context));
      }
      switch (field->cpp_type()) {
        case google::protobuf::FieldDescriptor::CppType::CPPTYPE_FLOAT: {
          float value = reflection->GetFloat(line_id, field);
          output() = internal::COMPARE_OPS.count(op_)
                         ? internal::compare(op_, value, float_operand_)
                         : internal::contains(op_, value, float_operand_set_);
          break;
        }
        case google::protobuf::FieldDescriptor::CppType::CPPTYPE_DOUBLE: {
          double value = reflection->GetDouble(line_id, field);
          output() = internal::COMPARE_OPS.count(op_)
                         ? internal::compare(op_, value, float_operand_)
                         : internal::contains(op_, value, float_operand_set_);
          break;
        }
        case google::protobuf::FieldDescriptor::CppType::CPPTYPE_INT32: {
          int64 value = reflection->GetInt32(line_id, field);
          output() = internal::COMPARE_OPS.count(op_)
                         ? internal::compare(op_, value, int_operand_)
                         : internal::contains(op_, value, int_operand_set_);
          break;
        }
        case google::protobuf::FieldDescriptor::CppType::CPPTYPE_INT64: {
          int64 value = reflection->GetInt64(line_id, field);
          output() = internal::COMPARE_OPS.count(op_)
                         ? internal::compare(op_, value, int_operand_)
                         : internal::contains(op_, value, int_operand_set_);
          break;
        }
        case google::protobuf::FieldDescriptor::CppType::CPPTYPE_UINT32: {
          int64 value = reflection->GetUInt32(line_id, field);
          output() = internal::COMPARE_OPS.count(op_)
                         ? internal::compare(op_, value, int_operand_)
                         : internal::contains(op_, value, int_operand_set_);
          break;
        }
        case google::protobuf::FieldDescriptor::CppType::CPPTYPE_UINT64: {
          uint64 value = reflection->GetUInt64(line_id, field);
          output() = internal::COMPARE_OPS.count(op_)
                         ? internal::compare(op_, value, uint_operand_)
                         : internal::contains(op_, value, uint_operand_set_);
          break;
        }
        case google::protobuf::FieldDescriptor::CppType::CPPTYPE_STRING: {
          std::string value = reflection->GetString(line_id, field);
          output() = false;
          if (op_ == "startswith") {
            for (const std::string &operand : string_operand_) {
              if (value.find(operand) == 0) {
                output() = true;
                break;
              }
            }
          } else if (op_ == "endswith") {
            for (const std::string &operand : string_operand_) {
              if (operand.size() <= value.size()) {
                bool found = std::equal(operand.rbegin(), operand.rend(),
                                        value.rbegin());
                if (found) {
                  output() = true;
                  break;
                }
              }
            }
          } else {
            output() =
                internal::COMPARE_OPS.count(op_)
                    ? internal::compare(op_, value, string_operand_)
                    : internal::contains(op_, value, string_operand_set_);
          }
          break;
        }
        default:
          output() = false;
          LOG(INFO) << "dtype is " << field->cpp_type();
          break;
      }
    } else {
      const int field_size = reflection->FieldSize(line_id, field);
      std::vector<int64> values;
      switch (field->cpp_type()) {
        case google::protobuf::FieldDescriptor::CppType::CPPTYPE_INT32:
          for (int i = 0; i < field_size; ++i) {
            values.push_back(reflection->GetRepeatedInt32(line_id, field, i));
          }
          break;
        case google::protobuf::FieldDescriptor::CppType::CPPTYPE_INT64:
          for (int i = 0; i < field_size; ++i) {
            values.push_back(reflection->GetRepeatedInt64(line_id, field, i));
          }
          break;
        case google::protobuf::FieldDescriptor::CppType::CPPTYPE_UINT32:
          for (int i = 0; i < field_size; ++i) {
            values.push_back(reflection->GetRepeatedUInt32(line_id, field, i));
          }
          break;
        case google::protobuf::FieldDescriptor::CppType::CPPTYPE_UINT64:
          for (int i = 0; i < field_size; ++i) {
            values.push_back(reflection->GetRepeatedUInt64(line_id, field, i));
          }
          break;
        default:
          LOG(INFO) << "dtype is " << field->cpp_type();
          break;
      }

      if (values.size() > 0) {
        output() = cmp(values);
      } else {
        output() = keep_empty_;
      }
    }
  }

 private:
  bool cmp(const std::vector<int64> &values) {
    std::set<int64> intersection;
    std::set_intersection(values.begin(), values.end(), int_operand_.begin(),
                          int_operand_.end(),
                          std::inserter(intersection, intersection.begin()));
    if (op_ == "any") {
      return intersection.size() > 0;
    } else if (op_ == "all") {
      return intersection.size() == int_operand_.size();
    } else if (op_ == "diff") {
      return intersection.size() == 0;
    } else {
      LOG(FATAL) << "Invalid op: " << op_;
      return false;
    }
  }

  const LineId &GetLineId(const Tensor &input_tensor) {
    if (variant_type_ == "instance") {
      return input_tensor.scalar<Variant>()().get<Instance>()->line_id();
    } else {
      return input_tensor.scalar<Variant>()().get<Example>()->line_id();
    }
  }

  Status EnsureLoadFilterValues(OpKernelContext *context)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    mutex_lock l(mu_);
    if (load_filter_values_finished_ || operand_filepath_.empty()) {
      return Status::OK();
    }

    std::string filter_values_serialized;
    TF_RETURN_IF_ERROR(ReadFileToString(context->env(), operand_filepath_,
                                        &filter_values_serialized));
    ::monolith::io::proto::FilterValues filter_values;
    if (!filter_values.ParseFromString(filter_values_serialized)) {
      return errors::InvalidArgument(
          "Unable to parse filter values, please make sure it is "
          "serialized version of message:FilterValues.");
    }

    const google::protobuf::Descriptor *descriptor = nullptr;
    const google::protobuf::FieldDescriptor *field = nullptr;
    descriptor = ::idl::matrix::proto::LineId::GetDescriptor();
    field = descriptor->FindFieldByName(field_name_);

    switch (field->cpp_type()) {
      case google::protobuf::FieldDescriptor::CppType::CPPTYPE_FLOAT:
      case google::protobuf::FieldDescriptor::CppType::CPPTYPE_DOUBLE: {
        if (!filter_values.has_float_list()) {
          return errors::InvalidArgument(
              "Filter values' type should be the same with field type.");
        }
        float_operand_set_.insert(filter_values.float_list().value().begin(),
                                  filter_values.float_list().value().end());
        break;
      }
      case google::protobuf::FieldDescriptor::CppType::CPPTYPE_INT32:
      case google::protobuf::FieldDescriptor::CppType::CPPTYPE_INT64: {
        if (!filter_values.has_int64_list()) {
          return errors::InvalidArgument(
              "Filter values' type should be the same with field type.");
        }
        int_operand_set_.insert(filter_values.int64_list().value().begin(),
                                filter_values.int64_list().value().end());
        break;
      }
      case google::protobuf::FieldDescriptor::CppType::CPPTYPE_UINT32:
      case google::protobuf::FieldDescriptor::CppType::CPPTYPE_UINT64: {
        if (!filter_values.has_int64_list()) {
          return errors::InvalidArgument(
              "Filter values' type should be the same with field type.");
        }
        uint_operand_set_.insert(filter_values.int64_list().value().begin(),
                                 filter_values.int64_list().value().end());
        break;
      }
      case google::protobuf::FieldDescriptor::CppType::CPPTYPE_STRING: {
        if (!filter_values.has_bytes_list()) {
          return errors::InvalidArgument(
              "Filter values' type should be the same with field type.");
        }
        string_operand_set_.insert(filter_values.bytes_list().value().begin(),
                                   filter_values.bytes_list().value().end());
        break;
      }
      default: {
        return errors::InvalidArgument("Invalid field type for filter.");
      }
    }
    load_filter_values_finished_ = true;

    return Status::OK();
  }

  mutex mu_;
  bool load_filter_values_finished_ TF_GUARDED_BY(mu_) = false;

  std::string field_name_;
  std::string op_;  // gt, ge, eq, lt, le, neq, between
  bool keep_empty_ = false;
  std::string operand_filepath_;

  std::vector<float> float_operand_;
  std::vector<int64> int_operand_;
  std::vector<uint64> uint_operand_;
  std::vector<std::string> string_operand_;

  std::unordered_set<float> float_operand_set_;
  std::unordered_set<int64> int_operand_set_;
  std::unordered_set<uint64> uint_operand_set_;
  std::unordered_set<std::string> string_operand_set_;

  std::string variant_type_;
};

class SpecialStrategyOp : public OpKernel {
 public:
  explicit SpecialStrategyOp(OpKernelConstruction *context)
      : OpKernel(context) {
    std::vector<int> special_strategy;
    OP_REQUIRES_OK(context,
                   context->GetAttr("special_strategies", &special_strategy));
    std::vector<float> sample_rate;
    OP_REQUIRES_OK(context, context->GetAttr("sample_rates", &sample_rate));
    std::vector<float> label;
    OP_REQUIRES_OK(context, context->GetAttr("labels", &label));

    OP_REQUIRES(
        context, special_strategy.size() == sample_rate.size(),
        errors::InvalidArgument(
            "length of sample_rates must identity with special_strategies"));
    OP_REQUIRES(
        context, special_strategy.size() == label.size() || label.size() == 0,
        errors::InvalidArgument(
            "length of labels must identity with special_strategies or zero"));

    for (size_t i = 0; i < special_strategy.size(); ++i) {
      strategy_to_rate_.emplace(special_strategy[i], sample_rate[i]);
    }

    if (label.size() > 0) {
      for (size_t i = 0; i < special_strategy.size(); ++i) {
        strategy_to_label_.emplace(special_strategy[i], label[i]);
      }
    }

    OP_REQUIRES_OK(context, context->GetAttr("strategy_list", &strategy_list_));
    OP_REQUIRES(context, strategy_list_.size() > 0,
                errors::InvalidArgument("strategy_list cannot be empty"));

    OP_REQUIRES_OK(context, context->GetAttr("keep_empty_strategy",
                                             &keep_empty_strategy_));
    OP_REQUIRES_OK(context, context->GetAttr("variant_type", &variant_type_));
  }

  void Compute(OpKernelContext *context) override {
    Tensor *input_tensor = const_cast<Tensor *>(&(context->input(0)));
    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor->shape(),
                                                     &output_tensor));
    auto output = output_tensor->scalar<bool>();
    output() = DoCompute(input_tensor);
  }

 private:
  const LineId &GetLineId(Tensor *input_tensor) {
    if (variant_type_ == "instance") {
      return input_tensor->scalar<Variant>()().get<Instance>()->line_id();
    } else {
      return input_tensor->scalar<Variant>()().get<Example>()->line_id();
    }
  }

  float *GetLabel(Tensor *input_tensor, int index = 0) {
    if (variant_type_ == "instance") {
      return input_tensor->scalar<Variant>()()
          .get<Instance>()
          ->mutable_label()
          ->Mutable(index);
    } else {
      return input_tensor->scalar<Variant>()()
          .get<Example>()
          ->mutable_label()
          ->Mutable(index);
    }
  }

  bool DoCompute(Tensor *input_tensor) {
    const LineId &line_id = GetLineId(input_tensor);
    const auto &strategies = line_id.special_strategies();

    if (strategies.size() > 1) {
      LOG(INFO) << "Size of special_strategies is bigger than one, pls. check!";
    }

    if (strategies.size() == 0) {
      // for unknow samples, drop
      if (keep_empty_strategy_) {
        // for special_strategies_neg_ins_keep_normal
        return true;
      } else {
        return false;
      }
    } else {
      for (auto &special_strategy : strategy_list_) {
        auto found =
            std::find(strategies.begin(), strategies.end(), special_strategy);
        if (found != strategies.end()) {
          auto rit = strategy_to_rate_.find(special_strategy);
          if (rit != strategy_to_rate_.end()) {
            float rate = rit->second;
            bool flag = false;
            if (rate == 1.0) {
              flag = true;
            } else {
              if (random_neg_sample_(generator_) <= rate) {
                flag = true;
              }
            }

            if (strategy_to_label_.size() > 0 && flag) {
              auto lit = strategy_to_label_.find(special_strategy);
              if (lit != strategy_to_label_.end()) {
                float new_label = lit->second;
                float *old_label = GetLabel(input_tensor);
                *old_label = new_label;
              }
            }

            return flag;
          } else {
            return true;
          }
        }
      }
    }
  }

  std::default_random_engine generator_;
  std::uniform_real_distribution<float> random_neg_sample_;

  bool keep_empty_strategy_ = true;
  std::unordered_map<int, float> strategy_to_rate_;
  std::unordered_map<int, float> strategy_to_label_;
  std::vector<int> strategy_list_;
  std::string variant_type_;
};

class NegativeSampleOp : public OpKernel {
 public:
  explicit NegativeSampleOp(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("drop_rate", &drop_rate_));
    OP_REQUIRES_OK(context, context->GetAttr("label_index", &label_index_));
    OP_REQUIRES_OK(context, context->GetAttr("threshold", &threshold_));
    OP_REQUIRES_OK(context, context->GetAttr("variant_type", &variant_type_));
  }

  void Compute(OpKernelContext *context) override {
    const Tensor &input_tensor = context->input(0);
    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->scalar<bool>();
    float label = GetLabel(input_tensor);

    if (label < threshold_) {
      thread_local std::mt19937 gen((std::random_device())());
      float random = gen() % 1000 / 1000.0;
      output() = random <= drop_rate_ ? false : true;
    } else {
      output() = true;
    }
  }

 private:
  float drop_rate_ = 0.0;
  int label_index_ = 0;
  float threshold_ = 0.0;
  std::string variant_type_ = "instance";

  float GetLabel(const Tensor &input_tensor) {
    auto input = input_tensor.scalar<Variant>();
    if (variant_type_ == "instance") {
      const Instance *instance = input().get<Instance>();
      return instance->label(label_index_);
    } else {
      const Example *example = input().get<Example>();
      return example->label(label_index_);
    }

    return 0;
  }
};

namespace {
REGISTER_KERNEL_BUILDER(Name("SetFilter").Device(DEVICE_CPU), SetFilterOp);

REGISTER_KERNEL_BUILDER(Name("ValueFilter").Device(DEVICE_CPU), ValueFilterOp);

REGISTER_KERNEL_BUILDER(Name("SpecialStrategy").Device(DEVICE_CPU),
                        SpecialStrategyOp);

REGISTER_KERNEL_BUILDER(Name("NegativeSample").Device(DEVICE_CPU),
                        NegativeSampleOp);

}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow
