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

#include "monolith/native_training/data/kernels/internal/line_id_value_filter.h"

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "idl/matrix/proto/example.pb.h"
#include "monolith/native_training/data/kernels/internal/relational_utils.h"
#include "third_party/nlohmann/json.hpp"

namespace tensorflow {
namespace monolith_tf {
namespace internal {

LineIdValueFilter::LineIdValueFilter(std::string field_name, std::string op,
                                     std::vector<float> float_operand,
                                     std::vector<int64> int_operand,
                                     std::vector<std::string> string_operand,
                                     std::string operand_filepath,
                                     bool keep_empty)
    : field_name_(std::move(field_name)),
      op_(std::move(op)),
      float_operand_(std::move(float_operand)),
      int_operand_(std::move(int_operand)),
      string_operand_(std::move(string_operand)),
      operand_filepath_(std::move(operand_filepath)),
      keep_empty_(keep_empty) {
  const auto descriptor = ::idl::matrix::proto::LineId::GetDescriptor();
  const auto reflection = ::idl::matrix::proto::LineId::GetReflection();
  field_ = descriptor->FindFieldByName(field_name_);
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
    std::vector<int> values(int_operand_.begin(), int_operand_.begin() + limit);
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
    string_operand_set_.insert(string_operand_.begin(), string_operand_.end());
  }
}

Status LineIdValueFilter::EnsureLoadFilterValues(tensorflow::Env *env) {
  absl::MutexLock l(&mu_);
  if (load_filter_values_finished_ || operand_filepath_.empty()) {
    return Status::OK();
  }

  std::string filter_values_serialized;
  TF_RETURN_IF_ERROR(
      ReadFileToString(env, operand_filepath_, &filter_values_serialized));
  ::monolith::io::proto::FilterValues filter_values;
  if (!filter_values.ParseFromString(filter_values_serialized)) {
    return errors::InvalidArgument(
        "Unable to parse filter values, please make sure it is "
        "serialized version of message:FilterValues.");
  }

  auto field = field_;
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

bool LineIdValueFilter::IsInstanceOfInterest(
    tensorflow::Env *env, const ::idl::matrix::proto::LineId &line_id) {
  bool output = false;

  const auto reflection = ::idl::matrix::proto::LineId::GetReflection();
  auto field = field_;
  if (field == nullptr) {
    output = false;
    return output;
  }

  if (!field->is_repeated()) {
    if ((op_ == internal::IN || op_ == internal::NOT_IN) &&
        !operand_filepath_.empty()) {
      TF_CHECK_OK(EnsureLoadFilterValues(env));
    }
    switch (field->cpp_type()) {
      case google::protobuf::FieldDescriptor::CppType::CPPTYPE_FLOAT: {
        float value = reflection->GetFloat(line_id, field);
        output = internal::COMPARE_OPS.count(op_)
                     ? internal::compare(op_, value, float_operand_)
                     : internal::contains(op_, value, float_operand_set_);
        break;
      }
      case google::protobuf::FieldDescriptor::CppType::CPPTYPE_DOUBLE: {
        double value = reflection->GetDouble(line_id, field);
        output = internal::COMPARE_OPS.count(op_)
                     ? internal::compare(op_, value, float_operand_)
                     : internal::contains(op_, value, float_operand_set_);
        break;
      }
      case google::protobuf::FieldDescriptor::CppType::CPPTYPE_INT32: {
        int64 value = reflection->GetInt32(line_id, field);
        output = internal::COMPARE_OPS.count(op_)
                     ? internal::compare(op_, value, int_operand_)
                     : internal::contains(op_, value, int_operand_set_);
        break;
      }
      case google::protobuf::FieldDescriptor::CppType::CPPTYPE_INT64: {
        int64 value = reflection->GetInt64(line_id, field);
        output = internal::COMPARE_OPS.count(op_)
                     ? internal::compare(op_, value, int_operand_)
                     : internal::contains(op_, value, int_operand_set_);
        break;
      }
      case google::protobuf::FieldDescriptor::CppType::CPPTYPE_UINT32: {
        int64 value = reflection->GetUInt32(line_id, field);
        output = internal::COMPARE_OPS.count(op_)
                     ? internal::compare(op_, value, int_operand_)
                     : internal::contains(op_, value, int_operand_set_);
        break;
      }
      case google::protobuf::FieldDescriptor::CppType::CPPTYPE_UINT64: {
        uint64 value = reflection->GetUInt64(line_id, field);
        output = internal::COMPARE_OPS.count(op_)
                     ? internal::compare(op_, value, uint_operand_)
                     : internal::contains(op_, value, uint_operand_set_);
        break;
      }
      case google::protobuf::FieldDescriptor::CppType::CPPTYPE_STRING: {
        std::string value = reflection->GetString(line_id, field);
        output = false;
        if (op_ == "startswith") {
          for (const std::string &operand : string_operand_) {
            if (value.find(operand) == 0) {
              output = true;
              break;
            }
          }
        } else if (op_ == "endswith") {
          for (const std::string &operand : string_operand_) {
            if (operand.size() <= value.size()) {
              bool found =
                  std::equal(operand.rbegin(), operand.rend(), value.rbegin());
              if (found) {
                output = true;
                break;
              }
            }
          }
        } else {
          output = internal::COMPARE_OPS.count(op_)
                       ? internal::compare(op_, value, string_operand_)
                       : internal::contains(op_, value, string_operand_set_);
        }
        break;
      }
      default:
        output = false;
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
      output = cmp(values);
    } else {
      output = keep_empty_;
    }
  }

  return output;
}

}  // namespace internal
}  // namespace monolith_tf
}  // namespace tensorflow
