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

#include "monolith/native_training/data/kernels/internal/value_filter_by_feature.h"

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "idl/matrix/proto/example.pb.h"
#include "monolith/native_training/data/kernels/internal/relational_utils.h"
#include "third_party/nlohmann/json.hpp"

namespace tensorflow {
namespace monolith_tf {
namespace internal {

using EFeature = ::monolith::io::proto::Feature;
using FilterValues = ::monolith::io::proto::FilterValues;

std::unordered_set<std::string> FeatureValueFilter::VALID_SET_OPS = {
    "any", "all", "diff", "startswith", "endswith"};

FeatureValueFilter::FeatureValueFilter(std::string field_name,
                                       std::string field_type, std::string op,
                                       std::vector<float> float_operand,
                                       std::vector<int64> int_operand,
                                       std::vector<std::string> string_operand,
                                       std::string operand_filepath,
                                       bool keep_empty)
    : field_name_(std::move(field_name)),
      field_type_(std::move(field_type)),
      op_(std::move(op)),
      feature_index_valid_score_(1.0),
      float_operand_(std::move(float_operand)),
      int_operand_(std::move(int_operand)),
      string_operand_(std::move(string_operand)),
      operand_filepath_(std::move(operand_filepath)),
      keep_empty_(keep_empty) {
  if (!internal::VALID_OPS.count(op_) && !VALID_SET_OPS.count(op_)) {
    std::string valid_ops_str = absl::StrJoin(internal::VALID_OPS, ", ");
    std::string valid_set_ops_str = absl::StrJoin(VALID_SET_OPS, ", ");
    LOG(FATAL) << absl::StrFormat(
        "Invalid op: %s, please choose one from [%s] or [%s]", op_,
        valid_ops_str, valid_set_ops_str);
  }

  nlohmann::json j;
  j["field_name"] = field_name_;
  j["field_type"] = field_type_;
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
    string_operand_set_.insert(string_operand_.begin(), string_operand_.end());
  }
}

Status FeatureValueFilter::EnsureLoadFilterValues(tensorflow::Env* env) {
  absl::MutexLock l(&load_filter_values_mu_);
  if (load_filter_values_finished_ || operand_filepath_.empty()) {
    return Status::OK();
  }

  std::string filter_values_serialized;
  TF_RETURN_IF_ERROR(
      ReadFileToString(env, operand_filepath_, &filter_values_serialized));
  FilterValues filter_values;
  if (!filter_values.ParseFromString(filter_values_serialized)) {
    return errors::InvalidArgument(
        "Unable to parse filter values, please make sure it is "
        "serialized version of message:FilterValues.");
  }
  switch (static_cast<int>(filter_values.type_case())) {
    case FilterValues::TypeCase::kFloatList: {
      if (field_type_ != "float") {
        return errors::InvalidArgument(
            "Filter values' type(float) should be the same with field type(",
            field_type_, ")");
      }
      float_operand_set_.insert(filter_values.float_list().value().begin(),
                                filter_values.float_list().value().end());
      break;
    }
    case FilterValues::TypeCase::kInt64List: {
      if (field_type_ != "int64") {
        return errors::InvalidArgument(
            "Filter values' type(int64) should be the same with field type(",
            field_type_, ")");
      }
      int_operand_set_.insert(filter_values.int64_list().value().begin(),
                              filter_values.int64_list().value().end());
      break;
    }
    case FilterValues::TypeCase::kBytesList: {
      if (field_type_ != "bytes") {
        return errors::InvalidArgument(
            "Filter values' type(bytes) should be the same with field type(",
            field_type_, ")");
      }
      string_operand_set_.insert(filter_values.bytes_list().value().begin(),
                                 filter_values.bytes_list().value().end());
      break;
    }
    case FilterValues::TypeCase::TYPE_NOT_SET:
      return errors::InvalidArgument("FilterValue TYPE_NOT_SET, field type(",
                                     field_type_, ")");
    default:
      return errors::InvalidArgument(
          "Invalid field type for feature value filter, field_type: ",
          field_type_, " FilterValues: ", filter_values.ShortDebugString());
  }
  load_filter_values_finished_ = true;
  return Status::OK();
}

bool FeatureValueFilter::CheckFeatureIndex(const Example& example,
                                           int* feature_index) {
  find_feature_index_mu_.ReaderLock();
  bool result = true;
  if (cached_feature_index_ == -1 ||
      cached_feature_index_ >= example.named_feature_size()) {
    result = false;
  } else {
    const auto& feature = example.named_feature(cached_feature_index_);
    if (feature.name() != field_name_) {
      result = false;
    }
  }
  if (result) {
    *feature_index = cached_feature_index_;
  }
  find_feature_index_mu_.ReaderUnlock();
  return result;
}

bool FeatureValueFilter::IsInstanceOfInterest(tensorflow::Env* env,
                                              const Example& example) {
  bool output = false;
  int feature_index = -1;
  if (!CheckFeatureIndex(example, &feature_index)) {
    for (int i = 0; i < example.named_feature_size(); i++) {
      const auto& feature = example.named_feature(i);
      if (feature.name() == field_name_) {
        feature_index = i;
      }
    }
    if (feature_index != -1) {
      absl::MutexLock l(&find_feature_index_mu_);
      cached_feature_index_ = feature_index;
    }
    double score = feature_index_valid_score_.load();
    score = 0.99 * score;
    feature_index_valid_score_.store(score);
    if (score < 0.7) {
      LOG_EVERY_N_SEC(ERROR, 15)
          << "Potential performance problem! feature index valid score: "
          << score;
    }
  } else {
    double score = feature_index_valid_score_.load();
    feature_index_valid_score_.store(0.99 * score + 0.01);
  }
  LOG_EVERY_N_SEC(INFO, 120)
      << "Feature index valid score (performance related): "
      << feature_index_valid_score_.load();
  if (feature_index == -1 && !keep_empty_) {
    output = false;
    LOG_EVERY_N_SEC(ERROR, 15) << "Feature not found!"
                               << " field name: " << field_name_;
    return output;
  }
  const auto& feature = example.named_feature(feature_index).feature();
  const auto& type_case = feature.type_case();
  // op是in/not_in，且feature是单值类型的场景
  if ((op_ == internal::IN || op_ == internal::NOT_IN) &&
      !operand_filepath_.empty() &&
      (type_case == EFeature::TypeCase::kFloatList ||
       type_case == EFeature::TypeCase::kDoubleList ||
       type_case == EFeature::TypeCase::kInt64List ||
       type_case == EFeature::TypeCase::kBytesList)) {
    TF_CHECK_OK(EnsureLoadFilterValues(env));
  }
  switch (static_cast<int>(type_case)) {
    case EFeature::TypeCase::TYPE_NOT_SET: {
      LOG_EVERY_N_SEC(ERROR, 15) << "Invalid data: feature not set!"
                                 << " field name: " << field_name_;
      break;
    }
    case EFeature::TypeCase::kFloatValue: {
      LOG_EVERY_N_SEC(ERROR, 15) << "Invalid data: float value is not "
                                    "supported, please use float list!"
                                 << " field name: " << field_name_;
      break;
    }
    case EFeature::TypeCase::kDoubleValue: {
      LOG_EVERY_N_SEC(ERROR, 15) << "Invalid data: double value is not "
                                    "supported, please use double list!"
                                 << " field name: " << field_name_;
      break;
    }
    case EFeature::TypeCase::kInt64Value: {
      LOG_EVERY_N_SEC(ERROR, 15) << "Invalid data: int64 value is not "
                                    "supported, please use double list!"
                                 << " field name: " << field_name_;
      break;
    }
    case EFeature::TypeCase::kBytesValue: {
      LOG_EVERY_N_SEC(ERROR, 15) << "Invalid data: bytes value is not "
                                    "supported, please use bytes list!"
                                 << " field name: " << field_name_;
      break;
    }
    default:
      break;
  }
  std::vector<int64> values;
  switch (static_cast<int>(type_case)) {
    case EFeature::TypeCase::kFloatList: {
      if (field_type_ != "float") {
        LOG_EVERY_N_SEC(ERROR, 15)
            << "Field type not match: field name: " << field_name_
            << " field type: " << field_type_
            << " but feature has float value.";
        break;
      }
      if (feature.float_list().value_size() == 1) {
        float value = feature.float_list().value(0);
        output = internal::COMPARE_OPS.count(op_)
                     ? internal::compare(op_, value, float_operand_)
                     : internal::contains(op_, value, float_operand_set_);
        return output;
      } else if (feature.float_list().value_size() > 1) {
        LOG_EVERY_N_SEC(ERROR, 15)
            << "Invalid data: float list with multiple elements is not "
               "supported, please investigate and retry!"
            << " field name: " << field_name_;
      }
      break;
    }
    case EFeature::TypeCase::kDoubleList: {
      if (field_type_ != "double") {
        LOG_EVERY_N_SEC(ERROR, 15)
            << "Field type not match: field name: " << field_name_
            << " field type: " << field_type_
            << " but feature has double value.";
        break;
      }
      if (feature.double_list().value_size() == 1) {
        double value = feature.double_list().value(0);
        output = internal::COMPARE_OPS.count(op_)
                     ? internal::compare(op_, value, float_operand_)
                     : internal::contains(op_, value, float_operand_set_);
        return output;
      } else if (feature.double_list().value_size() > 1) {
        LOG_EVERY_N_SEC(ERROR, 15)
            << "Invalid data: double_list with multiple elements is not "
               "supported, please investigate and retry!"
            << " field name: " << field_name_;
      }
      break;
    }
    case EFeature::TypeCase::kInt64List: {
      if (field_type_ != "int64") {
        LOG_EVERY_N_SEC(ERROR, 15)
            << "Field type not match: field name: " << field_name_
            << " field type: " << field_type_
            << " but feature has int64 value.";
        break;
      }
      if (VALID_SET_OPS.count(op_)) {
        for (const auto& value : feature.int64_list().value()) {
          values.push_back(value);
        }
      } else {
        if (feature.int64_list().value_size() == 1) {
          int64 value = feature.int64_list().value(0);
          output = internal::COMPARE_OPS.count(op_)
                       ? internal::compare(op_, value, int_operand_)
                       : internal::contains(op_, value, int_operand_set_);
          return output;
        } else if (feature.double_list().value_size() > 1) {
          LOG_EVERY_N_SEC(ERROR, 15)
              << "Invalid data: int64_list with multiple elements when not "
                 "using set_ops is not supported, please investigate and retry!"
              << " field name: " << field_name_ << " op: " << op_;
        }
      }
      break;
    }
    case EFeature::TypeCase::kBytesList: {
      if (field_type_ != "bytes") {
        LOG_EVERY_N_SEC(ERROR, 15)
            << "Field type not match: field name: " << field_name_
            << " field type: " << field_type_
            << " but feature has bytes value.";
        break;
      }
      if (feature.bytes_list().value_size() == 1) {
        std::string value = feature.bytes_list().value(0);
        output = false;
        if (op_ == "startswith") {
          for (const std::string& operand : string_operand_) {
            if (value.find(operand) == 0) {
              output = true;
              break;
            }
          }
        } else if (op_ == "endswith") {
          for (const std::string& operand : string_operand_) {
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
        return output;
      } else if (feature.bytes_list().value_size() > 1) {
        LOG_EVERY_N_SEC(ERROR, 15)
            << "Invalid data: bytes_list with multiple elements is not "
               "supported, please investigate and retry!"
            << " field name: " << field_name_;
      }
      break;
    }
    default: {
      output = false;
      const auto descriptor = EFeature::GetDescriptor();
      const auto reflection = EFeature::GetReflection();
      const auto oneof_descriptor = descriptor->FindOneofByName("type");
      std::string feature_dtype = "";
      if (oneof_descriptor != nullptr) {
        const auto field_descriptor =
            reflection->GetOneofFieldDescriptor(feature, oneof_descriptor);
        if (field_descriptor != nullptr) {
          if (field_descriptor->type() ==
              google::protobuf::FieldDescriptor::TYPE_MESSAGE) {
            // 处理嵌套消息类型
            const auto nested_descriptor = field_descriptor->message_type();
            if (nested_descriptor != nullptr) {
              feature_dtype = nested_descriptor->name();
            }
          } else if (field_descriptor->type() ==
                     google::protobuf::FieldDescriptor::TYPE_ENUM) {
            // 处理枚举类型
            const auto enum_descriptor = field_descriptor->enum_type();
            if (enum_descriptor != nullptr) {
              feature_dtype = enum_descriptor->name();
            }
          } else {
            feature_dtype = field_descriptor->type_name();
          }
        }
      }
      LOG(INFO) << "feature not match, feature dtype is: " << feature_dtype
                << ", supposed field type is: " << field_type_
                << " type case: " << int(type_case);
      break;
    }
  }
  if (values.size() > 0) {
    output = cmp(values);
  } else {
    output = keep_empty_;
  }
  return output;
}

}  // namespace internal
}  // namespace monolith_tf
}  // namespace tensorflow
