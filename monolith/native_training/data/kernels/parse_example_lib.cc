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

#include "monolith/native_training/data/kernels/parse_example_lib.h"

#include <algorithm>
#include <tuple>

#include "absl/strings/match.h"

#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {
namespace monolith_tf {

using Instance = ::parser::proto::Instance;
using LineId = ::idl::matrix::proto::LineId;
using EFeature = ::monolith::io::proto::Feature;
using Example = ::monolith::io::proto::Example;
using ExampleBatch = ::monolith::io::proto::ExampleBatch;
using FeatureListType = ::monolith::io::proto::FeatureListType;
using FieldDescriptor = ::google::protobuf::FieldDescriptor;
using FeatureConfigs = ::monolith::io::proto::FeatureConfigs;

BaseParser::BaseParser(const std::vector<std::string> &names,
                       const std::vector<int> &shapes,
                       const std::vector<DataType> &dtypes,
                       const std::vector<std::string> extra_names,
                       DataType input_dtype)
    : input_dtype_(input_dtype), extra_names_(extra_names) {
  for (size_t i = 0; i < names.size(); ++i) {
    name2info_.emplace(names[i], std::make_tuple(i, shapes[i], dtypes[i]));
    idx2info_.emplace(i, std::make_tuple(names[i], shapes[i], dtypes[i]));
    if (shapes[i] == -1 && dtypes[i] == DataType::DT_INT64) {
      ragged_names_.insert(names[i]);
      idx2info_.emplace(i + names.size(),
                        std::make_tuple(names[i], shapes[i], dtypes[i]));
    }
  }
}

void BaseParser::AllocateFeatures(OpKernelContext *ctx,
                                  std::vector<Tensor *> *out_tensors,
                                  OpOutputList *out_list, int batch_size) {
  profiler::TraceMe activity([]() { return "AllocateFeatures"; });
  std::string name;
  int shape;
  DataType dtype;

  for (size_t i = 0; i < name2info_.size(); ++i) {
    std::tie(name, shape, dtype) = idx2info_[i];
    if (shape == -1) {
      OP_REQUIRES(
          ctx, dtype == DataType::DT_INT64,
          errors::InvalidArgument("If shape is -1, then dtype must be int64"));
      OP_REQUIRES_OK(
          ctx, out_list->allocate(i, {batch_size + 1}, &out_tensors->at(i)));
    } else {
      OP_REQUIRES_OK(
          ctx, out_list->allocate(i, {batch_size, shape}, &out_tensors->at(i)));
    }
    std::memset(out_tensors->at(i)->data(), 0,
                out_tensors->at(i)->TotalBytes());
  }
}

void BaseParser::AllocateRaggedValues(OpKernelContext *ctx,
                                      std::vector<Tensor *> *out_tensors,
                                      OpOutputList *out_list, int batch_size) {
  profiler::TraceMe activity([]() { return "AllocateRaggedValues"; });
  int idx, shape;
  DataType dtype;
  for (const std::string &name : ragged_names_) {
    std::tie(idx, shape, dtype) = name2info_[name];
    Tensor *tensor = out_tensors->at(idx);
    shape = static_cast<int>(tensor->flat<int64>()(batch_size));
    idx += name2info_.size();
    OP_REQUIRES_OK(ctx,
                   out_list->allocate(idx, {shape}, &out_tensors->at(idx)));

    if (shape > 0) {
      std::memset(out_tensors->at(idx)->data(), 0,
                  out_tensors->at(idx)->TotalBytes());
    }
  }
}

// TODO: This function can be optimized further if needed:
//
// 1. Instead flat tensor inside, flat it outside (Reduce 2/3 running time)
// 2. Using switch instead of if
void BaseParser::FillFeature(OpKernelContext *ctx, const EFeature &feature,
                             Tensor *tensor, const std::string &name,
                             const int shape, const int offset) {
  if (feature.has_fid_v1_list()) {
    auto flat = tensor->flat<int64>();
    flat(offset + 1) = flat(offset) + feature.fid_v1_list().value_size();
  } else if (feature.has_fid_v2_list()) {
    auto flat = tensor->flat<int64>();
    flat(offset + 1) = flat(offset) + feature.fid_v2_list().value_size();
  } else {
    if (shape == -1) {
      auto flat = tensor->flat<int64>();
      flat(offset + 1) = flat(offset);
    }
  }

  if (feature.has_float_list()) {
    if (shape == 1) {
      CHECK_GT(feature.float_list().value_size(), 0);
      tensor->flat<float>()(offset) = feature.float_list().value(0);
    } else {
      auto matrix = tensor->matrix<float>();
      for (int j = 0; j < std::min(shape, feature.float_list().value_size());
           ++j) {
        matrix(offset, j) = feature.float_list().value(j);
      }
    }
  } else if (feature.has_double_list()) {
    if (shape == 1) {
      CHECK_GT(feature.double_list().value_size(), 0);
      tensor->flat<float>()(offset) = feature.double_list().value(0);
    } else {
      auto matrix = tensor->matrix<float>();
      for (int j = 0; j < std::min(shape, feature.double_list().value_size());
           ++j) {
        matrix(offset, j) = feature.double_list().value(j);
      }
    }
  } else if (feature.has_int64_list()) {
    if (shape == 1) {
      CHECK_GT(feature.int64_list().value_size(), 0);
      tensor->flat<int64>()(offset) = feature.int64_list().value(0);
    } else {
      auto matrix = tensor->matrix<int64>();
      for (int j = 0; j < std::min(shape, feature.int64_list().value_size());
           ++j) {
        matrix(offset, j) = feature.int64_list().value(j);
      }
    }
  } else if (feature.has_bytes_list()) {
    OP_REQUIRES(ctx, shape == 1,
                errors::InvalidArgument("shape must be 1 for bytes list!"));
    CHECK_GT(feature.bytes_list().value_size(), 0);
    tensor->flat<tstring>()(offset) = feature.bytes_list().value(0);
  } else {
    if (feature.has_fid_v2_lists() || feature.has_float_lists() ||
        feature.has_double_lists() || feature.has_int64_lists() ||
        feature.has_bytes_lists()) {
      LOG(ERROR) << "list of list is not support yet!";
    }
  }
}

void BaseParser::FillFromLineId(OpKernelContext *ctx, const LineId &line_id,
                                std::vector<Tensor *> *out_tensors,
                                const int offset) {
  int idx, shape;
  DataType dtype;
  for (const std::string &name : extra_names_) {
    std::tie(idx, shape, dtype) = name2info_[name];
    Tensor *tensor = out_tensors->at(idx);
    if (name == "req_time") {
      tensor->flat<int64>()(offset) = line_id.req_time();
    }
    if (name == "user_id") {
      tensor->flat<tstring>()(offset) = line_id.user_id();
    } else if (name == "uid") {
      tensor->flat<int64>()(offset) = line_id.uid();
    } else if (name == "actions") {
      if (shape > line_id.actions_size()) {
        LOG_EVERY_N(ERROR, 100)
            << absl::StrFormat("Expected actions' shape=%d while got %d", shape,
                               line_id.actions_size());
      }
      if (shape == 1) {
        if (line_id.actions_size()) {
          tensor->flat<int64>()(offset) = line_id.actions(0);
        }
      } else {
        auto matrix = tensor->matrix<int64>();
        for (int i = 0; i < std::min(shape, line_id.actions_size()); ++i) {
          matrix(offset, i) = line_id.actions(i);
        }
      }
    } else if (name == "sample_rate") {
      tensor->flat<float>()(offset) = line_id.sample_rate();
    } else if (name == "chnid") {
      tensor->flat<float>()(offset) = line_id.chnid();
    } else {
      const auto *field = descriptor->FindFieldByName(name);
      OP_REQUIRES_OK(ctx, FillFromLineIdByreflection(line_id, field, tensor,
                                                     shape, offset));
    }
  }
}

Status BaseParser::FillFromLineIdByreflection(const LineId &line_id,
                                              const FieldDescriptor *field,
                                              Tensor *tensor, int shape,
                                              int offset) {
  if (field->is_repeated()) {
    const int field_size = reflection->FieldSize(line_id, field);
    switch (field->cpp_type()) {
      case FieldDescriptor::CPPTYPE_INT32: {
        if (shape == 1) {
          tensor->flat<int64>()(offset) =
              reflection->GetRepeatedInt32(line_id, field, 0);
        } else {
          auto matrix = tensor->matrix<int64>();
          for (int i = 0; i < std::min(shape, field_size); ++i) {
            matrix(offset, i) = reflection->GetRepeatedInt32(line_id, field, i);
          }
        }
        break;
      }
      case FieldDescriptor::CPPTYPE_UINT32: {
        if (shape == 1) {
          tensor->flat<int64>()(offset) =
              reflection->GetRepeatedUInt32(line_id, field, 0);
        } else {
          auto matrix = tensor->matrix<int64>();
          for (int i = 0; i < std::min(shape, field_size); ++i) {
            matrix(offset, i) =
                reflection->GetRepeatedUInt32(line_id, field, i);
          }
        }
        break;
      }
      case FieldDescriptor::CPPTYPE_INT64: {
        if (shape == 1) {
          tensor->flat<int64>()(offset) =
              reflection->GetRepeatedInt64(line_id, field, 0);
        } else {
          auto matrix = tensor->matrix<int64>();
          for (int i = 0; i < std::min(shape, field_size); ++i) {
            matrix(offset, i) = reflection->GetRepeatedInt64(line_id, field, i);
          }
        }
        break;
      }
      case FieldDescriptor::CPPTYPE_UINT64: {
        if (shape == 1) {
          tensor->flat<int64>()(offset) =
              reflection->GetRepeatedUInt64(line_id, field, 0);
        } else {
          auto matrix = tensor->matrix<int64>();
          for (int i = 0; i < std::min(shape, field_size); ++i) {
            matrix(offset, i) =
                reflection->GetRepeatedUInt64(line_id, field, i);
          }
        }
        break;
      }
      case FieldDescriptor::CPPTYPE_FLOAT: {
        if (shape == 1) {
          tensor->flat<float>()(offset) =
              reflection->GetRepeatedFloat(line_id, field, 0);
        } else {
          auto matrix = tensor->matrix<float>();
          for (int i = 0; i < std::min(shape, field_size); ++i) {
            matrix(offset, i) = reflection->GetRepeatedFloat(line_id, field, i);
          }
        }
        break;
      }
      case FieldDescriptor::CPPTYPE_DOUBLE: {
        if (shape == 1) {
          tensor->flat<float>()(offset) =
              reflection->GetRepeatedDouble(line_id, field, 0);
        } else {
          auto matrix = tensor->matrix<float>();
          for (int i = 0; i < std::min(shape, field_size); ++i) {
            matrix(offset, i) =
                reflection->GetRepeatedDouble(line_id, field, i);
          }
        }
        break;
      }
      case FieldDescriptor::CPPTYPE_STRING: {
        if (shape == 1) {
          tensor->flat<tstring>()(offset) =
              reflection->GetRepeatedString(line_id, field, 0);
        } else {
          auto matrix = tensor->matrix<tstring>();
          for (int i = 0; i < std::min(shape, field_size); ++i) {
            matrix(offset, i) =
                reflection->GetRepeatedString(line_id, field, i);
          }
        }
        break;
      }
      default:
        return errors::InvalidArgument(field->name(),
                                       " Data type not match, only "
                                       "string/int32/int64/float32 "
                                       "supported.");
    }
  } else {
    switch (field->cpp_type()) {
      case FieldDescriptor::CPPTYPE_INT32: {
        auto flat = tensor->flat<int64>();
        flat(offset) = reflection->GetInt32(line_id, field);
        break;
      }
      case FieldDescriptor::CPPTYPE_UINT32: {
        auto flat = tensor->flat<int64>();
        flat(offset) = reflection->GetUInt32(line_id, field);
        break;
      }
      case FieldDescriptor::CPPTYPE_INT64: {
        auto flat = tensor->flat<int64>();
        flat(offset) = reflection->GetInt64(line_id, field);
        break;
      }
      case FieldDescriptor::CPPTYPE_UINT64: {
        auto flat = tensor->flat<int64>();
        flat(offset) = reflection->GetUInt64(line_id, field);
        break;
      }
      case FieldDescriptor::CPPTYPE_FLOAT: {
        auto flat = tensor->flat<float>();
        flat(offset) = reflection->GetFloat(line_id, field);
        break;
      }
      case FieldDescriptor::CPPTYPE_DOUBLE: {
        auto flat = tensor->flat<float>();
        flat(offset) = reflection->GetDouble(line_id, field);
        break;
      }
      case FieldDescriptor::CPPTYPE_STRING: {
        auto flat = tensor->flat<tstring>();
        flat(offset) = reflection->GetString(line_id, field);
        break;
      }
      default:
        return errors::InvalidArgument(field->name(),
                                       " Data type not match, only "
                                       "int32/int64/float32/string "
                                       "supported.");
    }
  }

  return Status::OK();
}

ExampleParser::ExampleParser(const std::vector<std::string> &names,
                             const std::vector<int> &shapes,
                             const std::vector<DataType> &dtypes,
                             const std::vector<std::string> extra_names,
                             DataType input_dtype, FeatureNameMapper *mapper)
    : BaseParser(names, shapes, dtypes, extra_names, input_dtype),
      mapper_(mapper) {
  std::unordered_set<std::string> extra_name_set(extra_names.begin(),
                                                 extra_names.end());
  std::vector<std::string> sparse_feature_names;
  for (size_t i = 0; i < names.size(); ++i) {
    if (!extra_name_set.count(names[i]) && shapes[i] == -1) {
      sparse_feature_names.push_back(names[i]);
    }
  }

  CHECK(mapper_->RegisterValidNames(sparse_feature_names));
}

void ExampleParser::Parse(OpKernelContext *ctx,
                          const std::vector<const Example *> &examples,
                          OpOutputList *out_list) {
  int batch_size = examples.size();
  std::vector<Tensor *> out_tensors;
  out_tensors.resize(idx2info_.size());

  int idx, shape;
  DataType dtype;

  // 1) allocate output tensors for ragged splits and other non-ragged
  AllocateFeatures(ctx, &out_tensors, out_list, batch_size);

  // 2) fill all tensors expect ragged values
  int offset = 0;
  {
    profiler::TraceMe activity(
        []() { return "FillAllTensorsExceptRaggedValues"; });
    for (const Example *example : examples) {
      std::unordered_set<std::string> appeared;
      appeared.reserve(example->named_feature_size());
      for (const auto &named_feature : example->named_feature()) {
        // FeatureNameMapper
        const std::string &name = named_feature.name();
        auto it = name2info_.find(name);
        if (it == name2info_.end()) continue;

        std::tie(idx, shape, dtype) = it->second;
        FillFeature(ctx, named_feature.feature(), out_tensors[idx], name, shape,
                    offset);
        appeared.insert(name);
      }

      for (const auto &ragged : ragged_names_) {
        if (appeared.find(ragged) == appeared.end()) {
          std::tie(idx, shape, dtype) = name2info_[ragged];
          auto flat = out_tensors[idx]->flat<int64>();
          flat(offset + 1) += flat(offset);
        }
      }

      // for label
      auto it = name2info_.find("label");
      if (it != name2info_.end()) {
        std::tie(idx, shape, dtype) = it->second;
        Tensor *tensor = out_tensors[idx];
        if (shape == 1) {
          tensor->flat<float>()(offset) = example->label(0);
        } else {
          auto matrix = tensor->matrix<float>();
          for (int j = 0; j < std::min(shape, example->label_size()); ++j) {
            matrix(offset, j) = example->label(j);
          }
        }
      }

      // for instance_weight
      it = name2info_.find("instance_weight");
      if (it != name2info_.end()) {
        std::tie(idx, shape, dtype) = it->second;
        Tensor *tensor = out_tensors[idx];
        float instance_weight = example->instance_weight();
        tensor->flat<float>()(offset) =
            instance_weight > 0 ? instance_weight : 1.0;
      }

      // for extra fields in line_id
      if (!extra_names_.empty()) {
        const LineId &line_id = example->line_id();
        FillFromLineId(ctx, line_id, &out_tensors, offset);
      }

      offset++;
    }
  }

  // 3) allocate output tensors for ragged values
  AllocateRaggedValues(ctx, &out_tensors, out_list, batch_size);

  // 4) fill ragged values
  if (ragged_names_.size()) {
    profiler::TraceMe activity([]() { return "FillRaggedValues"; });
    offset = 0;
    for (const Example *example : examples) {
      for (const auto &named_feature : example->named_feature()) {
        const auto &name = named_feature.name();
        auto it = ragged_names_.find(name);
        if (it != ragged_names_.end()) {
          std::tie(idx, shape, dtype) = name2info_[name];
          auto splits = out_tensors[idx]->flat<int64>();
          auto values = out_tensors[idx + name2info_.size()]->flat<int64>();
          int start = static_cast<int>(splits(offset));
          const auto &feature = named_feature.feature();
          if (feature.has_fid_v1_list()) {
            for (int i = 0; i < feature.fid_v1_list().value_size(); ++i) {
              values(start + i) =
                  convert_fid_v1_to_v2(feature.fid_v1_list().value(i));
            }
          } else if (feature.has_fid_v2_list()) {
            for (int i = 0; i < feature.fid_v2_list().value_size(); ++i) {
              values(start + i) = feature.fid_v2_list().value(i);
            }
          }
        }
      }

      offset++;
    }
  }
}

ExampleBatchParser::ExampleBatchParser(
    const std::vector<std::string> &names, const std::vector<int> &shapes,
    const std::vector<DataType> &dtypes,
    const std::vector<std::string> extra_names, DataType input_dtype)
    : BaseParser(names, shapes, dtypes, extra_names, input_dtype) {}

void ExampleBatchParser::Parse(OpKernelContext *ctx,
                               const ExampleBatch &example_batch,
                               OpOutputList *out_list) {
  int batch_size = example_batch.batch_size();
  std::vector<Tensor *> out_tensors;
  out_tensors.resize(idx2info_.size());

  std::string name;
  int idx, shape;
  DataType dtype;

  // 1) allocate output tensors for ragged splits and other non-ragged
  AllocateFeatures(ctx, &out_tensors, out_list, batch_size);

  // 2) fill all tensors expect ragged values
  for (const auto &named_feature_list : example_batch.named_feature_list()) {
    name = named_feature_list.name();
    if (name == "__LINE_ID__") {
      // for extra fields in line_id
      if (extra_names_.size() > 0) {
        int offset = 0;
        for (const auto &feature : named_feature_list.feature()) {
          LineId line_id;
          CHECK_GT(feature.bytes_list().value_size(), 0);
          const auto serialized = feature.bytes_list().value(0);
          OP_REQUIRES(
              ctx, line_id.ParseFromArray(serialized.data(), serialized.size()),
              errors::FailedPrecondition("Failed to parse the LineId."));
          FillFromLineId(ctx, line_id, &out_tensors, offset);
          offset++;
        }
      }
    } else if (name == "__LABEL__") {
      // for label
      auto it = name2info_.find("label");
      if (it != name2info_.end()) {
        std::tie(idx, shape, dtype) = it->second;
        Tensor *tensor = out_tensors[idx];

        int offset = 0;
        for (const auto &feature : named_feature_list.feature()) {
          if (shape == 1) {
            CHECK_GT(feature.float_list().value_size(), 0);
            tensor->flat<float>()(offset) = feature.float_list().value(0);
          } else {
            auto matrix = tensor->matrix<float>();
            for (int j = 0;
                 j < std::min(shape, feature.float_list().value_size()); ++j) {
              matrix(offset, j) = feature.float_list().value(j);
            }
          }
          offset++;
        }
      }
    } else if (name == "instance_weight") {
      auto it = name2info_.find("instance_weight");
      if (it != name2info_.end()) {
        std::tie(idx, shape, dtype) = it->second;
        Tensor *tensor = out_tensors[idx];
        int offset = 0;
        for (const auto &feature : named_feature_list.feature()) {
          CHECK_GT(feature.float_list().value_size(), 0);
          float instance_weight = feature.float_list().value(0);
          tensor->flat<float>()(offset) =
              instance_weight > 0 ? instance_weight : 1.0;
          offset++;
        }
      }
    } else {
      auto it = name2info_.find(name);
      if (it == name2info_.end()) continue;
      std::tie(idx, shape, dtype) = name2info_[name];
      Tensor *tensor = out_tensors[idx];

      if (named_feature_list.type() == FeatureListType::SHARED) {
        CHECK_GT(named_feature_list.feature_size(), 0);
        const auto &feature = named_feature_list.feature(0);
        for (int offset = 0; offset < batch_size; ++offset) {
          FillFeature(ctx, feature, tensor, name, shape, offset);
        }
      } else {
        int offset = 0;
        for (const auto &feature : named_feature_list.feature()) {
          FillFeature(ctx, feature, tensor, name, shape, offset);
          offset++;
        }
      }
    }
  }

  // 3) allocate output tensors for ragged values
  AllocateRaggedValues(ctx, &out_tensors, out_list, batch_size);

  // 4) fill ragged values
  if (ragged_names_.size()) {
    for (const auto &named_feature_list : example_batch.named_feature_list()) {
      name = named_feature_list.name();
      auto it = ragged_names_.find(name);
      if (it != ragged_names_.end()) {
        std::tie(idx, shape, dtype) = name2info_[name];
        auto splits = out_tensors[idx]->flat<int64>();
        auto values = out_tensors[idx + name2info_.size()]->flat<int64>();

        if (named_feature_list.type() == FeatureListType::SHARED) {
          const auto &feature = named_feature_list.feature(0);
          for (int offset = 0; offset < batch_size; ++offset) {
            int start = static_cast<int>(splits(offset));
            if (feature.has_fid_v1_list()) {
              for (int i = 0; i < feature.fid_v1_list().value_size(); ++i) {
                values(start + i) =
                    convert_fid_v1_to_v2(feature.fid_v1_list().value(i));
              }
            } else if (feature.has_fid_v2_list()) {
              for (int i = 0; i < feature.fid_v2_list().value_size(); ++i) {
                values(start + i) = feature.fid_v2_list().value(i);
              }
            }
          }
        } else {
          int offset = 0;
          for (const auto &feature : named_feature_list.feature()) {
            int start = static_cast<int>(splits(offset));
            if (feature.has_fid_v1_list()) {
              for (int i = 0; i < feature.fid_v1_list().value_size(); ++i) {
                values(start + i) =
                    convert_fid_v1_to_v2(feature.fid_v1_list().value(i));
              }
            } else if (feature.has_fid_v2_list()) {
              for (int i = 0; i < feature.fid_v2_list().value_size(); ++i) {
                values(start + i) = feature.fid_v2_list().value(i);
              }
            }
            offset++;
          }
        }
      }
    }
  }
}

ExampleBatchListParser::ExampleBatchListParser(
    const std::vector<std::string> &names, const std::vector<int> &shapes,
    const std::vector<DataType> &dtypes,
    const std::vector<std::string> &extra_names, DataType input_dtype)
    : BaseParser(names, shapes, dtypes, extra_names, input_dtype) {}

void ExampleBatchListParser::Parse(
    OpKernelContext *ctx, const ExampleBatch &example_batch,
    const std::vector<internal::TaskConfig> &label_config_,
    float positive_label, float negative_label, OpOutputList *out_list) {
  int batch_size = example_batch.batch_size();
  std::vector<Tensor *> out_tensors;
  out_tensors.resize(idx2info_.size());

  std::string name;
  int idx, shape;
  DataType dtype;

  // 1) allocate output tensors for ragged splits and other non-ragged
  AllocateFeatures(ctx, &out_tensors, out_list, batch_size);

  // 2) fill all tensors expect ragged values
  for (const auto &named_feature_list : example_batch.named_feature_list()) {
    name = named_feature_list.name();
    if (name == "__LINE_ID__") {
      auto it = name2info_.find("label");
      if (it != name2info_.end()) {
        std::tie(idx, shape, dtype) = it->second;
      }

      // for extra fields in line_id
      if (extra_names_.size() > 0) {
        int offset = 0;
        for (const auto &feature : named_feature_list.feature()) {
          LineId line_id;
          CHECK_GT(feature.bytes_list().value_size(), 0);
          const auto serialized = feature.bytes_list().value(0);
          OP_REQUIRES(
              ctx, line_id.ParseFromArray(serialized.data(), serialized.size()),
              errors::FailedPrecondition("Failed to parse the LineId."));
          FillFromLineId(ctx, line_id, &out_tensors, offset);
          if (it != name2info_.end()) {
            FillLabelFromLineId(ctx, line_id, label_config_, positive_label,
                                negative_label, out_tensors[idx], offset);
          }
          offset++;
        }
      }
    } else if (name == "instance_weight") {
      auto it = name2info_.find("instance_weight");
      if (it != name2info_.end()) {
        std::tie(idx, shape, dtype) = it->second;
        Tensor *tensor = out_tensors[idx];
        int offset = 0;
        for (const auto &feature : named_feature_list.feature()) {
          CHECK_GT(feature.float_list().value_size(), 0);
          float instance_weight = feature.float_list().value(0);
          tensor->flat<float>()(offset++) =
              instance_weight > 0 ? instance_weight : 1.0;
        }
      }
    } else {
      auto it = name2info_.find(name);
      if (it == name2info_.end()) continue;
      std::tie(idx, shape, dtype) = it->second;
      Tensor *tensor = out_tensors[idx];

      if (named_feature_list.type() == FeatureListType::SHARED) {
        const auto &feature = named_feature_list.feature(0);
        for (int offset = 0; offset < batch_size; ++offset) {
          FillFeature(ctx, feature, tensor, name, shape, offset);
        }
      } else {
        int offset = 0;
        for (const auto &feature : named_feature_list.feature()) {
          FillFeature(ctx, feature, tensor, name, shape, offset);
          offset++;
        }
      }
    }
  }

  // 3) allocate output tensors for ragged values
  AllocateRaggedValues(ctx, &out_tensors, out_list, batch_size);

  // 4) fill ragged values
  for (const auto &named_feature_list : example_batch.named_feature_list()) {
    name = named_feature_list.name();
    auto it = ragged_names_.find(name);
    if (it != ragged_names_.end()) {
      int slot = named_feature_list.id();
      std::tie(idx, shape, dtype) = name2info_[name];
      auto splits = out_tensors[idx]->flat<int64>();
      auto values = out_tensors[idx + name2info_.size()]->flat<int64>();

      if (named_feature_list.type() == FeatureListType::SHARED) {
        const auto &feature = named_feature_list.feature(0);
        for (int offset = 0; offset < batch_size; ++offset) {
          int start = static_cast<int>(splits(offset));
          if (feature.has_fid_v1_list()) {
            for (int i = 0; i < feature.fid_v1_list().value_size(); ++i) {
              values(start + i) =
                  GetFidV2(slot, feature.fid_v1_list().value(i));
            }
          } else if (feature.has_fid_v2_list()) {
            for (int i = 0; i < feature.fid_v2_list().value_size(); ++i) {
              values(start + i) =
                  GetFidV2(slot, feature.fid_v2_list().value(i));
            }
          }
        }
      } else {
        int offset = 0;
        for (const auto &feature : named_feature_list.feature()) {
          int start = static_cast<int>(splits(offset));
          if (feature.has_fid_v1_list()) {
            for (int i = 0; i < feature.fid_v1_list().value_size(); ++i) {
              values(start + i) =
                  GetFidV2(slot, feature.fid_v1_list().value(i));
            }
          } else if (feature.has_fid_v2_list()) {
            for (int i = 0; i < feature.fid_v2_list().value_size(); ++i) {
              values(start + i) =
                  GetFidV2(slot, feature.fid_v2_list().value(i));
            }
          }
          offset++;
        }
      }
    }
  }
}

void ExampleBatchListParser::FillLabelFromLineId(
    OpKernelContext *ctx, const ::idl::matrix::proto::LineId &line_id,
    const std::vector<internal::TaskConfig> &label_config_,
    float positive_label, float negative_label, Tensor *out_tensor,
    const int offset) {
  std::set<int32_t> actions(line_id.actions().begin(), line_id.actions().end());

  int label_idx = 0;
  auto matrix = out_tensor->matrix<float>();
  for (const auto &task_conf : label_config_) {
    if (internal::HasIntersection(task_conf.pos_actions, actions)) {
      matrix(offset, label_idx) = positive_label;
    } else {
      if (task_conf.neg_actions.empty()) {
        matrix(offset, label_idx) = negative_label;
      } else {
        if (internal::HasIntersection(task_conf.neg_actions, actions)) {
          matrix(offset, label_idx) = negative_label;
        } else {
          matrix(offset, label_idx) = internal::INVALID_LABEL;
        }
      }
    }

    label_idx++;
  }
}
}  // namespace monolith_tf
}  // namespace tensorflow
