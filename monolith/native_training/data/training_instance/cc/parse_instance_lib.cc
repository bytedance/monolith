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

#include <functional>
#include <memory>
#include <vector>

#include "glog/logging.h"
#include "tensorflow/core/platform/errors.h"

#include "idl/matrix/compression/float16.h"
#include "monolith/native_training/data/training_instance/cc/parse_instance_lib.h"
#include "monolith/native_training/data/training_instance/cc/reader_util.h"
#include "monolith/native_training/data/training_instance/cc/ue_compress.h"

namespace tensorflow {
namespace monolith_tf {
namespace {

using ::google::protobuf::FieldDescriptor;
using ::idl::matrix::proto::Feature;
using ::parser::proto::Instance;
using tensorflow::monolith_tf::UECompress;

// The spec that will be used by parser.
// It includes some preprocessed data
struct InstanceParserSpec : InstanceParserConfig {
  explicit InstanceParserSpec(const InstanceParserConfig &config)
      : InstanceParserConfig(config) {}

  Status Init() {
    fidv1_features_set = {fidv1_features.begin(), fidv1_features.end()};
    fidv2_features_set = {fidv2_features.begin(), fidv2_features.end()};
    int index = 0;
    for (int slot : fidv1_features) {
      slot_to_index[slot] = index++;
    }
    for (const std::string &name : fidv2_features) {
      fidv2_name_to_index[name] = index++;
    }
    n_ragged_tensors = fidv1_features.size() + fidv2_features.size();

    float_features_set = {float_features.begin(), float_features.end()};
    for (size_t i = 0; i < float_features.size(); ++i) {
      float_feature_name_to_index[float_features[i]] = i;
    }
    n_float_tensors = float_features.size();

    int64_features_set = {int64_features.begin(), int64_features.end()};
    for (size_t i = 0; i < int64_features.size(); ++i) {
      int64_feature_name_to_index[int64_features[i]] = i;
    }
    n_int64_tensors = int64_features.size();

    string_features_set = {string_features.begin(), string_features.end()};
    for (size_t i = 0; i < string_features.size(); ++i) {
      string_feature_name_to_index[string_features[i]] = i;
    }
    n_string_tensors = string_features.size();

    return Status::OK();
  }

  // Fid features attrs
  absl::flat_hash_set<int> fidv1_features_set;
  absl::flat_hash_map<int, int> slot_to_index;
  absl::flat_hash_set<std::string> fidv2_features_set;
  absl::flat_hash_map<std::string, int> fidv2_name_to_index;
  int n_ragged_tensors;

  // Float features attrs
  int n_float_tensors;
  absl::flat_hash_set<std::string> float_features_set;
  absl::flat_hash_map<std::string, int> float_feature_name_to_index;

  // Int64 features attrs
  int n_int64_tensors;
  absl::flat_hash_set<std::string> int64_features_set;
  absl::flat_hash_map<std::string, int> int64_feature_name_to_index;

  // String features attrs
  int n_string_tensors;
  absl::flat_hash_set<std::string> string_features_set;
  absl::flat_hash_map<std::string, int> string_feature_name_to_index;
};

class RaggedTensorProcessor {
 public:
  explicit RaggedTensorProcessor(const InstanceParserSpec *spec)
      : spec_(*spec) {}

  virtual ~RaggedTensorProcessor() = default;

  // Process the ragged tensor.
  // The output will be added to the output.
  virtual Status ParseRaggedTensors(OpKernelContext *ctx,
                                    absl::Span<const Instance> instances,
                                    InstanceParser::Output *output) = 0;

 protected:
  const InstanceParserSpec &spec() const { return spec_; }

  // A util function for child class to use.
  template <typename Func>
  void IterateFidFeatures(const Instance &instance, Func func) {
    const bool apply_fid_v2 = !spec_.fidv2_features.empty();
    for (const uint64_t fid : instance.fid()) {
      int slot_id = slot_id_v1(fid);
      if (!spec_.fidv1_features_set.contains(slot_id)) continue;
      uint64_t converted_fid =
          apply_fid_v2 ? convert_fid_v1_to_v2(slot_id, fid) : fid;
      func(spec_.slot_to_index.at(slot_id), converted_fid);
    }

    // Feature v2 should never have 2 features with the same feature name.
    for (const auto &feature : instance.feature()) {
      if (!spec_.fidv2_features_set.contains(feature.name())) continue;
      // This is a simple sample list.
      for (const auto &fid : feature.fid()) {
        func(spec_.fidv2_name_to_index.at(feature.name()), fid);
      }
      // this is a sequence feature list.
      for (const auto &fidlist : feature.fid_list()) {
        for (const auto &fid : fidlist.value()) {
          func(spec_.fidv2_name_to_index.at(feature.name()), fid);
        }
      }
    }
  }

 private:
  const InstanceParserSpec &spec_;
};

class RegularRaggedTensorProcessor : public RaggedTensorProcessor {
 public:
  explicit RegularRaggedTensorProcessor(const InstanceParserSpec *spec)
      : RaggedTensorProcessor(spec) {}

  Status ParseRaggedTensors(OpKernelContext *ctx,
                            absl::Span<const Instance> instances,
                            InstanceParser::Output *output) override {
    int batch_size = instances.size();
    std::vector<TTypes<int64>::Vec> splits_vec;
    splits_vec.reserve(spec().n_ragged_tensors);
    for (int i = 0; i < spec().n_ragged_tensors; ++i) {
      output->tensors.emplace_back();
      Tensor *t = &output->tensors.back();
      TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_INT64, {batch_size + 1}, t));
      auto vec = t->vec<int64>();
      vec(0) = 0;
      splits_vec.emplace_back(vec);
    }

    std::vector<int> nums(spec().n_ragged_tensors);
    for (int i = 0; i < batch_size; ++i) {
      IterateFidFeatures(instances[i],
                         [&nums](int idx, int64_t fid) { nums[idx]++; });
      for (int j = 0; j < spec().n_ragged_tensors; ++j) {
        splits_vec[j](i + 1) = nums[j];
      }
    }

    std::vector<TTypes<int64>::Vec> values_vec;
    for (int i = 0; i < spec().n_ragged_tensors; ++i) {
      output->tensors.emplace_back();
      Tensor *t = &output->tensors.back();
      TF_RETURN_IF_ERROR(
          ctx->allocate_temp(DT_INT64, {splits_vec[i](batch_size)}, t));
      values_vec.emplace_back(t->vec<int64>());
    }

    std::fill(nums.begin(), nums.end(), 0);
    for (int i = 0; i < batch_size; ++i) {
      IterateFidFeatures(instances[i],
                         [&nums, &values_vec](int idx, int64_t fid) {
                           values_vec[idx](nums[idx]) = fid;
                           nums[idx]++;
                         });
    }

    return Status::OK();
  }
};

class ConcatRaggedTensorProcessor : public RaggedTensorProcessor {
 public:
  explicit ConcatRaggedTensorProcessor(const InstanceParserSpec *spec)
      : RaggedTensorProcessor(spec) {}

  Status ParseRaggedTensors(OpKernelContext *ctx,
                            absl::Span<const Instance> instances,
                            InstanceParser::Output *output) override {
    int batch_size = instances.size();
    if (batch_size != 1) {
      return errors::InvalidArgument(
          "ConcatRaggedTensorProcessor only support batch_size == 1");
    }
    const Instance &instance = instances[0];
    Tensor t;
    TF_RETURN_IF_ERROR(
        ctx->allocate_temp(DT_INT64, {spec().n_ragged_tensors + 1}, &t));
    output->tensors.push_back(t);
    auto split = t.vec<int64>().setZero();
    IterateFidFeatures(instance,
                       [&split](int idx, int64_t fid) { ++split(idx + 1); });
    for (int i = 1; i <= spec().n_ragged_tensors; ++i) {
      split(i) = split(i) + split(i - 1);
    }

    TF_RETURN_IF_ERROR(
        ctx->allocate_temp(DT_INT64, {split(spec().n_ragged_tensors)}, &t));
    output->tensors.push_back(t);
    auto value = t.vec<int64>();
    std::vector<int> pos(spec().n_ragged_tensors);
    for (int i = 0; i < spec().n_ragged_tensors; ++i) {
      pos[i] = split(i);
    }
    IterateFidFeatures(instance, [&value, &pos](int idx, int64_t fid) {
      value(pos[idx]++) = fid;
    });
    return Status::OK();
  }
};

}  // namespace

class InstanceParser::Impl {
 public:
  explicit Impl(const InstanceParserConfig &config) : spec_(config) {
    switch (config.fid_output_type) {
      case InstanceParserConfig::REGULAR:
        ragged_tensor_processor_ =
            std::make_unique<RegularRaggedTensorProcessor>(&spec_);
        break;
      case InstanceParserConfig::CONCAT:
        ragged_tensor_processor_ =
            std::make_unique<ConcatRaggedTensorProcessor>(&spec_);
        break;
    }

    ue_compress_ = std::make_unique<UECompress>();
  }

  Status Init() { return spec_.Init(); }

  Status Parse(OpKernelContext *ctx, absl::Span<const Instance> instances,
               Output *output) {
    output->tensors.clear();
    TF_RETURN_IF_ERROR(
        ragged_tensor_processor_->ParseRaggedTensors(ctx, instances, output));
    TF_RETURN_IF_ERROR(FillFloatFeatures(ctx, instances, output));
    TF_RETURN_IF_ERROR(FillInt64Features(ctx, instances, output));
    TF_RETURN_IF_ERROR(FillStringFeatures(ctx, instances, output));
    TF_RETURN_IF_ERROR(ParseFloatTensors(ctx, instances, output));
    TF_RETURN_IF_ERROR(ParseInt64Tensors(ctx, instances, output));
    TF_RETURN_IF_ERROR(ParseStringTensors(ctx, instances, output));
    return Status::OK();
  }

 private:
  Status FillFloatFeatures(OpKernelContext *ctx,
                           absl::Span<const Instance> instances,
                           Output *output) {
    const int batch_size = instances.size();
    std::vector<TTypes<float>::Matrix> values_mat;
    for (int i = 0; i < spec_.n_float_tensors; ++i) {
      output->tensors.emplace_back();
      Tensor *t = &output->tensors.back();
      TF_RETURN_IF_ERROR(ctx->allocate_temp(
          DT_FLOAT, GetBatched1DShape(batch_size, spec_.float_feature_dims[i]),
          t));
      values_mat.emplace_back(
          t->shaped<float, 2>({batch_size, spec_.float_feature_dims[i]}));
      // To be safe, we initialize float tensors to zero by default.
      values_mat.back().setZero();
    }

    for (int i = 0; i < batch_size; ++i) {
      const Instance &instance = instances[i];
      for (const Feature &feature : instance.feature()) {
        if (spec_.float_features_set.contains(feature.name())) {
          std::vector<float> embedding;
          bool ret = ue_compress_->decompress_embeddings(
              feature, &embedding, UECompressMethod::COMPRESS_QTZ8);
          int idx = spec_.float_feature_name_to_index.at(feature.name());
          if (ret) {
            // Process data with qtz8 compression.
            if (spec_.float_feature_dims[idx] != embedding.size()) {
              return errors::Internal(
                  "Decompressed qtz8 data length doesn't match feature dim,",
                  " feature dim: ", spec_.float_feature_dims[idx],
                  ", uncompressed qtz8 size: ", embedding.size());
            }
            for (int j = 0; j < spec_.float_feature_dims[idx]; ++j) {
              values_mat[idx](i, j) = embedding[j];
            }
          } else if (spec_.float_feature_dims[idx] ==
                     feature.float_value_size()) {
            for (int j = 0; j < spec_.float_feature_dims[idx]; ++j) {
              values_mat[idx](i, j) = feature.float_value(j);
            }
          } else {
            // TODO(zouxuan) Set the default value to 0 for now. Xuan will make
            // an eventual fix for this later.
            for (int j = 0; j < spec_.float_feature_dims[idx]; ++j) {
              values_mat[idx](i, j) = 0;
            }
          }
        }
      }
    }
    return Status::OK();
  }

  Status FillInt64Features(OpKernelContext *ctx,
                           absl::Span<const Instance> instances,
                           Output *output) {
    const int batch_size = instances.size();
    std::vector<TTypes<int64>::Matrix> values_mat;
    for (int i = 0; i < spec_.n_int64_tensors; ++i) {
      output->tensors.emplace_back();
      Tensor *t = &output->tensors.back();
      TF_RETURN_IF_ERROR(ctx->allocate_temp(
          DT_INT64, GetBatched1DShape(batch_size, spec_.int64_feature_dims[i]),
          t));
      values_mat.emplace_back(
          t->shaped<int64, 2>({batch_size, spec_.int64_feature_dims[i]}));
      // To be safe, we initialize int64 tensors to zero by default.
      values_mat.back().setZero();
    }

    for (int i = 0; i < batch_size; ++i) {
      const Instance &instance = instances[i];
      for (const Feature &feature : instance.feature()) {
        if (spec_.int64_features_set.contains(feature.name())) {
          int idx = spec_.int64_feature_name_to_index.at(feature.name());
          if (spec_.int64_feature_dims[idx] == feature.int64_value_size()) {
            for (int j = 0; j < spec_.int64_feature_dims[idx]; ++j) {
              values_mat[idx](i, j) = feature.int64_value(j);
            }
          } else {
            // TODO(zouxuan) Set the default value to 0 for now. Xuan will make
            // an eventual fix for this later.
            for (int j = 0; j < spec_.int64_feature_dims[idx]; ++j) {
              values_mat[idx](i, j) = 0;
            }
          }
        }
      }
    }
    return Status::OK();
  }

  Status FillStringFeatures(OpKernelContext *ctx,
                            absl::Span<const Instance> instances,
                            Output *output) {
    const int batch_size = instances.size();
    std::vector<TTypes<tstring>::Matrix> values_mat;
    for (int i = 0; i < spec_.n_string_tensors; ++i) {
      output->tensors.emplace_back();
      Tensor *t = &output->tensors.back();
      TF_RETURN_IF_ERROR(ctx->allocate_temp(
          DT_STRING,
          GetBatched1DShape(batch_size, spec_.string_feature_dims[i]), t));
      values_mat.emplace_back(
          t->shaped<tstring, 2>({batch_size, spec_.string_feature_dims[i]}));
    }

    for (int i = 0; i < batch_size; ++i) {
      const Instance &instance = instances[i];
      for (const Feature &feature : instance.feature()) {
        if (spec_.string_features_set.contains(feature.name())) {
          int idx = spec_.string_feature_name_to_index.at(feature.name());
          if (spec_.string_feature_dims[idx] == feature.bytes_value_size()) {
            for (int j = 0; j < spec_.string_feature_dims[idx]; ++j) {
              values_mat[idx](i, j) = feature.bytes_value(j);
            }
          } else {
            for (int j = 0; j < spec_.string_feature_dims[idx]; ++j) {
              values_mat[idx](i, j) = "";
            }
          }
        }
      }
    }
    return Status::OK();
  }

  Status ParseFloatTensors(OpKernelContext *ctx,
                           absl::Span<const Instance> instances,
                           Output *output) {
    const int batch_size = instances.size();
    const auto *descriptor = ::idl::matrix::proto::LineId::GetDescriptor();
    const auto *reflection = ::idl::matrix::proto::LineId::GetReflection();
    for (size_t i = 0; i < spec_.misc_float_features.size(); ++i) {
      output->tensors.emplace_back();
      Tensor *t = &output->tensors.back();
      TF_RETURN_IF_ERROR(ctx->allocate_temp(
          DT_FLOAT, GetBatched1DShape(batch_size, spec_.misc_float_dims[i]),
          t));
      auto mat = t->shaped<float, 2>({batch_size, spec_.misc_float_dims[i]});
      // To be safe, we initialize float tensors to zero by default.
      mat.setZero();
      const std::string &name = spec_.misc_float_features[i];
      if (name == "label") {
        for (size_t j = 0; j < instances.size(); ++j) {
          const Instance &instance = instances[j];
          int dim = spec_.misc_float_dims[i];
          if (instance.label_size() < dim) {
            LOG_EVERY_N_SEC(ERROR, 60)
                << name << " Dim is smaller than expected "
                << instance.label_size() << " v.s. " << dim;
            dim = instance.label_size();
          }
          for (int k = 0; k < dim; ++k) {
            mat(j, k) = instance.label(k);
          }
        }
        continue;
      } else if (name == "instance_weight") {
        int dim = spec_.misc_float_dims[i];
        if (dim != 1) {
          LOG_EVERY_N_SEC(ERROR, 60) << name << " Dim is illegal, expected 1 "
                                     << " v.s. " << dim;
          dim = 1;
        }
        for (size_t j = 0; j < instances.size(); ++j) {
          const Instance &instance = instances[j];
          mat(j, 0) = instance.has_instance_weight()
                          ? instance.instance_weight()
                          : 1.0f;
        }
        continue;
      }
      const auto *field = descriptor->FindFieldByName(name);
      if (field == nullptr) {
        return errors::NotFound(name + " not found in misc_float_features!");
      }
      for (size_t j = 0; j < instances.size(); ++j) {
        const Instance &instance = instances[j];
        if (field->is_repeated()) {
          int dim = spec_.misc_float_dims[i];
          int field_size = reflection->FieldSize(instance.line_id(), field);
          if (field_size < dim) {
            LOG_EVERY_N_SEC(ERROR, 60)
                << name << " Dim is smaller than expected " << field_size
                << " v.s. " << dim;
            dim = field_size;
          }
          for (int k = 0; k < dim; ++k) {
            mat(j, k) =
                reflection->GetRepeatedFloat(instance.line_id(), field, k);
          }
        } else {
          mat(j, 0) = reflection->GetFloat(instance.line_id(), field);
        }
      }
    }
    return Status::OK();
  }

  Status ParseInt64Tensors(OpKernelContext *ctx,
                           absl::Span<const Instance> instances,
                           Output *output) {
    const int batch_size = instances.size();
    const auto *descriptor = ::idl::matrix::proto::LineId::GetDescriptor();
    const auto *reflection = ::idl::matrix::proto::LineId::GetReflection();

    for (size_t i = 0; i < spec_.misc_int64_features.size(); ++i) {
      output->tensors.emplace_back();
      Tensor *t = &output->tensors.back();
      TF_RETURN_IF_ERROR(ctx->allocate_temp(
          DT_INT64, GetBatched1DShape(batch_size, spec_.misc_int64_dims[i]),
          t));
      auto mat = t->shaped<int64, 2>({batch_size, spec_.misc_int64_dims[i]});
      // To be safe, we initialize int64 tensors to zero by default.
      mat.setZero();
      const std::string &name = spec_.misc_int64_features[i];
      const auto *field = descriptor->FindFieldByName(name);
      if (field == nullptr) {
        return errors::NotFound(name + " not found in misc_int64_features!");
      }
      if (field->is_repeated()) {
        for (int j = 0; j < batch_size; ++j) {
          int dim = spec_.misc_int64_dims[i];
          const Instance &instance = instances[j];
          const int field_size =
              reflection->FieldSize(instance.line_id(), field);
          if (field_size < dim) {
            LOG_EVERY_N_SEC(ERROR, 60)
                << name << " Dim is smaller than expected " << field_size
                << " v.s. " << dim;
            dim = field_size;
          }
          switch (field->cpp_type()) {
            case FieldDescriptor::CPPTYPE_INT32:
              for (int k = 0; k < dim; ++k) {
                mat(j, k) =
                    reflection->GetRepeatedInt32(instance.line_id(), field, k);
              }
              break;
            case FieldDescriptor::CPPTYPE_UINT32:
              for (int k = 0; k < dim; ++k) {
                mat(j, k) =
                    reflection->GetRepeatedUInt32(instance.line_id(), field, k);
              }
              break;
            case FieldDescriptor::CPPTYPE_INT64:
              for (int k = 0; k < dim; ++k) {
                mat(j, k) =
                    reflection->GetRepeatedInt64(instance.line_id(), field, k);
              }
              break;
            case FieldDescriptor::CPPTYPE_UINT64:
              for (int k = 0; k < dim; ++k) {
                mat(j, k) =
                    reflection->GetRepeatedUInt64(instance.line_id(), field, k);
              }
              break;
            default:
              return errors::InvalidArgument(
                  name,
                  " Data type not match, only int32/int64/float32 supported.");
          }
        }
      } else {
        for (int j = 0; j < batch_size; ++j) {
          const Instance &instance = instances[j];
          switch (field->cpp_type()) {
            case FieldDescriptor::CPPTYPE_INT32:
              mat(j, 0) = reflection->GetInt32(instance.line_id(), field);
              break;
            case FieldDescriptor::CPPTYPE_UINT32:
              mat(j, 0) = reflection->GetUInt32(instance.line_id(), field);
              break;
            case FieldDescriptor::CPPTYPE_INT64:
              mat(j, 0) = reflection->GetInt64(instance.line_id(), field);
              break;
            case FieldDescriptor::CPPTYPE_UINT64:
              mat(j, 0) = reflection->GetUInt64(instance.line_id(), field);
              break;
            default:
              return errors::InvalidArgument(
                  name,
                  " Data type not match, only int32/int64/float32 supported.");
          }
        }
      }
    }
    return Status::OK();
  }

  Status ParseStringTensors(OpKernelContext *ctx,
                            absl::Span<const Instance> instances,
                            Output *output) {
    const int batch_size = instances.size();
    const auto *descriptor = ::idl::matrix::proto::LineId::GetDescriptor();
    const auto *reflection = ::idl::matrix::proto::LineId::GetReflection();
    for (size_t i = 0; i < spec_.misc_string_features.size(); ++i) {
      output->tensors.emplace_back();
      Tensor *t = &output->tensors.back();
      TF_RETURN_IF_ERROR(ctx->allocate_temp(
          DT_STRING, GetBatched1DShape(batch_size, spec_.misc_string_dims[i]),
          t));
      auto mat = t->shaped<tstring, 2>({batch_size, spec_.misc_string_dims[i]});
      const std::string &name = spec_.misc_string_features[i];
      const auto *field = descriptor->FindFieldByName(name);
      if (field == nullptr) {
        return errors::NotFound(name + " not found in misc_string_features!");
      }
      for (size_t j = 0; j < instances.size(); ++j) {
        const Instance &instance = instances[j];
        if (field->is_repeated()) {
          int dim = spec_.misc_string_dims[i];
          int field_size = reflection->FieldSize(instance.line_id(), field);
          if (field_size < dim) {
            LOG_EVERY_N_SEC(ERROR, 60)
                << name << " Dim is smaller than expected " << field_size
                << " v.s. " << dim;
            dim = field_size;
          }
          for (int k = 0; k < dim; ++k) {
            mat(j, k) =
                reflection->GetRepeatedString(instance.line_id(), field, k);
          }
        } else {
          mat(j, 0) = reflection->GetString(instance.line_id(), field);
        }
      }
    }
    return Status::OK();
  }

  TensorShape GetBatched1DShape(int batch_size, int64 dim) {
    if (spec_.collapse_batch_dim) {
      return {dim};
    } else {
      return {batch_size, dim};
    }
  }

  InstanceParserSpec spec_;
  std::unique_ptr<RaggedTensorProcessor> ragged_tensor_processor_;
  std::unique_ptr<UECompress> ue_compress_;
};

InstanceParser::InstanceParser(const InstanceParserConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

InstanceParser::~InstanceParser() {}

Status InstanceParser::Init() { return impl_->Init(); }

Status InstanceParser::Parse(OpKernelContext *ctx,
                             absl::Span<const Instance> instances,
                             Output *output) const {
  return impl_->Parse(ctx, instances, output);
}

}  // namespace monolith_tf
}  // namespace tensorflow
