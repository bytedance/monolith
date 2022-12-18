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

#include <algorithm>

#include "absl/container/flat_hash_map.h"
#include "google/protobuf/descriptor.h"
#include "idl/matrix/proto/proto_parser.pb.h"
#include "monolith/native_training/data/training_instance/cc/parse_instance_lib.h"
#include "monolith/native_training/data/training_instance/cc/pb_variant.h"
#include "monolith/native_training/data/training_instance/cc/reader_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace monolith_tf {
namespace {

using Instance = ::parser::proto::Instance;

Status GetParserConfig(OpKernelConstruction *ctx, InstanceParserConfig *c) {
  TF_RETURN_IF_ERROR(ctx->GetAttr("fidv1_features", &c->fidv1_features));
  TF_RETURN_IF_ERROR(ctx->GetAttr("fidv2_features", &c->fidv2_features));
  TF_RETURN_IF_ERROR(
      ctx->GetAttr("float_feature_dims", &c->float_feature_dims));
  TF_RETURN_IF_ERROR(ctx->GetAttr("float_features", &c->float_features));
  if (c->float_features.size() != c->float_feature_dims.size()) {
    return errors::InvalidArgument(
        "Num of float features and float feature dims do not match");
  }

  TF_RETURN_IF_ERROR(
      ctx->GetAttr("int64_feature_dims", &c->int64_feature_dims));
  TF_RETURN_IF_ERROR(ctx->GetAttr("int64_features", &c->int64_features));
  if (c->int64_features.size() != c->int64_feature_dims.size()) {
    return errors::InvalidArgument(
        "Num of int64 features and int64 feature dims do not match");
  }

  TF_RETURN_IF_ERROR(
      ctx->GetAttr("string_feature_dims", &c->string_feature_dims));
  TF_RETURN_IF_ERROR(ctx->GetAttr("string_features", &c->string_features));
  if (c->string_features.size() != c->string_feature_dims.size()) {
    return errors::InvalidArgument(
        "Num of string features and string feature dims do not match");
  }

  TF_RETURN_IF_ERROR(
      ctx->GetAttr("misc_float_features", &c->misc_float_features));

  TF_RETURN_IF_ERROR(ctx->GetAttr("misc_float_dims", &c->misc_float_dims));

  if (c->misc_float_features.size() != c->misc_float_dims.size()) {
    return errors::InvalidArgument(
        "Num of float features do not match it dims the size of "
        "misc_float_features is ",
        c->misc_float_features.size(),
        ", while the size of misc_float_dims is ", c->misc_float_dims.size());
  }

  TF_RETURN_IF_ERROR(
      ctx->GetAttr("misc_int64_features", &c->misc_int64_features));

  TF_RETURN_IF_ERROR(ctx->GetAttr("misc_int64_dims", &c->misc_int64_dims));

  if (c->misc_int64_features.size() != c->misc_int64_dims.size()) {
    return errors::InvalidArgument(
        "Num of features do not match it dims the size of "
        "misc_features is ",
        c->misc_int64_features.size(), ", while the size of misc_dims is ",
        c->misc_int64_dims.size());
  }

  TF_RETURN_IF_ERROR(
      ctx->GetAttr("misc_string_features", &c->misc_string_features));

  TF_RETURN_IF_ERROR(ctx->GetAttr("misc_string_dims", &c->misc_string_dims));

  if (c->misc_string_features.size() != c->misc_string_dims.size()) {
    return errors::InvalidArgument(
        "Num of features do not match it dims the size of "
        "misc_features is ",
        c->misc_string_features.size(), ", while the size of misc_dims is ",
        c->misc_string_dims.size());
  }

  return Status::OK();
}

bool ParseInstance(const tstring &serialized, Instance *instance) {
  return instance->ParseFromArray(serialized.data(), serialized.size());
}

class ParseInstancesOp : public OpKernel {
 public:
  explicit ParseInstancesOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    InstanceParserConfig config;
    OP_REQUIRES_OK(ctx, GetParserConfig(ctx, &config));
    config.collapse_batch_dim = false;

    parser_ = std::make_unique<InstanceParser>(config);
    OP_REQUIRES_OK(ctx, parser_->Init());
  }

  void Compute(OpKernelContext *ctx) override {
    // Grab the input tensor
    const Tensor *serialized;
    OP_REQUIRES_OK(ctx, ctx->input("serialized", &serialized));
    TTypes<tstring>::ConstVec serialized_protos = serialized->vec<tstring>();
    const int batch_size = serialized_protos.dimension(0);
    std::vector<Instance> instances(batch_size);

    for (int i = 0; i < batch_size; ++i) {
      OP_REQUIRES(ctx, ParseInstance(serialized_protos(i), &instances[i]),
                  errors::FailedPrecondition("Failed to parse the Instance."));
    }

    InstanceParser::Output output;
    OP_REQUIRES_OK(ctx, parser_->Parse(ctx, instances, &output));
    for (int i = 0; i < static_cast<int>(output.tensors.size()); ++i) {
      ctx->set_output(i, output.tensors[i]);
    }
  }

 private:
  std::unique_ptr<InstanceParser> parser_;
};

// This class is mainly for testing parser.
// Do not use in the model code directly.
class RawParseInstanceOp : public OpKernel {
 public:
  explicit RawParseInstanceOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    InstanceParserConfig config;
    OP_REQUIRES_OK(ctx, GetParserConfig(ctx, &config));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("collapse_batch_dim", &config.collapse_batch_dim));
    std::string fid_output_type;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fid_output_type", &fid_output_type));
    absl::flat_hash_map<std::string, InstanceParserConfig::FidOutputType>
        str_to_enum = {
            {"REGULAR", InstanceParserConfig::REGULAR},
            {"CONCAT", InstanceParserConfig::CONCAT},
        };
    config.fid_output_type = str_to_enum.at(fid_output_type);
    parser_ = std::make_unique<InstanceParser>(config);
    OP_REQUIRES_OK(ctx, parser_->Init());
  }

  void Compute(OpKernelContext *ctx) override {
    // Grab the input tensor
    const Tensor *serialized;
    OP_REQUIRES_OK(ctx, ctx->input("serialized", &serialized));

    auto serialized_flat = serialized->flat<tstring>();
    std::vector<Instance> instances(serialized_flat.size());
    for (size_t i = 0; i < instances.size(); ++i) {
      OP_REQUIRES(ctx, ParseInstance(serialized_flat(i), &instances[i]),
                  errors::FailedPrecondition("Failed to parse the Instance."));
    }

    InstanceParser::Output output;
    OP_REQUIRES_OK(ctx, parser_->Parse(ctx, instances, &output));
    OpOutputList l;
    ctx->output_list("tensors", &l);
    for (size_t i = 0; i < output.tensors.size(); ++i) {
      l.set(i, output.tensors[i]);
    }
  }

 private:
  std::unique_ptr<InstanceParser> parser_;
};

REGISTER_KERNEL_BUILDER(Name("MonolithParseInstances").Device(DEVICE_CPU),
                        ParseInstancesOp);

REGISTER_KERNEL_BUILDER(Name("MonolithRawParseInstance").Device(DEVICE_CPU),
                        RawParseInstanceOp);
}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow
