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

#include "idl/matrix/proto/example.pb.h"
#include "idl/matrix/proto/proto_parser.pb.h"
#include "monolith/native_training/data/kernels/internal/datasource_utils.h"
#include "monolith/native_training/data/training_instance/cc/data_reader.h"
#include "monolith/native_training/data/training_instance/cc/pb_variant.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/coding.h"

namespace tensorflow {
namespace monolith_tf {
namespace {

using Example = ::monolith::io::proto::Example;
using ExampleBatch = ::monolith::io::proto::ExampleBatch;
using Instance = ::parser::proto::Instance;

class ReadHelper {
 public:
  explicit ReadHelper(DataFormatOptions options, bool has_header)
      : options_(options), has_header_(has_header) {}

  Status GetData(absl::string_view in, uint8_t* pb_type,
                 uint32_t* data_source_key, absl::string_view* out) {
    if (has_header_) {
      ZeroCopyStringViewStreamReader r(options_, in);
      TF_RETURN_IF_ERROR(r.ReadPBBytes(pb_type, data_source_key, out));
      return Status::OK();
    }
    *pb_type = 0;
    *data_source_key = 0;
    *out = in;
    return Status::OK();
  }

 private:
  DataFormatOptions options_;
  bool has_header_;
};

class StringToVariantOp : public OpKernel {
 public:
  using OpKernel::OpKernel;
  using ConstFlatSplits = typename TTypes<int64>::ConstFlat;

  explicit StringToVariantOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("input_type", &variant_type_));

    std::unordered_set<std::string> variant_type_set_ = {
        "instance", "example", "examplebatch", "example_batch"};
    OP_REQUIRES(
        ctx, variant_type_set_.count(variant_type_) != 0,
        errors::InvalidArgument("variant_type can only be instance, example "
                                "and examplebatch/example_batch"));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("has_header", &has_header_));
    if (has_header_) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("has_sort_id", &options_.has_sort_id));
      OP_REQUIRES_OK(
          ctx, ctx->GetAttr("lagrangex_header", &options_.lagrangex_header));
      OP_REQUIRES_OK(
          ctx, ctx->GetAttr("kafka_dump_prefix", &options_.kafka_dump_prefix));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("kafka_dump", &options_.kafka_dump));
    }

    std::vector<int64> chnid_list;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("chnids", &chnid_list));
    std::vector<std::string> datasource_list;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("datasources", &datasource_list));
    CHECK_EQ(chnid_list.size(), datasource_list.size());

    if (!chnid_list.empty()) {
      int i = 0;
      for (const std::string& sv : datasource_list) {
        uint32 code = internal::java_hash_code(sv);
        code = code << 8;
        chnid_to_code_.emplace(chnid_list.at(i), code);
        LOG(INFO) << "chnid: " << chnid_list.at(i) << ", code: " << code;
        i++;
      }
    }

    std::string default_datasource;
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("default_datasource", &default_datasource));
    uint32 default_code = internal::java_hash_code(default_datasource);
    default_code_ = (default_code << 8);
    LOG(INFO) << "default_code: " << default_code_;
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<tstring>();

    // Create an output tensor
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<Variant>();

    uint8_t pb_type;
    uint32_t data_source_key;
    ReadHelper reader(options_, has_header_);
    for (size_t i = 0; i < input.size(); ++i) {
      const tstring& buf = input(i);

      absl::string_view res;
      OP_REQUIRES_OK(context,
                     reader.GetData(buf, &pb_type, &data_source_key, &res));
      if (variant_type_ == "instance") {
        Instance pb;
        if (res.size() > 0) {
          CHECK(pb.ParseFromArray(res.data(), res.size()));
          UpdateDatasourceKey(pb.line_id().chnid(), &data_source_key);
          pb.set_data_source_key(data_source_key);
        }
        output_flat(i) = std::move(pb);
      } else if (variant_type_ == "example") {
        Example pb;
        if (res.size() > 0) {
          CHECK(pb.ParseFromArray(res.data(), res.size()));
          UpdateDatasourceKey(pb.line_id().chnid(), &data_source_key);
          pb.set_data_source_key(data_source_key);
        }
        output_flat(i) = std::move(pb);
      } else {
        ExampleBatch pb;
        if (res.size() > 0) {
          CHECK(pb.ParseFromArray(res.data(), res.size()));
          pb.set_data_source_key(data_source_key);
        }
        output_flat(i) = std::move(pb);
      }
    }
  }

 private:
  std::string variant_type_;
  bool has_header_ = false;
  DataFormatOptions options_;
  std::unordered_map<int64, uint32> chnid_to_code_;
  uint32 default_code_;

  void UpdateDatasourceKey(const int64& chnid, uint32_t* data_source_key) {
    if (has_header_ && options_.lagrangex_header) {
      return;
    } else if (!chnid_to_code_.empty()) {
      if (chnid_to_code_.count(chnid) != 0) {
        *data_source_key = chnid_to_code_[chnid];
      } else {
        *data_source_key = default_code_;
      }
    } else {
      *data_source_key = default_code_;
    }
  }
};

class VariantToZerosOp : public OpKernel {
 public:
  explicit VariantToZerosOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    // Create an output tensor
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<int64>();
    output_flat.setZero();
  }
};

class HasVariantOp : public OpKernel {
 public:
  explicit HasVariantOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("variant_type", &variant_type_));

    std::unordered_set<std::string> variant_type_set_ = {
        "instance", "example", "examplebatch", "example_batch"};
    OP_REQUIRES(
        ctx, variant_type_set_.count(variant_type_) != 0,
        errors::InvalidArgument("variant_type can only be instance, example "
                                "and examplebatch/example_batch"));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    // Create an output tensor
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_scalar = output_tensor->scalar<bool>();

    int byte_size = 0;
    if (variant_type_ == "instance") {
      const auto* instance = input_tensor.scalar<Variant>()().get<Instance>();
      byte_size = instance->ByteSize();
    } else if (variant_type_ == "example") {
      const auto* example = input_tensor.scalar<Variant>()().get<Example>();
      byte_size = example->ByteSize();
    } else {
      const auto* example_batch =
          input_tensor.scalar<Variant>()().get<ExampleBatch>();
      byte_size = example_batch->ByteSize();
    }

    output_scalar() = byte_size > 0;
  }

 private:
  std::string variant_type_;
};

REGISTER_KERNEL_BUILDER(Name("StringToVariant").Device(DEVICE_CPU),
                        StringToVariantOp);
REGISTER_KERNEL_BUILDER(Name("VariantToZeros").Device(DEVICE_CPU),
                        VariantToZerosOp);
REGISTER_KERNEL_BUILDER(Name("HasVariant").Device(DEVICE_CPU), HasVariantOp);
}  // namespace

}  // namespace monolith_tf
}  // namespace tensorflow
