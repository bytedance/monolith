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
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "google/protobuf/descriptor.h"
#include "idl/matrix/proto/example.pb.h"
#include "idl/matrix/proto/proto_parser.pb.h"
#include "monolith/native_training/data/kernels/feature_name_mapper_tf_bridge.h"
#include "monolith/native_training/data/kernels/internal/label_utils.h"
#include "monolith/native_training/data/kernels/parse_example_lib.h"
#include "monolith/native_training/data/training_instance/cc/data_reader.h"
#include "monolith/native_training/data/training_instance/cc/parse_instance_lib.h"
#include "monolith/native_training/data/training_instance/cc/pb_variant.h"
#include "monolith/native_training/data/training_instance/cc/reader_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/profiler/lib/traceme.h"

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "monolith/native_training/runtime/common/metrics.h"
namespace tensorflow {
namespace monolith_tf {
namespace {

using Instance = ::parser::proto::Instance;
using LineId = ::idl::matrix::proto::LineId;
using EFeature = ::monolith::io::proto::Feature;
using Example = ::monolith::io::proto::Example;
using ExampleBatch = ::monolith::io::proto::ExampleBatch;
using FieldDescriptor = ::google::protobuf::FieldDescriptor;
using ExampleParser = ::tensorflow::monolith_tf::ExampleParser;
using ExampleBatchParser = ::tensorflow::monolith_tf::ExampleBatchParser;
using ExampleBatchListParser =
    ::tensorflow::monolith_tf::ExampleBatchListParser;
using NamedFeatureList = ::monolith::io::proto::NamedFeatureList;
using FeatureConfigs = ::monolith::io::proto::FeatureConfigs;

class DataCounter {
 public:
  explicit DataCounter(std::string op, bool emit_mini_batch,
                       int64_t emit_every_n_batch = 2000)
      : op_(std::move(op)),
        emit_mini_batch_(emit_mini_batch),
        mini_batch_num_(0),
        emit_every_n_batch_(emit_every_n_batch),
        last_batch_size_(0) {
    CHECK_GT(emit_every_n_batch_, 0);
  }

  ~DataCounter() {
    LOG(INFO) << absl::StrFormat(
        "Finally metrics_emit(counter) [data_consume_num] op=%s, "
        "batch_size=%d, total_mini_batch_num=%llu",
        op_, last_batch_size_, mini_batch_num_);
    int64_t remainder = mini_batch_num_ % emit_every_n_batch_;
    if (remainder) {
      monolith::GetMetrics()->emit_counter("data_consume_num",
                                           last_batch_size_ * remainder,
                                           absl::StrFormat("op=%s", op_));
      if (emit_mini_batch_) {
        monolith::GetMetrics()->emit_counter("mini_batch_num", remainder,
                                             absl::StrFormat("op=%s", op_));
      }
    }
  }

  void EmitDataConsumeNumCounter(int batch_size) {
    mini_batch_num_ += 1;
    last_batch_size_ = batch_size;
    LOG_EVERY_N_SEC(INFO, 300) << absl::StrFormat(
        "metrics_emit(counter) [data_consume_num] op=%s, "
        "batch_size=%d, total_mini_batch_num=%llu",
        op_, batch_size, mini_batch_num_);
    if (mini_batch_num_ % emit_every_n_batch_ == 0) {
      monolith::GetMetrics()->emit_counter("data_consume_num",
                                           batch_size * emit_every_n_batch_,
                                           absl::StrFormat("op=%s", op_));
      if (emit_mini_batch_) {
        monolith::GetMetrics()->emit_counter("mini_batch_num",
                                             emit_every_n_batch_,
                                             absl::StrFormat("op=%s", op_));
      }
    }
  }

 private:
  std::string op_;
  bool emit_mini_batch_;
  int64_t mini_batch_num_;
  int64_t emit_every_n_batch_;
  int64_t last_batch_size_;
};

Status GetParserConfig(OpKernelConstruction *ctx, InstanceParserConfig *c,
                       std::vector<int> *index) {
  TF_RETURN_IF_ERROR(ctx, ctx->GetAttr("fidv1_features", &(c->fidv1_features)));
  TF_RETURN_IF_ERROR(ctx, ctx->GetAttr("fidv2_features", &(c->fidv2_features)));

  std::vector<std::string> names;
  std::vector<int> shapes;
  std::vector<DataType> dtypes;
  std::vector<std::string> extra_names;
  std::unordered_set<std::string> misc({"label", "instance_weight"});
  TF_RETURN_IF_ERROR(ctx, ctx->GetAttr("names", &names));
  TF_RETURN_IF_ERROR(ctx, ctx->GetAttr("shapes", &shapes));
  TF_RETURN_IF_ERROR(ctx, ctx->GetAttr("dtypes", &dtypes));
  TF_RETURN_IF_ERROR(ctx, ctx->GetAttr("extra_names", &extra_names));

  int ragged_size = c->fidv1_features.size() + c->fidv2_features.size();
  if (names.size() != shapes.size() ||
      shapes.size() + ragged_size != dtypes.size()) {
    return errors::InvalidArgument(
        "Num of names, shapes and dtypes do not match");
  }

  for (size_t i = 0; i < names.size(); ++i) {
    if (i < ragged_size) {
      continue;  // skip fidv1/fidv2
    }
    std::string name = names[i];
    int dim = shapes[i];
    DataType dtype = dtypes[i];
    auto eit = std::find(extra_names.begin(), extra_names.end(), name);
    if (eit != extra_names.end() || misc.find(name) != misc.end()) {  // extra
      switch (dtype) {
        case DataType::DT_INT64:
          c->misc_int64_features.push_back(name);
          c->misc_int64_dims.push_back(dim);
          break;
        case DataType::DT_FLOAT:
          c->misc_float_features.push_back(name);
          c->misc_float_dims.push_back(dim);
          break;
        case DataType::DT_STRING:
          c->misc_string_features.push_back(name);
          c->misc_string_dims.push_back(dim);
          break;
        default:
          return errors::InvalidArgument("Unsupported data type!");
      }
    } else {  // dense
      switch (dtype) {
        case DataType::DT_INT64:
          c->int64_features.push_back(name);
          c->int64_feature_dims.push_back(dim);
          break;
        case DataType::DT_FLOAT:
          c->float_features.push_back(name);
          c->float_feature_dims.push_back(dim);
          break;
        case DataType::DT_STRING:
          c->string_features.push_back(name);
          c->string_feature_dims.push_back(dim);
          break;
        default:
          return errors::InvalidArgument("Unsupported data type!");
      }
    }
  }

  std::vector<std::string> new_names;
  new_names.reserve(names.size());
  new_names.insert(new_names.end(), names.begin(), names.begin() + ragged_size);
  new_names.insert(new_names.end(), c->float_features.begin(),
                   c->float_features.end());
  new_names.insert(new_names.end(), c->int64_features.begin(),
                   c->int64_features.end());
  new_names.insert(new_names.end(), c->string_features.begin(),
                   c->string_features.end());
  new_names.insert(new_names.end(), c->misc_float_features.begin(),
                   c->misc_float_features.end());
  new_names.insert(new_names.end(), c->misc_int64_features.begin(),
                   c->misc_int64_features.end());
  new_names.insert(new_names.end(), c->misc_string_features.begin(),
                   c->misc_string_features.end());

  index->reserve(dtypes.size());
  std::unordered_map<std::string, int> name_to_idx;
  for (size_t i = 0; i < names.size(); ++i) {
    name_to_idx.emplace(names[i], i);
  }

  for (size_t i = 0; i < new_names.size(); ++i) {
    int idx = name_to_idx[new_names[i]];
    if (i < ragged_size) {
      (*index)[i] = idx;
      (*index)[i + ragged_size] = idx + names.size();
    } else {
      (*index)[i + ragged_size] = idx;
    }
  }

  return Status::OK();
}

class ParseStringInstancesOp : public OpKernel {
 public:
  explicit ParseStringInstancesOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    InstanceParserConfig config;
    OP_REQUIRES_OK(ctx, GetParserConfig(ctx, &config, &index_));
    config.collapse_batch_dim = false;

    parser_ = std::make_unique<InstanceParser>(config);
    OP_REQUIRES_OK(ctx, parser_->Init());
    counter_ = std::make_unique<DataCounter>("ParseStringInstancesOp", true);
  }

  void Compute(OpKernelContext *ctx) override {
    // Grab the input tensor
    const Tensor *pb_input;
    OP_REQUIRES_OK(ctx, ctx->input("pb_input", &pb_input));
    const auto &serialized_flat = pb_input->flat<tstring>();
    const int batch_size = serialized_flat.size();
    std::vector<Instance> instances(batch_size);  // has alocated memory

    for (int i = 0; i < batch_size; ++i) {
      const auto &serialized = serialized_flat(i);
      OP_REQUIRES(ctx, instances[i].ParseFromArray(serialized.data(),
                                                   serialized.size()),
                  errors::FailedPrecondition("Failed to parse the Instance."));
    }

    InstanceParser::Output output;
    OP_REQUIRES_OK(ctx, parser_->Parse(ctx, instances, &output));

    OpOutputList out_list;
    OP_REQUIRES_OK(ctx, ctx->output_list("tensors", &out_list));

    OP_REQUIRES(ctx, output.tensors.size() == out_list.size(),
                errors::FailedPrecondition("output tensor size doesn't match"));
    for (size_t i = 0; i < output.tensors.size(); ++i) {
      out_list.set(index_[i], output.tensors[i]);
    }

    counter_->EmitDataConsumeNumCounter(batch_size);
  }

 protected:
  const std::vector<int> &GetIndex() const { return index_; }
  InstanceParser *GetParse() const { return parser_.get(); }
  DataCounter *GetCounter() const { return counter_.get(); }

 private:
  std::vector<int> index_;
  std::unique_ptr<InstanceParser> parser_;
  std::unique_ptr<DataCounter> counter_;
};

class ParseStringInstancesV2Op : public ParseStringInstancesOp {
 public:
  explicit ParseStringInstancesV2Op(OpKernelConstruction *ctx)
      : ParseStringInstancesOp(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    // Grab the input tensor
    const Tensor *pb_input;
    OP_REQUIRES_OK(ctx, ctx->input("pb_input", &pb_input));
    const auto &serialized_flat = pb_input->flat<tstring>();
    int batch_size = serialized_flat.size();
    std::vector<Instance> instances(batch_size);  // has alocated memory

    for (int i = 0; i < batch_size; ++i) {
      const auto &serialized = serialized_flat(i);
      OP_REQUIRES(ctx, instances[i].ParseFromArray(serialized.data(),
                                                   serialized.size()),
                  errors::FailedPrecondition("Failed to parse the Instance."));
    }

    InstanceParser::Output output;
    OP_REQUIRES_OK(ctx, ParseStringInstancesOp::GetParse()->Parse(
                            ctx, instances, &output));

    OpOutputList out_list;
    OP_REQUIRES_OK(ctx, ctx->output_list("tensors", &out_list));

    OP_REQUIRES(ctx, output.tensors.size() == out_list.size(),
                errors::FailedPrecondition("output tensor size doesn't match"));
    for (size_t i = 0; i < output.tensors.size(); ++i) {
      out_list.set(ParseStringInstancesOp::GetIndex()[i], output.tensors[i]);
    }

    Tensor *instance_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("sparse_features", TensorShape({
                                                                    batch_size,
                                                                }),
                                             &instance_tensor));
    for (size_t i = 0; i < batch_size; ++i) {
      instance_tensor->flat<Variant>()(i) = std::move(instances[i]);
    }
    ParseStringInstancesOp::GetCounter()->EmitDataConsumeNumCounter(batch_size);
  }
};

class ParseVariantInstancesOp : public OpKernel {
 public:
  explicit ParseVariantInstancesOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    InstanceParserConfig config;
    OP_REQUIRES_OK(ctx, GetParserConfig(ctx, &config, &index_));
    config.collapse_batch_dim = false;

    parser_ = std::make_unique<InstanceParser>(config);
    OP_REQUIRES_OK(ctx, parser_->Init());
    counter_ = std::make_unique<DataCounter>("ParseVariantInstancesOp", true);
  }

  void Compute(OpKernelContext *ctx) override {
    // Grab the input tensor
    const Tensor *pb_input;
    OP_REQUIRES_OK(ctx, ctx->input("pb_input", &pb_input));

    TTypes<Variant>::ConstVec pb_variant_tensor = pb_input->vec<Variant>();
    const int batch_size = pb_variant_tensor.dimension(0);
    std::vector<Instance> instances;  // not allocated memory
    instances.reserve(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      instances.push_back(*pb_variant_tensor(i).get<Instance>());
    }

    InstanceParser::Output output;
    OP_REQUIRES_OK(ctx, parser_->Parse(ctx, instances, &output));

    OpOutputList out_list;
    OP_REQUIRES_OK(ctx, ctx->output_list("tensors", &out_list));
    for (size_t i = 0; i < output.tensors.size(); ++i) {
      out_list.set(index_[i], output.tensors[i]);
    }

    counter_->EmitDataConsumeNumCounter(batch_size);
  }

 private:
  std::vector<int> index_;
  std::unique_ptr<InstanceParser> parser_;
  std::unique_ptr<DataCounter> counter_;
};

class ParseVariantInstancesV2Op : public ParseVariantInstancesOp {
 public:
  explicit ParseVariantInstancesV2Op(OpKernelConstruction *ctx)
      : ParseVariantInstancesOp(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    ParseVariantInstancesOp::Compute(ctx);
    OP_REQUIRES_OK(ctx, ctx->set_output("sparse_features", ctx->input(0)));
  }
};

class ParseStringExamplesOp : public OpKernel {
 public:
  explicit ParseStringExamplesOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    std::vector<std::string> names;
    std::vector<int> shapes;
    std::vector<DataType> dtypes;
    std::vector<std::string> extra_names;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("names", &names));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shapes", &shapes));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtypes", &dtypes));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("extra_names", &extra_names));

    auto creator = [this](FeatureNameMapperTfBridge **out_mapper) {
      TF_RETURN_IF_ERROR(FeatureNameMapperTfBridge::New(out_mapper));
      return Status::OK();
    };
    ResourceMgr *resource_mgr = ctx->resource_manager();
    OP_REQUIRES_OK(ctx,
                   resource_mgr->LookupOrCreate<FeatureNameMapperTfBridge>(
                       resource_mgr->default_container(),
                       FeatureNameMapperTfBridge::kName, &mapper_, creator));
    parser_ = std::make_unique<ExampleParser>(names, shapes, dtypes,
                                              extra_names, DataType::DT_STRING,
                                              mapper_->GetFeatureNameMapper());
    counter_ = std::make_unique<DataCounter>("ParseStringExamplesOp", true);
  }

  ~ParseStringExamplesOp() override { mapper_->Unref(); }

  void Compute(OpKernelContext *ctx) override {
    // Grab the input tensor
    const Tensor *pb_input;
    OP_REQUIRES_OK(ctx, ctx->input("pb_input", &pb_input));
    const auto &serialized_flat = pb_input->flat<tstring>();
    int batch_size = serialized_flat.size();
    std::vector<Example> examples(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      const auto &serialized = serialized_flat(i);
      OP_REQUIRES(
          ctx, examples[i].ParseFromArray(serialized.data(), serialized.size()),
          errors::FailedPrecondition("Failed to parse the Example."));
      ExtendExample(&examples[i]);
    }

    OpOutputList out_list;
    OP_REQUIRES_OK(ctx, ctx->output_list("tensors", &out_list));
    std::vector<const Example *> example_ptrs;
    example_ptrs.reserve(examples.size());
    for (const auto &example : examples) {
      example_ptrs.push_back(&example);
    }

    parser_->Parse(ctx, example_ptrs, &out_list);
    counter_->EmitDataConsumeNumCounter(batch_size);
  }

 private:
  FeatureNameMapperTfBridge *mapper_ = nullptr;

 protected:
  ExampleParser *GetParse() const { return parser_.get(); }
  DataCounter *GetCounter() const { return counter_.get(); }
  FeatureNameMapper *GetFeatureNameMapper() const {
    return mapper_->GetFeatureNameMapper();
  }

 private:
  std::unique_ptr<ExampleParser> parser_;
  std::unique_ptr<DataCounter> counter_;
};

class ParseStringExamplesV2Op : public ParseStringExamplesOp {
 public:
  explicit ParseStringExamplesV2Op(OpKernelConstruction *ctx)
      : ParseStringExamplesOp(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    // Grab the input tensor
    const Tensor *pb_input;
    OP_REQUIRES_OK(ctx, ctx->input("pb_input", &pb_input));
    const auto &serialized_flat = pb_input->flat<tstring>();
    int batch_size = serialized_flat.size();

    Tensor *example_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("sparse_features", TensorShape({
                                                                    batch_size,
                                                                }),
                                             &example_tensor));
    google::protobuf::Arena arena;
    std::vector<const Example *> example_ptrs;
    example_ptrs.reserve(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
      const auto &serialized = serialized_flat(i);
      auto *example_ptr =
          google::protobuf::Arena::CreateMessage<Example>(&arena);
      example_tensor->flat<Variant>()(i) = std::move(*example_ptr);
      auto example = example_tensor->flat<Variant>()(i).get<Example>();
      OP_REQUIRES(ctx,
                  example->ParseFromArray(serialized.data(), serialized.size()),
                  errors::FailedPrecondition("Failed to parse the Example."));
      ExtendExample(example);
      example_ptrs.push_back(example);
    }

    OpOutputList out_list;
    OP_REQUIRES_OK(ctx, ctx->output_list("tensors", &out_list));
    ParseStringExamplesOp::GetParse()->Parse(ctx, example_ptrs, &out_list);
    ParseStringExamplesOp::GetCounter()->EmitDataConsumeNumCounter(batch_size);
  }
};

class ParseVariantExamplesOp : public OpKernel {
 public:
  explicit ParseVariantExamplesOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    std::vector<std::string> names;
    std::vector<int> shapes;
    std::vector<DataType> dtypes;
    std::vector<std::string> extra_names;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("names", &names));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shapes", &shapes));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtypes", &dtypes));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("extra_names", &extra_names));

    auto creator = [this](FeatureNameMapperTfBridge **out_mapper) {
      TF_RETURN_IF_ERROR(FeatureNameMapperTfBridge::New(out_mapper));
      return Status::OK();
    };
    ResourceMgr *resource_mgr = ctx->resource_manager();
    OP_REQUIRES_OK(ctx,
                   resource_mgr->LookupOrCreate<FeatureNameMapperTfBridge>(
                       resource_mgr->default_container(),
                       FeatureNameMapperTfBridge::kName, &mapper_, creator));
    parser_ = std::make_unique<ExampleParser>(names, shapes, dtypes,
                                              extra_names, DataType::DT_VARIANT,
                                              mapper_->GetFeatureNameMapper());
    counter_ = std::make_unique<DataCounter>("ParseVariantExamplesOp", true);
  }

  ~ParseVariantExamplesOp() override { mapper_->Unref(); }

  void Compute(OpKernelContext *ctx) override {
    const Tensor *pb_input;
    OP_REQUIRES_OK(ctx, ctx->input("pb_input", &pb_input));
    const auto &pb_variant_tensor = pb_input->vec<Variant>();
    int batch_size = pb_variant_tensor.dimension(0);
    std::vector<const Example *> examples;
    examples.reserve(batch_size);

    for (int i = 0; i < batch_size; ++i) {
      const auto *example = pb_variant_tensor(i).get<Example>();
      CHECK_NOTNULL(example);
      examples.push_back(example);
    }
    OpOutputList out_list;
    OP_REQUIRES_OK(ctx, ctx->output_list("tensors", &out_list));
    {
      profiler::TraceMe activity([]() { return "Parse"; });
      parser_->Parse(ctx, examples, &out_list);
    }
    {
      profiler::TraceMe activity([]() { return "EmitDataConsumeNumCounter"; });
      counter_->EmitDataConsumeNumCounter(batch_size);
    }
  }

 private:
  FeatureNameMapperTfBridge *mapper_ = nullptr;
  std::unique_ptr<ExampleParser> parser_;
  std::unique_ptr<DataCounter> counter_;
};

class ParseVariantExamplesV2Op : public ParseVariantExamplesOp {
 public:
  explicit ParseVariantExamplesV2Op(OpKernelConstruction *ctx)
      : ParseVariantExamplesOp(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    ParseVariantExamplesOp::Compute(ctx);
    OP_REQUIRES_OK(ctx, ctx->set_output("sparse_features", ctx->input(0)));
  }
};

class ParseStringExampleBatchOp : public OpKernel {
 public:
  explicit ParseStringExampleBatchOp(OpKernelConstruction *ctx)
      : OpKernel(ctx) {
    std::vector<std::string> names;
    std::vector<int> shapes;
    std::vector<DataType> dtypes;
    std::vector<std::string> extra_names;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("names", &names));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shapes", &shapes));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtypes", &dtypes));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("extra_names", &extra_names));

    parser_ = std::make_unique<ExampleBatchParser>(
        names, shapes, dtypes, extra_names, DataType::DT_STRING);
    counter_ =
        std::make_unique<DataCounter>("ParseStringExampleBatchOp", false);
  }

  void Compute(OpKernelContext *ctx) override {
    // Grab the input tensor
    const Tensor *pb_input;
    OP_REQUIRES_OK(ctx, ctx->input("pb_input", &pb_input));
    const auto &serialized = pb_input->flat<tstring>()(0);
    google::protobuf::Arena arena;
    auto *example_batch =
        google::protobuf::Arena::CreateMessage<ExampleBatch>(&arena);
    OP_REQUIRES(ctx, example_batch->ParseFromArray(serialized.data(),
                                                   serialized.size()),
                errors::FailedPrecondition("Failed to parse the Instance."));

    OpOutputList out_list;
    OP_REQUIRES_OK(ctx, ctx->output_list("tensors", &out_list));
    parser_->Parse(ctx, *example_batch, &out_list);

    counter_->EmitDataConsumeNumCounter(example_batch->batch_size());
  }

 protected:
  ExampleBatchParser *GetParse() const { return parser_.get(); }
  DataCounter *GetCounter() const { return counter_.get(); }

 private:
  std::unique_ptr<ExampleBatchParser> parser_;
  std::unique_ptr<DataCounter> counter_;
};

class ParseStringExampleBatchV2Op : public ParseStringExampleBatchOp {
 public:
  explicit ParseStringExampleBatchV2Op(OpKernelConstruction *ctx)
      : ParseStringExampleBatchOp(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    // Grab the input tensor
    const Tensor *pb_input;
    OP_REQUIRES_OK(ctx, ctx->input("pb_input", &pb_input));
    const auto &serialized = pb_input->flat<tstring>()(0);
    google::protobuf::Arena arena;
    auto *example_batch_ptr =
        google::protobuf::Arena::CreateMessage<ExampleBatch>(&arena);
    Tensor *example_batch_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("sparse_features", TensorShape({
                                                                    1,
                                                                }),
                                             &example_batch_tensor));
    example_batch_tensor->scalar<Variant>()() = std::move(*example_batch_ptr);
    auto example_batch =
        example_batch_tensor->scalar<Variant>()().get<ExampleBatch>();
    OP_REQUIRES(ctx, example_batch->ParseFromArray(serialized.data(),
                                                   serialized.size()),
                errors::FailedPrecondition("Failed to parse the Instance."));

    OpOutputList out_list;
    OP_REQUIRES_OK(ctx, ctx->output_list("tensors", &out_list));
    ParseStringExampleBatchOp::GetParse()->Parse(ctx, *example_batch,
                                                 &out_list);

    ParseStringExampleBatchOp::GetCounter()->EmitDataConsumeNumCounter(
        example_batch->batch_size());
  }
};

class ParseVariantExampleBatchOp : public OpKernel {
 public:
  explicit ParseVariantExampleBatchOp(OpKernelConstruction *ctx)
      : OpKernel(ctx) {
    std::vector<std::string> names;
    std::vector<int> shapes;
    std::vector<DataType> dtypes;
    std::vector<std::string> extra_names;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("names", &names));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shapes", &shapes));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtypes", &dtypes));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("extra_names", &extra_names));

    parser_ = std::make_unique<ExampleBatchParser>(
        names, shapes, dtypes, extra_names, DataType::DT_VARIANT);
    counter_ =
        std::make_unique<DataCounter>("ParseVariantExampleBatchOp", false);
  }

  void Compute(OpKernelContext *ctx) override {
    // Grab the input tensor
    const Tensor *pb_input;
    OP_REQUIRES_OK(ctx, ctx->input("pb_input", &pb_input));
    const auto &variant = pb_input->flat<Variant>()(0);
    const ExampleBatch *example_batch = variant.get<ExampleBatch>();

    OpOutputList out_list;
    OP_REQUIRES_OK(ctx, ctx->output_list("tensors", &out_list));
    parser_->Parse(ctx, *example_batch, &out_list);

    counter_->EmitDataConsumeNumCounter(example_batch->batch_size());
  }

 private:
  std::unique_ptr<ExampleBatchParser> parser_;
  std::unique_ptr<DataCounter> counter_;
};

class ParseVariantExampleBatchV2Op : public ParseVariantExampleBatchOp {
 public:
  explicit ParseVariantExampleBatchV2Op(OpKernelConstruction *ctx)
      : ParseVariantExampleBatchOp(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    ParseVariantExampleBatchOp::Compute(ctx);
    OP_REQUIRES_OK(ctx, ctx->set_output("sparse_features", ctx->input(0)));
  }
};

class ParseVariantExampleBatchListOp : public OpKernel {
 public:
  explicit ParseVariantExampleBatchListOp(OpKernelConstruction *ctx)
      : OpKernel(ctx) {
    std::string label_config;
    std::vector<std::string> names;
    std::vector<int> shapes;
    std::vector<DataType> dtypes;
    std::vector<std::string> extra_names;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("label_config", &label_config));
    internal::ParseTaskConfig(label_config, &label_config_);

    OP_REQUIRES_OK(ctx, ctx->GetAttr("names", &names));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shapes", &shapes));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtypes", &dtypes));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("extra_names", &extra_names));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("positive_label", &positive_label_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("negative_label", &negative_label_));

    parser_ = std::make_unique<ExampleBatchListParser>(
        names, shapes, dtypes, extra_names, DataType::DT_VARIANT);
    counter_ =
        std::make_unique<DataCounter>("ParseVariantExampleBatchListOp", false);
  }

  void Compute(OpKernelContext *ctx) override {
    // Grab the input tensor
    OpInputList inputs;
    OP_REQUIRES_OK(ctx, ctx->input_list("inputs", &inputs));
    ExampleBatch example_batch;
    int batch_size = 0;
    for (auto iter = inputs.begin(); iter != inputs.end(); ++iter) {
      const ExampleBatch *sub_eb =
          iter->scalar<Variant>()().get<ExampleBatch>();
      batch_size += sub_eb->batch_size();
      example_batch.MergeFrom(*sub_eb);
    }

    OpOutputList out_list;
    OP_REQUIRES_OK(ctx, ctx->output_list("tensors", &out_list));
    parser_->Parse(ctx, example_batch, label_config_, positive_label_,
                   negative_label_, &out_list);

    counter_->EmitDataConsumeNumCounter(batch_size);
  }

 private:
  std::unique_ptr<ExampleBatchListParser> parser_;
  std::unique_ptr<DataCounter> counter_;
  std::vector<internal::TaskConfig> label_config_;
  float positive_label_ = 1.0f, negative_label_ = 0.0f;
};

REGISTER_KERNEL_BUILDER(
    Name("ParseInstances").Device(DEVICE_CPU).TypeConstraint<tstring>("T"),
    ParseStringInstancesOp);

REGISTER_KERNEL_BUILDER(
    Name("ParseInstances").Device(DEVICE_CPU).TypeConstraint<Variant>("T"),
    ParseVariantInstancesOp);

REGISTER_KERNEL_BUILDER(
    Name("ParseInstancesV2").Device(DEVICE_CPU).TypeConstraint<tstring>("T"),
    ParseStringInstancesV2Op);

REGISTER_KERNEL_BUILDER(
    Name("ParseInstancesV2").Device(DEVICE_CPU).TypeConstraint<Variant>("T"),
    ParseVariantInstancesV2Op);

REGISTER_KERNEL_BUILDER(
    Name("ParseExamples").Device(DEVICE_CPU).TypeConstraint<tstring>("T"),
    ParseStringExamplesOp);

REGISTER_KERNEL_BUILDER(
    Name("ParseExamples").Device(DEVICE_CPU).TypeConstraint<Variant>("T"),
    ParseVariantExamplesOp);

REGISTER_KERNEL_BUILDER(
    Name("ParseExamplesV2").Device(DEVICE_CPU).TypeConstraint<tstring>("T"),
    ParseStringExamplesV2Op);
REGISTER_KERNEL_BUILDER(
    Name("ParseExamplesV2").Device(DEVICE_CPU).TypeConstraint<Variant>("T"),
    ParseVariantExamplesV2Op);

REGISTER_KERNEL_BUILDER(
    Name("ParseExampleBatch").Device(DEVICE_CPU).TypeConstraint<tstring>("T"),
    ParseStringExampleBatchOp);

REGISTER_KERNEL_BUILDER(
    Name("ParseExampleBatch").Device(DEVICE_CPU).TypeConstraint<Variant>("T"),
    ParseVariantExampleBatchOp);

REGISTER_KERNEL_BUILDER(
    Name("ParseExampleBatchV2").Device(DEVICE_CPU).TypeConstraint<tstring>("T"),
    ParseStringExampleBatchV2Op);
REGISTER_KERNEL_BUILDER(
    Name("ParseExampleBatchV2").Device(DEVICE_CPU).TypeConstraint<Variant>("T"),
    ParseVariantExampleBatchV2Op);

REGISTER_KERNEL_BUILDER(Name("ParseExampleBatchList").Device(DEVICE_CPU),
                        ParseVariantExampleBatchListOp);
}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow
