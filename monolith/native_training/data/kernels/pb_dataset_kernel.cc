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

#include <utility>

#include "absl/strings/str_format.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"

#include "monolith/native_training/data/kernels/feature_name_mapper_tf_bridge.h"
#include "monolith/native_training/data/training_instance/cc/data_reader.h"
#include "monolith/native_training/runtime/common/metrics.h"
#include "third_party/nlohmann/json.hpp"

namespace tensorflow {
namespace data {
namespace monolith_tf {
namespace {
using Example = ::monolith::io::proto::Example;
using ExampleBatch = ::monolith::io::proto::ExampleBatch;
using Instance = ::parser::proto::Instance;
using ::tensorflow::monolith_tf::ExampleBatchIterator;
using ::tensorflow::monolith_tf::ExampleToInstance;
using ::tensorflow::monolith_tf::FeatureNameMapper;
using ::tensorflow::monolith_tf::FeatureNameMapperTfBridge;
using ::tensorflow::monolith_tf::FeaturePruningType;
using ::tensorflow::monolith_tf::InstanceToExample;
using ::tensorflow::monolith_tf::PBIterator;
using ::tensorflow::monolith_tf::DataFormatOptions;
using ::tensorflow::monolith_tf::BaseStreamReader;
using ::tensorflow::monolith_tf::StdinStreamReader;
using ::tensorflow::monolith_tf::FileStreamReader;

const std::string EXAMPLEBATCH = "examplebatch";
const std::string EXAMPLE = "example";
const std::string INSTANCE = "instance";
const std::string PLAINTEXT = "plaintext";

struct DsOptions : DataFormatOptions {
  bool use_snappy = false;
  int64 buffer_size = 64 * 1024 * 1024;
};

}  // namespace

// This is the instance dataset op and used in the estimator as input fn.
class PBDatasetOp : public DatasetOpKernel {
 public:
  static constexpr const char *const kDatasetType = "PbDataset";
  static constexpr const char *const kFileName = "file_name";
  static constexpr const char *const kBufferSize = "buffer_size";
  static constexpr const char *const kUseSnappy = "use_snappy";
  static constexpr const char *const kLagrangexHeader = "lagrangex_header";
  static constexpr const char *const kHasSortId = "has_sort_id";
  static constexpr const char *const kKafkaDump = "kafka_dump";
  static constexpr const char *const kKafkaDumpPrefix = "kafka_dump_prefix";
  static constexpr const char *const kInputPbType = "input_pb_type";
  static constexpr const char *const kOutputPbType = "output_pb_type";
  static constexpr const char *const kOutType = "out_type";
  static constexpr const char *const kFeaturePruningType =
      "feature_pruning_type";
  static constexpr const char *const kFeatureNameList = "feature_name_list";
  static constexpr const char *const kFeatureIdList = "feature_id_list";

  explicit PBDatasetOp(OpKernelConstruction *ctx) : DatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutType, &out_type_));

    auto creator = [this](FeatureNameMapperTfBridge **out_mapper) {
      TF_RETURN_IF_ERROR(FeatureNameMapperTfBridge::New(out_mapper));
      return Status::OK();
    };
    ResourceMgr *resource_mgr = ctx->resource_manager();
    OP_REQUIRES_OK(ctx,
                   resource_mgr->LookupOrCreate<FeatureNameMapperTfBridge>(
                       resource_mgr->default_container(),
                       FeatureNameMapperTfBridge::kName, &mapper_, creator));
  }

  ~PBDatasetOp() override { mapper_->Unref(); };

 private:
  void MakeDataset(OpKernelContext *ctx, DatasetBase **output) override {
    tstring file_name;
    DsOptions options;
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument<tstring>(ctx, kFileName, &file_name));
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument<bool>(ctx, kUseSnappy, &options.use_snappy));
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument<bool>(ctx, kHasSortId, &options.has_sort_id));
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument<bool>(ctx, kKafkaDump, &options.kafka_dump));
    OP_REQUIRES_OK(ctx, ParseScalarArgument<bool>(ctx, kKafkaDumpPrefix,
                                                  &options.kafka_dump_prefix));
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, kBufferSize,
                                                   &options.buffer_size));
    OP_REQUIRES_OK(ctx, ParseScalarArgument<bool>(ctx, kLagrangexHeader,
                                                  &options.lagrangex_header));
    tstring input_pb_type;
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument<tstring>(ctx, kInputPbType, &input_pb_type));
    tstring output_pb_type;
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument<tstring>(ctx, kOutputPbType, &output_pb_type));

    int feature_pruning_type = 0;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int>(ctx, kFeaturePruningType,
                                                 &feature_pruning_type));

    std::vector<tstring> feature_name_list;
    OP_REQUIRES_OK(ctx, ParseVectorArgument<tstring>(ctx, kFeatureNameList,
                                                     &feature_name_list));
    std::vector<int32_t> feature_id_list;
    OP_REQUIRES_OK(ctx, ParseVectorArgument<int32_t>(ctx, kFeatureIdList,
                                                     &feature_id_list));
    if (feature_name_list.size() != feature_id_list.size()) {
      LOG(FATAL) << absl::StrFormat(
          "feature_name_list/feature_id_list size should match, while got %ld "
          "vs %ld",
          feature_name_list.size(), feature_id_list.size());
    }
    std::unordered_set<std::string> feature_name_set(feature_name_list.begin(),
                                                     feature_name_list.end());
    std::unordered_set<int32_t> feature_id_set(feature_id_list.begin(),
                                               feature_id_list.end());
    if (feature_name_list.size() != feature_name_set.size()) {
      LOG(FATAL)
          << "feature name list has duplicates, please investigate and retry !";
    }
    if (feature_id_set.size() > feature_name_set.size()) {
      LOG(FATAL) << "feature_name -> feature_id should be  non-injective and "
                    "surjective, that is feature_id_set.size() should be <= "
                    "feature_name_set.size(), please investigate and retry !";
    }
    output_ =
        new Dataset(ctx, file_name, options, input_pb_type, output_pb_type,
                    out_type_, feature_pruning_type, feature_name_list,
                    feature_id_list, mapper_->GetFeatureNameMapper());
    *output = output_;

    nlohmann::json j;
    j[kFileName] = file_name;
    j[kUseSnappy] = options.use_snappy;
    j[kHasSortId] = options.has_sort_id;
    j[kKafkaDump] = options.kafka_dump;
    j[kKafkaDumpPrefix] = options.kafka_dump_prefix;
    j[kBufferSize] = options.buffer_size;
    j[kLagrangexHeader] = options.lagrangex_header;
    j[kInputPbType] = input_pb_type;
    j[kOutputPbType] = output_pb_type;
    j[kFeaturePruningType] = feature_pruning_type;

    LOG(INFO) << j.dump();
  }

  class Dataset : public DatasetBase {
   public:
    explicit Dataset(OpKernelContext *ctx, tstring file_name,
                     const DsOptions &options, std::string input_pb_type,
                     std::string output_pb_type, DataType out_type,
                     int feature_pruning_type,
                     std::vector<tstring> feature_name_list,
                     std::vector<int32_t> feature_id_list,
                     FeatureNameMapper *mapper)
        : DatasetBase(DatasetContext(ctx)),
          file_name_(std::move(file_name)),
          options_(options),
          input_pb_type_(std::move(input_pb_type)),
          output_pb_type_(std::move(output_pb_type)),
          out_type_(out_type),
          feature_pruning_type_(feature_pruning_type),
          feature_name_list_(std::move(feature_name_list)),
          feature_id_list_(std::move(feature_id_list)),
          mapper_(mapper) {
      absl::flat_hash_map<std::string, int32_t> name_to_id;
      absl::flat_hash_map<int32_t, std::vector<std::string>> id_to_name;
      for (size_t i = 0; i < feature_name_list_.size(); ++i) {
        name_to_id.insert({feature_name_list_[i], feature_id_list_[i]});
        id_to_name[feature_id_list_[i]].push_back(feature_name_list_[i]);
      }
      CHECK(mapper_->SetMapping(name_to_id, id_to_name));
      if (input_pb_type == "examplebatch" && output_pb_type == "example") {
        mapper_->TurnOn();
      }
      LOG_FIRST_N(INFO, 1) << "NameToId: " << mapper_->DebugString();
    }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string &prefix) const override {
      return absl::make_unique<Iterator>(
          Iterator::Params{this, strings::StrCat(prefix, "::", kDatasetType)},
          mapper_);
    }

    const DataTypeVector &output_dtypes() const override {
      static auto *dtypes = new DataTypeVector({out_type_});
      return *dtypes;
    }

    const std::vector<PartialTensorShape> &output_shapes() const override {
      static auto *shapes =
          new std::vector<PartialTensorShape>{TensorShape({})};
      return *shapes;
    }

    string DebugString() const override {
      return ("This is the customized Instance Dataset: " + file_name_);
    }

    Status CheckExternalState() const override { return Status::OK(); }

   private:
    Status AsGraphDefInternal(SerializationContext *ctx,
                              DatasetGraphDefBuilder *b,
                              Node **output) const override {
      Node *filename = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(file_name_, &filename));
      Node *use_snappy = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(options_.use_snappy, &use_snappy));
      Node *has_sort_id = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(options_.has_sort_id, &has_sort_id));
      Node *kafka_dump = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(options_.kafka_dump, &kafka_dump));
      Node *kafka_dump_prefix = nullptr;
      TF_RETURN_IF_ERROR(
          b->AddScalar(options_.kafka_dump_prefix, &kafka_dump_prefix));
      Node *buffer_size = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(options_.buffer_size, &buffer_size));
      Node *lagrangex_header = nullptr;
      TF_RETURN_IF_ERROR(
          b->AddScalar(options_.lagrangex_header, &lagrangex_header));
      Node *input_pb_type = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(input_pb_type_, &input_pb_type));
      Node *output_pb_type = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(output_pb_type_, &output_pb_type));
      Node *feature_pruning_type = nullptr;
      TF_RETURN_IF_ERROR(
          b->AddScalar(feature_pruning_type_, &feature_pruning_type));
      Node *feature_name_list = nullptr;
      TF_RETURN_IF_ERROR(b->AddVector(feature_name_list_, &feature_name_list));
      Node *feature_id_list = nullptr;
      TF_RETURN_IF_ERROR(b->AddVector(feature_id_list_, &feature_id_list));
      AttrValue out_type;
      b->BuildAttrValue(out_type_, &out_type);

      TF_RETURN_IF_ERROR(b->AddDataset(
          this,
          {filename, use_snappy, has_sort_id, kafka_dump, kafka_dump_prefix,
           buffer_size, lagrangex_header, input_pb_type, output_pb_type,
           feature_pruning_type, feature_name_list, feature_id_list},
          {{kOutType, out_type}}, output));
      return Status::OK();
    }

    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params &params, FeatureNameMapper *mapper)
          : DatasetIterator<Dataset>(params), mapper_(mapper) {
        mutex_lock l(mu_);
        offset_ = 0;
      }

      Status GetNextInternal(IteratorContext *ctx,
                             std::vector<Tensor> *out_tensors,
                             bool *end_of_sequence) override {
        out_tensors->reserve(1);
        mutex_lock l(mu_);
        if (!reader_) {
          TF_RETURN_IF_ERROR(SetupStreamsLocked(ctx->env()));
        }
        out_tensors->emplace_back(ctx->allocator({}), dataset()->out_type_,
                                  TensorShape({}));
        Status s;
        size_t size;
        if (dataset()->output_pb_type_ == PLAINTEXT) {
          tstring serialized;
          uint32_t data_source_key;
          s = reader_->next(&offset_, &data_source_key, &serialized);
          out_tensors->back().scalar<tstring>()() = std::move(serialized);
        } else if (dataset()->input_pb_type_ == EXAMPLE &&
                   dataset()->output_pb_type_ == INSTANCE) {
          Example exa_pb;
          s = reader_->next(&offset_, &exa_pb);
          Instance ins_pb;
          ExampleToInstance(&exa_pb, &ins_pb);
          size = ins_pb.ByteSize();
          out_tensors->back().scalar<Variant>()() = std::move(ins_pb);
        } else if (dataset()->input_pb_type_ == INSTANCE &&
                   dataset()->output_pb_type_ == EXAMPLE) {
          Instance ins_pb;
          s = reader_->next(&offset_, &ins_pb);
          Example exa_pb;
          InstanceToExample(&ins_pb, &exa_pb);
          size = exa_pb.ByteSize();
          out_tensors->back().scalar<Variant>()() = std::move(exa_pb);
        } else if (dataset()->output_pb_type_ == EXAMPLE) {  // any -> example
          Example exa_pb;
          s = reader_->next(&offset_, &exa_pb);
          size = exa_pb.ByteSize();
          out_tensors->back().scalar<Variant>()() = std::move(exa_pb);
        } else if (dataset()->output_pb_type_ == INSTANCE) {  // any -> instance
          Instance ins_pb;
          s = reader_->next(&offset_, &ins_pb);
          size = ins_pb.ByteSize();
          out_tensors->back().scalar<Variant>()() = std::move(ins_pb);
        } else {  // any -> example_batch
          ExampleBatch eb_pb;
          s = reader_->next(&offset_, &eb_pb);
          size = eb_pb.ByteSize();
          out_tensors->back().scalar<Variant>()() = std::move(eb_pb);
        }

        if (s.ok()) {
          static monitoring::CounterCell *bytes_counter =
              metrics::GetTFDataBytesReadCounter(kDatasetType);
          bytes_counter->IncrementBy(size);
          *end_of_sequence = false;
          num_random_samples_++;
          offset_ = reader_->GetOffset();
          if (num_random_samples_ % metric_emit_step_ == 0) {
            LOG_EVERY_N_SEC(INFO, 300) << absl::StrFormat(
                "metrics_emit(counter) [instance_num] emit=%llu, "
                "total_instance_num=%lld",
                metric_emit_step_, num_random_samples_);
            monolith::GetMetrics()->emit_counter("instance_num",
                                                 metric_emit_step_);
          }
          return Status::OK();
        }

        out_tensors->pop_back();
        ResetStreamsLocked();
        if (errors::IsOutOfRange(s)) {
          *end_of_sequence = true;
          int64 unsubmit_instance_num = num_random_samples_ % metric_emit_step_;
          if (unsubmit_instance_num > 0) {
            LOG(INFO) << absl::StrFormat(
                "metrics_emit(counter) [instance_num] emit=%lld, "
                "total_instance_num=%lld, end_of_sequence",
                unsubmit_instance_num, num_random_samples_);
            monolith::GetMetrics()->emit_counter("instance_num",
                                                 unsubmit_instance_num);
          }
          return Status::OK();
        }
        return s;
      }

     private:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext *ctx, model::Node::Args args) const override {
        return model::MakeSourceNode(std::move(args));
      }

      Status SaveInternal(SerializationContext *ctx,
                          IteratorStateWriter *writer) override {
        mutex_lock l(mu_);
        LOG(INFO) << "Save function is not supported yet.";
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("num_random_samples"),
                                               num_random_samples_));
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("offset"), offset_));
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext *ctx,
                             IteratorStateReader *reader) override {
        mutex_lock l(mu_);
        LOG(INFO) << "Restore function is not supported yet.";
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("num_random_samples"),
                                              &num_random_samples_));
        int64 offset;
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("offset"), &offset));
        if (dataset()->file_name_.empty()) {
          offset_ = 0;
        } else {
          offset_ = offset;
        }
        return Status::OK();
      }

      // Sets up reader streams to read from filename
      Status SetupStreamsLocked(Env *env) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        std::unique_ptr<BaseStreamReader> stream_reader;
        if (dataset()->file_name_.empty()) {
          stream_reader =
              std::make_unique<StdinStreamReader>(dataset()->options_);
        } else {
          std::unique_ptr<RandomAccessFile> f;
          TF_RETURN_IF_ERROR(
              env->NewRandomAccessFile(dataset()->file_name_, &f));
          stream_reader = std::make_unique<FileStreamReader>(
              dataset()->options_, std::move(f),
              dataset()->options_.use_snappy);
        }
        if (dataset()->input_pb_type_ == "instance" ||
            dataset()->input_pb_type_ == "example") {
          reader_ = absl::make_unique<PBIterator>(
              std::move(stream_reader), static_cast<FeaturePruningType>(
                                            dataset()->feature_pruning_type_));
        } else {
          reader_ = absl::make_unique<ExampleBatchIterator>(
              std::move(stream_reader),
              static_cast<FeaturePruningType>(dataset()->feature_pruning_type_),
              mapper_);
        }

        return Status::OK();
      }

      // Resets all reader streams.
      void ResetStreamsLocked() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        reader_.reset();
      }

      mutex mu_;
      std::unique_ptr<PBIterator> reader_ TF_GUARDED_BY(mu_);
      int64 num_random_samples_ TF_GUARDED_BY(mu_) = 0;
      uint64 offset_ TF_GUARDED_BY(mu_) = 0;
      uint64 metric_emit_step_ TF_GUARDED_BY(mu_) = 10000;
      FeatureNameMapper *mapper_ = nullptr;
    };

    tstring file_name_, input_pb_type_, output_pb_type_;
    DsOptions options_;
    DataType out_type_;
    std::vector<tstring> feature_name_list_;
    std::vector<int32_t> feature_id_list_;
    int feature_pruning_type_ = FeaturePruningType::PRUNING_RAW_FEATURE;
    FeatureNameMapper *mapper_ = nullptr;
  };

  Dataset *output_ = nullptr;
  DataType out_type_;
  FeatureNameMapperTfBridge *mapper_ = nullptr;
};

namespace {
REGISTER_KERNEL_BUILDER(Name("PBDataset").Device(DEVICE_CPU), PBDatasetOp);
}  // namespace

}  // namespace monolith_tf
}  // namespace data
}  // namespace tensorflow
