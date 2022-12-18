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

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_inputstream.h"

#include "monolith/native_training/data/training_instance/cc/data_reader.h"

namespace tensorflow {
namespace data {
namespace monolith_tf {

using ::tensorflow::monolith_tf::PBIterator;
using ::tensorflow::monolith_tf::PRUNING_RAW_FEATURE;
using ::tensorflow::monolith_tf::DataFormatOptions;
using ::tensorflow::monolith_tf::BaseStreamReader;
using ::tensorflow::monolith_tf::StdinStreamReader;
using ::tensorflow::monolith_tf::FileStreamReader;

struct DsOptions : DataFormatOptions {
  bool use_snappy = false;
};

// This is the instance dataset op and used in the estimator as input fn.
class InstanceDatasetOp : public DatasetOpKernel {
 public:
  static constexpr const char* const kDatasetType = "PbInstance";
  static constexpr const char* const kFileName = "file_name";
  static constexpr const char* const kUseSnappy = "use_snappy";
  static constexpr const char* const kHasSortId = "has_sort_id";
  static constexpr const char* const kKafkaDump = "kafka_dump";
  static constexpr const char* const kKafkaDumpPrefix = "kafka_dump_prefix";

  explicit InstanceDatasetOp(OpKernelConstruction* ctx)
      : DatasetOpKernel(ctx) {}
  ~InstanceDatasetOp() {}

 private:
  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
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
    output_ = new Dataset(ctx, file_name, options);
    *output = output_;
  }

  class Dataset : public DatasetBase {
   public:
    explicit Dataset(OpKernelContext* ctx, const tstring& file_name,
                     const DsOptions& options)
        : DatasetBase(DatasetContext(ctx)),
          file_name_(file_name),
          options_(options) {}

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return absl::make_unique<Iterator>(
          Iterator::Params{this, strings::StrCat(prefix, "::", kDatasetType)});
    }

    const DataTypeVector& output_dtypes() const override {
      static DataTypeVector* dtypes = new DataTypeVector({DT_STRING});
      return *dtypes;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      static std::vector<PartialTensorShape>* shapes =
          new std::vector<PartialTensorShape>{TensorShape({})};
      return *shapes;
    }

    string DebugString() const override {
      return ("This is the customized Instance Dataset: " + file_name_);
    }

    Status CheckExternalState() const override { return Status::OK(); }

   private:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* filename = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(file_name_, &filename));
      Node* use_snappy = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(options_.use_snappy, &use_snappy));
      Node* has_sort_id = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(options_.has_sort_id, &has_sort_id));
      Node* kafka_dump = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(options_.kafka_dump, &kafka_dump));
      Node* kafka_dump_prefix = nullptr;
      TF_RETURN_IF_ERROR(
          b->AddScalar(options_.kafka_dump_prefix, &kafka_dump_prefix));
      TF_RETURN_IF_ERROR(b->AddDataset(this, {filename, use_snappy, has_sort_id,
                                              kafka_dump, kafka_dump_prefix},
                                       output));
      return Status::OK();
    }

    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        out_tensors->reserve(1);
        mutex_lock l(mu_);
        if (!reader_) {
          TF_RETURN_IF_ERROR(SetupStreamsLocked(ctx->env()));
        }
        out_tensors->emplace_back(ctx->allocator({}), DT_STRING,
                                  TensorShape({}));
        uint32_t data_source_key;
        Status s = reader_->next(&offset_, &data_source_key,
                                 &out_tensors->back().scalar<tstring>()());
        if (s.ok()) {
          static monitoring::CounterCell* bytes_counter =
              metrics::GetTFDataBytesReadCounter(kDatasetType);
          bytes_counter->IncrementBy(
              out_tensors->back().scalar<tstring>()().size());
          *end_of_sequence = false;
          num_random_samples_++;
          offset_ = reader_->GetOffset();
          return Status::OK();
        }
        out_tensors->pop_back();
        ResetStreamsLocked();
        if (errors::IsOutOfRange(s)) {
          *end_of_sequence = true;
          return Status::OK();
        }
        return s;
      }

     private:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeSourceNode(std::move(args));
      }

      Status SaveInternal(SerializationContext* ctx,
                          IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        LOG(INFO) << "Save function is not supported yet.";
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("num_random_samples"),
                                               num_random_samples_));
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("offset_"), offset_));
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        LOG(INFO) << "Restore function is not supported yet.";
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("num_random_samples"),
                                              &num_random_samples_));
        int64 offset;
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("offset_"), &offset));
        if (dataset()->file_name_.empty()) {
          offset_ = 0;
        } else {
          offset_ = offset;
        }
        return Status::OK();
      }

      // Sets up reader streams to read from filename
      Status SetupStreamsLocked(Env* env) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
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
        reader_ = absl::make_unique<PBIterator>(std::move(stream_reader),
                                                PRUNING_RAW_FEATURE);
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
    };

    tstring file_name_;
    DsOptions options_;
  };
  Dataset* output_ = nullptr;
};

namespace {
REGISTER_KERNEL_BUILDER(Name("InstanceDataset").Device(DEVICE_CPU),
                        InstanceDatasetOp);
}  // namespace

}  // namespace monolith_tf
}  // namespace data
}  // namespace tensorflow
