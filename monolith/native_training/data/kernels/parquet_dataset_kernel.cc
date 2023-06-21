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

#include "monolith/native_training/data/kernels/internal/parquet_example_reader.h"
#include "parquet/api/reader.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "third_party/nlohmann/json.hpp"

namespace tensorflow {
namespace data {
namespace monolith_tf {

using monolith::io::proto::Example;
using monolith::io::proto::NamedFeature;
using monolith::io::proto::NamedFeatureList;

class ParquetDatasetOp : public DatasetOpKernel {
 public:
  static const char* const kDatasetType;
  static const char* const kFileName;
  static const char* const kOutputPbType;
  static const char* const kBatchSize;
  static const char* const kDropRemainder;
  static const char* const kSelectColumns;
  static const char* const kSelectColumnsType;

  explicit ParquetDatasetOp(OpKernelConstruction* ctx) : DatasetOpKernel(ctx) {
    // select_columns
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kBatchSize, &batch_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kDropRemainder, &drop_remainder_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kSelectColumns, &select_columns_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr(kSelectColumnsType, &select_columns_type_));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    tstring file_name;
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument<tstring>(ctx, kFileName, &file_name));
    tstring output_pb_type;
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument<tstring>(ctx, kOutputPbType, &output_pb_type));

    *output =
        new Dataset(ctx, file_name, output_pb_type, batch_size_,
                    drop_remainder_, select_columns_, select_columns_type_);

    // config log
    nlohmann::json j;
    j[kFileName] = file_name;
    j[kOutputPbType] = output_pb_type;
    j[kBatchSize] = batch_size_;
    j[kSelectColumns] = select_columns_.size();
    j[kSelectColumnsType] = select_columns_type_.size();
    LOG(INFO) << j.dump();
  }

 private:
  class Dataset : public DatasetBase {
   public:
    explicit Dataset(OpKernelContext* ctx, tstring file_name,
                     tstring output_pb_type, int32_t batch_size,
                     bool drop_remainder, std::vector<tstring> select_columns,
                     std::vector<tstring> select_columns_type)
        : DatasetBase(DatasetContext(ctx)),
          file_name_(std::move(file_name)),
          output_pb_type_(std::move(output_pb_type)),
          batch_size_(batch_size),
          drop_remainder_(drop_remainder),
          select_columns_(std::move(select_columns)),
          select_columns_type_(std::move(select_columns_type)) {}

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return absl::make_unique<Iterator>(
          Iterator::Params{this, strings::StrCat(prefix, "::", kDatasetType)});
    }

    const DataTypeVector& output_dtypes() const override {
      static DataTypeVector* dtypes = nullptr;
      if (!dtypes) {
        if (output_pb_type_ == "example" || output_pb_type_ == "examplebatch") {
          dtypes = new DataTypeVector({DT_VARIANT});
        } else {
          dtypes = new DataTypeVector({DT_STRING});
        }
      }
      return *dtypes;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      static auto* shapes =
          new std::vector<PartialTensorShape>{TensorShape({})};
      return *shapes;
    }

    string DebugString() const override { return "ParquetDatasetOp::Dataset"; }

    Status CheckExternalState() const override { return Status::OK(); }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* file_name = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(file_name_, &file_name));
      Node* output_pb_type = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(output_pb_type_, &output_pb_type));
      AttrValue batch_size;
      b->BuildAttrValue(batch_size_, &batch_size);
      AttrValue drop_remainder;
      b->BuildAttrValue(drop_remainder_, &drop_remainder);
      AttrValue select_columns;
      b->BuildAttrValue(select_columns_, &select_columns);
      AttrValue select_columns_type;
      b->BuildAttrValue(select_columns_type_, &select_columns_type);

      TF_RETURN_IF_ERROR(
          b->AddDataset(this, {file_name, output_pb_type},
                        {{kBatchSize, batch_size},
                         {kDropRemainder, drop_remainder},
                         {kSelectColumns, select_columns},
                         {kSelectColumnsType, select_columns_type}},
                        output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        out_tensors->clear();
        out_tensors->reserve(1);
        if (!parquet_reader_) {
          parquet_reader_.reset(
              new tensorflow::data::ParquetExampleReader(ctx->env()));
          std::vector<string> select_col_str(dataset()->select_columns_.begin(),
                                             dataset()->select_columns_.end());
          std::vector<string> select_col_type_str(
              dataset()->select_columns_type_.begin(),
              dataset()->select_columns_type_.end());

          TF_RETURN_IF_ERROR(parquet_reader_->Init(
              dataset()->file_name_, select_col_str, select_col_type_str));
        }

        if (dataset()->output_pb_type_ == "example") {
          Example example;
          TF_RETURN_IF_ERROR(GetNextExample(example, end_of_sequence));
          Tensor record_tensor(ctx->allocator({}), DT_VARIANT, {});
          record_tensor.scalar<Variant>()() = std::move(example);
          out_tensors->emplace_back(std::move(record_tensor));
          if (*end_of_sequence) {
            LOG(INFO) << "end_of_sequence of " << dataset()->file_name_;
          } else {
            counter_++;
            if (counter_ % 1000 == 0) {
              LOG(INFO) << "consume " << counter_ << "examples from "
                        << dataset()->file_name_;
            }
          }
        } else if (dataset()->output_pb_type_ == "examplebatch") {
          ExampleBatch example_batch;
          TF_RETURN_IF_ERROR(
              GetNextExampleBatch(example_batch, end_of_sequence));
          if (!(*end_of_sequence)) {
            if (dataset()->drop_remainder_ &&
                example_batch.batch_size() < dataset()->batch_size_) {
              LOG(INFO) << "last example batch size="
                        << example_batch.batch_size() << " dropped";
              *end_of_sequence = true;
            } else {
              Tensor record_tensor(ctx->allocator({}), DT_VARIANT, {});
              record_tensor.scalar<Variant>()() = std::move(example_batch);
              out_tensors->emplace_back(std::move(record_tensor));
              counter_++;
              if (counter_ % 100 == 0) {
                LOG(INFO) << "consume " << counter_ << "example_batch from "
                          << dataset()->file_name_;
              }
            }
          }
        } else if (dataset()->output_pb_type_ == "plaintext") {
          // only for debug use, generate examplebatch pb string
          ExampleBatch example_batch;
          TF_RETURN_IF_ERROR(
              GetNextExampleBatch(example_batch, end_of_sequence));
          Tensor record_tensor(ctx->allocator({}), DT_STRING, {});
          std::string out;
          example_batch.SerializeToString(&out);
          record_tensor.scalar<tstring>()() = out;
          out_tensors->emplace_back(std::move(record_tensor));
          if (*end_of_sequence) {
            LOG(INFO) << "end_of_sequence of " << dataset()->file_name_;
          }
        } else {
          return errors::InvalidArgument(
              "output_pb_type is ", dataset()->output_pb_type_,
              ",should be example or examplebatch or plaintext");
        }

        return Status::OK();
      }

      Status GetNextExample(Example& example, bool* end_of_sequence) {
        if (parquet_reader_->IsEOF()) {
          *end_of_sequence = true;
        } else {
          *end_of_sequence = false;
          example.Clear();
          parquet_reader_->GetNextExample(example);
        }

        return Status::OK();
      }

      Status GetNextExampleBatch(ExampleBatch& example_batch,
                                 bool* end_of_sequence) {
        profiler::TraceMe activity(
            []() { return "ParquetDatasetOp::GetNextExampleBatch"; });
        if (parquet_reader_->IsEOF()) {
          *end_of_sequence = true;
        } else {
          *end_of_sequence = false;
          example_batch.Clear();
          parquet_reader_->GetNextExampleBatch(example_batch,
                                               dataset()->batch_size_);
        }
        return Status::OK();
      }

      NamedFeatureList* AddNamedFeatureList(ExampleBatch& example_batch,
                                            const std::string& name,
                                            int32_t id) {
        NamedFeatureList* named_feature_list =
            example_batch.add_named_feature_list();
        named_feature_list->set_id(id);
        named_feature_list->set_name(name);
        return named_feature_list;
      }

     protected:
      Status SaveInternal(SerializationContext* ctx,
                          IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        // do nothing
        LOG(INFO) << "Save function is not supported yet.";
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        // do nothing
        LOG(INFO) << "Restore function is not supported yet.";
        return Status::OK();
      }

     private:
      mutex mu_;
      int64_t counter_ = 0;
      std::unique_ptr<tensorflow::data::ParquetExampleReader> parquet_reader_;
    };

    // original inputs/attrs
    tstring file_name_;
    tstring output_pb_type_;
    int32_t batch_size_;
    bool drop_remainder_;
    std::vector<tstring> select_columns_;
    std::vector<tstring> select_columns_type_;
  };

  Dataset* output_ = nullptr;
  int32_t batch_size_;
  bool drop_remainder_;
  std::vector<tstring> select_columns_;
  std::vector<tstring> select_columns_type_;
};

const char* const ParquetDatasetOp::kDatasetType = "ParquetDataset";
const char* const ParquetDatasetOp::kFileName = "file_name";
const char* const ParquetDatasetOp::kOutputPbType = "output_pb_type";
const char* const ParquetDatasetOp::kBatchSize = "batch_size";
const char* const ParquetDatasetOp::kSelectColumns = "select_columns";
const char* const ParquetDatasetOp::kSelectColumnsType = "select_columns_type";
const char* const ParquetDatasetOp::kDropRemainder = "drop_remainder";

namespace {
REGISTER_KERNEL_BUILDER(Name("ParquetDataset").Device(DEVICE_CPU),
                        ParquetDatasetOp);
}

}  // namespace monolith_tf
}  // namespace data
}  // namespace tensorflow
