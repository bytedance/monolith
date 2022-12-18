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

#include <queue>
#include "monolith/native_training/data/kernels/internal/file_match_split_provider.h"

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_inputstream.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace data {
namespace monolith_tf {

class DynamicMatchingFilesDatasetOp : public DatasetOpKernel {
 public:
  using DatasetOpKernel::DatasetOpKernel;

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    const Tensor* patterns_t;
    OP_REQUIRES_OK(ctx, ctx->input("patterns", &patterns_t));
    const auto patterns = patterns_t->flat<tstring>();
    size_t num_patterns = static_cast<size_t>(patterns.size());
    std::vector<std::string> pattern_strs;
    pattern_strs.reserve(num_patterns);

    for (size_t i = 0; i < num_patterns; i++) {
      LOG(INFO) << "pattern " << patterns(i) << ", num_patterns "
                << num_patterns;
      pattern_strs.push_back(patterns(i));
    }

    *output = new Dataset(ctx, std::move(pattern_strs));
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, std::vector<std::string> patterns)
        : DatasetBase(DatasetContext(ctx)), patterns_(std::move(patterns)) {}

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return absl::make_unique<Iterator>(Iterator::Params{
          this, strings::StrCat(prefix, "::DynamicMatchingFiles")});
    }

    const DataTypeVector& output_dtypes() const override {
      static DataTypeVector* dtypes = new DataTypeVector({DT_STRING});
      return *dtypes;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      static std::vector<PartialTensorShape>* shapes =
          new std::vector<PartialTensorShape>({{}});
      return *shapes;
    }

    string DebugString() const override {
      return "DynamicMatchingFilesDatasetOp::Dataset";
    }

    Status InputDatasets(
        std::vector<const DatasetBase*>* inputs) const override {
      return Status::OK();
    }

    Status CheckExternalState() const override { return Status::OK(); }

    Status MakeSplitProvider(
        std::unique_ptr<SplitProvider>* split_provider) const override {
      split_provider->reset(new FileMatchSplitProvider(patterns_));
      return Status::OK();
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* patterns_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddVector(patterns_, &patterns_node));
      TF_RETURN_IF_ERROR(b->AddDataset(this, {patterns_node}, output));
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
        if (!split_provider_) {
          LOG(INFO) << "Begin to get split_provider from ctx!";
          split_provider_ = ctx->split_provider();
          if (!split_provider_) {
            LOG(INFO) << "No split_provider in ctx, call MakeSplitProvider!";
            std::unique_ptr<SplitProvider> split_provider;
            TF_RETURN_IF_ERROR(dataset()->MakeSplitProvider(&split_provider));
            split_provider_.reset(split_provider.release());
          } else {
            LOG(INFO) << "Got split_provider from IteratorContext";
          }
          LOG(INFO) << "Get split_provider done!";
        }

        if (end_of_sequence_) {
          *end_of_sequence = true;
          out_tensors->clear();
          return Status::OK();
        }

        Tensor split;
        Status s = split_provider_->GetNext(&split, end_of_sequence);
        if (errors::IsOutOfRange(s)) {
          out_tensors->clear();
          *end_of_sequence = true;
          end_of_sequence_ = true;
          LOG(INFO) << s.error_message();
        } else if (s.ok()) {
          *end_of_sequence = false;
          out_tensors->emplace_back(std::move(split));
        } else {
          return s;
        }

        return Status::OK();
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeSourceNode(std::move(args));
      }

      Status SaveInternal(SerializationContext* ctx,
                          IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        if (split_provider_ != nullptr) {
          split_provider_->Save(
              [this](std::string name) { return FullName(prefix(), name); },
              writer);
        }
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        if (!split_provider_) {
          split_provider_ = ctx->split_provider();
        }
        split_provider_->Restore(
            [this](std::string name) { return FullName(prefix(), name); },
            reader);
        return Status::OK();
      }

     private:
      mutex mu_;
      bool end_of_sequence_ = false;
      std::shared_ptr<SplitProvider> split_provider_;
    };

    const std::vector<std::string> patterns_;
  };
};

namespace {
REGISTER_KERNEL_BUILDER(Name("DynamicMatchingFilesDataset").Device(DEVICE_CPU),
                        DynamicMatchingFilesDatasetOp);
}  // namespace
}  // namespace monolith_tf
}  // namespace data
}  // namespace tensorflow
