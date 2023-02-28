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

#include "monolith/native_training/data/kernels/cache_one_dataset_kernel.h"

#include <deque>

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/stats_aggregator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"

namespace tensorflow {
namespace data {
namespace monolith_tf {

class CacheOneDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input)
      : DatasetBase(DatasetContext(ctx)), input_(input) {
    input_->Ref();
    output_dtypes_ = input->output_dtypes();
    output_dtypes_.push_back(DT_BOOL);
    output_shapes_ = input->output_shapes();
    output_shapes_.push_back({});
  }

  ~Dataset() override { input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(
        Iterator::Params{this, absl::StrCat(prefix, ":: CacheOneDataset")});
  }

  const DataTypeVector& output_dtypes() const override {
    return output_dtypes_;
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  string DebugString() const override {
    return "This is the customized Dataset: CacheOneDataset";
  }

  int64 Cardinality() const override { return input_->Cardinality(); }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
    inputs->push_back(input_);
    return Status::OK();
  }

  Status CheckExternalState() const override {
    return input_->CheckExternalState();
  }

 private:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    TF_RETURN_IF_ERROR(b->AddDataset(this, {input_graph_node}, {}, output));
    return Status::OK();
  }

  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params) {}

    Status Initialize(IteratorContext* ctx) override {
      absl::MutexLock l(&mu_);
      return dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_);
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      absl::MutexLock l(&mu_);
      if (first_element_) {
        first_element_ = false;
        TF_RETURN_IF_ERROR(
            input_impl_->GetNext(ctx, &buffered_tensors_, end_of_sequence));
        if (*end_of_sequence) {
          // This is the special case that input dataset contains no data.
          // Here we just throw it out.
          return Status::OK();
        }
      }

      // We run out of the data.
      if (eof_) {
        *end_of_sequence = true;
        return Status::OK();
      }

      *out_tensors = std::move(buffered_tensors_);
      buffered_tensors_.clear();
      TF_RETURN_IF_ERROR(input_impl_->GetNext(ctx, &buffered_tensors_, &eof_));
      Tensor eof_tensor(ctx->allocator({}), DT_BOOL, {});
      eof_tensor.scalar<bool>()() = eof_;
      out_tensors->push_back(eof_tensor);
      *end_of_sequence = false;
      return Status::OK();
    }

    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeKnownRatioNode(std::move(args), 1);
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      return errors::Unimplemented("Not Implemented");
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      return errors::Unimplemented("Not Implemented");
    }

    absl::Mutex mu_;
    std::unique_ptr<IteratorBase> input_impl_;
    std::vector<Tensor> buffered_tensors_;
    bool first_element_ = true;
    bool eof_ = false;
  };

 private:
  const DatasetBase* const input_;
  DataTypeVector output_dtypes_;
  std::vector<PartialTensorShape> output_shapes_;
};

void CacheOneDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                    DatasetBase** output) {
  *output = new Dataset(ctx, input);
}

CacheOneDatasetOp::CacheOneDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {}

namespace {
REGISTER_KERNEL_BUILDER(Name("MonolithCacheOneDataset").Device(DEVICE_CPU),
                        CacheOneDatasetOp);
}  // namespace

}  // namespace monolith_tf
}  // namespace data
}  // namespace tensorflow
