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

#include <bitset>
#include "monolith/native_training/data/training_instance/cc/pb_variant.h"
#include "monolith/native_training/data/training_instance/cc/reader_util.h"
#include "monolith/native_training/data/kernels/df_resource_kernel.h"
#include "monolith/native_training/data/kernels/internal/datasource_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/resource_mgr.h"

namespace tensorflow {
namespace data {
namespace monolith_tf {
using Instance = ::parser::proto::Instance;
using Example = ::monolith::io::proto::Example;
using Item = ::tensorflow::monolith_tf::Item;
using QueueResource = ::tensorflow::monolith_tf::QueueResource;
using VariantType = ::tensorflow::monolith_tf::VariantType;

static mutex input_mu_;

class SplitFlowDatasetOp : public UnaryDatasetOpKernel {
 public:
  static constexpr const char *const kDatasetType = "dataflow_dataset";
  static constexpr const char *const kDataFlow = "data_flow";
  static constexpr const char *const kIndex = "index";
  static constexpr const char *const kMaxQueueSize = "max_queue_size";
  static constexpr const char *const kVariantType = "variant_type";

  explicit SplitFlowDatasetOp(OpKernelConstruction *ctx);

 protected:
  void MakeDataset(OpKernelContext *ctx, DatasetBase *input,
                   DatasetBase **output) override;

 private:
  class Dataset;
  std::vector<std::string> data_flows_;
  int index_;
  int max_queue_size_;
  VariantType variant_type_;
};

class SplitFlowDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext *ctx, const DatasetBase *input,
          const std::vector<std::string> &data_flows, int index,
          int max_queue_size, const VariantType &variant_type)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        data_flows_(data_flows),
        index_(index),
        max_queue_size_(max_queue_size),
        variant_type_(variant_type) {
    input_->Ref();
  }

  ~Dataset() override { input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string &prefix) const override {
    return absl::make_unique<Iterator>(
        Iterator::Params{this, strings::StrCat(prefix, "::", kDatasetType)});
  }

  const DataTypeVector &output_dtypes() const override {
    return input_->output_dtypes();
  }

  const std::vector<PartialTensorShape> &output_shapes() const override {
    return input_->output_shapes();
  }

  string DebugString() const override {
    return "This is the customized Dataset: SplitFlowDataset";
  }

  Status InputDatasets(
      std::vector<const DatasetBase *> *inputs) const override {
    inputs->push_back(input_);
    return Status::OK();
  }

  Status CheckExternalState() const override {
    return input_->CheckExternalState();
  }

  void SetContainer(const std::string container) { container_ = container; }

  std::string GetContainer() const { return container_; }

 protected:
  Status AsGraphDefInternal(SerializationContext *ctx,
                            DatasetGraphDefBuilder *b,
                            Node **output) const override {
    Node *input_graph_node;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));

    AttrValue data_flows_node;
    b->BuildAttrValue(data_flows_, &data_flows_node);
    AttrValue index_node;
    b->BuildAttrValue(index_, &index_node);
    AttrValue max_queue_size_node;
    b->BuildAttrValue(max_queue_size_, &max_queue_size_node);

    AttrValue variant_type_node;
    if (variant_type_ == VariantType::PBInstance) {
      b->BuildAttrValue("instance", &variant_type_node);
    } else {
      b->BuildAttrValue("example", &variant_type_node);
    }

    TF_RETURN_IF_ERROR(
        b->AddDataset(this,                // dataset
                      {input_graph_node},  // inputs
                      {{kDataFlow, data_flows_node},
                       {kIndex, index_node},
                       {kMaxQueueSize, max_queue_size_node},
                       {kVariantType, variant_type_node}},  // attrs
                      output));                             // Node**

    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params &params)
        : DatasetIterator<Dataset>(params),
          mu_(std::make_shared<mutex>()),
          output_mu_(std::make_shared<mutex>()) {}

    ~Iterator() override {
      CancelThreads();
      if (deregister_fn_) deregister_fn_();
    }

    void CancelThreads() TF_LOCKS_EXCLUDED(mu_) {
      cancellation_manager_->StartCancel();
      mutex_lock l(*mu_);
      cancelled_ = true;
    }

    Status Initialize(IteratorContext *ctx) override {
      mutex_lock l(*mu_);
      name_ = dataset()->data_flows_[dataset()->index_];
      cancellation_manager_ = absl::make_unique<CancellationManager>();
      TF_RETURN_IF_ERROR(
          ::tensorflow::monolith_tf::RegisterCancellationCallback(
              ctx->cancellation_manager(), [this]() { CancelThreads(); },
              &deregister_fn_));

      IteratorContext::Params params(ctx);
      params.cancellation_manager = cancellation_manager_.get();
      Status s = dataset()->input_->MakeIterator(IteratorContext(params), this,
                                                 prefix(), &input_impl_);

      std::function<Status(QueueResource **)> creator =
          [this](QueueResource **queue) -> Status {
        *queue = new QueueResource(dataset()->max_queue_size_);
        return Status::OK();
      };

      {
        mutex_lock input_l(input_mu_);
        for (size_t i = 0; i < dataset()->data_flows_.size(); ++i) {
          // 1) get data_flow_name and hash it into uint32
          std::string data_flows_name = dataset()->data_flows_[i];
          uint32 df_code = static_cast<uint32>(::tensorflow::monolith_tf::internal::java_hash_code(data_flows_name));
          df_code = df_code << 8;

          // 2) get resource
          QueueResource *resource = nullptr;
          s.Update(ctx->resource_mgr()->LookupOrCreate(
              dataset()->GetContainer(), data_flows_name, &resource, creator));
          df_to_queue_.emplace(df_code, resource);
          if (i == dataset()->index_) {
            data_flow_ = df_code;
            queue_ = resource;
          }
        }
      }

      return s;
    }

    Status GetNextInternal(IteratorContext *ctx,
                           std::vector<Tensor> *out_tensors,
                           bool *end_of_sequence) override {
      // std::thread::id this_id = std::this_thread::get_id();
      {
        mutex_lock l(*mu_);
        if (dataset()->index_ == 0) {
          TF_RETURN_IF_ERROR(EnsureThreadStarted(ctx));
        }
      }

      {
        mutex_lock output_l(*output_mu_);
        out_tensors->reserve(1);

        Item item;
        bool poped = false;
        while (!poped) {
          // the queue is empty and the fetch threas is cancelled or finished
          if (cancelled_ || prefetch_thread_finished_) {
            out_tensors->clear();
            *end_of_sequence = true;
            return Status::OK();
          }
          poped = queue_->TryPop(item, 100);
        }

        if (!poped || item.end_of_sequence) {
          out_tensors->clear();
          *end_of_sequence = true;
        } else {
          for (const auto &tensor : item.out_tensors) {
            out_tensors->push_back(tensor);
          }
          *end_of_sequence = item.end_of_sequence;
        }
      }

      return Status::OK();
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext *ctx, model::Node::Args args) const override {
      return model::MakeUnknownRatioNode(std::move(args));
    }

    Status SaveInternal(SerializationContext *ctx,
                        IteratorStateWriter *writer) override {
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext *ctx,
                           IteratorStateReader *reader) override {
      return Status::OK();
    }

   private:
    const std::shared_ptr<mutex> mu_;
    const std::shared_ptr<mutex> output_mu_;
    std::function<void()> deregister_fn_;
    std::unique_ptr<CancellationManager> cancellation_manager_;
    bool cancelled_ TF_GUARDED_BY(*mu_) = false;
    bool prefetch_thread_started_ TF_GUARDED_BY(*mu_) = false;
    bool prefetch_thread_finished_ TF_GUARDED_BY(*mu_) = false;

    uint32 data_flow_;
    std::string name_;
    QueueResource *queue_;
    std::unique_ptr<IteratorBase> input_impl_;
    std::unique_ptr<Thread> prefetch_thread_;
    std::unordered_map<uint32, QueueResource *> df_to_queue_;

    Status EnsureThreadStarted(IteratorContext *ctx)
        TF_EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      if (!prefetch_thread_started_) {
        prefetch_thread_started_ = true;
        std::string name = dataset()->data_flows_[dataset()->index_];
        std::shared_ptr<IteratorContext> new_ctx =
            std::make_shared<IteratorContext>(*ctx);
        prefetch_thread_ = ctx->StartThread(
            name, [new_ctx, name, this]() { PrefetchThread(new_ctx, name); });
      }

      return Status::OK();
    }

    void PrefetchThread(const std::shared_ptr<IteratorContext> &ctx,
                        std::string name) {
      while (true) {
        {
          mutex_lock l(*mu_);
          if (cancelled_) {
            prefetch_thread_finished_ = true;
            break;
          }
        }

        if (!prefetch_thread_finished_) {
          Item item;
          input_impl_->GetNext(ctx.get(), &item.out_tensors,
                               &item.end_of_sequence);

          if (item.end_of_sequence) {
            mutex_lock l(*mu_);
            item.end_of_sequence = true;
            if (!cancelled_ && !prefetch_thread_finished_) {
              for (auto kv : df_to_queue_) {
                kv.second->Push(item);
              }
            }
            break;
          } else {
            uint32 code;
            if (dataset()->variant_type_ == VariantType::PBInstance) {
              code = item.out_tensors[0]
                         .scalar<Variant>()()
                         .get<Instance>()
                         ->data_source_key();
            } else {
              code = item.out_tensors[0]
                         .scalar<Variant>()()
                         .get<Example>()
                         ->data_source_key();
            }

            bool pushed = false;
            do {
              if (cancelled_ || prefetch_thread_finished_) {
                break;
              }
              pushed = df_to_queue_[code]->TryPush(item);
            } while (!pushed);
          }
        } else {
          break;
        }
      }
    }
  };

  const DatasetBase *const input_;
  std::vector<std::string> data_flows_;
  int index_;
  int max_queue_size_;
  VariantType variant_type_;
  std::string container_;
};

SplitFlowDatasetOp::SplitFlowDatasetOp(OpKernelConstruction *ctx)
    : UnaryDatasetOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kDataFlow, &data_flows_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kIndex, &index_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kMaxQueueSize, &max_queue_size_));

  std::string variant_type;
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kVariantType, &variant_type));
  if (variant_type == "instance") {
    variant_type_ = VariantType::PBInstance;
  } else if (variant_type == "example") {
    variant_type_ = VariantType::PBExample;
  } else {
    LOG(ERROR) << "invalid variant_type: " << variant_type;
    ctx->SetStatus(Status(tensorflow::error::Code::INVALID_ARGUMENT,
                          "invalid variant_type"));
  }
}

void SplitFlowDatasetOp::MakeDataset(OpKernelContext *ctx, DatasetBase *input,
                                     DatasetBase **output) {
  *output = new Dataset(ctx, input, data_flows_, index_, max_queue_size_,
                        variant_type_);

  std::string container;
  // OP_REQUIRES_OK(ctx, GetNodeAttr(def(), "container", &container));
  static_cast<Dataset *>(*output)->SetContainer("");
}

namespace {
REGISTER_KERNEL_BUILDER(Name("SplitFlowDataset").Device(DEVICE_CPU),
                        SplitFlowDatasetOp);
}  // namespace
}  // namespace monolith_tf
}  // namespace data
}  // namespace tensorflow
