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

#include "absl/strings/str_cat.h"
#include "monolith/native_training/data/training_instance/cc/pb_variant.h"
#include "monolith/native_training/data/kernels/df_resource_kernel.h"
#include "tensorflow/core/framework/cancellation.h"
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

class MergeFlowDatasetOp : public DatasetOpKernel {
 public:
  static constexpr const char *const kDatasetType = "merge_dataset";
  static constexpr const char *const kDataFlow = "data_flow";
  static constexpr const char *const kMaxQueueSize = "max_queue_size";
  static constexpr const char *const kVariantType = "variant_type";

  explicit MergeFlowDatasetOp(OpKernelConstruction *ctx);

 protected:
  void MakeDataset(OpKernelContext *ctx, DatasetBase **output) override;

 private:
  class Dataset;
  std::vector<std::string> data_flows_;
  int max_queue_size_;
  VariantType variant_type_;
};

class MergeFlowDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext *ctx, const std::vector<const DatasetBase *> &inputs,
          const std::vector<std::string> &data_flows, int max_queue_size,
          const VariantType &variant_type)
      : DatasetBase(DatasetContext(ctx)),
        inputs_(inputs),
        data_flows_(data_flows),
        max_queue_size_(max_queue_size),
        variant_type_(variant_type) {
    for (const auto input : inputs_) {
      input->Ref();
    }
  }

  ~Dataset() override {
    for (const auto input : inputs_) {
      input->Unref();
    }
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string &prefix) const override {
    return absl::make_unique<Iterator>(
        Iterator::Params{this, strings::StrCat(prefix, "::", kDatasetType)});
  }

  const DataTypeVector &output_dtypes() const override {
    return inputs_[0]->output_dtypes();
  }

  const std::vector<PartialTensorShape> &output_shapes() const override {
    return inputs_[0]->output_shapes();
  }

  string DebugString() const override {
    return "This is the customized Dataset: DataFlowDataset";
  }

  Status InputDatasets(
      std::vector<const DatasetBase *> *inputs) const override {
    for (const auto input : inputs_) {
      inputs->push_back(input);
    }
    return Status::OK();
  }

  Status CheckExternalState() const override {
    for (const auto input : inputs_) {
      Status s = input->CheckExternalState();
      if (!s.ok()) {
        return s;
      }
    }

    return Status::OK();
  }

  void SetContainer(const std::string &container) { container_ = container; }

  std::string GetContainer() const { return container_; }

 protected:
  Status AsGraphDefInternal(SerializationContext *ctx,
                            DatasetGraphDefBuilder *b,
                            Node **output) const override {
    std::vector<Node *> input_graph_nodes;
    input_graph_nodes.reserve(inputs_.size());
    for (const auto &input : inputs_) {
      Node *input_node;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input, &input_node));
      input_graph_nodes.emplace_back(input_node);
    }

    AttrValue data_flows_node;
    b->BuildAttrValue(data_flows_, &data_flows_node);
    AttrValue max_queue_size_node;
    b->BuildAttrValue(max_queue_size_, &max_queue_size_node);

    AttrValue variant_type_node;
    if (variant_type_ == VariantType::PBInstance) {
      b->BuildAttrValue("instance", &variant_type_node);
    } else {
      b->BuildAttrValue("example", &variant_type_node);
    }

    TF_RETURN_IF_ERROR(
        b->AddDataset(this,                                        // dataset
                      {}, {std::make_pair(0, input_graph_nodes)},  // inputs
                      {{kDataFlow, data_flows_node},
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

      for (const std::string &name : dataset()->data_flows_) {
        auto iter = df_to_queue_.find(name);
        if (iter != df_to_queue_.end()) {
          if (iter->second != nullptr) {
            delete iter->second;
          }
          df_to_queue_.erase(iter);
        }
      }
    }

    void CancelThreads() TF_LOCKS_EXCLUDED(mu_) {
      cancellation_manager_->StartCancel();
      mutex_lock l(*mu_);
      cancelled_ = true;
    }

    Status Initialize(IteratorContext *ctx) override {
      mutex_lock l(*mu_);
      cancellation_manager_ = absl::make_unique<CancellationManager>();
      IteratorContext::Params params(ctx);
      params.cancellation_manager = cancellation_manager_.get();
      TF_RETURN_IF_ERROR(
          ::tensorflow::monolith_tf::RegisterCancellationCallback(
              ctx->cancellation_manager(), [this]() { CancelThreads(); },
              &deregister_fn_));

      Status s = Status::OK();
      input_impls_.reserve(dataset()->inputs_.size());
      for (const auto input : dataset()->inputs_) {
        std::unique_ptr<IteratorBase> input_impl;
        s.Update(input->MakeIterator(IteratorContext(params), this, prefix(),
                                     &input_impl));
        input_impls_.push_back(input_impl.release());
      }

      for (size_t i = 0; i < dataset()->data_flows_.size(); ++i) {
        std::string data_flows_name = dataset()->data_flows_[i];
        QueueResource *queue = new QueueResource(dataset()->max_queue_size_);
        df_to_queue_.emplace(data_flows_name, queue);
        prefetch_thread_finished_.push_back(false);
      }

      return s;
    }

    Status GetNextInternal(IteratorContext *ctx,
                           std::vector<Tensor> *out_tensors,
                           bool *end_of_sequence) override {
      out_tensors->reserve(1);
      {
        mutex_lock l(*mu_);
        TF_RETURN_IF_ERROR(EnsureThreadStarted(ctx));
      }

      {
        mutex_lock output_l(*output_mu_);
        do {
          for (size_t i = 0; i < dataset()->data_flows_.size(); ++i) {
            std::string name = dataset()->data_flows_[cur_];
            const QueueResource *queue = df_to_queue_[name];
            cur_ = (cur_ + 1) % dataset()->data_flows_.size();
            if (queue->Empty()) {
              continue;
            }

            Item item = queue->Pop();
            if (item.end_of_sequence) {
              out_tensors->clear();
              *end_of_sequence = true;
            } else {
              for (const auto &tensor : item.out_tensors) {
                out_tensors->push_back(tensor);
                Instance *inst =
                    out_tensors->at(0).scalar<Variant>()().get<Instance>();
                inst->mutable_line_id()->set_data_source_name(
                    absl::StrCat("data_source", inst->data_source_key()));
              }
              *end_of_sequence = item.end_of_sequence;
            }

            return Status::OK();
          }

          bool finished = true;
          for (bool f : prefetch_thread_finished_) {
            finished = finished && f;
          }

          if (cancelled_ || finished) {
            out_tensors->clear();
            *end_of_sequence = true;
            break;
          }
        } while (true);
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
    std::vector<bool> prefetch_thread_finished_ TF_GUARDED_BY(*mu_);

    size_t cur_ = 0;
    std::vector<IteratorBase *> input_impls_;
    std::vector<Thread *> prefetch_threads_;
    std::unordered_map<std::string, QueueResource *> df_to_queue_;

    Status EnsureThreadStarted(IteratorContext *ctx)
        TF_EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      if (!prefetch_thread_started_) {
        prefetch_thread_started_ = true;
        for (size_t i = 0; i < dataset()->data_flows_.size(); ++i) {
          std::string name = dataset()->data_flows_[i];
          std::shared_ptr<IteratorContext> new_ctx =
              std::make_shared<IteratorContext>(*ctx);
          std::unique_ptr<Thread> prefetch_thread_ = ctx->StartThread(
              name,
              [new_ctx, i, name, this]() { PrefetchThread(new_ctx, i, name); });
          prefetch_threads_.push_back(prefetch_thread_.release());
        }
      }
      return Status::OK();
    }

    void PrefetchThread(const std::shared_ptr<IteratorContext> &ctx, size_t i,
                        std::string name) {
      while (true) {
        {
          mutex_lock l(*mu_);
          if (cancelled_) {
            prefetch_thread_finished_[i] = true;
            break;
          }
        }

        if (!prefetch_thread_finished_[i]) {
          Item item;
          input_impls_[i]->GetNext(ctx.get(), &item.out_tensors,
                                   &item.end_of_sequence);
          if (item.end_of_sequence) {
            prefetch_thread_finished_[i] = true;
            break;
          }

          bool pushed = false;
          do {
            if (cancelled_ || prefetch_thread_finished_[i]) {
              break;
            }
            pushed = df_to_queue_[name]->TryPush(item);
          } while (!pushed);
        }
      }
    }
  };

  const std::vector<const DatasetBase *> inputs_;
  std::vector<std::string> data_flows_;
  int max_queue_size_;
  VariantType variant_type_;
  std::string container_;
};

MergeFlowDatasetOp::MergeFlowDatasetOp(OpKernelConstruction *ctx)
    : DatasetOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kDataFlow, &data_flows_));
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

void MergeFlowDatasetOp::MakeDataset(OpKernelContext *ctx,
                                     DatasetBase **output) {
  OpInputList iplist;
  OP_REQUIRES_OK(ctx, ctx->input_list("inputs", &iplist));
  std::vector<const DatasetBase *> inputs;
  for (const auto &ds : iplist) {
    DatasetBase *input;
    OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ds, &input));
    inputs.push_back(input);
  }

  *output =
      new Dataset(ctx, inputs, data_flows_, max_queue_size_, variant_type_);

  std::string container;
  // OP_REQUIRES_OK(ctx, GetNodeAttr(def(), "container", &container));
  static_cast<Dataset *>(*output)->SetContainer("");
}

namespace {
REGISTER_KERNEL_BUILDER(Name("MergeFlowDataset").Device(DEVICE_CPU),
                        MergeFlowDatasetOp);
}  // namespace
}  // namespace monolith_tf
}  // namespace data
}  // namespace tensorflow
