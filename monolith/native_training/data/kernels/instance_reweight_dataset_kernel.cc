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

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/stats_aggregator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/str_util.h"

#include "monolith/native_training/data/kernels/instance_reweight_dataset_kernel.h"
#include "monolith/native_training/data/training_instance/cc/pb_variant.h"
#include "third_party/nlohmann/json.hpp"

namespace {
const unsigned int NONEXIST_PRIORITY = 2000;
const unsigned int UNKNOWN_PRIORITY = 1000;
}  // namespace

namespace tensorflow {
namespace data {
namespace monolith_tf {
using IFeature = ::idl::matrix::proto::Feature;
using Instance = ::parser::proto::Instance;
using Example = ::monolith::io::proto::Example;
using EFeature = ::monolith::io::proto::Feature;
using LineId = ::idl::matrix::proto::LineId;
using Action = google::protobuf::RepeatedField<int>;

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

/* static */ constexpr const char
    *const InstanceReweightDatasetOp::kDatasetType;
/* static */ constexpr const char
    *const InstanceReweightDatasetOp::kInputDataset;
/* static */ constexpr const char *const InstanceReweightDatasetOp::kMethod;
/* static */ constexpr const char *const InstanceReweightDatasetOp::kActions;
/* static */ constexpr const char *const InstanceReweightDatasetOp::kWeights;
/* static */ constexpr const char *const InstanceReweightDatasetOp::kLabels;
/* static */ constexpr const char *const InstanceReweightDatasetOp::kPriority;
/* static */ constexpr const char
    *const InstanceReweightDatasetOp::kVariantType;

class InnerIterator {
 public:
  InnerIterator(IteratorBase *input_impl, int instance_reweight_method,
                const std::vector<int32> &actions,
                const std::vector<int32> &weights,
                const std::vector<int32> &labels,
                const std::vector<int32> &priorities, std::string variant_type)
      : instance_reweight_method_(instance_reweight_method),
        variant_type_(std::string(variant_type)) {
    input_impl_.reset(input_impl);
    for (size_t i = 0; i < actions.size(); ++i) {
      reweight_[actions[i]] = weights[i];
      relabel_[actions[i]] = labels[i];
    }
    int idx = 1;
    for (const auto &value : priorities) {
      action_priority_[value] = idx++;
    }

    tensors_ = new std::vector<Tensor>();
    tensors_->reserve(1);
  }

  ~InnerIterator() { delete tensors_; }

  Status GetNext(IteratorContext *ctx, std::vector<Tensor> *out_tensors,
                 bool *end_of_sequence) {
    Status s = NextInternal(ctx);
    *end_of_sequence = end_of_sequence_;
    out_tensors->clear();
    if (s.ok() && !end_of_sequence_) {
      out_tensors->push_back(tensors_->back());
    }

    return s;
  }

 private:
  Status NextInternal(IteratorContext *ctx) {
    std::lock_guard<std::mutex> lck(mu_);
    while ((replicas_ == 0 || index_ == replicas_) && !end_of_sequence_) {
      tensors_->clear();
      TF_RETURN_IF_ERROR(
          input_impl_->GetNext(ctx, tensors_, &end_of_sequence_));
      if (!end_of_sequence_) {
        if (variant_type_ == "instance") {
          Instance *instance = GetCurrentInstance();
          if (instance->label_size()) {
            float *int_label = instance->mutable_label()->Mutable(0);
            replicas_ = CalReplicas(instance->line_id(), int_label);
          } else {
            replicas_ = 0;
            LOG_EVERY_N_SEC(ERROR, 60) << "label is empty, please investigate!";
          }
        } else if (variant_type_ == "example") {
          Example *example = GetCurrentExample();
          if (example->label_size()) {
            float *int_label = example->mutable_label()->Mutable(0);
            replicas_ = CalReplicas(example->line_id(), int_label);
          } else {
            replicas_ = 0;
            LOG_EVERY_N_SEC(ERROR, 60) << "label is empty, please investigate!";
          }
        } else {
          return errors::InvalidArgument(
              absl::StrCat(variant_type_, " variant_type is invalid!"));
        }
      } else {
        replicas_ = 0;
        return Status::OK();
      }
      index_ = 0;
    }

    index_++;
    return Status::OK();
  }

  inline Instance *GetCurrentInstance() {
    Variant *variant = &tensors_->back().scalar<Variant>()();
    return variant->get<Instance>();
  }

  inline Example *GetCurrentExample() {
    Variant *variant = &tensors_->back().scalar<Variant>()();
    return variant->get<Example>();
  }
  int CalReplicas(const LineId &lineid, float *ins_label) {
    // if priority is NONEXIST_PRIORITY or UNKNOWN_PRIORITY, action will be
    // undefined
    auto find_most_prior_action = [&](const Action &actions, int64_t *priority,
                                      int64_t *action) {
      *priority = NONEXIST_PRIORITY;
      if (actions.size() != 0) {
        *priority = UNKNOWN_PRIORITY;
        for (auto &act : actions) {
          auto action_iter = action_priority_.find(act);
          if (action_iter != action_priority_.end() &&
              action_iter->second < *priority) {
            *priority = action_iter->second;
            *action = act;
          }
        }
      }
    };
    auto get_pre_priority = [&]() {
      int64_t priority, action;
      find_most_prior_action(lineid.pre_actions(), &priority, &action);
      return priority;
    };
    auto get_cur_priority = [&]() {
      int64_t priority, action;
      find_most_prior_action(lineid.actions(), &priority, &action);
      return priority;
    };
    auto get_label = [&](const Action &actions, int64_t *label) {
      int64_t priority, action = 0;
      find_most_prior_action(actions, &priority, &action);
      if (priority != NONEXIST_PRIORITY && priority != UNKNOWN_PRIORITY) {
        auto label_iter = relabel_.find(action);
        if (label_iter != relabel_.end()) {
          *label = label_iter->second;
          return true;
        }
      }
      return false;
    };
    // return true if we can actually get the label
    auto get_pre_label = [&](int64_t *label) {
      return get_label(lineid.pre_actions(), label);
    };
    // return true if we can actually get the label & relabel if needed
    auto get_cur_label = [&](int64_t *label) {
      if (get_label(lineid.actions(), label)) {
        *ins_label = *label;
      }
      *label = *ins_label;
      return true;
    };
    auto get_cnt = [&](const Action &actions) {
      int64_t ins_num = actions.size() != 0 ? 1 : 0;
      int64_t priority, action = 0;
      find_most_prior_action(actions, &priority, &action);
      for (auto act : actions) {
        if (action_priority_.contains(act) && act != action) {
          continue;
        }
        // set the ins_num if the act need reweight.
        auto reweight_iter = reweight_.find(act);
        if (reweight_iter != reweight_.end()) {
          if (instance_reweight_method_ == 1) {
            ins_num += reweight_iter->second;
          } else {
            ins_num *= reweight_iter->second;
          }
        }
      }
      return ins_num;
    };
    auto get_pre_cnt = [&]() { return get_cnt(lineid.pre_actions()); };
    auto get_cur_cnt = [&]() { return get_cnt(lineid.actions()); };
    auto reverse_label = [&]() { *ins_label = -(*ins_label); };
    // start from here
    int64_t pre_priority = get_pre_priority();
    int64_t cur_priority = get_cur_priority();
    if (pre_priority > cur_priority) {
      int64_t pre_label;  // fast emit label, it's a negative sample
      int64_t cur_label;  // the real label, can be positive or negative
      // Note: the same sample with different label (+1/-1), equal there is no
      // sample
      if (get_cur_label(&cur_label) && get_pre_label(&pre_label)) {
        // for the real sample (the second one) of fast emit
        if (pre_label != cur_label) {  // for real positive sample
          // pre_cnt(-1) + pre_cnt(1) + cur_cnt(1) => cur_cnt(1)
          return get_pre_cnt() + get_cur_cnt();
        } else {  // the real negative sample
          auto pre_cnt = get_pre_cnt();
          auto cur_cnt = get_cur_cnt();
          if (pre_cnt > cur_cnt) {
            reverse_label();
            // pre_cnt(-1) + (pre_cnt(1) - cur_cnt(1)) => cur_cnt(-1)
            return pre_cnt - cur_cnt;
          } else {
            // pre_cnt(-1) + (cur_cnt(-1) - pre_cnt(-1))  => cur_cnt(-1)
            return cur_cnt - pre_cnt;
          }
        }
      } else {
        // for fast emit sample (the first one, negative) or non fast emit
        // sample
        return get_cur_cnt();
      }
    } else if (pre_priority == NONEXIST_PRIORITY &&
               cur_priority == NONEXIST_PRIORITY) {
      return 1;
    } else {
      return 0;
    }
  }

  std::mutex mu_;
  int index_ = 0;
  int replicas_ = 0;
  bool end_of_sequence_ = false;
  std::vector<Tensor> *tensors_ = nullptr;
  std::shared_ptr<IteratorBase> input_impl_ = nullptr;
  int instance_reweight_method_;
  absl::flat_hash_map<int, int> reweight_;
  absl::flat_hash_map<int, int> relabel_;
  absl::flat_hash_map<int, int> action_priority_;
  std::string variant_type_;
};

class InstanceReweightDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext *ctx, const DatasetBase *input,
          int instance_reweight_method, const std::vector<int32> &actions,
          const std::vector<int32> &weights, const std::vector<int32> &labels,
          const std::vector<int32> &priorities, std::string variant_type)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        instance_reweight_method_(instance_reweight_method),
        actions_(actions),
        weights_(weights),
        labels_(labels),
        priorities_(priorities),
        variant_type_(std::move(variant_type)) {
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
    return "This is the customized Dataset: InstanceReweight";
  }

  Status InputDatasets(
      std::vector<const DatasetBase *> *inputs) const override {
    inputs->push_back(input_);
    return Status::OK();
  }

  Status CheckExternalState() const override {
    return input_->CheckExternalState();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext *ctx,
                            DatasetGraphDefBuilder *b,
                            Node **output) const override {
    Node *input_graph_node;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));

    AttrValue method_node;
    b->BuildAttrValue(instance_reweight_method_, &method_node);
    AttrValue actions_node;
    b->BuildAttrValue(actions_, &actions_node);
    AttrValue weights_node;
    b->BuildAttrValue(weights_, &weights_node);
    AttrValue labels_node;
    b->BuildAttrValue(labels_, &labels_node);
    AttrValue priorities_node;
    b->BuildAttrValue(priorities_, &priorities_node);
    AttrValue variant_type_node;
    b->BuildAttrValue(variant_type_, &variant_type_node);

    TF_RETURN_IF_ERROR(
        b->AddDataset(this,                // dataset
                      {input_graph_node},  // inputs
                      {{kMethod, method_node},
                       {kActions, actions_node},
                       {kWeights, weights_node},
                       {kLabels, labels_node},
                       {kPriority, priorities_node},
                       {kVariantType, variant_type_node}},  // attrs
                      output));                             // Node**

    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params &params)
        : DatasetIterator<Dataset>(params) {}

    Status Initialize(IteratorContext *ctx) override {
      std::unique_ptr<IteratorBase> input_impl;
      Status s =
          dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl);
      LOG(INFO) << "Initialize InnerIterator ...";
      iter_ = std::make_unique<InnerIterator>(
          input_impl.release(), dataset()->instance_reweight_method_,
          dataset()->actions_, dataset()->weights_, dataset()->labels_,
          dataset()->priorities_, dataset()->variant_type_);
      return s;
    }

    Status GetNextInternal(IteratorContext *ctx,
                           std::vector<Tensor> *out_tensors,
                           bool *end_of_sequence) override {
      out_tensors->reserve(1);
      TF_RETURN_IF_ERROR(iter_->GetNext(ctx, out_tensors, end_of_sequence));
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
    std::unique_ptr<InnerIterator> iter_;
  };

  const DatasetBase *const input_;
  int instance_reweight_method_;
  std::vector<int32> actions_;
  std::vector<int32> weights_;
  std::vector<int32> labels_;
  std::vector<int32> priorities_;
  std::string variant_type_;
};

InstanceReweightDatasetOp::InstanceReweightDatasetOp(OpKernelConstruction *ctx)
    : UnaryDatasetOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kMethod, &instance_reweight_method_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kActions, &actions_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kWeights, &weights_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kLabels, &labels_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kPriority, &priorities_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kVariantType, &variant_type_));

  nlohmann::json j;
  j[kMethod] = instance_reweight_method_;
  j[kActions] = actions_;
  j[kWeights] = weights_;
  j[kLabels] = labels_;
  j[kPriority] = priorities_;
  j[kVariantType] = variant_type_;

  LOG(INFO) << j.dump();
}

void InstanceReweightDatasetOp::MakeDataset(OpKernelContext *ctx,
                                            DatasetBase *input,
                                            DatasetBase **output) {
  *output = new Dataset(ctx, input, instance_reweight_method_, actions_,
                        weights_, labels_, priorities_, variant_type_);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("InstanceReweightDataset").Device(DEVICE_CPU),
                        InstanceReweightDatasetOp);
}  // namespace

}  // namespace monolith_tf
}  // namespace data
}  // namespace tensorflow
