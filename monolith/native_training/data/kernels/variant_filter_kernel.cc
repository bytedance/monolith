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

#include <climits>
#include <cstdio>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_format.h"
#include "monolith/native_training/data/kernels/feature_name_mapper_tf_bridge.h"
#include "monolith/native_training/data/kernels/internal/relational_utils.h"
#include "monolith/native_training/data/kernels/internal/value_filter_by_feature.h"
#include "monolith/native_training/data/kernels/internal/value_filter_by_line_id.h"
#include "monolith/native_training/data/training_instance/cc/instance_utils.h"
#include "monolith/native_training/data/training_instance/cc/pb_variant.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/variant.h"
#include "third_party/nlohmann/json.hpp"

namespace tensorflow {
namespace monolith_tf {

using IFeature = ::idl::matrix::proto::Feature;
using Instance = ::parser::proto::Instance;
using Example = ::monolith::io::proto::Example;
using LineId = ::idl::matrix::proto::LineId;
using tensorflow::monolith_tf::internal::LineIdValueFilter;
using tensorflow::monolith_tf::internal::FeatureValueFilter;

class SetFilterOp : public OpKernel {
 public:
  explicit SetFilterOp(OpKernelConstruction *context) : OpKernel(context) {
    std::vector<int64> filter_fids;
    OP_REQUIRES_OK(context, context->GetAttr("filter_fids", &filter_fids));
    filter_fids_.insert(filter_fids.begin(), filter_fids.end());

    std::vector<int64> has_fids;
    OP_REQUIRES_OK(context, context->GetAttr("has_fids", &has_fids));
    has_fids_.insert(has_fids.begin(), has_fids.end());

    std::vector<int64> select_fids;
    OP_REQUIRES_OK(context, context->GetAttr("select_fids", &select_fids));
    select_fids_.insert(select_fids.begin(), select_fids.end());

    std::vector<int64> has_actions;
    OP_REQUIRES_OK(context, context->GetAttr("has_actions", &has_actions));
    has_actions_.insert(has_actions.begin(), has_actions.end());

    OP_REQUIRES_OK(context, context->GetAttr("variant_type", &variant_type_));

    OP_REQUIRES_OK(context, context->GetAttr("req_time_min", &req_time_min_));

    std::vector<int32_t> select_slots;
    OP_REQUIRES_OK(context, context->GetAttr("select_slots", &select_slots));
    for (int32_t slot : select_slots) {
      CHECK_GE(slot, 0);
    }
    select_slots_.insert(select_slots.begin(), select_slots.end());

    auto creator = [this](FeatureNameMapperTfBridge **out_mapper) {
      TF_RETURN_IF_ERROR(FeatureNameMapperTfBridge::New(out_mapper));
      return Status::OK();
    };
    ResourceMgr *resource_mgr = context->resource_manager();
    OP_REQUIRES_OK(context,
                   resource_mgr->LookupOrCreate<FeatureNameMapperTfBridge>(
                       resource_mgr->default_container(),
                       FeatureNameMapperTfBridge::kName, &mapper_, creator));
    if (variant_type_ == "example") {
      std::vector<std::pair<int, int>> valid_ids;
      for (uint32_t slot : select_slots_) {
        valid_ids.emplace_back(slot, slot);
      }
      for (uint64_t fid : filter_fids_) {
        valid_ids.emplace_back(slot_id_v1(fid), slot_id_v2(fid));
      }
      for (uint64_t fid : has_fids_) {
        valid_ids.emplace_back(slot_id_v1(fid), slot_id_v2(fid));
      }
      for (uint64_t fid : select_fids_) {
        valid_ids.emplace_back(slot_id_v1(fid), slot_id_v2(fid));
      }

      OP_REQUIRES_OK(context, mapper_->RegisterValidIds(valid_ids));
    }
  }

  ~SetFilterOp() override { mapper_->Unref(); }

  void Compute(OpKernelContext *context) override {
    const Tensor &input_tensor = context->input(0);
    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(0, input_tensor.shape(), &output_tensor));
    auto output = output_tensor->scalar<bool>();
    output() = IsInstanceOfInterest(input_tensor);
  }

 private:
  bool IsInstanceOfInterest(const Tensor &input_tensor) {
    auto input = input_tensor.scalar<Variant>();
    if (variant_type_ == "instance") {
      const auto *instance = input().get<Instance>();
      return monolith_tf::IsInstanceOfInterest(
          *instance, filter_fids_, has_fids_, select_fids_, has_actions_,
          req_time_min_, select_slots_);
    } else {
      const auto *example = input().get<Example>();
      return monolith_tf::IsInstanceOfInterest(
          *example, filter_fids_, has_fids_, select_fids_, has_actions_,
          req_time_min_, select_slots_);
    }
  }

  std::set<uint64_t> filter_fids_;
  std::set<uint64_t> has_fids_;
  std::set<uint64_t> select_fids_;
  std::set<int32_t> has_actions_;
  std::string variant_type_ = "instance";
  int req_time_min_;
  std::set<uint32_t> select_slots_;
  FeatureNameMapperTfBridge *mapper_;
};

class FeatureValueFilterOp : public OpKernel {
 public:
  explicit FeatureValueFilterOp(OpKernelConstruction *context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("field_name", &field_name_));
    OP_REQUIRES_OK(context, context->GetAttr("op", &op_));
    OP_REQUIRES_OK(context, context->GetAttr("float_operand", &float_operand_));
    OP_REQUIRES_OK(context, context->GetAttr("int_operand", &int_operand_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("string_operand", &string_operand_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("operand_filepath", &operand_filepath_));
    OP_REQUIRES_OK(context, context->GetAttr("keep_empty", &keep_empty_));
    OP_REQUIRES_OK(context, context->GetAttr("field_type", &field_type_));
    OP_REQUIRES(context,
                field_type_ == "int64" || field_type_ == "float" ||
                    field_type_ == "double" || field_type_ == "bytes",
                errors::Unknown(
                    "field_type unknown! need to be int64/float/double/bytes"));
    feature_value_filter_ = std::make_unique<FeatureValueFilter>(
        field_name_, field_type_, op_, float_operand_, int_operand_,
        string_operand_, operand_filepath_, keep_empty_);
  }

  void Compute(OpKernelContext *context) override {
    const Tensor &input_tensor = context->input(0);
    const Variant &variant = input_tensor.scalar<Variant>()();
    OP_REQUIRES(context, variant.TypeId() == TypeIndex::Make<Example>(),
                errors::InvalidArgument("input must be Example proto"));
    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(0, input_tensor.shape(), &output_tensor));
    auto output = output_tensor->scalar<bool>();
    // only support Example input
    output() = feature_value_filter_->IsInstanceOfInterest(
        context->env(), *(input_tensor.scalar<Variant>()().get<Example>()));
  }

 private:
  std::string field_name_;
  std::string op_;  // gt, ge, eq, lt, le, neq, between
  bool keep_empty_ = false;
  std::string operand_filepath_;

  std::vector<float> float_operand_;
  std::vector<int64> int_operand_;
  std::vector<std::string> string_operand_;

  std::unique_ptr<FeatureValueFilter> feature_value_filter_;
  std::string field_type_;
};

class ValueFilterOp : public OpKernel {
 public:
  explicit ValueFilterOp(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("field_name", &field_name_));
    OP_REQUIRES_OK(context, context->GetAttr("op", &op_));
    OP_REQUIRES_OK(context, context->GetAttr("float_operand", &float_operand_));
    OP_REQUIRES_OK(context, context->GetAttr("int_operand", &int_operand_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("string_operand", &string_operand_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("operand_filepath", &operand_filepath_));
    OP_REQUIRES_OK(context, context->GetAttr("keep_empty", &keep_empty_));
    OP_REQUIRES_OK(context, context->GetAttr("variant_type", &variant_type_));
    line_id_value_filter_ = std::make_unique<LineIdValueFilter>(
        field_name_, op_, float_operand_, int_operand_, string_operand_,
        operand_filepath_, keep_empty_);
  }

  void Compute(OpKernelContext *context) override {
    const Tensor &input_tensor = context->input(0);
    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(0, input_tensor.shape(), &output_tensor));
    auto output = output_tensor->scalar<bool>();
    const LineId &line_id = GetLineId(input_tensor);
    output() =
        line_id_value_filter_->IsInstanceOfInterest(context->env(), line_id);
  }

 private:
  const LineId &GetLineId(const Tensor &input_tensor) {
    if (variant_type_ == "instance") {
      return input_tensor.scalar<Variant>()().get<Instance>()->line_id();
    } else {
      return input_tensor.scalar<Variant>()().get<Example>()->line_id();
    }
  }

  std::string field_name_;
  std::string op_;  // gt, ge, eq, lt, le, neq, between
  bool keep_empty_ = false;
  std::string operand_filepath_;

  std::vector<float> float_operand_;
  std::vector<int64> int_operand_;
  std::vector<std::string> string_operand_;

  std::unique_ptr<LineIdValueFilter> line_id_value_filter_;

  std::string variant_type_;
};

class SpecialStrategyOp : public OpKernel {
 public:
  explicit SpecialStrategyOp(OpKernelConstruction *context)
      : OpKernel(context) {
    std::vector<int> special_strategy;
    OP_REQUIRES_OK(context,
                   context->GetAttr("special_strategies", &special_strategy));
    std::vector<float> sample_rate;
    OP_REQUIRES_OK(context, context->GetAttr("sample_rates", &sample_rate));
    std::vector<float> label;
    OP_REQUIRES_OK(context, context->GetAttr("labels", &label));

    OP_REQUIRES(
        context, special_strategy.size() == sample_rate.size(),
        errors::InvalidArgument(
            "length of sample_rates must identity with special_strategies"));
    OP_REQUIRES(
        context, special_strategy.size() == label.size() || label.size() == 0,
        errors::InvalidArgument(
            "length of labels must identity with special_strategies or zero"));

    for (size_t i = 0; i < special_strategy.size(); ++i) {
      strategy_to_rate_.emplace(special_strategy[i], sample_rate[i]);
    }

    if (label.size() > 0) {
      for (size_t i = 0; i < special_strategy.size(); ++i) {
        strategy_to_label_.emplace(special_strategy[i], label[i]);
      }
    }

    OP_REQUIRES_OK(context, context->GetAttr("strategy_list", &strategy_list_));
    OP_REQUIRES(context, strategy_list_.size() > 0,
                errors::InvalidArgument("strategy_list cannot be empty"));

    OP_REQUIRES_OK(
        context,
        context->GetAttr("keep_empty_strategy", &keep_empty_strategy_));
    OP_REQUIRES_OK(context, context->GetAttr("variant_type", &variant_type_));
  }

  void Compute(OpKernelContext *context) override {
    Tensor *input_tensor = const_cast<Tensor *>(&(context->input(0)));
    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(0, input_tensor->shape(), &output_tensor));
    auto output = output_tensor->scalar<bool>();
    output() = DoCompute(input_tensor);
  }

 private:
  const LineId &GetLineId(Tensor *input_tensor) {
    if (variant_type_ == "instance") {
      return input_tensor->scalar<Variant>()().get<Instance>()->line_id();
    } else {
      return input_tensor->scalar<Variant>()().get<Example>()->line_id();
    }
  }

  float *GetLabel(Tensor *input_tensor, int index = 0) {
    if (variant_type_ == "instance") {
      return input_tensor->scalar<Variant>()()
          .get<Instance>()
          ->mutable_label()
          ->Mutable(index);
    } else {
      return input_tensor->scalar<Variant>()()
          .get<Example>()
          ->mutable_label()
          ->Mutable(index);
    }
  }

  bool DoCompute(Tensor *input_tensor) {
    const LineId &line_id = GetLineId(input_tensor);
    const auto &strategies = line_id.special_strategies();

    if (strategies.size() > 1) {
      LOG(INFO) << "Size of special_strategies is bigger than one, pls. check!";
    }

    if (strategies.size() == 0) {
      // for unknow samples, drop
      if (keep_empty_strategy_) {
        // for special_strategies_neg_ins_keep_normal
        return true;
      } else {
        return false;
      }
    } else {
      for (auto &special_strategy : strategy_list_) {
        auto found =
            std::find(strategies.begin(), strategies.end(), special_strategy);
        if (found != strategies.end()) {
          auto rit = strategy_to_rate_.find(special_strategy);
          if (rit != strategy_to_rate_.end()) {
            float rate = rit->second;
            bool flag = false;
            if (rate == 1.0) {
              flag = true;
            } else {
              if (random_neg_sample_(generator_) <= rate) {
                flag = true;
              }
            }

            if (strategy_to_label_.size() > 0 && flag) {
              auto lit = strategy_to_label_.find(special_strategy);
              if (lit != strategy_to_label_.end()) {
                float new_label = lit->second;
                float *old_label = GetLabel(input_tensor);
                *old_label = new_label;
              }
            }

            return flag;
          } else {
            return true;
          }
        }
      }
    }
  }

  std::default_random_engine generator_;
  std::uniform_real_distribution<float> random_neg_sample_;

  bool keep_empty_strategy_ = true;
  std::unordered_map<int, float> strategy_to_rate_;
  std::unordered_map<int, float> strategy_to_label_;
  std::vector<int> strategy_list_;
  std::string variant_type_;
};

class NegativeSampleOp : public OpKernel {
 public:
  explicit NegativeSampleOp(OpKernelConstruction *context) : OpKernel(context) {
    std::vector<int> priorities;
    std::vector<int> actions;
    std::vector<float> per_action_drop_rate;
    OP_REQUIRES_OK(context, context->GetAttr("drop_rate", &drop_rate_));
    OP_REQUIRES_OK(context, context->GetAttr("label_index", &label_index_));
    OP_REQUIRES_OK(context, context->GetAttr("threshold", &threshold_));
    OP_REQUIRES_OK(context, context->GetAttr("priorities", &priorities));
    OP_REQUIRES_OK(context, context->GetAttr("actions", &actions));
    OP_REQUIRES_OK(context, context->GetAttr("per_action_drop_rate",
                                             &per_action_drop_rate));
    OP_REQUIRES_OK(context, context->GetAttr("variant_type", &variant_type_));

    OP_REQUIRES(context, actions.size() == per_action_drop_rate.size(),
                errors::Unknown("internal error"));

    for (size_t i = 0; i < actions.size(); i++) {
      action_drop_rate_map_.emplace(actions[i], per_action_drop_rate[i]);
    }
    for (size_t i = 0; i < priorities.size(); i++) {
      action_priorities_map_.emplace(priorities[i], i);
    }
    if (actions.size() > 0) {
      enable_drop_by_action_ = true;
    }
  }

  void Compute(OpKernelContext *context) override {
    const Tensor &input_tensor = context->input(0);
    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(0, input_tensor.shape(), &output_tensor));
    auto output = output_tensor->scalar<bool>();
    float label = GetLabel(input_tensor);

    if (label < threshold_) {
      float sample_drop_rate = drop_rate_;
      if (enable_drop_by_action_) {
        sample_drop_rate = GetNegDropRate(input_tensor);
      }
      thread_local std::mt19937 gen((std::random_device())());
      float random = gen() % 1000 / 1000.0;
      output() = random < sample_drop_rate ? false : true;
    } else {
      output() = true;
    }
  }

 private:
  float drop_rate_ = 0.0;
  int label_index_ = 0;
  float threshold_ = 0.0;
  bool enable_drop_by_action_ = false;
  std::unordered_map<int, int> action_priorities_map_;
  std::unordered_map<int, float> action_drop_rate_map_;
  std::string variant_type_ = "instance";

  float GetLabel(const Tensor &input_tensor) {
    auto input = input_tensor.scalar<Variant>();
    if (variant_type_ == "instance") {
      const Instance *instance = input().get<Instance>();
      return instance->label(label_index_);
    } else {
      const Example *example = input().get<Example>();
      return example->label(label_index_);
    }

    return 0;
  }

  const LineId *GetLineId(const Tensor &input_tensor) {
    auto input = input_tensor.scalar<Variant>();
    if (variant_type_ == "instance") {
      const Instance *instance = input().get<Instance>();
      return &instance->line_id();
    } else {
      const Example *example = input().get<Example>();
      return &example->line_id();
    }
  }

  int FindMostPriorAction(const Tensor &input_tensor) {
    const LineId *line_id = GetLineId(input_tensor);
    CHECK(line_id != nullptr);
    int most_prior = INT_MAX;
    int record_action = -1;
    for (int action : line_id->actions()) {
      auto it = action_priorities_map_.find(action);
      if (it != action_priorities_map_.end() && it->second < most_prior) {
        most_prior = it->second;
        record_action = action;
      }
    }
    return record_action;
  }

  float GetNegDropRate(const Tensor &input_tensor) {
    int prior_action = FindMostPriorAction(input_tensor);
    if (prior_action > 0) {
      auto it = action_drop_rate_map_.find(prior_action);
      if (it != action_drop_rate_map_.end()) {
        return it->second;
      }
    }
    return drop_rate_;
  }
};

namespace {
REGISTER_KERNEL_BUILDER(Name("SetFilter").Device(DEVICE_CPU), SetFilterOp);

REGISTER_KERNEL_BUILDER(Name("FeatureValueFilter").Device(DEVICE_CPU),
                        FeatureValueFilterOp);

REGISTER_KERNEL_BUILDER(Name("ValueFilter").Device(DEVICE_CPU), ValueFilterOp);

REGISTER_KERNEL_BUILDER(Name("SpecialStrategy").Device(DEVICE_CPU),
                        SpecialStrategyOp);

REGISTER_KERNEL_BUILDER(Name("NegativeSample").Device(DEVICE_CPU),
                        NegativeSampleOp);

}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow
