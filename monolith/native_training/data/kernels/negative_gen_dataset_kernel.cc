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

#include <deque>
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
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

#include "monolith/native_training/data/kernels/item_pool_kernels.h"
#include "monolith/native_training/data/kernels/negative_gen_dataset_kernel.h"
#include "monolith/native_training/data/training_instance/cc/pb_variant.h"
#include "monolith/native_training/data/training_instance/cc/reader_util.h"

namespace tensorflow {
namespace data {
namespace monolith_tf {
using IFeature = ::idl::matrix::proto::Feature;
using Instance = ::parser::proto::Instance;
using LineId = ::idl::matrix::proto::LineId;
using Action = google::protobuf::RepeatedField<int>;
using Label = google::protobuf::RepeatedField<float>;
using Example = ::monolith::io::proto::Example;
using ::tensorflow::monolith_tf::FeatureNameMapper;
using ::tensorflow::monolith_tf::FeatureNameMapperTfBridge;
using EFeature = ::monolith::io::proto::NamedFeature;
using ItemPoolResource = ::tensorflow::monolith_tf::ItemPoolResource;
using ItemFeatures = ::tensorflow::monolith_tf::internal::ItemFeatures;

static const int32 INVALID_NEGATIVE_ACTION = -99999;
constexpr char kInputImplEmpty[] = "input_impl_empty";

static constexpr const char *const kDatasetType = "negtive_gen_dataset";
static constexpr const char *const kNegNum = "neg_num";
static constexpr const char *const kPerChannel = "per_channel";
static constexpr const char *const kChannelFeature = "channel_feature";
static constexpr const char *const kItemFeature = "item_features";
static constexpr const char *const kLabelIndex = "label_index";
static constexpr const char *const kPositiveLabel = "positive_label";
static constexpr const char *const kNegativeLabel = "negative_label";
static constexpr const char *const kNegativeAction = "negative_action";
static constexpr const char *const kActionPriority = "action_priority";
static constexpr const char *const kPositiveActions = "positive_actions";
static constexpr const char *const kCacheOnlyPos = "cache_only_pos";
static constexpr const char *const kIndexFeature = "index_feature";
static constexpr const char *const kThrowOrigin = "throw_origin";
static constexpr const char *const kThrowOriginNeg = "throw_origin_neg";
static constexpr const char *const kRealNegInstanceWeight =
    "real_neg_instance_weight";
static constexpr const char *const kSampledNegInstanceWeight =
    "sampled_neg_instance_weight";
static constexpr const char *const kUnbiasSampledNeg = "unbias_sampled_neg";
static constexpr const char *const kOriginNegInPoolProba =
    "origin_neg_in_pool_proba";
static constexpr const char *const kNegSampleDeclayFactor =
    "neg_sample_declay_factor";
static constexpr const char *const kHardEasyRatio = "hard_easy_ratio";
static constexpr const char *const kVariantType = "variant_type";

class InnerIterator {
 public:
  InnerIterator(IteratorBase *input_impl, ItemPoolResource *resource,
                int32 neg_num, bool per_channel,
                const std::string &channel_feature,
                const std::vector<std::string> &item_features,
                int32 label_index, int32 positive_label, int32 negative_label,
                int32 negative_action, const std::string &action_priority,
                const std::vector<int32> &positive_actions,
                const std::string &index_feature, bool throw_origin,
                bool throw_origin_neg, bool cache_only_pos,
                float real_neg_instance_weight,
                float sampled_neg_instance_weight, bool unbias_sampled_neg,
                float origin_neg_in_pool_proba, float neg_sample_declay_factor,
                float hard_easy_ratio, const std::string &variant_type)
      : resource_(resource),
        index_(0),
        need_new_ins_(true),
        input_real_negative_instance_num_(0),
        input_instance_num_(0),
        output_instance_num_(0),
        generate_instance_num_(0),
        hard_sample_num_(0),
        easy_sample_num_(0),
        neg_num_(neg_num),
        per_channel_(per_channel),
        channel_feature_(channel_feature),
        item_features_(item_features.begin(), item_features.end()),
        label_index_(label_index),
        positive_label_(positive_label),
        negative_label_(negative_label),
        negative_action_(negative_action),
        positive_actions_(positive_actions.begin(), positive_actions.end()),
        index_feature_(index_feature),
        throw_origin_(throw_origin),
        throw_origin_neg_(throw_origin_neg),
        cache_only_pos_(cache_only_pos),
        real_neg_instance_weight_(real_neg_instance_weight),
        sampled_neg_instance_weight_(sampled_neg_instance_weight),
        unbias_sampled_neg_(unbias_sampled_neg),
        origin_neg_in_pool_proba_(origin_neg_in_pool_proba),
        neg_sample_declay_factor_(neg_sample_declay_factor),
        hard_easy_ratio_(hard_easy_ratio) {
    input_impl_ = input_impl;
    tensors_ = new std::vector<Tensor>();
    tensors_->reserve(1);
    std::vector<absl::string_view> action_priority_items =
        absl::StrSplit(action_priority, ",");
    for (size_t i = 0; i < action_priority_items.size(); ++i) {
      int32 action;
      if (action_priority_items[i].empty()) {
        continue;
      }
      CHECK(absl::SimpleAtoi(action_priority_items[i], &action));
      action_priority_.insert({action, static_cast<int32>(i)});
    }

    if (variant_type == "instance") {
      variant_type_ = VariantType::PBInstance;
      if (index_feature_.empty()) {
        has_index_feature_ = false;
        index_slot_ = 0;
      } else {
        has_index_feature_ = true;
        CHECK(absl::SimpleAtoi(index_feature_, &index_slot_));
      }

      if (channel_feature_.empty()) {
        channel_slot_ = 3;
      } else {
        CHECK(absl::SimpleAtoi(channel_feature_, &channel_slot_));
      }

      for (const auto &fname : item_features_) {
        int32 slot;
        CHECK(absl::SimpleAtoi(fname, &slot));
        item_slots_.insert(slot);
      }
    } else {
      variant_type_ = VariantType::PBExample;
      index_slot_ = 0;
      has_index_feature_ = !index_feature_.empty();
      channel_slot_ = 3;
    }
  }

  ~InnerIterator() { delete tensors_; }

  Status GetNext(IteratorContext *ctx, std::vector<Tensor> *out_tensors,
                 bool *end_of_sequence) {
    if (end_of_sequence_) {
      *end_of_sequence = end_of_sequence_;
      out_tensors->clear();
    }

    while (!end_of_sequence_) {
      Status s = MaybeGetNextRealInstance(ctx);
      if (!s.ok()) {
        return s;
      }

      if (end_of_sequence_) {
        *end_of_sequence = end_of_sequence_;
        out_tensors->clear();
        break;
      }

      bool is_positive = IsPositive();
      if (index_ == 0 && Cacheable(is_positive)) {
        SaveToCache(is_positive);
      }
      if (index_ == 0 && !is_positive) {
        input_real_negative_instance_num_++;
      }

      if (is_positive && index_ < neg_num_) {
        Tensor tensor;
        if (BuildNegativeTensor(ctx, &tensor)) {
          *end_of_sequence = end_of_sequence_;
          out_tensors->push_back(std::move(tensor));
          index_++;
          generate_instance_num_++;
          output_instance_num_++;
          break;
        }
      }

      need_new_ins_ = true;
      if (Emitable(is_positive)) {
        *end_of_sequence = end_of_sequence_;
        if (is_positive) {
          SetInstanceWeight(&tensors_->back(), 1.0);
        } else {
          float instance_weight = real_neg_instance_weight_ > 0.00001
                                      ? real_neg_instance_weight_
                                      : 1.0;
          SetInstanceWeight(&tensors_->back(), instance_weight);
        }
        out_tensors->push_back(tensors_->back());
        output_instance_num_++;
        break;
      }
    }

    LOG_EVERY_N_SEC(INFO, 180) << "input_instance_num: " << input_instance_num_;
    LOG_EVERY_N_SEC(INFO, 180) << "input_real_negative_instance_num: "
                               << input_real_negative_instance_num_;
    LOG_EVERY_N_SEC(INFO, 180) << "output_instance_num: "
                               << output_instance_num_;
    LOG_EVERY_N_SEC(INFO, 180) << "generate_instance_num: "
                               << generate_instance_num_;
    LOG_EVERY_N_SEC(INFO, 180) << "hard_sample_num: " << hard_sample_num_;
    LOG_EVERY_N_SEC(INFO, 180) << "easy_sample_num: " << easy_sample_num_;
    return Status::OK();
  }

 private:
  Status MaybeGetNextRealInstance(IteratorContext *ctx) {
    if (need_new_ins_ && !end_of_sequence_) {
      tensors_->clear();
      TF_RETURN_IF_ERROR(
          input_impl_->GetNext(ctx, tensors_, &end_of_sequence_));
      if (end_of_sequence_) {
        input_impl_ = nullptr;
        return Status::OK();
      }
      need_new_ins_ = false;
      gcids_ = GetGidAndChannelId();
      ++input_instance_num_;
      index_ = 0;  // Got one new instance, reset the neg count
    }
    return Status::OK();
  }

  template <typename T>
  inline T *GetCurrent() {
    Variant *variant = &tensors_->back().scalar<Variant>()();
    return variant->get<T>();
  }

  inline const LineId *GetLineId() {
    if (variant_type_ == VariantType::PBInstance) {
      Instance *instance = GetCurrent<Instance>();
      return &instance->line_id();
    } else if (variant_type_ == VariantType::PBExample) {
      Example *example = GetCurrent<Example>();
      return &example->line_id();
    } else {
      return nullptr;
    }
  }

  inline const Label *GetLabel() {
    if (variant_type_ == VariantType::PBInstance) {
      Instance *instance = GetCurrent<Instance>();
      return &instance->label();
    } else if (variant_type_ == VariantType::PBExample) {
      Example *example = GetCurrent<Example>();
      return &example->label();
    } else {
      return nullptr;
    }
  }

  bool IsPositive() {
    bool is_pos = false;
    const LineId *line_id = GetLineId();
    const Label *label = GetLabel();

    if (!positive_actions_.empty() && line_id != nullptr) {
      if (!line_id->actions().empty()) {
        int64_t action;
        FindMostPriorAction(line_id->actions(), &action);
        auto iter = positive_actions_.find(action);
        is_pos = iter != positive_actions_.end();
      }
    } else if (label != nullptr) {
      if (label_index_ < label->size()) {
        is_pos = label->at(label_index_) == positive_label_;
      } else {
        LOG_EVERY_N_SEC(ERROR, 60) << absl::StrFormat(
            "label_index_ should be less than label_size, while got %d vs %d",
            label_index_, label->size());
      }
    }

    return is_pos;
  }

  inline bool Cacheable(bool is_positive) {
    return (!cache_only_pos_ || is_positive);
  }

  inline bool Emitable(bool is_positive) {
    return (!throw_origin_ && (!throw_origin_neg_ || is_positive));
  }

  std::pair<uint64_t, uint64_t> GetGidAndChannelId() {
    uint64_t gid = 0, cid = 3;
    if (variant_type_ == VariantType::PBInstance) {
      const Instance *instance = GetCurrent<Instance>();
      gid = instance->line_id().item_id();
      if (per_channel_ || has_index_feature_) {
        for (const auto fid : instance->fid()) {
          int32 slot = slot_id_v1(fid);
          if (per_channel_ && channel_slot_ == slot) {
            cid = fid;
          }
          if (has_index_feature_ && slot == index_slot_) {
            gid = fid;
          }
        }
      }
    } else if (variant_type_ == VariantType::PBExample) {
      const Example *example = GetCurrent<Example>();
      gid = example->line_id().item_id();
      if (per_channel_ || has_index_feature_) {
        for (const auto &named_feature : example->named_feature()) {
          const std::string &feature_name = named_feature.name();
          auto &feature_value = named_feature.feature();
          if (per_channel_ && channel_feature_ == feature_name) {
            if (feature_value.type_case() ==
                    ::monolith::io::proto::Feature::kFidV1List &&
                feature_value.fid_v1_list().value_size() > 0) {
              cid = feature_value.fid_v1_list().value(0);
              LOG_EVERY_N_SEC(INFO, 180) << "Use Fidv1.";
            } else if (feature_value.type_case() ==
                           ::monolith::io::proto::Feature::kFidV2List &&
                       feature_value.fid_v2_list().value_size() > 0) {
              cid = feature_value.fid_v2_list().value(0);
              LOG_EVERY_N_SEC(INFO, 180) << "Use Fidv2.";
            } else {
              LOG_EVERY_N_SEC(INFO, 180) << "Use Default cid.";
            }
          }
          if (has_index_feature_ && index_feature_ == feature_name) {
            if (feature_value.type_case() ==
                    ::monolith::io::proto::Feature::kFidV1List &&
                feature_value.fid_v1_list().value_size() > 0) {
              gid = feature_value.fid_v1_list().value(0);
            } else if (feature_value.type_case() ==
                           ::monolith::io::proto::Feature::kFidV2List &&
                       feature_value.fid_v2_list().value_size() > 0) {
              gid = feature_value.fid_v2_list().value(0);
            }
          }
        }
      }
    }

    return std::pair<uint64_t, uint64_t>(gid, cid);
  }

  void SetInstanceWeight(Tensor *tensor, float instance_weight) {
    if (variant_type_ == VariantType::PBInstance) {
      auto *instance = tensor->scalar<Variant>()().get<Instance>();
      instance->set_instance_weight(instance_weight);
    } else {
      auto *example = tensor->scalar<Variant>()().get<Example>();
      example->set_instance_weight(instance_weight);
    }
  }

  void SaveToCache(bool is_positive) {
    std::shared_ptr<ItemFeatures> item_features =
        std::make_shared<ItemFeatures>();
    uint64_t item_id = gcids_.first;
    uint64_t channel_id = gcids_.second;

    if (channel_id == 0) {
      return;
    }

    if (!is_positive && origin_neg_in_pool_proba_ >= 0 &&
        origin_neg_in_pool_proba_ < 1) {
      float proba = (std::rand() % 100) / 100.0;
      if (proba > origin_neg_in_pool_proba_) {
        return;
      }
    }

    if (variant_type_ == VariantType::PBExample) {
      Example *example = GetCurrent<Example>();
      if (is_positive) {
        named_feature_list_.Clear();
      }
      for (auto &named_feature : example->named_feature()) {
        const std::string &feature_name = named_feature.name();
        if (item_features_.count(feature_name) != 0) {
          item_features->example_features[feature_name] = named_feature;
        } else if (is_positive) {
          named_feature_list_.Add(named_feature);
        }
      }
    } else {
      Instance *instance = GetCurrent<Instance>();
      if (is_positive) {
        fid_list_.Clear();
      }
      for (auto fid : instance->fid()) {
        int32 slot = slot_id_v1(fid);
        if (item_slots_.count(slot) != 0) {  // only cache group slots
          item_features->fids.emplace_back(fid);
        } else if (is_positive) {
          fid_list_.Add(fid);
        }
      }
    }

    item_features->item_id = item_id;
    resource_->Add(channel_id, item_id, item_features);
  }

  template <typename T>
  void SetLabelAndLineId(T *neg, uint64_t item_id) {
    if (label_index_ < neg->label_size()) {
      neg->set_label(label_index_, negative_label_);
    } else {
      LOG_EVERY_N_SEC(ERROR, 60) << absl::StrFormat(
          "label_index_ should be less than label_size, while got %d vs %d",
          label_index_, neg->label_size());
    }

    neg->mutable_line_id()->set_item_id(item_id);
    if (negative_action_ != INVALID_NEGATIVE_ACTION) {
      neg->mutable_line_id()->clear_actions();
      neg->mutable_line_id()->add_actions(negative_action_);
    }
  }

  bool BuildNegativeTensor(IteratorContext *ctx, Tensor *res) {
    // hard_easy neg when per_channel enabled
    uint64_t channel_id = gcids_.second;
    if (per_channel_ && NeedEasyNeg(hard_easy_ratio_)) {
      resource_->SampleChannelID(&channel_id);
      easy_sample_num_++;
    } else {
      hard_sample_num_++;
    }

    if (channel_id == 0) {
      return false;
    }
    double freq_factor, time_factor;
    std::shared_ptr<const ItemFeatures> cached_item =
        resource_->Sample(channel_id, &freq_factor, &time_factor);
    if (!cached_item) {
      return false;
    }
    uint64_t item_id = cached_item->item_id;
    Tensor tensor(ctx->allocator({}), DT_VARIANT, TensorShape({}));

    float instance_weight;
    if (sampled_neg_instance_weight_ > 0.00001) {
      instance_weight = sampled_neg_instance_weight_;
    } else if (unbias_sampled_neg_) {
      instance_weight = 1.0 +
                        neg_num_ *
                            std::pow(time_factor, neg_sample_declay_factor_) *
                            freq_factor;
    } else {
      instance_weight = 1.0;
    }

    if (variant_type_ == VariantType::PBExample) {
      Example *example = GetCurrent<Example>();
      Example new_example;
      new_example.mutable_line_id()->CopyFrom(example->line_id());
      for (const auto &label : example->label()) {
        new_example.add_label(label);
      }

      const auto &cached_example_features = cached_item->example_features;
      auto *mutable_named_feature = new_example.mutable_named_feature();
      for (const auto &nf : named_feature_list_) {
        mutable_named_feature->Add()->CopyFrom(nf);
      }
      for (const auto &nf : cached_example_features) {
        mutable_named_feature->Add()->CopyFrom(nf.second);
      }
      SetLabelAndLineId(&new_example, item_id);
      new_example.set_instance_weight(instance_weight);
      tensor.scalar<Variant>()() = std::move(new_example);
    } else {
      Instance *instance = GetCurrent<Instance>();
      Instance new_instance;
      new_instance.CopyFrom(*instance);

      const auto &cached_fid_list = cached_item->fids;
      google::protobuf::RepeatedField<::google::protobuf::uint64> fid_list =
          fid_list_;  // copy
      for (auto fid : cached_fid_list) {
        fid_list.Add(fid);
      }
      new_instance.mutable_fid()->Swap(&fid_list);
      SetLabelAndLineId(&new_instance, item_id);
      new_instance.set_instance_weight(instance_weight);
      tensor.scalar<Variant>()() = std::move(new_instance);
    }

    *res = std::move(tensor);
    return true;
  }

  bool FindMostPriorAction(const Action &actions, int64_t *action) {
    if (actions.size() != 0) {
      if (action_priority_.empty() || actions.size() == 1) {
        *action = actions[0];
      } else {
        int64_t priority = std::numeric_limits<int64_t>::max();
        for (auto &act : actions) {
          auto iter = action_priority_.find(act);
          if (iter != action_priority_.end() && iter->second < priority) {
            *action = iter->first;
            priority = iter->second;
          }
        }

        if (priority == std::numeric_limits<int64_t>::max())
          *action = actions[0];
      }
      return true;
    }

    return false;
  }

  bool NeedEasyNeg(float hard_easy_ratio) {
    return static_cast<float>(std::rand()) / RAND_MAX < hard_easy_ratio;
  }

  ItemPoolResource *resource_ = nullptr;
  bool end_of_sequence_ = false;
  std::vector<Tensor> *tensors_ = nullptr;
  IteratorBase *input_impl_ = nullptr;
  int index_ = 0;
  bool need_new_ins_ = true;
  // stats variables
  int64 input_real_negative_instance_num_ = 0;
  int64 input_instance_num_ = 0;
  int64 output_instance_num_ = 0;
  int64 generate_instance_num_ = 0;
  // hard & easy stats
  int64 hard_sample_num_ = 0;
  int64 easy_sample_num_ = 0;

  int32 neg_num_;
  bool per_channel_;
  std::string channel_feature_;
  int32 channel_slot_;
  std::unordered_set<std::string> item_features_;
  std::unordered_set<int32> item_slots_;
  int32 positive_label_;
  int32 negative_label_;
  int32 negative_action_;
  std::unordered_set<int32> positive_actions_;
  int32 label_index_;
  std::unordered_map<int32, int32> action_priority_;
  std::string index_feature_;
  int32 index_slot_;
  bool has_index_feature_;
  bool throw_origin_;
  bool throw_origin_neg_;
  bool cache_only_pos_;
  float real_neg_instance_weight_;
  float sampled_neg_instance_weight_;
  bool unbias_sampled_neg_;
  float origin_neg_in_pool_proba_;
  float neg_sample_declay_factor_;
  float hard_easy_ratio_;
  VariantType variant_type_;

  std::pair<uint64_t, uint64_t> gcids_;
  google::protobuf::RepeatedField<::google::protobuf::uint64> fid_list_;
  google::protobuf::RepeatedField<::monolith::io::proto::NamedFeature>
      named_feature_list_;
};

class InstanceNegativeGenDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext *ctx, const DatasetBase *input, int32 neg_num,
          bool per_channel, const std::string &channel_feature,
          const std::vector<std::string> &item_features, int32 label_index,
          int32 positive_label, int32 negative_label, int32 negative_action,
          std::string action_priority,
          const std::vector<int32> &positive_actions,
          const std::string &index_feature, bool throw_origin,
          bool throw_origin_neg, bool cache_only_pos,
          float real_neg_instance_weight, float sampled_neg_instance_weight,
          bool unbias_sampled_neg, float origin_neg_in_pool_proba,
          float neg_sample_declay_factor, float hard_easy_ratio,
          const std::string &variant_type, FeatureNameMapper *mapper)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        neg_num_(neg_num),
        per_channel_(per_channel),
        channel_feature_(channel_feature),
        item_features_(item_features),
        label_index_(label_index),
        positive_label_(positive_label),
        negative_label_(negative_label),
        negative_action_(negative_action),
        action_priority_(action_priority),
        positive_actions_(positive_actions),
        index_feature_(index_feature),
        throw_origin_(throw_origin),
        throw_origin_neg_(throw_origin_neg),
        cache_only_pos_(cache_only_pos),
        real_neg_instance_weight_(real_neg_instance_weight),
        sampled_neg_instance_weight_(sampled_neg_instance_weight),
        unbias_sampled_neg_(unbias_sampled_neg),
        origin_neg_in_pool_proba_(origin_neg_in_pool_proba),
        neg_sample_declay_factor_(neg_sample_declay_factor),
        hard_easy_ratio_(hard_easy_ratio),
        variant_type_(variant_type),
        mapper_(mapper) {
    input_->Ref();

    const Tensor *pool_tensor_;
    OP_REQUIRES_OK(ctx, ctx->input("pool", &pool_tensor_));
    handle_ = pool_tensor_->scalar<ResourceHandle>()();
    OP_REQUIRES_OK(ctx, LookupResource(ctx, handle_, &resource_));

    if (variant_type_ == "example") {
      std::vector<std::string> valid_feature_names = item_features_;
      if (!channel_feature_.empty()) {
        valid_feature_names.push_back(channel_feature_);
      }
      if (!index_feature_.empty()) {
        valid_feature_names.push_back(index_feature_);
      }
      mapper_->RegisterValidNames(valid_feature_names);
    }
  }

  ~Dataset() override {
    input_->Unref();
    core::ScopedUnref unref(resource_);
  }

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
    return "This is the customized Dataset: NegativeGenV2";
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

    Tensor handle(DT_RESOURCE, TensorShape({}));
    handle.scalar<ResourceHandle>()() = handle_;
    Node *pool_node;
    TF_RETURN_IF_ERROR(b->AddTensor(handle, &pool_node));

    AttrValue neg_num_node;
    b->BuildAttrValue(neg_num_, &neg_num_node);

    AttrValue per_channel_node;
    b->BuildAttrValue(per_channel_, &per_channel_node);

    AttrValue channel_feature_node;
    b->BuildAttrValue(channel_feature_, &channel_feature_node);

    AttrValue item_features_node;
    b->BuildAttrValue(item_features_, &item_features_node);

    AttrValue label_index_node;
    b->BuildAttrValue(label_index_, &label_index_node);

    AttrValue positive_label_node;
    b->BuildAttrValue(positive_label_, &positive_label_node);

    AttrValue negative_label_node;
    b->BuildAttrValue(negative_label_, &negative_label_node);

    AttrValue negative_action_node;
    b->BuildAttrValue(negative_action_, &negative_action_node);

    AttrValue action_priority_node;
    b->BuildAttrValue(action_priority_, &action_priority_node);

    AttrValue positive_actions_node;
    b->BuildAttrValue(positive_actions_, &positive_actions_node);

    AttrValue index_feature_node;
    b->BuildAttrValue(index_feature_, &index_feature_node);

    AttrValue throw_origin_node;
    b->BuildAttrValue(throw_origin_, &throw_origin_node);

    AttrValue throw_origin_neg_node;
    b->BuildAttrValue(throw_origin_neg_, &throw_origin_neg_node);

    AttrValue cache_only_pos_node;
    b->BuildAttrValue(cache_only_pos_, &cache_only_pos_node);

    AttrValue real_neg_instance_weight_node;
    b->BuildAttrValue(real_neg_instance_weight_,
                      &real_neg_instance_weight_node);

    AttrValue sampled_neg_instance_weight_node;
    b->BuildAttrValue(sampled_neg_instance_weight_,
                      &sampled_neg_instance_weight_node);

    AttrValue unbias_sampled_neg_node;
    b->BuildAttrValue(unbias_sampled_neg_, &unbias_sampled_neg_node);

    AttrValue origin_neg_in_pool_proba_node;
    b->BuildAttrValue(origin_neg_in_pool_proba_,
                      &origin_neg_in_pool_proba_node);

    AttrValue neg_sample_declay_factor_node;
    b->BuildAttrValue(neg_sample_declay_factor_,
                      &neg_sample_declay_factor_node);

    AttrValue hard_easy_ratio_node;
    b->BuildAttrValue(hard_easy_ratio_, &hard_easy_ratio_node);

    AttrValue variant_type_node;
    b->BuildAttrValue(variant_type_, &variant_type_node);

    TF_RETURN_IF_ERROR(b->AddDataset(
        this,                           // dataset
        {input_graph_node, pool_node},  // inputs
        {{kNegNum, neg_num_node},
         {kPerChannel, per_channel_node},
         {kChannelFeature, channel_feature_node},
         {kItemFeature, item_features_node},
         {kLabelIndex, label_index_node},
         {kPositiveLabel, positive_label_node},
         {kNegativeLabel, negative_label_node},
         {kNegativeAction, negative_action_node},
         {kActionPriority, action_priority_node},
         {kPositiveActions, positive_actions_node},
         {kIndexFeature, index_feature_node},
         {kThrowOrigin, throw_origin_node},
         {kThrowOriginNeg, throw_origin_neg_node},
         {kCacheOnlyPos, cache_only_pos_node},
         {kRealNegInstanceWeight, real_neg_instance_weight_node},
         {kSampledNegInstanceWeight, sampled_neg_instance_weight_node},
         {kUnbiasSampledNeg, unbias_sampled_neg_node},
         {kOriginNegInPoolProba, origin_neg_in_pool_proba_node},
         {kNegSampleDeclayFactor, neg_sample_declay_factor_node},
         {kHardEasyRatio, hard_easy_ratio_node},
         {kVariantType, variant_type_node}},
        output));  // Node**

    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params &params)
        : DatasetIterator<Dataset>(params) {}

    ~Iterator() override {
      mutex_lock l(mu_);
      if (input_impl_ != nullptr) {
        input_impl_.reset();
      }
    }

    Status Initialize(IteratorContext *ctx) override {
      Status s =
          dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_);
      iter_ = std::make_unique<InnerIterator>(
          input_impl_.get(), dataset()->resource_, dataset()->neg_num_,
          dataset()->per_channel_, dataset()->channel_feature_,
          dataset()->item_features_, dataset()->label_index_,
          dataset()->positive_label_, dataset()->negative_label_,
          dataset()->negative_action_, dataset()->action_priority_,
          dataset()->positive_actions_, dataset()->index_feature_,
          dataset()->throw_origin_, dataset()->throw_origin_neg_,
          dataset()->cache_only_pos_, dataset()->real_neg_instance_weight_,
          dataset()->sampled_neg_instance_weight_,
          dataset()->unbias_sampled_neg_, dataset()->origin_neg_in_pool_proba_,
          dataset()->neg_sample_declay_factor_, dataset()->hard_easy_ratio_,
          dataset()->variant_type_);
      return s;
    }

    Status GetNextInternal(IteratorContext *ctx,
                           std::vector<Tensor> *out_tensors,
                           bool *end_of_sequence) override {
      mutex_lock l(mu_);
      out_tensors->reserve(1);
      TF_RETURN_IF_ERROR(iter_->GetNext(ctx, out_tensors, end_of_sequence));
      if (*end_of_sequence) {
        input_impl_.reset();
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
      mutex_lock l(mu_);
      if (!input_impl_) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kInputImplEmpty), ""));
      } else {
        TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
      }
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext *ctx,
                           IteratorStateReader *reader) override {
      mutex_lock l(mu_);
      if (!reader->Contains(full_name(kInputImplEmpty))) {
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
      } else {
        input_impl_.reset();
      }
      return Status::OK();
    }

   private:
    mutex mu_;
    std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
    std::unique_ptr<InnerIterator> iter_;
  };

  const DatasetBase *const input_;
  int32 neg_num_;
  bool per_channel_;
  std::string channel_feature_;
  std::vector<std::string> item_features_;
  int32 positive_label_;
  int32 negative_label_;
  int32 negative_action_;
  std::vector<int32> positive_actions_;
  int32 label_index_;
  std::string action_priority_;
  std::string index_feature_;
  bool throw_origin_;
  bool throw_origin_neg_;
  bool cache_only_pos_;
  float real_neg_instance_weight_;
  float sampled_neg_instance_weight_;
  bool unbias_sampled_neg_;
  float origin_neg_in_pool_proba_;
  float neg_sample_declay_factor_;
  float hard_easy_ratio_;
  std::string variant_type_;

  ResourceHandle handle_;
  ItemPoolResource *resource_;
  FeatureNameMapper *mapper_ = nullptr;
};

InstanceNegativeGenDatasetOp::InstanceNegativeGenDatasetOp(
    OpKernelConstruction *ctx)
    : UnaryDatasetOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kNegNum, &neg_num_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kPerChannel, &per_channel_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kChannelFeature, &channel_feature_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kItemFeature, &item_features_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kLabelIndex, &label_index_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kPositiveLabel, &positive_label_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kNegativeLabel, &negative_label_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kNegativeAction, &negative_action_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kPositiveActions, &positive_actions_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kActionPriority, &action_priority_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kIndexFeature, &index_feature_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kThrowOrigin, &throw_origin_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kThrowOriginNeg, &throw_origin_neg_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kCacheOnlyPos, &cache_only_pos_));
  OP_REQUIRES_OK(
      ctx, ctx->GetAttr(kRealNegInstanceWeight, &real_neg_instance_weight_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kSampledNegInstanceWeight,
                                   &sampled_neg_instance_weight_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kUnbiasSampledNeg, &unbias_sampled_neg_));
  OP_REQUIRES_OK(
      ctx, ctx->GetAttr(kOriginNegInPoolProba, &origin_neg_in_pool_proba_));
  OP_REQUIRES_OK(
      ctx, ctx->GetAttr(kNegSampleDeclayFactor, &neg_sample_declay_factor_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kHardEasyRatio, &hard_easy_ratio_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kVariantType, &variant_type_));

  auto creator = [this](FeatureNameMapperTfBridge **out_mapper) {
    TF_RETURN_IF_ERROR(FeatureNameMapperTfBridge::New(out_mapper));
    return Status::OK();
  };
  ResourceMgr *resource_mgr = ctx->resource_manager();
  OP_REQUIRES_OK(ctx, resource_mgr->LookupOrCreate<FeatureNameMapperTfBridge>(
                          resource_mgr->default_container(),
                          FeatureNameMapperTfBridge::kName, &mapper_, creator));
}

void InstanceNegativeGenDatasetOp::MakeDataset(OpKernelContext *ctx,
                                               DatasetBase *input,
                                               DatasetBase **output) {
  *output = new Dataset(
      ctx, input, neg_num_, per_channel_, channel_feature_, item_features_,
      label_index_, positive_label_, negative_label_, negative_action_,
      action_priority_, positive_actions_, index_feature_, throw_origin_,
      throw_origin_neg_, cache_only_pos_, real_neg_instance_weight_,
      sampled_neg_instance_weight_, unbias_sampled_neg_,
      origin_neg_in_pool_proba_, neg_sample_declay_factor_, hard_easy_ratio_,
      variant_type_, mapper_->GetFeatureNameMapper());
}

namespace {
REGISTER_KERNEL_BUILDER(Name("InstanceNegativeGenDataset").Device(DEVICE_CPU),
                        InstanceNegativeGenDatasetOp);
}  // namespace

}  // namespace monolith_tf
}  // namespace data
}  // namespace tensorflow
