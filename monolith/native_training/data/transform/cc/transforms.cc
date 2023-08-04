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

#include "monolith/native_training/data/transform/cc/transforms.h"

#include <random>
#include <utility>

#include "absl/base/internal/cycleclock.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "glog/logging.h"
#include "monolith/native_training/data/kernels/internal/label_utils.h"
#include "monolith/native_training/data/kernels/internal/line_id_value_filter.h"
#include "monolith/native_training/data/kernels/internal/relational_utils.h"
#include "monolith/native_training/data/training_instance/cc/instance_utils.h"
#include "monolith/native_training/runtime/common/linalg_utils.h"
#include "tensorflow/core/platform/logging.h"
#include "third_party/nlohmann/json.hpp"

namespace tensorflow {
namespace monolith_tf {

using ::google::protobuf::RepeatedField;
using ::idl::matrix::proto::LineId;
using internal::LineIdValueFilter;
using ::monolith::common::IsAlmostEqual;
using ::monolith::io::proto::Example;
using ::monolith::io::proto::ExampleBatch;
using ::parser::proto::Instance;

class LogEveryNSecState {
 public:
  bool ShouldLog(double seconds) {
    LossyIncrement(&counter_);
    const int64 now_cycles = absl::base_internal::CycleClock::Now();
    int64 next_cycles = next_log_time_cycles_.load(std::memory_order_relaxed);
    do {
      if (now_cycles <= next_cycles) return false;
    } while (!next_log_time_cycles_.compare_exchange_weak(
        next_cycles,
        now_cycles + seconds * absl::base_internal::CycleClock::Frequency(),
        std::memory_order_relaxed, std::memory_order_relaxed));
    return true;
  }

  uint32 counter() { return counter_.load(std::memory_order_relaxed); }

 private:
  // The following code behaves like AtomicStatsCounter::LossyAdd() for
  // speed since it is fine to lose occasional updates.
  // Returns old value of *counter.
  uint32 LossyIncrement(std::atomic<uint32>* counter) {
    const uint32 value = counter->load(std::memory_order_relaxed);
    counter->store(value + 1, std::memory_order_relaxed);
    return value;
  }

  std::atomic<uint32> counter_{0};
  // Cycle count according to CycleClock that we should next log at.
  std::atomic<int64> next_log_time_cycles_{0};
};

class TransformSummary : public TransformInterface {
 public:
  explicit TransformSummary(std::unique_ptr<TransformInterface> transform,
                            bool print_summary = false)
      : transform_(std::move(transform)),
        offset_(0),
        input_total_(0),
        output_total_(0) {}

  ~TransformSummary() override {
    LOG(INFO) << "Finally " << this->DebugString();
  }

  std::string Name() override { return transform_->Name(); }

  std::string DebugString() {
    float rate = input_total_ == 0
                     ? 0
                     : static_cast<float>(output_total_) / input_total_;
    return absl::StrFormat(
        "%s, input = %ld, output = %ld, retention rate = %.2f",
        transform_->Name(), input_total_, output_total_, rate);
  }

  void Transform(std::shared_ptr<Instance> instance,
                 std::vector<std::shared_ptr<Instance>>* output) override {
    offset_ = output->size();
    input_total_ += 1;
    transform_->Transform(instance, output);
    output_total_ += output->size() - offset_;
    if (every_n_sec_state_.ShouldLog(60 * 5)) {
      LOG(INFO) << DebugString();
    }
  }

  void Transform(std::shared_ptr<Example> example,
                 std::vector<std::shared_ptr<Example>>* output) override {
    offset_ = output->size();
    input_total_ += 1;
    transform_->Transform(example, output);
    output_total_ += output->size() - offset_;
    if (every_n_sec_state_.ShouldLog(60 * 5)) {
      LOG(INFO) << DebugString();
    }
  }

 private:
  std::unique_ptr<TransformInterface> transform_;

  int64_t offset_;

  int64_t input_total_;

  int64_t output_total_;

  LogEveryNSecState every_n_sec_state_;
};

class Identity : public TransformInterface {
 public:
  std::string Name() override { return "Identity"; }

  void Transform(std::shared_ptr<Instance> instance,
                 std::vector<std::shared_ptr<Instance>>* output) override {
    output->push_back(instance);
  }

  void Transform(std::shared_ptr<Example> example,
                 std::vector<std::shared_ptr<Example>>* output) override {
    output->push_back(example);
  }
};

class FilterByFid : public TransformInterface {
 public:
  explicit FilterByFid(FilterByFidConfig config) : config_(std::move(config)) {
    filter_fids_.insert(config_.filter_fids().begin(),
                        config_.filter_fids().end());
    has_fids_.insert(config_.has_fids().begin(), config_.has_fids().end());
    select_fids_.insert(config_.select_fids().begin(),
                        config_.select_fids().end());
    req_time_min_ = 0;
  }

  std::string Name() override { return "FilterByFid"; }

  void Transform(std::shared_ptr<Instance> instance,
                 std::vector<std::shared_ptr<Instance>>* output) override {
    if (tensorflow::monolith_tf::IsInstanceOfInterest(*instance, filter_fids_,
                                                      has_fids_, select_fids_,
                                                      {}, req_time_min_, {})) {
      output->push_back(instance);
    }
  }

  void Transform(std::shared_ptr<Example> example,
                 std::vector<std::shared_ptr<Example>>* output) override {
    if (tensorflow::monolith_tf::IsInstanceOfInterest(*example, filter_fids_,
                                                      has_fids_, select_fids_,
                                                      {}, req_time_min_, {})) {
      output->push_back(example);
    }
  }

 private:
  std::set<uint64_t> filter_fids_;
  std::set<uint64_t> has_fids_;
  std::set<uint64_t> select_fids_;
  int64_t req_time_min_;
  FilterByFidConfig config_;
};

class FilterByAction : public TransformInterface {
 public:
  explicit FilterByAction(FilterByActionConfig config)
      : config_(std::move(config)) {
    has_actions_.insert(config_.has_actions().begin(),
                        config_.has_actions().end());
  }

  std::string Name() override { return "FilterByAction"; }

  void Transform(std::shared_ptr<Instance> instance,
                 std::vector<std::shared_ptr<Instance>>* output) override {
    if (tensorflow::monolith_tf::IsInstanceOfInterest(*instance, {}, {}, {},
                                                      has_actions_, 0, {})) {
      output->push_back(instance);
    }
  }

  void Transform(std::shared_ptr<Example> example,
                 std::vector<std::shared_ptr<Example>>* output) override {
    if (tensorflow::monolith_tf::IsInstanceOfInterest(*example, {}, {}, {},
                                                      has_actions_, 0, {})) {
      output->push_back(example);
    }
  }

 private:
  std::set<int32_t> has_actions_;
  FilterByActionConfig config_;
};

class FilterByLabel : public TransformInterface {
 public:
  explicit FilterByLabel(FilterByLabelConfig config)
      : config_(std::move(config)) {}

  std::string Name() override { return "FilterByLabel"; }

  void Transform(std::shared_ptr<Instance> instance,
                 std::vector<std::shared_ptr<Instance>>* output) override {
    if (IsInstanceOfInterest(instance->label())) {
      output->push_back(instance);
    }
  }

  void Transform(std::shared_ptr<Example> example,
                 std::vector<std::shared_ptr<Example>>* output) override {
    if (IsInstanceOfInterest(example->label())) {
      output->push_back(example);
    }
  }

 private:
  bool IsInstanceOfInterest(const RepeatedField<float>& labels) const {
    if (labels.size() < config_.thresholds_size()) {
      LOG_EVERY_N_SEC(ERROR, 60) << absl::StrFormat(
          "Label size(=%ld) should be >= label_threshold size(=%ld), please "
          "investigate!",
          labels.size(), config_.thresholds_size());
      return false;
    }

    for (int i = 0; i < config_.thresholds_size(); ++i) {
      if (labels.Get(i) >= config_.thresholds(i)) {
        return true;
      }
    }

    return false;
  }

  FilterByLabelConfig config_;
};

class FilterByValue : public TransformInterface {
 public:
  explicit FilterByValue(FilterByValueConfig config)
      : config_(std::move(config)) {
    field_name_ = config_.field_name();
    op_ = config_.op();
    float_operand_.insert(float_operand_.end(), config_.float_operand().begin(),
                          config_.float_operand().end());
    int_operand_.insert(int_operand_.end(), config_.int_operand().begin(),
                        config_.int_operand().end());
    string_operand_.insert(string_operand_.end(),
                           config_.string_operand().begin(),
                           config_.string_operand().end());
    keep_empty_ = config_.keep_empty();
    operand_filepath_ = config_.operand_filepath();
    line_id_value_filter_ = std::make_unique<LineIdValueFilter>(
        field_name_, op_, float_operand_, int_operand_, string_operand_,
        operand_filepath_, keep_empty_);
  }

  std::string Name() override { return "FilterByValue"; }

  void Transform(std::shared_ptr<Instance> instance,
                 std::vector<std::shared_ptr<Instance>>* output) override {
    if (IsInstanceOfInterest(instance->line_id())) {
      output->push_back(instance);
    }
  }

  void Transform(std::shared_ptr<Example> example,
                 std::vector<std::shared_ptr<Example>>* output) override {
    if (IsInstanceOfInterest(example->line_id())) {
      output->push_back(example);
    }
  }

 private:
  bool IsInstanceOfInterest(const LineId& line_id) const {
    tensorflow::Env* env = tensorflow::Env::Default();
    return line_id_value_filter_->IsInstanceOfInterest(env, line_id);
  }

  FilterByValueConfig config_;

  std::string field_name_;
  std::string op_;  // gt, ge, eq, lt, le, neq, between
  bool keep_empty_ = false;
  std::string operand_filepath_;

  std::vector<float> float_operand_;
  std::vector<int64> int_operand_;
  std::vector<std::string> string_operand_;

  std::unique_ptr<LineIdValueFilter> line_id_value_filter_;
};

class AddLabel : public TransformInterface {
 public:
  explicit AddLabel(AddLabelConfig config) : config_(std::move(config)) {
    task_configs_.reserve(config_.task_label_configs_size());
    for (const auto& t : config_.task_label_configs()) {
      std::set<int32_t> pos_actions, neg_actions;
      CHECK(!t.pos_actions().empty());
      pos_actions.insert(t.pos_actions().begin(), t.pos_actions().end());
      neg_actions.insert(t.neg_actions().begin(), t.neg_actions().end());

      CHECK(!internal::HasIntersection(pos_actions, neg_actions));

      float sample_rate = t.sample_rate();
      CHECK_GE(sample_rate, 0);
      CHECK_LE(sample_rate, 1.0);

      task_configs_.push_back({pos_actions, neg_actions, sample_rate});
    }

    for (size_t i = 0; i < task_configs_.size(); ++i) {
      LOG(INFO) << absl::StrFormat("Task #%d config: %s", i + 1,
                                   task_configs_[i].ToString());
    }
    LOG(INFO) << absl::StrFormat("sample_rate = %.4f",
                                 config_.new_sample_rate());
    std::size_t seed =
        std::chrono::system_clock::now().time_since_epoch().count();
    random_generator_.seed(seed);
    random_neg_sample_ = std::uniform_real_distribution<float>(0.0, 1.0);
  }

  std::string Name() override { return "AddLabel"; }

  void Transform(std::shared_ptr<Instance> instance,
                 std::vector<std::shared_ptr<Instance>>* output) override {
    DoAddLabel(instance->mutable_line_id(), instance->mutable_label());
    output->push_back(instance);
  }

  void Transform(std::shared_ptr<Example> example,
                 std::vector<std::shared_ptr<Example>>* output) override {
    DoAddLabel(example->mutable_line_id(), example->mutable_label());
    output->push_back(example);
  }

 private:
  void DoAddLabel(LineId* mutable_line_id,
                  google::protobuf::RepeatedField<float>* mutable_label) {
    std::set<int32_t> actions(mutable_line_id->actions().begin(),
                              mutable_line_id->actions().end());

    if (!mutable_label->empty() && mutable_label->Get(0) <= 0) {
      mutable_label->Set(0, internal::INVALID_LABEL);
    }

    for (const auto& t : task_configs_) {
      bool has_pos = internal::HasIntersection(actions, t.pos_actions);
      bool has_neg = internal::HasIntersection(actions, t.neg_actions);

      if (!t.neg_actions.empty()) {
        // If there is given neg_actions
        if (!has_pos && !has_neg) {
          mutable_label->Add(internal::INVALID_LABEL);
        } else if (has_pos) {
          // (has_pos && !has_neg) || (has_pos && has_neg)
          mutable_label->Add(internal::POSITIVE_LABEL);
        } else {
          // !has_pos && has_neg
          if (SelectedByNegativeSampling(t)) {
            mutable_label->Add(config_.negative_value());
          } else {
            mutable_label->Add(internal::INVALID_LABEL);
          }
        }
      } else {
        // If there is no given neg_actions
        if (has_pos) {
          mutable_label->Add(internal::POSITIVE_LABEL);
        } else {
          if (SelectedByNegativeSampling(t)) {
            mutable_label->Add(config_.negative_value());
          } else {
            mutable_label->Add(internal::INVALID_LABEL);
          }
        }
      }
    }

    mutable_line_id->set_sample_rate(config_.new_sample_rate());
  }

  bool SelectedByNegativeSampling(const internal::TaskConfig& t) {
    return IsAlmostEqual(t.sample_rate, 1.0f) ||
           random_neg_sample_(random_generator_) < t.sample_rate;
  }

  std::vector<internal::TaskConfig> task_configs_;

  std::default_random_engine random_generator_;

  std::uniform_real_distribution<float> random_neg_sample_;

  AddLabelConfig config_;
};

class LogicalOrTransform : public TransformInterface {
 public:
  LogicalOrTransform(std::unique_ptr<TransformInterface> t1,
                     std::unique_ptr<TransformInterface> t2)
      : t1_(std::move(t1)), t2_(std::move(t2)) {}

  std::string Name() override {
    return absl::StrFormat("(%s or %s)", t1_->Name(), t2_->Name());
  }

  void Transform(std::shared_ptr<Instance> instance,
                 std::vector<std::shared_ptr<Instance>>* output) override {
    std::vector<std::shared_ptr<Instance>> intermediates;
    t1_->Transform(instance, &intermediates);
    t2_->Transform(instance, &intermediates);
    if (!intermediates.empty()) {
      CHECK_LE(intermediates.size(), 2);
      output->push_back(intermediates.front());
    }
  }

  void Transform(std::shared_ptr<Example> example,
                 std::vector<std::shared_ptr<Example>>* output) override {
    std::vector<std::shared_ptr<Example>> intermediates;
    t1_->Transform(example, &intermediates);
    t2_->Transform(example, &intermediates);
    if (!intermediates.empty()) {
      CHECK_LE(intermediates.size(), 2);
      output->push_back(intermediates.front());
    }
  }

 private:
  std::unique_ptr<TransformInterface> t1_;
  std::unique_ptr<TransformInterface> t2_;
};

class CombinedTransform : public TransformInterface {
 public:
  CombinedTransform(std::unique_ptr<TransformInterface> t1,
                    std::unique_ptr<TransformInterface> t2)
      : t1_(std::move(t1)), t2_(std::move(t2)) {}

  std::string Name() override {
    return absl::StrFormat("(%s and %s)", t1_->Name(), t2_->Name());
  }

  void Transform(std::shared_ptr<Instance> instance,
                 std::vector<std::shared_ptr<Instance>>* output) override {
    std::vector<std::shared_ptr<Instance>> intermediates;
    t1_->Transform(instance, &intermediates);
    for (const auto& intermediate : intermediates) {
      t2_->Transform(intermediate, output);
    }
  }

  void Transform(std::shared_ptr<Example> example,
                 std::vector<std::shared_ptr<Example>>* output) override {
    std::vector<std::shared_ptr<Example>> intermediates;
    t1_->Transform(example, &intermediates);
    for (const auto& intermediate : intermediates) {
      t2_->Transform(intermediate, output);
    }
  }

 private:
  std::unique_ptr<TransformInterface> t1_;
  std::unique_ptr<TransformInterface> t2_;
};

std::unique_ptr<TransformInterface> NewTransformSummary(
    std::unique_ptr<TransformInterface> transform, bool print_summary) {
  return std::make_unique<TransformSummary>(std::move(transform),
                                            print_summary);
}

std::unique_ptr<TransformInterface> NewIdentity() {
  return std::make_unique<Identity>();
}

std::unique_ptr<TransformInterface> NewFilterByFid(FilterByFidConfig config) {
  return std::make_unique<FilterByFid>(std::move(config));
}

std::unique_ptr<TransformInterface> NewFilterByAction(
    FilterByActionConfig config) {
  return std::make_unique<FilterByAction>(std::move(config));
}

std::unique_ptr<TransformInterface> NewFilterByLabel(
    FilterByLabelConfig config) {
  return std::make_unique<FilterByLabel>(std::move(config));
}

std::unique_ptr<TransformInterface> NewAddLabel(AddLabelConfig config) {
  return std::make_unique<AddLabel>(std::move(config));
}

std::unique_ptr<TransformInterface> NewFilterByValue(
    FilterByValueConfig config) {
  return std::make_unique<FilterByValue>(std::move(config));
}

std::unique_ptr<TransformInterface> CombineTransforms(
    std::unique_ptr<TransformInterface> t1,
    std::unique_ptr<TransformInterface> t2) {
  return std::make_unique<CombinedTransform>(std::move(t1), std::move(t2));
}

std::unique_ptr<TransformInterface> CombineLogicalOrTransforms(
    std::unique_ptr<TransformInterface> t1,
    std::unique_ptr<TransformInterface> t2) {
  return std::make_unique<LogicalOrTransform>(std::move(t1), std::move(t2));
}

std::unique_ptr<TransformInterface> NewTransformFromBasicConfig(
    BasicTransformConfig config) {
  std::string name;
  std::unique_ptr<TransformInterface> transform = nullptr;
  switch (config.type_case()) {
    case (BasicTransformConfig::kFilterByFid):
      transform = NewFilterByFid(std::move(*config.mutable_filter_by_fid()));
      break;
    case (BasicTransformConfig::kFilterByAction):
      transform =
          NewFilterByAction(std::move(*config.mutable_filter_by_action()));
      break;
    case (BasicTransformConfig::kFilterByLabel):
      transform =
          NewFilterByLabel(std::move(*config.mutable_filter_by_label()));
      break;
    case (BasicTransformConfig::kAddLabel):
      transform = NewAddLabel(std::move(*config.mutable_add_label()));
      break;
    case (BasicTransformConfig::kFilterByValue):
      transform =
          NewFilterByValue(std::move(*config.mutable_filter_by_value()));
      break;
    default:
      throw std::invalid_argument(absl::StrFormat(
          "transform is not implemented yet. %s", config.ShortDebugString()));
  }

  return NewTransformSummary(std::move(transform));
}

std::unique_ptr<TransformInterface> NewTransformFromConfig(
    const TransformConfig& config) {
  std::unique_ptr<TransformInterface> transform = nullptr;
  for (const auto& c : config.configs()) {
    std::unique_ptr<TransformInterface> t;
    if (c.has_basic_config()) {
      t = NewTransformFromBasicConfig(c.basic_config());
    } else if (c.has_logical_or_config()) {
      std::unique_ptr<TransformInterface> t1 =
          NewTransformFromBasicConfig(c.logical_or_config().x());
      std::unique_ptr<TransformInterface> t2 =
          NewTransformFromBasicConfig(c.logical_or_config().y());
      t = CombineLogicalOrTransforms(std::move(t1), std::move(t2));
    }

    AssignOrCombine(&transform, std::move(t), CombineTransforms);
  }
  return NewTransformSummary(std::move(transform), true);
}

}  // namespace monolith_tf
}  // namespace tensorflow
