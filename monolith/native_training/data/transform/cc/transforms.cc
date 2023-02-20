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
#include "absl/strings/str_format.h"
#include "glog/logging.h"
#include "monolith/native_training/data/kernels/internal/label_utils.h"
#include "monolith/native_training/data/training_instance/cc/instance_utils.h"
#include "tensorflow/core/platform/logging.h"
#include "third_party/nlohmann/json.hpp"

namespace tensorflow {
namespace monolith_tf {

using ::google::protobuf::RepeatedField;
using ::idl::matrix::proto::LineId;
using ::monolith::io::proto::Example;
using ::monolith::io::proto::ExampleBatch;
using ::parser::proto::Instance;

class SampleCounter : public TransformInterface {
 public:
  explicit SampleCounter(std::unique_ptr<TransformInterface> transform,
                         std::string transform_name = "")
      : transform_(std::move(transform)),
        transform_name_(std::move(transform_name)),
        offset_(0),
        input_total_(0),
        output_total_(0) {}

  ~SampleCounter() override {
    LOG(INFO) << absl::StrFormat(
        "Finally transform: %s, input_num = %ld, output_num = %ld.",
        transform_name_, input_total_, output_total_);
  }

  void Transform(std::shared_ptr<Instance> instance,
                 std::vector<std::shared_ptr<Instance>>* output) override {
    offset_ = output->size();
    input_total_ += 1;
    transform_->Transform(instance, output);
    output_total_ += output->size() - offset_;
    LOG_EVERY_N_SEC(INFO, 60 * 5)
        << absl::StrFormat("transform: %s, input_num = %ld, output_num = %ld.",
                           transform_name_, input_total_, output_total_);
  }

  void Transform(std::shared_ptr<Example> example,
                 std::vector<std::shared_ptr<Example>>* output) override {
    offset_ = output->size();
    input_total_ += 1;
    transform_->Transform(example, output);
    output_total_ += output->size() - offset_;
    LOG_EVERY_N_SEC(INFO, 60 * 5)
        << absl::StrFormat("transform: %s, input_num = %ld, output_num = %ld.",
                           transform_name_, input_total_, output_total_);
  }

  void Transform(std::shared_ptr<ExampleBatch> example_batch,
                 std::vector<std::shared_ptr<ExampleBatch>>* output) override {
    offset_ = output->size();
    input_total_ += 1;
    transform_->Transform(example_batch, output);
    output_total_ += output->size() - offset_;
    LOG_EVERY_N_SEC(INFO, 60 * 5)
        << absl::StrFormat("transform: %s, input_num = %ld, output_num = %ld.",
                           transform_name_, input_total_, output_total_);
  }

 private:
  std::unique_ptr<TransformInterface> transform_;

  std::string transform_name_;

  int64_t offset_;

  int64_t input_total_;

  int64_t output_total_;
};

class Identity : public TransformInterface {
 public:
  void Transform(std::shared_ptr<Instance> instance,
                 std::vector<std::shared_ptr<Instance>>* output) override {
    output->push_back(instance);
  }

  void Transform(std::shared_ptr<Example> example,
                 std::vector<std::shared_ptr<Example>>* output) override {
    output->push_back(example);
  }

  void Transform(std::shared_ptr<ExampleBatch> example_batch,
                 std::vector<std::shared_ptr<ExampleBatch>>* output) override {
    output->push_back(example_batch);
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

  void Transform(
      std::shared_ptr<::monolith::io::proto::ExampleBatch> example_batch,
      std::vector<std::shared_ptr<::monolith::io::proto::ExampleBatch>>* output)
      override {
    throw std::runtime_error("not implemented!");
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

  void Transform(
      std::shared_ptr<::monolith::io::proto::ExampleBatch> example_batch,
      std::vector<std::shared_ptr<::monolith::io::proto::ExampleBatch>>* output)
      override {
    throw std::runtime_error("not implemented!");
  }

 private:
  std::set<int32_t> has_actions_;
  FilterByActionConfig config_;
};

class FilterByLabel : public TransformInterface {
 public:
  explicit FilterByLabel(FilterByLabelConfig config)
      : config_(std::move(config)) {}

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

  void Transform(
      std::shared_ptr<::monolith::io::proto::ExampleBatch> example_batch,
      std::vector<std::shared_ptr<::monolith::io::proto::ExampleBatch>>* output)
      override {
    throw std::runtime_error("not implemented!");
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

  void Transform(
      std::shared_ptr<::monolith::io::proto::ExampleBatch> example_batch,
      std::vector<std::shared_ptr<::monolith::io::proto::ExampleBatch>>* output)
      override {
    throw std::runtime_error("not implemented!");
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
    return internal::IsAlmostEqual(t.sample_rate, 1.0f) ||
           random_neg_sample_(random_generator_) < t.sample_rate;
  }

  std::vector<internal::TaskConfig> task_configs_;

  std::default_random_engine random_generator_;

  std::uniform_real_distribution<float> random_neg_sample_;

  AddLabelConfig config_;
};

class CombinedTransform : public TransformInterface {
 public:
  CombinedTransform(std::unique_ptr<TransformInterface> t1,
                    std::unique_ptr<TransformInterface> t2)
      : t1_(std::move(t1)), t2_(std::move(t2)) {}

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

  void Transform(std::shared_ptr<ExampleBatch> example_batch,
                 std::vector<std::shared_ptr<ExampleBatch>>* output) override {
    std::vector<std::shared_ptr<ExampleBatch>> intermediates;
    t1_->Transform(example_batch, &intermediates);
    for (const auto& intermediate : intermediates) {
      t2_->Transform(intermediate, output);
    }
  }

 private:
  std::unique_ptr<TransformInterface> t1_;
  std::unique_ptr<TransformInterface> t2_;
};

std::unique_ptr<TransformInterface> NewSampleCounter(
    std::unique_ptr<TransformInterface> transform,
    const std::string& transform_name) {
  return std::make_unique<SampleCounter>(std::move(transform), transform_name);
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

std::unique_ptr<TransformInterface> CombineTransforms(
    std::unique_ptr<TransformInterface> t1,
    std::unique_ptr<TransformInterface> t2) {
  return std::make_unique<CombinedTransform>(std::move(t1), std::move(t2));
}

std::unique_ptr<TransformInterface> NewTransformFromConfig(
    TransformConfig_OneTransformConfig config) {
  std::string name;
  std::unique_ptr<TransformInterface> transform = nullptr;
  switch (config.type_case()) {
    case (TransformConfig_OneTransformConfig::kFilterByFid):
      name = "FilterByFid";
      transform = NewFilterByFid(std::move(*config.mutable_filter_by_fid()));
      break;
    case (TransformConfig_OneTransformConfig::kFilterByAction):
      name = "FilterByAction";
      transform =
          NewFilterByAction(std::move(*config.mutable_filter_by_action()));
      break;
    case (TransformConfig_OneTransformConfig::kFilterByLabel):
      name = "FilterByLabel";
      transform =
          NewFilterByLabel(std::move(*config.mutable_filter_by_label()));
      break;
    case (TransformConfig_OneTransformConfig::kAddLabel):
      name = "AddLabel";
      transform = NewAddLabel(std::move(*config.mutable_add_label()));
      break;
    default:
      throw std::invalid_argument(absl::StrFormat(
          "transform is not implemented yet. %s", config.ShortDebugString()));
  }

  return NewSampleCounter(std::move(transform), name);
}

std::unique_ptr<TransformInterface> NewTransformFromConfig(
    const TransformConfig& config) {
  std::unique_ptr<TransformInterface> transform = nullptr;
  for (const auto& c : config.configs()) {
    std::unique_ptr<TransformInterface> t = NewTransformFromConfig(c);
    AssignOrCombine(&transform, std::move(t), CombineTransforms);
  }
  return std::move(transform);
}

}  // namespace monolith_tf
}  // namespace tensorflow
