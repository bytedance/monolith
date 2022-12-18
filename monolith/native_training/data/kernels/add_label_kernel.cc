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

#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_format.h"
#include "monolith/native_training/data/kernels/internal/label_utils.h"
#include "monolith/native_training/data/training_instance/cc/instance_utils.h"
#include "monolith/native_training/data/training_instance/cc/pb_variant.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace monolith_tf {

using IFeature = ::idl::matrix::proto::Feature;
using Instance = ::parser::proto::Instance;
using Example = ::monolith::io::proto::Example;
using LineId = ::idl::matrix::proto::LineId;

class AddLabelOp : public OpKernel {
 public:
  explicit AddLabelOp(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("config", &config_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("negative_value", &negative_value_));
    OP_REQUIRES_OK(context, context->GetAttr("sample_rate", &sample_rate_));
    OP_REQUIRES_OK(context, context->GetAttr("variant_type", &variant_type_));

    if (variant_type_ != "instance" && variant_type_ != "example") {
      LOG(FATAL) << "Invalid 'variant_type', please choose on from "
                    "['instance', 'example']!";
    }

    internal::ParseTaskConfig(config_, &task_configs_);
    for (size_t i = 0; i < task_configs_.size(); ++i) {
      LOG(INFO) << absl::StrFormat("Task #%d config: %s", i + 1,
                                   task_configs_[i].ToString());
    }
    LOG(INFO) << absl::StrFormat("sample_rate = %.4f", sample_rate_);
    std::size_t seed =
        std::chrono::system_clock::now().time_since_epoch().count();
    random_generator_.seed(seed);
    random_neg_sample_ = std::uniform_real_distribution<float>(0.0, 1.0);
  }

  void Compute(OpKernelContext *context) override {
    const Tensor &input_tensor = context->input(0);
    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));

    bool is_instance = variant_type_ == "instance";
    if (is_instance) {
      Instance instance;
      instance.CopyFrom(*input_tensor.scalar<Variant>()().get<Instance>());
      output_tensor->scalar<Variant>()() = std::move(instance);
    } else {
      Example example;
      example.CopyFrom(*input_tensor.scalar<Variant>()().get<Example>());
      output_tensor->scalar<Variant>()() = std::move(example);
    }

    LineId *line_id = GetLineId(output_tensor, is_instance);
    auto label = GetLabel(output_tensor, is_instance);
    std::set<int32_t> actions(line_id->actions().begin(),
                              line_id->actions().end());

    if (!label->empty() && label->Get(0) <= 0) {
      label->Set(0, internal::INVALID_LABEL);
    }

    for (const auto &t : task_configs_) {
      bool has_pos = internal::HasIntersection(actions, t.pos_actions);
      bool has_neg = internal::HasIntersection(actions, t.neg_actions);

      if (!t.neg_actions.empty()) {
        // If there is given neg_actions
        if (!has_pos && !has_neg) {
          label->Add(internal::INVALID_LABEL);
        } else if (has_pos) {
          // (has_pos && !has_neg) || (has_pos && has_neg)
          label->Add(internal::POSITIVE_LABEL);
        } else {
          // !has_pos && has_neg
          if (SelectedByNegativeSampling(t)) {
            label->Add(negative_value_);
          } else {
            label->Add(internal::INVALID_LABEL);
          }
        }
      } else {
        // If there is no given neg_actions
        if (has_pos) {
          label->Add(internal::POSITIVE_LABEL);
        } else {
          if (SelectedByNegativeSampling(t)) {
            label->Add(negative_value_);
          } else {
            label->Add(internal::INVALID_LABEL);
          }
        }
      }
    }

    line_id->set_sample_rate(sample_rate_);
  }

 private:
  bool SelectedByNegativeSampling(const internal::TaskConfig &t) {
    return internal::IsAlmostEqual(t.sample_rate, 1.0f) ||
           random_neg_sample_(random_generator_) < t.sample_rate;
  }

  static LineId *GetLineId(Tensor *output_tensor, bool is_instance) {
    if (is_instance) {
      return output_tensor->scalar<Variant>()()
          .get<Instance>()
          ->mutable_line_id();
    } else {
      return output_tensor->scalar<Variant>()()
          .get<Example>()
          ->mutable_line_id();
    }
  }

  static ::google::protobuf::RepeatedField<float> *GetLabel(
      Tensor *output_tensor, bool is_instance) {
    if (is_instance) {
      return output_tensor->scalar<Variant>()()
          .get<Instance>()
          ->mutable_label();
    } else {
      return output_tensor->scalar<Variant>()().get<Example>()->mutable_label();
    }
  }

  float negative_value_;
  float sample_rate_;
  std::string config_;
  std::string variant_type_;
  std::vector<internal::TaskConfig> task_configs_;
  std::default_random_engine random_generator_;
  std::uniform_real_distribution<float> random_neg_sample_;
};

namespace {

REGISTER_KERNEL_BUILDER(Name("AddLabel").Device(DEVICE_CPU), AddLabelOp)

}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow
