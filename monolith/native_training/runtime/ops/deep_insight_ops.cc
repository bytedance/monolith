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

#include "monolith/native_training/runtime/ops/deep_insight_client_tf_bridge.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"

using monolith::deep_insight::ExtraField;
using monolith::deep_insight::FloatExtraField;
using monolith::deep_insight::Int64ExtraField;
using monolith::deep_insight::StringExtraField;

namespace tensorflow {
namespace monolith_tf {

class MonolithCreateDeepInsightClientOp
    : public ResourceOpKernel<DeepInsightClientTfBridge> {
 public:
  explicit MonolithCreateDeepInsightClientOp(OpKernelConstruction* ctx)
      : ResourceOpKernel(ctx) {
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("enable_metrics_counter", &enable_metrics_counter_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("is_fake", &is_fake_));
  }

  ~MonolithCreateDeepInsightClientOp() override {}

 private:
  Status CreateResource(DeepInsightClientTfBridge** deep_insight_client_bridge)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    auto deep_insight_client =
        std::make_unique<monolith::deep_insight::DeepInsight>(
            enable_metrics_counter_, is_fake_);
    *deep_insight_client_bridge =
        new DeepInsightClientTfBridge(std::move(deep_insight_client));
    return Status::OK();
  }

  bool enable_metrics_counter_;
  bool is_fake_;
};

class MonolithWriteDeepInsightOp : public OpKernel {
 public:
  explicit MonolithWriteDeepInsightOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("model_name", &model_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("target", &target_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sample_ratio", &sample_ratio_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("return_msgs", &return_msgs_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("use_zero_train_time", &use_zero_train_time_));
  }

  void Compute(OpKernelContext* ctx) override {
    DeepInsightClientTfBridge* deep_insight_client = nullptr;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0),
                                       &deep_insight_client));
    core::ScopedUnref unref(deep_insight_client);

    auto uids_vec = ctx->input(1).vec<int64_t>();
    auto req_times_vec = ctx->input(2).vec<int64_t>();
    auto labels_vec = ctx->input(3).vec<float>();
    auto preds_vec = ctx->input(4).vec<float>();
    auto sample_rates_vec = ctx->input(5).vec<float>();

    int64_t train_time =
        use_zero_train_time_ ? 0 : deep_insight_client->GenerateTrainingTime();

    int64_t batch_size = labels_vec.dimension(0);
    Tensor* msgs;
    ctx->allocate_output(0, {batch_size}, &msgs);
    auto msgs_vec = msgs->vec<tstring>();

    std::vector<std::string> targets;
    targets.push_back(target_);

    for (uint32_t i = 0; i < batch_size; i++) {
      std::vector<float> labels, preds, sample_rates;
      std::vector<std::shared_ptr<ExtraField>> extra_fields;
      labels.push_back(labels_vec(i));
      preds.push_back(preds_vec(i));
      sample_rates.push_back(sample_rates_vec(i));
      std::string msg = deep_insight_client->SendV2(
          model_name_, targets, uids_vec(i), req_times_vec(i), train_time,
          labels, preds, sample_rates, sample_ratio_, extra_fields,
          return_msgs_);

      msgs_vec(i) = msg;
    }
  };

 private:
  std::string model_name_;
  std::string target_;
  float sample_ratio_;
  bool return_msgs_;
  bool use_zero_train_time_;
};

class MonolithWriteDeepInsightV2 : public OpKernel {
 public:
  explicit MonolithWriteDeepInsightV2(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("model_name", &model_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("targets", &targets_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sample_ratio", &sample_ratio_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("return_msgs", &return_msgs_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("use_zero_train_time", &use_zero_train_time_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("extra_fields_keys", &extra_fields_keys_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Tfields", &extra_fields_dtypes_));
  }

  void Compute(OpKernelContext* ctx) override {
    DeepInsightClientTfBridge* deep_insight_client = nullptr;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0),
                                       &deep_insight_client));
    core::ScopedUnref unref(deep_insight_client);

    auto req_times_vec = ctx->input(1).vec<int64_t>();      // batch
    auto labels_mat = ctx->input(2).matrix<float>();        // num_heads x batch
    auto preds_mat = ctx->input(3).matrix<float>();         // num_heads x batch
    auto sample_rates_mat = ctx->input(4).matrix<float>();  // num_heads x batch

    OpInputList extra_fields_values;
    OP_REQUIRES_OK(
        ctx, ctx->input_list("extra_fields_values", &extra_fields_values));

    int64_t train_time =
        use_zero_train_time_ ? 0 : deep_insight_client->GenerateTrainingTime();

    int64_t batch_size = labels_mat.dimension(1);

    std::vector<std::vector<std::shared_ptr<ExtraField>>>
        batched_extra_fields;  // batch x num_keys
    std::vector<int64_t> uids_vec;
    for (uint32_t b = 0; b < batch_size; b++) {
      batched_extra_fields.emplace_back();
      auto& extra_fields = batched_extra_fields.back();
      for (size_t i = 0; i < extra_fields_dtypes_.size(); i++) {
        if (extra_fields_dtypes_.at(i) == tensorflow::DT_FLOAT) {
          extra_fields.push_back(std::make_shared<FloatExtraField>(
              extra_fields_keys_.at(i),
              extra_fields_values[i].vec<float>()(b)));
        } else if (extra_fields_dtypes_.at(i) == tensorflow::DT_INT64) {
          if (extra_fields_keys_.at(i) == "uid") {
            uids_vec.push_back(extra_fields_values[i].vec<int64_t>()(b));
          } else {
            extra_fields.push_back(std::make_shared<Int64ExtraField>(
                extra_fields_keys_.at(i),
                extra_fields_values[i].vec<int64_t>()(b)));
          }
        } else if (extra_fields_dtypes_.at(i) == tensorflow::DT_STRING) {
          extra_fields.push_back(std::make_shared<StringExtraField>(
              extra_fields_keys_.at(i),
              extra_fields_values[i].vec<tstring>()(b)));
        }
      }
    }

    Tensor* msgs;
    ctx->allocate_output(0, {batch_size}, &msgs);
    auto msgs_vec = msgs->vec<tstring>();

    for (uint32_t i = 0; i < batch_size; i++) {
      std::vector<float> labels, preds, sample_rates;
      for (int j = 0; j < targets_.size(); j++) {
        labels.push_back(labels_mat(j, i));
        preds.push_back(preds_mat(j, i));
        sample_rates.push_back(sample_rates_mat(j, i));
      }
      std::string msg = deep_insight_client->SendV2(
          model_name_, targets_, uids_vec.at(i), req_times_vec(i), train_time,
          labels, preds, sample_rates, sample_ratio_,
          batched_extra_fields.at(i), return_msgs_);

      msgs_vec(i) = msg;
    }
  };

 private:
  std::string model_name_;
  std::vector<std::string> targets_;
  float sample_ratio_;
  bool return_msgs_;
  bool use_zero_train_time_;
  std::vector<tstring> extra_fields_keys_;
  std::vector<DataType> extra_fields_dtypes_;
};

REGISTER_OP("MonolithCreateDeepInsightClient")
    .Output("handle: resource")
    .Attr("enable_metrics_counter: bool = false")
    .Attr("is_fake: bool = false")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(
    Name("MonolithCreateDeepInsightClient").Device(DEVICE_CPU),
    MonolithCreateDeepInsightClientOp);

REGISTER_OP("MonolithWriteDeepInsight")
    .Input("deep_insight_client_handle: resource")
    .Input("uids: int64")
    .Input("req_times: int64")
    .Input("labels: float")
    .Input("preds: float")
    .Input("sample_rates: float")
    .Output("msgs: string")
    .Attr("model_name: string")
    .Attr("target: string = 'ctr_head'")
    .Attr("sample_ratio: float = 0.01")
    .Attr("return_msgs: bool = false")
    .Attr("use_zero_train_time: bool = false")
    .SetShapeFn([](shape_inference::InferenceContext* ctx) {
      ctx->set_output(0, ctx->input(1));
      return Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name("MonolithWriteDeepInsight").Device(DEVICE_CPU),
                        MonolithWriteDeepInsightOp);

REGISTER_OP("MonolithWriteDeepInsightV2")
    .Input("deep_insight_client_handle: resource")
    .Input("req_times: int64")
    .Input("labels: float")
    .Input("preds: float")
    .Input("sample_rates: float")
    .Input("extra_fields_values: Tfields")
    .Output("msgs: string")
    .Attr("model_name: string")
    .Attr("extra_fields_keys: list(string)")
    .Attr("Tfields: list(type)")
    .Attr("targets: list(string)")
    .Attr("sample_ratio: float = 0.01")
    .Attr("return_msgs: bool = false")
    .Attr("use_zero_train_time: bool = false")
    .SetShapeFn([](shape_inference::InferenceContext* ctx) {
      ctx->set_output(0, ctx->input(1));
      return Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name("MonolithWriteDeepInsightV2").Device(DEVICE_CPU),
                        MonolithWriteDeepInsightV2);

}  // namespace monolith_tf
}  // namespace tensorflow
