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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// forked from:
//   https://github.com/tensorflow/serving/blob/2.4.0/tensorflow_serving/experimental/tensorflow/ops/remote_predict/kernels/remote_predict_op_kernel.h
//   https://github.com/tensorflow/serving/blob/2.4.0/tensorflow_serving/experimental/tensorflow/ops/remote_predict/kernels/remote_predict_op_kernel.cc
//   https://github.com/tensorflow/serving/blob/2.4.0/tensorflow_serving/experimental/tensorflow/ops/remote_predict/ops/remote_predict_op.cc
// with:
//   "#ifndef
//   TENSORFLOW_SERVING_EXPERIMENTAL_TENSORFLOW_OPS_REMOTE_PREDICT_KERNELS_REMOTE_PREDICT_OP_KERNEL_H_
//   ..." removed

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "glog/logging.h"
#include "google/protobuf/map.h"
#include "google/protobuf/wrappers.pb.h"
#include "monolith/native_training/runtime/common/metrics.h"
#include "monolith/native_training/runtime/ops/agent_heartbeat.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/protobuf/named_tensor.pb.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/apis/predict.pb.h"

namespace tensorflow {
namespace monolith_tf {

typedef google::protobuf::Map<tensorflow::string, tensorflow::TensorProto>
    AliasTensorMap;
using ::tensorflow::serving::PredictRequest;
using ::tensorflow::serving::PredictResponse;

// Remote Predict Op kernel implementation class templated on different
// PredictionServiceStubTypes.
template <typename PredictionServiceStubType, typename AgentHeartbeat>
class RemotePredictOp : public AsyncOpKernel {
 public:
  explicit RemotePredictOp(OpKernelConstruction *context)
      : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("model_name", &model_name_));
    OP_REQUIRES_OK(context, context->GetAttr("model_version", &model_version_));
    OP_REQUIRES_OK(context, context->GetAttr("max_rpc_deadline_millis",
                                             &max_rpc_deadline_millis_));
    OP_REQUIRES_OK(context, context->GetAttr("fail_op_on_rpc_error",
                                             &fail_op_on_rpc_error_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("signature_name", &signature_name_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("old_model_name", &old_model_name_));
    OP_REQUIRES_OK(context, context->GetAttr("task", &task_));

    if (AgentHeartbeat::GetInstance().api_version() == 0 &&
        old_model_name_.size() > 0) {
      req_model_name_ = old_model_name_;
    } else {
      req_model_name_ = model_name_;
    }
  }

  void ComputeAsync(OpKernelContext *context, DoneCallback done) override {
    auto activity =
        std::make_shared<profiler::TraceMe>([this]() { return name(); });
    auto remote_predict_op_latency_start = std::chrono::system_clock::now();
    // Get the input tensor alias names.
    const auto &input_tensor_aliases = context->input(0).flat<tstring>();

    // Get the input tensors.
    OpInputList input_tensors;
    OP_REQUIRES_OK_ASYNC(
        context, context->input_list("input_tensors", &input_tensors), done);
    // Get the output tensor alias names.
    // Directly index to output_tensor_aliases by moving past all the input
    // before it, including the input_tensor_aliases and input_tensors.
    auto output_tensor_aliases =
        context->input(1 + input_tensors.size()).flat<tstring>();

    // Build the PredictRequest.
    std::shared_ptr<PredictRequest> request(new PredictRequest);

    request->mutable_model_spec()->set_name(req_model_name_);

    request->mutable_model_spec()->set_signature_name(signature_name_);

    if (model_version_ >= 0) {
      request->mutable_model_spec()->mutable_version()->set_value(
          model_version_);
    }

    AliasTensorMap &inputs = *request->mutable_inputs();
    for (int i = 0; i < input_tensor_aliases.size(); ++i) {
      tensorflow::TensorProto proto;
      input_tensors[i].AsProtoField(&proto);
      inputs[input_tensor_aliases(i)] = proto;
    }

    for (int i = 0; i < output_tensor_aliases.size(); ++i) {
      request->add_output_filter(tensorflow::string(output_tensor_aliases(i)));
    }

    std::shared_ptr<PredictResponse> response(new PredictResponse());

    std::shared_ptr<PredictionServiceStubType> prediction_service = nullptr;
    if (AgentHeartbeat::GetInstance().api_version() == 0) {
      if (old_model_name_.size() > 0) {
        size_t pos = old_model_name_.find("_");
        std::string real_model_name = old_model_name_;
        if (pos != std::string::npos) {
          real_model_name.replace(pos, 1, ":");
        }
        LOG_FIRST_N(INFO, 3) << "GetPredictionService by old_model_name_:"
                             << real_model_name;
        prediction_service =
            AgentHeartbeat::GetInstance().GetPredictionService(real_model_name);
      } else {
        LOG_FIRST_N(INFO, 3) << "GetPredictionService by task_:" << task_;
        prediction_service =
            AgentHeartbeat::GetInstance().GetPredictionServiceByIdx(task_);
      }
    } else {
      prediction_service =
          AgentHeartbeat::GetInstance().GetPredictionService(model_name_);
    }

    OP_REQUIRES_ASYNC(
        context, prediction_service != nullptr,
        errors::Unavailable("No available remote servers. model_name=",
                            model_name_, ",signature_name=", signature_name_),
        done);

    auto serving_latency_start = std::chrono::system_clock::now();
    auto callback = [this, context, request, response, activity,
                     output_tensor_aliases, done, serving_latency_start,
                     remote_predict_op_latency_start, prediction_service](
        const absl::Status &status, DoneCallback &&rpc_done) {
      std::ostringstream tagkv;
      tagkv << "model_name=" << model_name_
            << "|signature_name=" << signature_name_;

      auto serving_latency_end = std::chrono::system_clock::now();
      std::chrono::duration<double> serving_latency_diff =
          std::chrono::duration_cast<std::chrono::microseconds>(
              serving_latency_end - serving_latency_start);
      monolith::GetMetrics()->emit_timer(
          "serving_latency", serving_latency_diff.count(), tagkv.str());
      LOG_EVERY_N(INFO, 1000) << "emit_timer serving_latency " << tagkv.str();
      PostProcessResponse(context, response.get(), status,
                          fail_op_on_rpc_error_, output_tensor_aliases,
                          std::forward<DoneCallback>(rpc_done));
      auto remote_predict_op_latency_end = std::chrono::system_clock::now();
      std::chrono::duration<double> remote_predict_op_latency_diff =
          std::chrono::duration_cast<std::chrono::microseconds>(
              remote_predict_op_latency_end - remote_predict_op_latency_start);
      monolith::GetMetrics()->emit_timer("remote_predict_op_latency",
                                         remote_predict_op_latency_diff.count(),
                                         tagkv.str());
      monolith::GetMetrics()->emit_counter("remote_predict_op_throughput", 1,
                                           tagkv.str());
      LOG_EVERY_N(INFO, 1000) << "emit_timer remote_predict_op_latency "
                              << tagkv.str();
    };
    // Make the RPC call.
    prediction_service->Predict(request.get(), response.get(), callback,
                                max_rpc_deadline_millis_, done);
  }

  void PostProcessResponse(OpKernelContext *context, PredictResponse *response,
                           const absl::Status &rpc_status,
                           bool fail_op_on_rpc_error,
                           TTypes<const tstring>::Flat output_tensor_aliases,
                           DoneCallback rpc_done) {
    auto rpc_cleaner = gtl::MakeCleanup([&] { rpc_done(); });
    Tensor *status_code;
    OP_REQUIRES_OK_ASYNC(
        context, context->allocate_output(0, TensorShape({}), &status_code),
        rpc_cleaner.release());
    status_code->scalar<int>()() = static_cast<int>(rpc_status.code());
    Tensor *status_error_message;
    OP_REQUIRES_OK_ASYNC(
        context,
        context->allocate_output(1, TensorShape({}), &status_error_message),
        rpc_cleaner.release());
    status_error_message->scalar<tstring>()() = rpc_status.message();
    OpOutputList output_tensors_list;
    OP_REQUIRES_OK_ASYNC(
        context, context->output_list("output_tensors", &output_tensors_list),
        rpc_cleaner.release());
    // Process the response.
    if (!rpc_status.ok()) {
      if (fail_op_on_rpc_error) {
        OP_REQUIRES_OK_ASYNC(
            context, tensorflow::Status(static_cast<tensorflow::error::Code>(
                                            rpc_status.code()),
                                        rpc_status.message()),
            rpc_cleaner.release());
      } else {
        // Allocate some empty output for the output_tensors.
        for (int i = 0; i < output_tensors_list.size(); ++i) {
          Tensor *unused;
          OP_REQUIRES_OK_ASYNC(context, output_tensors_list.allocate(
                                            i, TensorShape({}), &unused),
                               rpc_cleaner.release());
        }
        return;
      }
    }
    OP_REQUIRES_ASYNC(
        context, output_tensors_list.size() == output_tensor_aliases.size(),
        errors::Internal(
            "Response doesn't have the right number of outputs; actual: ",
            output_tensors_list.size(), " expected: ",
            output_tensor_aliases.size()),
        rpc_cleaner.release());
    AliasTensorMap &outputs = *response->mutable_outputs();
    for (int i = 0; i < output_tensor_aliases.size(); i++) {
      Tensor output_tensor;
      OP_REQUIRES_ASYNC(
          context, output_tensor.FromProto(outputs[output_tensor_aliases(i)]),
          errors::Internal("Response tensor proto: ",
                           tensorflow::string(output_tensor_aliases(i)),
                           " cannot be converted back to a tensor."),
          rpc_cleaner.release());
      output_tensors_list.set(i, output_tensor);
    }
  }

 private:
  std::string model_name_;
  int64 model_version_;
  bool fail_op_on_rpc_error_;
  int64 max_rpc_deadline_millis_;
  std::string signature_name_;
  std::string old_model_name_;
  int task_;

  std::string req_model_name_;
};

}  // namespace monolith_tf
}  // namespace tensorflow
