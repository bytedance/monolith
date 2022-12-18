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

#include "monolith/native_training/runtime/ops/remote_predict_op.h"
#include "monolith/native_training/runtime/ops/prediction_service_grpc.h"

namespace tensorflow {
namespace monolith_tf {
namespace {

REGISTER_KERNEL_BUILDER(
    Name("TfServingRemotePredict").Device(DEVICE_CPU),
    RemotePredictOp<::tensorflow::monolith_tf::PredictionServiceGrpc,
                    ::tensorflow::monolith_tf::AgentHeartbeat<
                        ::tensorflow::monolith_tf::PredictionServiceGrpc>>);

REGISTER_OP("TfServingRemotePredict")
    .Attr("T: list(type)")
    .Attr("model_name: string = ''")
    .Attr("model_version: int = -1")
    .Attr("fail_op_on_rpc_error: bool = true")
    .Attr("max_rpc_deadline_millis: int = 30")
    .Attr("signature_name: string = 'serving_default'")
    .Attr("old_model_name: string = ''")
    .Attr("task: int = -1")
    .Input("input_tensor_aliases: string")
    .Input("input_tensors: T")
    .Input("output_tensor_aliases: string")
    .Output("status_code: int32")
    .Output("status_error_message: string")
    .Output("output_tensors: output_types")
    .Attr("output_types: list(type)")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      shape_inference::ShapeHandle unused;
      // Checks the length of input_tensor_aliases with that of input_tensors.
      std::vector<shape_inference::ShapeHandle> input_aliases_handle;
      TF_RETURN_IF_ERROR(
          c->input("input_tensor_aliases", &input_aliases_handle));
      TF_RETURN_IF_ERROR(c->WithRank(input_aliases_handle[0], 1, &unused));
      std::vector<shape_inference::ShapeHandle> inputs_handle;
      TF_RETURN_IF_ERROR(c->input("input_tensors", &inputs_handle));
      if (c->Value(c->NumElements(input_aliases_handle[0])) !=
          inputs_handle.size()) {
        return errors::InvalidArgument(
            "'input_tensors' should be equal in length to "
            "'input_tensor_aliases'. Length of 'input_tensors': ",
            inputs_handle.size(), ", length of 'input_tensor_aliases': ",
            c->Value(c->NumElements(input_aliases_handle[0])));
      }

      // Checks the length of output_tensor_aliases with that of output_types.
      DataTypeVector output_types;
      TF_RETURN_IF_ERROR(c->GetAttr("output_types", &output_types));
      std::vector<shape_inference::ShapeHandle> output_aliases_handle;
      TF_RETURN_IF_ERROR(
          c->input("output_tensor_aliases", &output_aliases_handle));
      if (c->Value(c->NumElements(output_aliases_handle[0])) !=
          output_types.size()) {
        return errors::InvalidArgument(
            "'output_types' should be equal in length to "
            "'output_tensor_aliases'. Length of 'output_types': ",
            output_types.size(), ", length of 'output_tensor_aliases': ",
            c->Value(c->NumElements(output_aliases_handle[0])));
      }

      // We know the shape of the first 2 outputs, but not the rest.
      TF_RETURN_IF_ERROR(c->set_output("status_code", {c->Scalar()}));
      TF_RETURN_IF_ERROR(c->set_output("status_error_message", {c->Scalar()}));
      for (int i = 2; i < c->num_outputs(); ++i) {
        c->set_output(i, c->UnknownShape());
      }

      return Status::OK();
    })
    .SetIsStateful()
    .Doc(R"doc(
Invokes Predict on a remote graph.
fail_op_on_rpc_error: If set true, the Op fails if the rpc fails, and returns
  the status code as 0 and an empty status_message. Otherwise the
  Op returns the status of the rpc call, along with the output tensors, if any.
  Set true by default.
max_rpc_deadline_millis: The rpc deadline for remote predict. The actual
deadline is min(incoming_rpc_deadline, max_rpc_deadline_millis).
task: The task id of ps/entry, if the server_type is entry, it would be 0.
signature_name: the signature def for remote graph inference, defaulting to 
"serving_default".
model_name: Model name of the remote TF graph.
model_version: the target version for the Predict call. When unset, the
  default value (-1) implies the latest available version should be used.
input_tensor_aliases: Tensor of strings for the input tensor alias names to supply
  to the RemotePredict call.
input_tensors: List of tensors to provide as input. Should be equal in length
  to 'input_tensor_aliases'.
output_tensor_aliases: Tensor of strings for the output tensor alias names to
  supply to the Predict call.
status_code: Returns the status code of the rpc call; basically converting
  tensorflow::error::Code to it's int value, so 0 means OK.
status_error_message: Returns the error message in the rpc status.
output_tensors: Tensors returned by the Predict call on the remote graph, which 
  are in the same order as output_tensor_aliases.
output_types: A list of types of the output tensors. Length of this list should
  be equal to the length of 'output_tensor_aliases'.
old_model_name: the model name used in previous monolith version (deprecated).
task: the index of old_model_name (deprecated).
)doc");

}  // namespace

}  // namespace monolith_tf
}  // namespace tensorflow
