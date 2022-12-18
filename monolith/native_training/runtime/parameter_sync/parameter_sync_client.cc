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

#include "monolith/native_training/runtime/parameter_sync/parameter_sync_client.h"

#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "grpc/impl/codegen/gpr_types.h"
#include "tensorflow/core/platform/default/logging.h"

namespace monolith {
namespace parameter_sync {

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using tensorflow::serving::PredictionService;
using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;

ParameterSyncClient::ConvertResult ParameterSyncClient::Convert(
    const PushRequest& request) {
  ConvertResult result;
  PredictRequest& predict_request = result.req;
  predict_request.mutable_model_spec()->set_name(request.model_name());
  predict_request.mutable_model_spec()->set_signature_name(
      request.signature_name());
  auto& inputs = *predict_request.mutable_inputs();
  int& total = result.total;
  if (request.delta_multi_hash_tables_size() > 0) {
    tensorflow::TensorProto id_tensor, id_split_tensor, emb_tensor;
    id_tensor.set_dtype(tensorflow::DataType::DT_INT64);
    id_split_tensor.set_dtype(tensorflow::DataType::DT_INT64);
    emb_tensor.set_dtype(tensorflow::DataType::DT_FLOAT);
    int64_t split = 0;
    id_split_tensor.add_int64_val(split);
    for (const PushRequest::DeltaEmbeddingHashTable& table :
         request.delta_multi_hash_tables()) {
      total += table.fids_size();
      split += table.fids_size();
      id_split_tensor.add_int64_val(split);
      for (int64_t id : table.fids()) {
        id_tensor.add_int64_val(id);
      }
      for (float value : table.embeddings()) {
        emb_tensor.add_float_val(value);
      }
    }
    id_tensor.mutable_tensor_shape()->add_dim()->set_size(
        id_tensor.int64_val_size());
    id_split_tensor.mutable_tensor_shape()->add_dim()->set_size(
        id_split_tensor.int64_val_size());
    emb_tensor.mutable_tensor_shape()->add_dim()->set_size(
        emb_tensor.float_val_size());
    // names here should match what we write in `saved_model_exporters`
    inputs["id"] = std::move(id_tensor);
    inputs["id_split"] = std::move(id_split_tensor);
    inputs["flat_value"] = std::move(emb_tensor);
  } else {
    for (const auto& delta : request.delta_hash_tables()) {
      int num_update = delta.fids().size();
      total += num_update;
      tensorflow::TensorProto proto_fid, proto_emb;
      proto_fid.set_dtype(tensorflow::DataType::DT_INT64);
      proto_emb.set_dtype(tensorflow::DataType::DT_FLOAT);
      for (int64_t id : delta.fids()) {
        proto_fid.add_int64_val(id);
      }
      for (float value : delta.embeddings()) {
        proto_emb.add_float_val(value);
      }

      int dimension = delta.dim_size();
      proto_fid.mutable_tensor_shape()->add_dim()->set_size(num_update);
      proto_emb.mutable_tensor_shape()->add_dim()->set_size(num_update);
      proto_emb.mutable_tensor_shape()->add_dim()->set_size(dimension);
      inputs[delta.unique_id() + "_id"] = proto_fid;
      inputs[delta.unique_id() + "_value"] = proto_emb;
    }
  }
  return result;
}

grpc::Status ParameterSyncClient::Push(const PushRequest& request,
                                       PushResponse* response) const {
  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  ConvertResult convert_result = Convert(request);
  const PredictRequest& predict_request = convert_result.req;
  PredictResponse predict_response;
  ClientContext context;
  gpr_timespec ts;
  ts.tv_sec = request.timeout_in_ms() / 1000;
  ts.tv_nsec = (request.timeout_in_ms() % 1000) * 1000 * 1000;
  ts.clock_type = GPR_TIMESPAN;
  context.set_deadline(ts);

  // TODO(zhangbiao.david): predict_request.DebugString() causes a segment
  //  fault, but I have no idea about it.
  // LOG(INFO) << "PredictRequest\n" << predict_request.DebugString() <<
  // std::endl;

  // The actual RPC.
  Status status;
  status = stub_->Predict(&context, predict_request, &predict_response);
  response->set_status_code(status.error_code());
  response->set_error_message(status.error_message());

  // Act upon its status.
  if (status.ok()) {
    response->set_update_num(convert_result.total);
    return status;
  } else {
    response->set_update_num(0);
    LOG_EVERY_N_SEC(ERROR, 10) << status.error_code() << ": "
                               << status.error_message();
    return status;
  }
}

}  // namespace parameter_sync
}  // namespace monolith
