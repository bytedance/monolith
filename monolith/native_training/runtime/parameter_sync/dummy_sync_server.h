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

#ifndef MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_PARAMETER_SYNC_DUMMY_SYNC_SERVER_H_
#define MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_PARAMETER_SYNC_DUMMY_SYNC_SERVER_H_

#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "glog/logging.h"
#include "grpcpp/ext/proto_server_reflection_plugin.h"
#include "grpcpp/grpcpp.h"
#include "grpcpp/health_check_service_interface.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"

#include "monolith/native_training/runtime/parameter_sync/parameter_sync.grpc.pb.h"

namespace monolith {
namespace parameter_sync {

// Test only
class PredictionServiceImpl final
    : public tensorflow::serving::PredictionService::Service {
  grpc::Status Predict(
      grpc::ServerContext* context,
      const tensorflow::serving::PredictRequest* request,
      tensorflow::serving::PredictResponse* response) override {
    // TODO(zhangbiao.david): remove
    LOG(INFO) << "PredictionServiceImpl" << std::endl;
    for (const auto& kv : request->inputs()) {
      std::vector<std::string> output;
      if (absl::EndsWith(kv.first, "_id")) {
        std::transform(kv.second.int64_val().begin(),
                       kv.second.int64_val().end(), std::back_inserter(output),
                       [](int64_t id) { return std::to_string(id); });
      } else if (absl::EndsWith(kv.first, "_value")) {
        std::transform(kv.second.float_val().begin(),
                       kv.second.float_val().end(), std::back_inserter(output),
                       [](float value) { return std::to_string(value); });
      } else {
        LOG(FATAL) << "Inputs' key should end with '_id' or '_value'";
      }

      LOG(INFO) << absl::StrFormat("%s: %s", kv.first,
                                   absl::StrJoin(output, " "));
    }

    response->mutable_model_spec()->CopyFrom(request->model_spec());
    return grpc::Status::OK;
  }
};

class DummySyncServer {
 public:
  explicit DummySyncServer(std::string target);

  void Shutdown() const;

  const std::string& GetTarget() const;

  int GetSelectedPort() const;

 private:
  std::string target_;

  int selected_port_;

  PredictionServiceImpl service_;

  std::unique_ptr<grpc::Server> server_;
};

}  // namespace parameter_sync
}  // namespace monolith

#endif  // MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_PARAMETER_SYNC_DUMMY_SYNC_SERVER_H_
