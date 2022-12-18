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

#ifndef MONOLITH_NATIVE_TRAINING_RUNTIME_PARAMETER_SYNC_PARAMETER_SYNC_CLIENT_H_
#define MONOLITH_NATIVE_TRAINING_RUNTIME_PARAMETER_SYNC_PARAMETER_SYNC_CLIENT_H_

#include "glog/logging.h"
#include "grpcpp/grpcpp.h"
#include "monolith/native_training/runtime/parameter_sync/parameter_sync.grpc.pb.h"
#include "monolith/native_training/runtime/parameter_sync/sync_client_interface.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"

namespace monolith {
namespace parameter_sync {

class ParameterSyncClient final : public SyncClientInterface {
 public:
  explicit ParameterSyncClient(std::string target)
      : ParameterSyncClient(CreateStub(target)) {
    target_ = std::move(target);
  }

  explicit ParameterSyncClient(
      std::unique_ptr<tensorflow::serving::PredictionService::StubInterface>
          stub)
      : stub_(std::move(stub)) {}

  // Assembles the client's payload, sends it and presents the response back
  // from the server.
  grpc::Status Push(const PushRequest& request,
                    PushResponse* response) const override;

  // Ideally we should mock stub to simulate the behavior of this class.
  // However, there are some problems to generate mock class.
  // We just verify request conversion here.
  struct ConvertResult {
    tensorflow::serving::PredictRequest req;
    int total = 0;
  };
  static ConvertResult Convert(const PushRequest& req);

 private:
  static std::unique_ptr<tensorflow::serving::PredictionService::Stub>
  CreateStub(const std::string& target) {
    // 32M
    const int MAX_MESSAGE_LENGTH = 32 * 1024 * 1024;
    grpc::ChannelArguments arguments;
    arguments.SetMaxSendMessageSize(MAX_MESSAGE_LENGTH);
    arguments.SetMaxReceiveMessageSize(MAX_MESSAGE_LENGTH);
    auto channel = grpc::CreateCustomChannel(
        target, grpc::InsecureChannelCredentials(), arguments);
    return tensorflow::serving::PredictionService::NewStub(channel);
  }

  std::string target_;

  std::unique_ptr<tensorflow::serving::PredictionService::StubInterface> stub_;
};

}  // namespace parameter_sync
}  // namespace monolith

#endif  // MONOLITH_NATIVE_TRAINING_RUNTIME_PARAMETER_SYNC_PARAMETER_SYNC_CLIENT_H_
