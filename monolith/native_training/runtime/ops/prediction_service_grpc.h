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

/* Copyright 2020 Google Inc. All Rights Reserved.
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

#ifndef MONOLITH_NATIVE_TRAINING_RUNTIME_OPS_PREDICTION_SERVICE_GRPC_H_
#define MONOLITH_NATIVE_TRAINING_RUNTIME_OPS_PREDICTION_SERVICE_GRPC_H_
#include <string>
#include <thread>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"

namespace tensorflow {
namespace monolith_tf {

class RemotePredictCQTag {
 public:
  RemotePredictCQTag(
      ::grpc::CompletionQueue *cq, ::grpc::ClientContext *rpc,
      std::unique_ptr<::tensorflow::serving::PredictionService::Stub> *stub_,
      ::tensorflow::serving::PredictRequest *request,
      ::tensorflow::serving::PredictResponse *response,
      std::function<void(grpc::Status)> callback)
      : response_(response), callback_(std::move(callback)) {
    std::unique_ptr<
        grpc::ClientAsyncResponseReader<::tensorflow::serving::PredictResponse>>
        rpc_call = (*stub_)->AsyncPredict(rpc, *request, cq);
    rpc_call->Finish(response, &status_, reinterpret_cast<void *>(this));
  };
  ~RemotePredictCQTag() {}

  // OnCompleted is invoked when the RPC has finished.
  // Implementations of OnCompleted can delete *this.
  void OnCompleted(bool ok) {
    callback_(status_);
    delete this;
  }

 private:
  ::tensorflow::serving::PredictResponse *response_;
  std::function<void(grpc::Status)> callback_;
  grpc::Status status_;
};

class CompletionQueueWithThreads {
 public:
  explicit CompletionQueueWithThreads(const size_t thread_num);

  ~CompletionQueueWithThreads();

  ::grpc::CompletionQueue *GetCompletionQueue();

 private:
  std::atomic_ullong queue_idx_;
  std::vector<::grpc::CompletionQueue> queues_;
  std::vector<std::unique_ptr<std::thread>> cq_threads_;
};

::grpc::CompletionQueue *GetSharedCompletionQueue();

// gRPC based communication point with PredictionService.
class PredictionServiceGrpcPerAddress {
 public:
  using DoneCallback = std::function<void()>;
  explicit PredictionServiceGrpcPerAddress(const std::string &target_address);

  void Predict(
      ::tensorflow::serving::PredictRequest *request,
      ::tensorflow::serving::PredictResponse *response,
      std::function<void(absl::Status status, DoneCallback &&)> callback,
      int64_t max_rpc_deadline_millis, DoneCallback op_done);

 private:
  std::unique_ptr<::tensorflow::serving::PredictionService::Stub> stub_;
};

class PredictionServiceGrpc {
 public:
  using DoneCallback = std::function<void()>;

  void Predict(
      tensorflow::serving::PredictRequest *request,
      tensorflow::serving::PredictResponse *response,
      std::function<void(absl::Status status, DoneCallback &&)> callback,
      int64_t max_rpc_deadline_millis, DoneCallback op_done) {
    size_t idx = std::rand() % services_.size();
    services_[idx]->Predict(request, response, callback,
                            max_rpc_deadline_millis, op_done);
  }

  explicit PredictionServiceGrpc(const std::vector<std::string> &address_list) {
    size_t n = address_list.size();
    services_.reserve(n);
    for (const auto &addr : address_list) {
      services_.push_back(
          std::make_unique<PredictionServiceGrpcPerAddress>(addr));
    }
  }

 private:
  std::vector<std::unique_ptr<PredictionServiceGrpcPerAddress>> services_;
};

}  // namespace monolith_tf
}  // namespace tensorflow

#endif  // MONOLITH_NATIVE_TRAINING_RUNTIME_OPS_PREDICTION_SERVICE_GRPC_H_
