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
#include "monolith/native_training/runtime/ops/prediction_service_grpc.h"

#include "absl/time/clock.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"

namespace tensorflow {
namespace monolith_tf {
namespace {

absl::Status FromGrpcStatus(const ::grpc::Status& s) {
  if (s.ok()) {
    return absl::Status();
  }
  return absl::Status(static_cast<absl::StatusCode>(s.error_code()),
                      s.error_message());
}

int GetCallbackThreadNum() {
  const char * thread_num_str = std::getenv("MONOLITH_GRPC_REMOTE_CALLBACK_THREADS");
  if (thread_num_str == nullptr) {
    return 10;
  }
  return std::stoi(std::string(thread_num_str));
}

}  // namespace

::grpc::CompletionQueue* GetSharedCompletionQueue() {
  static CompletionQueueWithThreads* cq_with_threads = new CompletionQueueWithThreads(GetCallbackThreadNum());
  return cq_with_threads->GetCompletionQueue();
}

CompletionQueueWithThreads::CompletionQueueWithThreads(const size_t thread_num) {
  queues_ = std::vector<::grpc::CompletionQueue>(thread_num);
  for (size_t i = 0; i < thread_num; ++i) {
    auto* cq = &queues_[i];
    auto pooling_fn = [cq]() {
      void* p_tag;
      bool ok;
      while (cq->Next(&p_tag, &ok)) {
        RemotePredictCQTag* cq_tag = static_cast<RemotePredictCQTag*>(p_tag);
        cq_tag->OnCompleted(ok);
      }
    };
    cq_threads_.emplace_back(std::make_unique<std::thread>(pooling_fn));
  }
}

CompletionQueueWithThreads::~CompletionQueueWithThreads() {
  for (size_t i = 0; i < queues_.size(); ++i) {
    queues_[i].Shutdown();
  }
  for (size_t i = 0; i < cq_threads_.size(); ++i) {
    cq_threads_[i]->join();
  }
}

::grpc::CompletionQueue * CompletionQueueWithThreads::GetCompletionQueue() {
  return &queues_[queue_idx_++ % queues_.size()];
}

PredictionServiceGrpcPerAddress::PredictionServiceGrpcPerAddress(
    const std::string& target_address) {
  // TODO(b/159739577): Set security channel from incoming rpc request.
  auto channel = ::grpc::CreateChannel(target_address,
                                       ::grpc::InsecureChannelCredentials());
  stub_ = tensorflow::serving::PredictionService::NewStub(channel);
}

void PredictionServiceGrpcPerAddress::Predict(
    tensorflow::serving::PredictRequest* request,
    tensorflow::serving::PredictResponse* response,
    std::function<void(absl::Status status, DoneCallback&&)> callback,
    int64_t max_rpc_deadline_millis, DoneCallback op_done) {
  ::grpc::ClientContext* rpc = new ::grpc::ClientContext;
  DoneCallback rpc_done = [rpc, done = op_done]() {
    delete rpc;
    done();
  };
  std::function<void(::grpc::Status)> wrapped_callback =
      [callback,
       rpc_done = std::move(rpc_done)](::grpc::Status status) mutable {
        callback(FromGrpcStatus(status), std::forward<DoneCallback>(rpc_done));
      };

  new RemotePredictCQTag(GetSharedCompletionQueue(), rpc, &stub_, request, response,
                         std::move(wrapped_callback));
}

}  // namespace monolith_tf
}  // namespace tensorflow
