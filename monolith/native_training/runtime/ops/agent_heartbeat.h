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

#ifndef MONOLITH_NATIVE_TRAINING_RUNTIME_OPS_AGENT_HEARTBEAT_H_
#define MONOLITH_NATIVE_TRAINING_RUNTIME_OPS_AGENT_HEARTBEAT_H_

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <thread>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "glog/logging.h"
#include "grpcpp/channel.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "monolith/agent_service/agent_service.grpc.pb.h"
#include "monolith/agent_service/agent_service.pb.h"
#include "monolith/native_training/runtime/ops/net_utils.h"
#include "tensorflow/core/platform/default/logging.h"

namespace tensorflow {
namespace monolith_tf {

extern const char *const kAgentPortEnvVar;

std::unique_ptr<monolith::serving::agent_service::AgentService::Stub>
NewAgentStub();

void RemoveOtherAddrsIfThereIsLocalAddr(
    const std::string &host,
    google::protobuf::RepeatedPtrField<std::string> *addrs);

int GetApiVersion(const absl::flat_hash_map<
                  std::string, std::vector<std::string>> &model_addrs);

std::string GetModelKey(absl::string_view model_name,
                        absl::string_view server_type, int index);

std::string GetModelPsKey(absl::string_view model_name, int index);

// Provide getting PredictionServiceType by task,
// while update cache data periodically by calling agent service.
template <typename PredictionServiceType>
class AgentHeartbeat {
 public:
  using AgentService = monolith::serving::agent_service::AgentService;

  AgentHeartbeat() : AgentHeartbeat(NewAgentStub(), absl::Seconds(15)) {}

  ~AgentHeartbeat() {
    stopped_.Notify();
    heartbeat_thread_->join();
  }

  explicit AgentHeartbeat(
      std::unique_ptr<AgentService::StubInterface> agent_stub,
      absl::Duration heartbeat_interval)
      : agent_stub_(std::move(agent_stub)),
        heartbeat_interval_(heartbeat_interval) {
    // Manual update once
    UpdateAddrs();
    {
      absl::ReaderMutexLock l(&mu_);
      api_version_ = GetApiVersion(model_addrs_);
    }
    heartbeat_thread_ = std::make_unique<std::thread>(HeartbeatFunc, this);
  }

  static const AgentHeartbeat &GetInstance() {
    static AgentHeartbeat *instance = new AgentHeartbeat();
    return *instance;
  }

  AgentHeartbeat(AgentHeartbeat const &) = delete;
  void operator=(AgentHeartbeat const &) = delete;

  // Old API encodes in this way:
  // API version 0:
  // model key: `ps:1`
  // model_name: `ps_1`
  //
  // API version 1:
  // model key: RealModel:ps:1
  // model name: RealModel:ps:1
  int api_version() const { return api_version_; }

  // Old APIs. Going to be deprecated.
  std::shared_ptr<PredictionServiceType> GetPredictionServiceByIdx(
      int idx) const {
    return GetPredictionService(absl::StrCat("ps:", idx));
  }

  std::shared_ptr<PredictionServiceType> GetPredictionService(
      absl::string_view model_key) const {
    absl::ReaderMutexLock l(&mu_);
    auto iter = service_by_model_.find(model_key);
    if (iter == service_by_model_.end()) {
      LOG(ERROR) << "model key doesn't exist: " << model_key;
      return nullptr;
    }
    return iter->second;
  }

  void TestOnly_UpdateAddrs() { UpdateAddrs(); }
  absl::flat_hash_map<std::string, std::vector<std::string>>
  TestOnly_GetModelAddrs() {
    absl::ReaderMutexLock l(&mu_);
    return model_addrs_;
  }

 private:
  static void HeartbeatFunc(AgentHeartbeat *agent) {
    absl::Time now = absl::Now();
    while (!agent->stopped_.WaitForNotificationWithTimeout(
        now + agent->heartbeat_interval_ - absl::Now())) {
      now = absl::Now();
      agent->UpdateAddrs();
    }
  }

  // Updates the current model addresses.
  void GetAddrs(monolith::serving::agent_service::ServerType server_type,
                absl::flat_hash_map<std::string, std::vector<std::string>>& new_model_addrs) {
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() +
                         absl::ToChronoSeconds(absl::Seconds(5)));
    monolith::serving::agent_service::HeartBeatRequest req;
    req.set_server_type(server_type);
    monolith::serving::agent_service::HeartBeatResponse resp;
    grpc::Status status = agent_stub_->HeartBeat(&context, req, &resp);
    if (!status.ok()) {
      LOG(ERROR) << "agent_service->HeartBeat error, code: "
                 << status.error_code() << ", msg: " << status.error_message();
      return;
    }

    const std::string my_host_ip = GetMyHostIp();

    for (auto &kv : *resp.mutable_addresses()) {
      const std::string &model = kv.first;
      auto *resp_addrs = kv.second.mutable_address();
      std::vector<std::string> addr_list;
      addr_list.reserve(resp_addrs->size());
      for (const std::string &addr : *resp_addrs) {
        addr_list.push_back(addr);
      }
      new_model_addrs.insert({model, std::move(addr_list)});
    }
  }

  void UpdateAddrs() {
    absl::flat_hash_map<std::string, std::vector<std::string>> new_model_addrs;
    GetAddrs(monolith::serving::agent_service::PS, new_model_addrs);
    GetAddrs(monolith::serving::agent_service::DENSE, new_model_addrs);

    bool same;
    {
      absl::ReaderMutexLock l(&mu_);
      same = (new_model_addrs == model_addrs_);
    }

    if (!same) {
      absl::flat_hash_map<std::string, std::shared_ptr<PredictionServiceType>>
          new_service_by_model;
      for (auto &kv : new_model_addrs) {
        new_service_by_model.emplace(
            kv.first, std::make_shared<PredictionServiceType>(kv.second));
      }

      {
        absl::MutexLock l(&mu_);
        model_addrs_.swap(new_model_addrs);
        service_by_model_.swap(new_service_by_model);
      }
    }
  }

  absl::Notification stopped_;
  std::unique_ptr<AgentService::StubInterface> agent_stub_;
  absl::Duration heartbeat_interval_;
  int api_version_;

  // This is for the public API to use.
  absl::flat_hash_map<std::string, std::vector<std::string>> model_addrs_;
  mutable absl::Mutex mu_;
  absl::flat_hash_map<std::string, std::shared_ptr<PredictionServiceType>>
      service_by_model_ GUARDED_BY(mu_);

  std::unique_ptr<std::thread> heartbeat_thread_;
};

// Gets the model key.
std::string GetModelPsKey(absl::string_view model_name, int index);
std::string GetModelKey(absl::string_view model_name,
                        absl::string_view server_type, int index);

}  // namespace monolith_tf
}  // namespace tensorflow
#endif  // MONOLITH_NATIVE_TRAINING_RUNTIME_OPS_AGENT_HEARTBEAT_H_
