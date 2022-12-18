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

#include "monolith/native_training/runtime/ops/agent_heartbeat.h"

namespace tensorflow {
namespace monolith_tf {

const char *const kAgentPortEnvVar = "PORT2";

std::unique_ptr<monolith::serving::agent_service::AgentService::Stub>
NewAgentStub() {
  const char *agent_port = getenv(kAgentPortEnvVar);
  if (agent_port == nullptr) {
    LOG(FATAL) << "missing env " << kAgentPortEnvVar;
    return nullptr;
  }
  auto channel = grpc::CreateChannel("localhost:" + std::string(agent_port),
                                     grpc::InsecureChannelCredentials());
  return monolith::serving::agent_service::AgentService::NewStub(channel);
}
void RemoveOtherAddrsIfThereIsLocalAddr(
    const std::string &host,
    google::protobuf::RepeatedPtrField<std::string> *addrs) {
  std::string local_shard;
  for (const std::string &addr : *addrs) {
    if (addr.find(host) == 0) {
      local_shard = addr;
      break;
    }
  }
  if (!local_shard.empty()) {
    addrs->Clear();
    addrs->Add(std::move(local_shard));
  }
}
int GetApiVersion(
    const absl::flat_hash_map<std::string, std::vector<std::string>>
        &model_addrs) {
  for (const auto it : model_addrs) {
    const std::string &model = it.first;
    if (model.find(":ps") != std::string::npos) {
      return 1;
    }
  }
  return 0;
}

std::string GetModelKey(absl::string_view model_name,
                        absl::string_view server_type, int index) {
  return absl::StrCat(model_name, ":", server_type, ":", index);
}

std::string GetModelPsKey(absl::string_view model_name, int index) {
  return GetModelKey(model_name, "ps", index);
}

}  // namespace monolith_tf
}  // namespace tensorflow
