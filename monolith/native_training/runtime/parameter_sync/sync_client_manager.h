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

#ifndef MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_PARAMETER_SYNC_SYNC_CLIENT_MANAGER_H_
#define MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_PARAMETER_SYNC_SYNC_CLIENT_MANAGER_H_

#include "absl/synchronization/mutex.h"

#include "monolith/native_training/runtime/parameter_sync/request_splitter.h"
#include "monolith/native_training/runtime/parameter_sync/sync_client_interface.h"

namespace monolith {
namespace parameter_sync {

class SyncClientManager {
 public:
  SyncClientManager(
      std::function<std::unique_ptr<SyncClientInterface>(const std::string&)>
          client_factory);

  PushResult Push(const PushRequest& request, const std::string& model_name,
                  const std::string& signature_name) const
      ABSL_SHARED_LOCKS_REQUIRED(mu_);

  bool TryReplace(
      const google::protobuf::RepeatedPtrField<std::string>& targets)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

 private:
  std::string PushRequestDebugString(const PushRequest& request, int index,
                                     int total) const;

 private:
  RequestSplitter request_splitter_;

  // We create an individual ParameterSyncClient for each target, which is an
  // online ps shard replica. Typically, each target has corresponding hash
  // tables like hash_tables_.
  std::map<std::string, std::unique_ptr<SyncClientInterface>> clients_
      ABSL_GUARDED_BY(mu_);

  std::function<std::unique_ptr<SyncClientInterface>(const std::string&)>
      client_factory_;

  mutable absl::Mutex mu_;
};

}  // namespace parameter_sync
}  // namespace monolith

#endif  // MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_PARAMETER_SYNC_SYNC_CLIENT_MANAGER_H_
