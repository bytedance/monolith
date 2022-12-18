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

#ifndef MONOLITH_NATIVE_TRAINING_RUNTIME_OPS_PARAMETER_SYNC_TF_BRIDGE_H_
#define MONOLITH_NATIVE_TRAINING_RUNTIME_OPS_PARAMETER_SYNC_TF_BRIDGE_H_

#include <memory>
#include <utility>

#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "monolith/native_training/runtime/ops/embedding_hash_table_tf_bridge.h"
#include "monolith/native_training/runtime/ops/multi_hash_table.h"
#include "monolith/native_training/runtime/parameter_sync/dummy_sync_server.h"
#include "monolith/native_training/runtime/parameter_sync/sync_client_interface.h"
#include "monolith/native_training/runtime/parameter_sync/sync_client_manager.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"

namespace tensorflow {
namespace monolith_tf {

// 64MB
const size_t MAX_TOUCHED_KEYS = 64 * 1024 * 1024 / (8 * 4);

class DummySyncServerTfBridge : public ResourceBase {
 public:
  using DummySyncServer = monolith::parameter_sync::DummySyncServer;

  explicit DummySyncServerTfBridge(const std::string& target) {
    server_ = std::make_unique<DummySyncServer>(target);
  }

  void Shutdown() const { server_->Shutdown(); }

  std::string GetTarget() const { return server_->GetTarget(); }

  int GetSelectedPort() const { return server_->GetSelectedPort(); }

  std::string DebugString() const override {
    return absl::StrFormat("DummySyncServerTfBridge target = %s",
                           server_->GetTarget());
  }

 private:
  std::unique_ptr<DummySyncServer> server_;
};

class ParameterSyncClientTfBridge : public ResourceBase {
 public:
  using SyncClientInterface = monolith::parameter_sync::SyncClientInterface;
  using PushResult = monolith::parameter_sync::PushResult;
  using SyncClientManager = monolith::parameter_sync::SyncClientManager;

  ParameterSyncClientTfBridge(
      bool is_dummy_sync_client,
      std::function<std::unique_ptr<SyncClientInterface>(const std::string&)>
          client_factory)
      : is_dummy_sync_client_(is_dummy_sync_client) {
    sync_client_manager_ =
        std::make_unique<SyncClientManager>(std::move(client_factory));
    if (!IsDummySyncClient()) {
      touched_key_set_ = std::move(
          std::make_unique<HopscotchHashSet<std::pair<int64_t, const void*>>>(
              MAX_TOUCHED_KEYS, 1024));
    }
  }

  Status Push(const std::string& model_name, const std::string& signature_name,
              int64_t timeout_in_ms, PushResult* result) const;

  Status TryReplace(
      const google::protobuf::RepeatedPtrField<std::string>& targets);

  Status AddHashTableResource(const std::string& name,
                              EmbeddingHashTableTfBridge* hash_table)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    absl::WriterMutexLock l(&mu_);
    DCHECK(!hash_tables_.count(name));
    if (mtable_ != nullptr) {
      return errors::InvalidArgument(
          "Only one type of tables can be set. MultiHashTable is set.");
    }
    hash_tables_[name] = hash_table;
    return Status::OK();
  }

  Status SetMultiHashTableResource(MultiHashTable* mtable)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    absl::WriterMutexLock l(&mu_);
    if (mtable_ != nullptr) {
      return errors::AlreadyExists(
          "The sync client is set a mtable resource already.");
    }
    if (hash_tables_.size() > 0) {
      return errors::InvalidArgument(
          "Only one type of tables can be set. HashTable is set.");
    }
    mtable_ = mtable;
    return Status::OK();
  }

  std::string DebugString() const override {
    std::vector<std::string> hash_table_names;
    hash_table_names.reserve(hash_tables_.size());
    std::transform(hash_tables_.begin(), hash_tables_.end(),
                   std::back_inserter(hash_table_names),
                   [](const auto& kv) { return kv.first; });
    return absl::StrFormat("hash tables = [%s]",
                           absl::StrJoin(hash_table_names, ", "));
  }

  bool IsDummySyncClient() const { return is_dummy_sync_client_; }

  HopscotchHashSet<std::pair<int64_t, const void*>>* GetTouchedKeySet() {
    return touched_key_set_.get();
  }

 private:
  // hash table name -> hash table resource
  std::map<std::string, EmbeddingHashTableTfBridge*> hash_tables_
      ABSL_GUARDED_BY(mu_);

  MultiHashTable* mtable_ = nullptr;
  std::unique_ptr<SyncClientManager> sync_client_manager_;
  std::unique_ptr<HopscotchHashSet<std::pair<int64_t, const void*>>>
      touched_key_set_;

  mutable absl::Mutex mu_;

  bool is_dummy_sync_client_;
};

}  // namespace monolith_tf
}  // namespace tensorflow

#endif  // MONOLITH_NATIVE_TRAINING_RUNTIME_OPS_PARAMETER_SYNC_TF_BRIDGE_H_
