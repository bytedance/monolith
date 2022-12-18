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

#include "monolith/native_training/runtime/ops/parameter_sync_tf_bridge.h"

#include "absl/strings/str_cat.h"

namespace tensorflow {
namespace monolith_tf {
namespace {

using ::monolith::parameter_sync::PushRequest;
using ::monolith::parameter_sync::PushResult;

void AddIdToDelta(
    const std::string& name, const EmbeddingHashTableTfBridge& table,
    const std::vector<int64_t>& ids,
    monolith::parameter_sync::PushRequest_DeltaEmbeddingHashTable* delta) {
  int dim_size = static_cast<int>(table.dim_size());
  std::vector<float> embedding(dim_size);
  delta->set_unique_id(name);
  delta->set_dim_size(dim_size);

  int delta_size = static_cast<int>(ids.size());
  auto* mutable_fids = delta->mutable_fids();
  auto* embeddings = delta->mutable_embeddings();
  mutable_fids->Reserve(delta_size);
  embeddings->Reserve(delta_size * dim_size);

  for (int64_t id : ids) {
    mutable_fids->Add(id);
    table.Lookup(nullptr, id, embedding.data());
    embeddings->Add(embedding.data(), embedding.data() + dim_size);
  }
}

}  // namespace

Status ParameterSyncClientTfBridge::Push(const std::string& model_name,
                                         const std::string& signature_name,
                                         int64_t timeout_in_ms,
                                         PushResult* result) const {
  try {
    PushRequest request;
    const bool is_mtable = mtable_ != nullptr;
    request.set_model_name(model_name);
    if (is_mtable) {
      // TODO(leqi.zou): Currently it is hard coded.
      // Will revisit this part later.
      request.set_signature_name(
          absl::StrCat(mtable_->shared_name(), "/raw_assign"));
      request.mutable_delta_multi_hash_tables()->Reserve(mtable_->size());
    } else {
      request.set_signature_name(signature_name);
      request.mutable_delta_hash_tables()->Reserve(hash_tables_.size());
    }
    request.set_timeout_in_ms(timeout_in_ms);
    std::vector<std::pair<int64_t, const void*>> fids_and_tables =
        touched_key_set_->GetAndClear();
    std::unordered_map<const void*, std::vector<int64_t>> table_to_fids;
    for (const auto& fid_and_table : fids_and_tables) {
      table_to_fids[fid_and_table.second].push_back(fid_and_table.first);
    }

    if (is_mtable) {
      for (int i = 0; i < mtable_->size(); ++i) {
        AddIdToDelta(mtable_->name(i), *mtable_->table(i),
                     table_to_fids[mtable_->table(i)],
                     request.mutable_delta_multi_hash_tables()->Add());
      }
    } else {
      for (const auto& kv : hash_tables_) {
        const std::string& name = kv.first;
        const auto* table = kv.second;
        AddIdToDelta(name, *table, table_to_fids[table],
                     request.mutable_delta_hash_tables()->Add());
      }
    }

    if (fids_and_tables.size() > 0) {
      *result = sync_client_manager_->Push(request, model_name, signature_name);
      LOG_EVERY_N_SEC(INFO, 600) << "Response: " << result->ShortDebugString();
    } else {
      LOG_EVERY_N_SEC(INFO, 600) << "No updated FIDs!";
    }
    return Status::OK();
  } catch (const std::exception& e) {
    return errors::InvalidArgument(e.what());
  }
}

Status ParameterSyncClientTfBridge::TryReplace(
    const google::protobuf::RepeatedPtrField<std::string>& targets) {
  try {
    sync_client_manager_->TryReplace(targets);
    return Status::OK();
  } catch (const std::exception& e) {
    return errors::InvalidArgument(e.what());
  }
}

}  // namespace monolith_tf
}  // namespace tensorflow
