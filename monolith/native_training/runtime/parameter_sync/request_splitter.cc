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

#include "monolith/native_training/runtime/parameter_sync/request_splitter.h"

#include "glog/logging.h"

namespace monolith {
namespace parameter_sync {
namespace {

void SplitTable(const PushRequest::DeltaEmbeddingHashTable& table,
                int split_num, int i,
                PushRequest::DeltaEmbeddingHashTable* target_table) {
  size_t delta_size = table.fids_size();
  int dim_size = table.dim_size();
  size_t q = delta_size / split_num;
  size_t part_size = i + 1 == split_num ? q + delta_size % split_num : q;
  target_table->set_unique_id(table.unique_id());
  target_table->set_dim_size(table.dim_size());
  auto* mutable_fids = target_table->mutable_fids();
  auto* mutable_embeddings = target_table->mutable_embeddings();

  // TODO(leqi.zou): This seems not very mem efficient.
  for (size_t j = 0; j < part_size; ++j) {
    int index = i * q + j;
    int64_t fid = table.fids(index);
    mutable_fids->Add(fid);
    const float* embedding = table.embeddings().data() + index * dim_size;
    mutable_embeddings->Add(embedding, embedding + dim_size);
  }
}

}  // namespace

std::vector<PushRequest> RequestSplitter::Split(
    const PushRequest& push_request, int64_t max_message_length) const {
  DCHECK_GT(max_message_length, 0);
  size_t byte_size = push_request.ByteSizeLong();
  if (byte_size <= max_message_length) {
    return {push_request};
  }

  size_t split_num = (byte_size + max_message_length - 1) / max_message_length;
  const std::string& model_name = push_request.model_name();
  const std::string& signature_name = push_request.signature_name();
  int64_t timeout_in_ms = push_request.timeout_in_ms();

  std::vector<PushRequest> requests;
  requests.reserve(split_num);
  for (size_t i = 0; i < split_num; ++i) {
    requests.emplace_back();
    PushRequest* request = &requests.back();
    request->set_model_name(model_name);
    request->set_signature_name(signature_name);
    request->mutable_delta_hash_tables()->Reserve(
        push_request.delta_hash_tables_size());
    request->mutable_delta_multi_hash_tables()->Reserve(
        push_request.delta_multi_hash_tables_size());
    request->set_timeout_in_ms(timeout_in_ms);

    for (const auto& table : push_request.delta_hash_tables()) {
      auto* delta_hash_table = request->mutable_delta_hash_tables()->Add();
      SplitTable(table, split_num, i, delta_hash_table);
    }

    for (const auto& table : push_request.delta_multi_hash_tables()) {
      auto* delta_multi_hash_table =
          request->mutable_delta_multi_hash_tables()->Add();
      SplitTable(table, split_num, i, delta_multi_hash_table);
    }
  }

  return requests;
}
}  // namespace parameter_sync
}  // namespace monolith
