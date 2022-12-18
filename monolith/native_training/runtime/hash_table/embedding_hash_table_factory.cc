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

#include "monolith/native_training/runtime/hash_table/embedding_hash_table_factory.h"

#include <exception>

#include "absl/strings/str_format.h"
#include "monolith/native_training/runtime/hash_table/cuckoohash/cuckoo_embedding_hash_table.h"
#include "monolith/native_training/runtime/hash_table/entry_accessor.h"

namespace monolith {
namespace hash_table {

std::unique_ptr<EmbeddingHashTableInterface> NewEmbeddingHashTableFromConfig(
    EmbeddingHashTableConfig config, cudaStream_t stream) {
  std::unique_ptr<EntryAccessorInterface> accessor =
      NewEntryAccessor(config.entry_config());
  switch (config.type_case()) {
    case EmbeddingHashTableConfig::kCuckoo:
      return NewCuckooEmbeddingHashTable(
          config.cuckoo(), std::move(accessor), config.entry_type(),
          config.initial_capacity(), config.slot_expire_time_config());
    default:
      throw std::invalid_argument(absl::StrFormat(
          "Unknown type of hash table. %s", config.ShortDebugString()));
  }
}

}  // namespace hash_table
}  // namespace monolith
