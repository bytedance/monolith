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

#ifndef MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_UTILS
#define MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_UTILS
namespace monolith {
namespace hash_table {

inline void* AddOffset(void* p, int offset) {
  return reinterpret_cast<char*>(p) + offset;
}

inline const void* AddOffset(const void* p, int offset) {
  return reinterpret_cast<const char*>(p) + offset;
}

template <bool compute_keys_per_table = true>
std::pair<int, int> ComputeFusedOffsets(
    const int* slot_size_vec,  // num_tables * num_shards
    const int* table_dims,     // num_tables
    int num_tables, int num_shards,
    int* key_offsets,     // num_tables * num_shards + 1
    int* emb_offsets,     // num_tables * num_shards + 1
    int* keys_per_table,  // num_tables
    int* emb_splits       // num_shards
) {
  if (compute_keys_per_table)
    std::fill(keys_per_table, keys_per_table + num_tables, 0);
  int total_keys = 0;
  int total_embs = 0;
  int prev_total_emb = 0;
  key_offsets[0] = emb_offsets[0] = 0;
  for (int shard_id = 0; shard_id < num_shards; shard_id++) {
    for (int table_id = 0; table_id < num_tables; table_id++) {
      int idx = num_tables * shard_id + table_id;
      int slot_sz = slot_size_vec[idx];
      int segment_dim = table_dims[table_id] * slot_sz;

      if (compute_keys_per_table) keys_per_table[table_id] += slot_sz;
      total_keys += slot_sz;
      total_embs += segment_dim;

      key_offsets[idx + 1] = key_offsets[idx] + slot_sz;
      emb_offsets[idx + 1] = emb_offsets[idx] + segment_dim;
    }
    emb_splits[shard_id] = total_embs - prev_total_emb;
    prev_total_emb = total_embs;
  }
  return std::make_pair(total_keys, total_embs);
}

}  // namespace hash_table
}  // namespace monolith
#endif  // MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_UTILS
