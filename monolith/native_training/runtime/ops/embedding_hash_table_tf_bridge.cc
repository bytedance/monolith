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

#include "monolith/native_training/runtime/ops/embedding_hash_table_tf_bridge.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <exception>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "monolith/native_training/runtime/common/metrics.h"
#include "monolith/native_training/runtime/hash_table/embedding_hash_table_factory.h"
#include "monolith/native_training/runtime/hash_table/optimizer/avx_utils.h"

namespace tensorflow {
namespace monolith_tf {
namespace {

namespace hash_table = ::monolith::hash_table;
using ::monolith::hash_table::EmbeddingHashTableConfig;

constexpr int64_t kSecPerHour = 60 * 60;

Status ValidateDim(const Tensor& t, int64 expected_dim) {
  if (TF_PREDICT_FALSE(t.NumElements() != expected_dim)) {
    return errors::InvalidArgument("The dim doesn't match expectation. ",
                                   t.NumElements(), " vs ", expected_dim);
  }
  return Status::OK();
}

}  // namespace

Status EmbeddingHashTableTfBridge::New(
    monolith::hash_table::EmbeddingHashTableConfig config,
    HashFilterTfBridge* hash_filter, EmbeddingHashTableTfBridge** new_bridge,
    const std::string& name, cudaStream_t stream) {
  auto bridge = core::RefCountPtr<EmbeddingHashTableTfBridge>(
      new EmbeddingHashTableTfBridge(hash_filter));
  bridge->config_ = config;
  bridge->name_ = name;
  bridge->dim_size_ = 0;
  for (const auto& segment : config.entry_config().segments()) {
    bridge->dim_size_ += segment.dim_size();
  }
  try {
    bridge->table_ =
        hash_table::NewEmbeddingHashTableFromConfig(config, stream);
  } catch (const std::exception& e) {
    return errors::InvalidArgument(e.what());
  }

  bridge->max_update_ts_sec_ = std::make_unique<std::atomic<int64_t>>(0);
  bridge->last_evict_ts_sec_ = std::make_unique<std::atomic<int64_t>>(0);

  auto& bridge_ref = *bridge;
  bridge_ref.evict_finished_ = true;
  if (config.enable_feature_eviction()) {
    bridge_ref.evict_finished_ = false;
    const int evict_features_every_n_hours =
        config.feature_evict_every_n_hours();
    auto evict_func = [&bridge_ref, evict_features_every_n_hours]() {
      while (!bridge_ref.evict_finished_) {
        const int64_t last_evict_ts_sec = bridge_ref.last_evict_ts_sec();
        const int64_t current_ts_sec =
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count();
        if (last_evict_ts_sec == 0) {
          bridge_ref.set_last_evict_ts_sec(current_ts_sec);
        }
        if (last_evict_ts_sec != 0 &&
            current_ts_sec - last_evict_ts_sec >=
                evict_features_every_n_hours * kSecPerHour) {
          LOG_EVERY_N_SEC(INFO, 60)
              << "embedding_hash_table_tf_bridge: started feature eviction, "
                 "current_ts_sec : "
              << current_ts_sec << " last_evict_ts_sec : " << last_evict_ts_sec
              << " max_update_ts_sec: " << bridge_ref.max_update_ts_sec();
          bridge_ref.table_->Evict(bridge_ref.max_update_ts_sec());
          bridge_ref.set_last_evict_ts_sec(current_ts_sec);
          LOG_EVERY_N_SEC(INFO, 60)
              << "embedding_hash_table_tf_bridge: finished feature eviction";
        }
        std::this_thread::sleep_for(std::chrono::seconds(10));
      }
    };
    bridge_ref.evict_thread_ = std::make_unique<std::thread>(evict_func);
  }

  *new_bridge = bridge.release();
  return Status::OK();
}

Status EmbeddingHashTableTfBridge::BatchLookup(OpKernelContext* ctx,
                                               const int num_ids, int64_t* ids,
                                               float* out_embedding,
                                               int64_t* hit_fid_count) const {
  try {
    std::vector<absl::Span<float>> out_embeddings;
    out_embeddings.reserve(num_ids);
    for (int i = 0; i < num_ids; ++i) {
      out_embeddings.push_back(
          absl::MakeSpan(out_embedding + i * dim_size(), dim_size()));
    }

    *hit_fid_count = table_->BatchLookup(absl::MakeSpan(ids, num_ids),
                                         absl::MakeSpan(out_embeddings));

    if (IsServingEntryType() && num_ids) {
      const std::string tagkv = absl::StrFormat("name=%s", name_);
      float hit_rate = *hit_fid_count / static_cast<float>(num_ids);
      monolith::GetMetrics()->emit_timer("lookup_fid_hit_rate", hit_rate,
                                         tagkv);
    }
    return Status::OK();
  } catch (const std::exception& e) {
    return errors::InvalidArgument(e.what());
  }
}


Status EmbeddingHashTableTfBridge::BatchLookupEntry(
    OpKernelContext* ctx, const int num_ids, int64_t* ids,
    EntryDump* out_entries) const {
  try {
    table_->BatchLookupEntry(absl::MakeSpan(ids, num_ids),
                             absl::MakeSpan(out_entries, num_ids));
    return Status::OK();
  } catch (const std::exception& e) {
    return errors::InvalidArgument(e.what());
  }
}

Status EmbeddingHashTableTfBridge::Lookup(OpKernelContext* ctx, int64 id,
                                          float* out_embedding) const {
  try {
    table_->Lookup(id, absl::MakeSpan(out_embedding, dim_size()));
    return Status::OK();
  } catch (const std::exception& e) {
    return errors::InvalidArgument(e.what());
  }
}

Status EmbeddingHashTableTfBridge::LookupEntry(OpKernelContext* ctx, int64 id,
                                               EntryDump* out_entry) const {
  try {
    table_->LookupEntry(id, absl::MakeSpan(out_entry, 1));
    return Status::OK();
  } catch (const std::exception& e) {
    return errors::InvalidArgument(e.what());
  }
}

Status EmbeddingHashTableTfBridge::Assign(OpKernelContext* ctx, int num_ids,
                                          const int64_t* ids,
                                          const float* embeddings,
                                          int64_t update_time) const {
  try {
    int64_t max_value = std::max(update_time, max_update_ts_sec_->load());
    max_update_ts_sec_->store(max_value);
    std::vector<int64_t> ids_after_filter;
    std::vector<absl::Span<const float>> embeddings_after_filter;
    ids_after_filter.reserve(num_ids);
    embeddings_after_filter.reserve(num_ids);
    for (int i = 0; i < num_ids; ++i) {
      int64_t id = ids[i];
      if (!table_->Contains(id) &&
          hash_filter_->ShouldBeFiltered(id, table_.get())) {
        continue;
      }
      ids_after_filter.push_back(id);
      embeddings_after_filter.emplace_back(
          absl::MakeSpan(embeddings + i * dim_size(), dim_size()));
    }

    table_->Assign(absl::MakeSpan(ids_after_filter),
                   absl::MakeSpan(embeddings_after_filter), update_time);
    return Status::OK();
  } catch (const std::exception& e) {
    return errors::InvalidArgument(e.what());
  }
}

Status EmbeddingHashTableTfBridge::AssignAdd(OpKernelContext* ctx, int64 id,
                                             const Tensor& tensor,
                                             int64_t update_time) const {
  // Here max_update_ts_sec_ only need a fuzzy maximum value.
  // We don't need use strict locking or compare_and_change
  // to change this max_update_ts_sec_. And we can save some performance here.
  int64_t max_value = std::max(update_time, max_update_ts_sec_->load());
  max_update_ts_sec_->store(max_value);

  if (!table_->Contains(id) &&
      hash_filter_->ShouldBeFiltered(id, table_.get())) {
    return Status::OK();
  }

  try {
    TF_RETURN_IF_ERROR(ValidateDim(tensor, dim_size_));
    auto span =
        absl::MakeConstSpan(static_cast<float*>(tensor.data()), dim_size());
    table_->AssignAdd(id, span, update_time);
    return Status::OK();
  } catch (const std::exception& e) {
    return errors::InvalidArgument(e.what());
  }
}

Status EmbeddingHashTableTfBridge::AssignAdd2(int64 id,
                                              absl::Span<const float> value,
                                              int64_t update_time) {
  int64_t max_value = std::max(update_time, max_update_ts_sec_->load());
  max_update_ts_sec_->store(max_value);

  if (hash_filter_->ShouldBeFiltered(id, table_.get())) {
    return Status::OK();
  }

  try {
    table_->AssignAdd(id, value, update_time);
    return Status::OK();
  } catch (const std::exception& e) {
    return errors::InvalidArgument(e.what());
  }
}

Status EmbeddingHashTableTfBridge::BatchOptimize(
    OpKernelContext* ctx, size_t num_ids, const int64_t* ids,
    const float* tensor, absl::Span<const float> learning_rates,
    int64_t update_time, bool enable_dedup, const int64_t global_step) const {
  int max_value = std::max(update_time, max_update_ts_sec_->load());
  max_update_ts_sec_->store(max_value);

  try {
    std::vector<int64_t> ids_after_filter;
    std::vector<absl::Span<const float>> grads_after_filter;
    // TODO(zouxuan): add theadlocal cache instead of allocating everytime.
    std::unique_ptr<float[]> cache_grads;
    if (enable_dedup) {
      // To avoid repeated alloc, we do a conservative block allocation.
      cache_grads = std::make_unique<float[]>(num_ids * dim_size());

      // The first step we do a dedup, where all grads and occurences are
      // grouped by IDs.
      absl::flat_hash_map<int64_t, float*> ids_to_grads;
      absl::flat_hash_map<int64_t, uint32_t> ids_to_counts;
      ids_to_grads.reserve(num_ids);
      ids_to_counts.reserve(num_ids);
      for (int i = 0; i < num_ids; ++i) {
        int64_t id = ids[i];
        if (!ids_to_grads.count(id)) {
          ids_to_counts[id] = 1;
          const float* grad_src = tensor + i * dim_size();
          float* grad_dest =
              cache_grads.get() + ids_to_grads.size() * dim_size();
          std::memcpy(grad_dest, grad_src, dim_size() * sizeof(float));
          ids_to_grads[id] = grad_dest;
        } else {
          const float* grad_src = tensor + i * dim_size();
          float* grad_dest = ids_to_grads[id];
          hash_table::ReduceSum(grad_dest, grad_src, grad_dest, dim_size());
          ++(ids_to_counts[id]);
        }
      }
      // The second step is to perform a filtering, and creates the vect of IDs
      // and grads for update.
      ids_after_filter.reserve(num_ids);
      grads_after_filter.reserve(num_ids);
      for (const auto& entry : ids_to_counts) {
        int64_t id = entry.first;
        uint32_t filter_count = entry.second;
        if (!table_->Contains(id) &&
            hash_filter_->ShouldBeFiltered(id, filter_count, table_.get())) {
          continue;
        }
        ids_after_filter.emplace_back(id);
        grads_after_filter.emplace_back(
            absl::MakeSpan(ids_to_grads[id], dim_size()));
      }
    } else {
      // We do simple increments (by 1) on the hash filters.
      ids_after_filter.reserve(num_ids);
      grads_after_filter.reserve(num_ids);
      for (int i = 0; i < num_ids; ++i) {
        int64_t id = ids[i];
        if (!table_->Contains(id) &&
            hash_filter_->ShouldBeFiltered(id, table_.get())) {
          continue;
        }

        ids_after_filter.emplace_back(id);
        grads_after_filter.emplace_back(
            absl::MakeSpan(tensor + i * dim_size(), dim_size()));
      }
    }

    // The final step is to perform an update based on the optimizer it uses.
    table_->BatchOptimize(absl::MakeSpan(ids_after_filter),
                          absl::MakeSpan(grads_after_filter), learning_rates,
                          update_time, global_step);
    if (hash_set_ != nullptr) {
      for (int64_t id : ids_after_filter) {
        hash_set_->insert(std::make_pair(id, this));
      }
    }
    return Status::OK();
  } catch (const std::exception& e) {
    return errors::InvalidArgument(e.what());
  }
}

Status EmbeddingHashTableTfBridge::Optimize(
    OpKernelContext* ctx, int64 id, absl::Span<const float> grads,
    absl::Span<const float> learning_rates, int64_t update_time,
    int64_t global_step) const {
  int max_value = std::max(update_time, max_update_ts_sec_->load());
  max_update_ts_sec_->store(max_value);

  try {
    table_->Optimize(id, grads, learning_rates, update_time, global_step);
    if (hash_set_ != nullptr) {
      hash_set_->insert(std::make_pair(id, this));
    }
    return Status::OK();
  } catch (const std::exception& e) {
    return errors::InvalidArgument(e.what());
  }
}

Status EmbeddingHashTableTfBridge::Save(OpKernelContext* ctx, DumpShard shard,
                                        WriteFn write_fn,
                                        DumpIterator* iter) const {
  try {
    table_->Save(shard, std::move(write_fn), iter);
    return Status::OK();
  } catch (const std::exception& e) {
    return errors::ResourceExhausted(e.what());
  }
}

Status EmbeddingHashTableTfBridge::Restore(
    OpKernelContext* ctx, DumpShard shard,
    std::function<bool(EntryDump*, int64_t*)> get_fn) const {
  try {
    int64_t update_time = table_->Restore(shard, std::move(get_fn));

    // Here we make sure max value is updated correctly when there are
    // multiple threads to update this value simultaneously.
    // There is no overhead since this operation is called once for each shard.
    while (true) {
      int64_t old_value = max_update_ts_sec_->load();
      int64_t new_value = std::max(old_value, update_time);
      bool ret =
          max_update_ts_sec_->compare_exchange_weak(old_value, new_value);
      if (ret == true) {
        break;
      }
    }

    return Status::OK();
  } catch (const std::exception& e) {
    return errors::ResourceExhausted(e.what());
  }
}

std::string EmbeddingHashTableTfBridge::DebugString() const {
  return config_.DebugString();
}

int32 EmbeddingHashTableTfBridge::dim_size() const { return dim_size_; }
int32 EmbeddingHashTableTfBridge::slice_size() const {
  return table_->SliceSize();
}
int64 EmbeddingHashTableTfBridge::max_update_ts_sec() const {
  return max_update_ts_sec_->load();
}

int64 EmbeddingHashTableTfBridge::last_evict_ts_sec() const {
  return last_evict_ts_sec_->load();
}

void EmbeddingHashTableTfBridge::set_last_evict_ts_sec(
    const int64_t last_evict_ts_sec) {
  *last_evict_ts_sec_ = last_evict_ts_sec;
}

bool EmbeddingHashTableTfBridge::IsServingEntryType() const {
  return config_.entry_config().entry_type() ==
         hash_table::EntryConfig_EntryType_SERVING;
}

std::vector<std::pair<int64_t, const void*>>
EmbeddingHashTableTfBridge::TouchedKeySet() const {
  if (hash_set_) {
    return hash_set_->GetAndClear();
  }
  return {};
}

void EmbeddingHashTableTfBridge::SetHopscotchHashSet(
    HopscotchHashSet<std::pair<int64_t, const void*>>* hash_set) {
  CHECK(hash_set_ == nullptr);
  hash_set_ = hash_set;
}

const EmbeddingHashTableConfig& EmbeddingHashTableTfBridge::GetConfig() const {
  return config_;
}

EmbeddingHashTableTfBridge::~EmbeddingHashTableTfBridge() {
  // Let the eviction thread stop
  evict_finished_ = true;
  if (evict_thread_) {
    evict_thread_->join();
  }
  hash_set_ = nullptr;
}

}  // namespace monolith_tf
}  // namespace tensorflow
