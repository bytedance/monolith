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

#ifndef MONOLITH_NATIVE_TRAINING_RUNTIME_OPS_EMBEDDING_HASH_TABLE_TF_BRIDGE_H_
#define MONOLITH_NATIVE_TRAINING_RUNTIME_OPS_EMBEDDING_HASH_TABLE_TF_BRIDGE_H_
#include <atomic>
#include <memory>
#include <mutex>
#include <thread>

#include "monolith/native_training/runtime/hash_table/embedding_hash_table.pb.h"
#include "monolith/native_training/runtime/hash_table/embedding_hash_table_interface.h"
#include "monolith/native_training/runtime/hopscotch/hopscotch_hash_set.h"
#include "monolith/native_training/runtime/ops/hash_filter_tf_bridge.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace monolith_tf {

template <class T>
using HopscotchHashSet = monolith::hopscotch::HopscotchHashSet<T>;

// A hash table which can be used in TF runtime.
// It captures all potential exceptions and convert them into error.
class EmbeddingHashTableTfBridge : public ResourceBase {
 public:
  using EntryDump = monolith::hash_table::EntryDump;

  static Status New(monolith::hash_table::EmbeddingHashTableConfig config,
                    HashFilterTfBridge* hash_filter,
                    EmbeddingHashTableTfBridge** new_bridge,
                    const std::string& name, cudaStream_t stream = 0);

  ~EmbeddingHashTableTfBridge();

  // BatchLookup |ids| and write it into |embeddings|
  Status BatchLookup(OpKernelContext* ctx, const int num_ids, int64_t* ids,
                     float* out_embedding, int64_t* hit_fid_count) const;
  Status BatchLookupEntry(OpKernelContext* ctx, const int num_ids, int64_t* ids,
                          EntryDump* out_entries) const;
  // Lookup |id| and write it into |embedding|
  Status Lookup(OpKernelContext* ctx, int64 id, float* out_embedding) const;
  Status LookupEntry(OpKernelContext* ctx, int64 id,
                     EntryDump* out_entry) const;
  // TODO(leqi.zou): Unify the API here.
  // 1. Remove all batch APIs.
  // 2. Replace int64_t to int64
  Status Assign(OpKernelContext* ctx, int num_ids, const int64_t* ids,
                const float* embeddings, int64_t update_time) const;
  // TODO(leqi.zou): Replace this API by AssignAdd2.
  Status AssignAdd(OpKernelContext* ctx, int64 id, const Tensor& tensor,
                   int64_t update_time) const;
  Status AssignAdd2(int64 id, absl::Span<const float> value,
                    int64_t update_time);
  Status BatchOptimize(OpKernelContext* ctx, size_t num_ids, const int64_t* ids,
                       const float* tensor,
                       absl::Span<const float> learning_rates,
                       int64_t update_time, bool enable_dedup,
                       const int64_t global_step) const;
  Status Optimize(OpKernelContext* ctx, int64 id, absl::Span<const float> grads,
                  absl::Span<const float> learning_rates, int64_t update_time,
                  int64_t global_step) const;

  using DumpShard =
      monolith::hash_table::EmbeddingHashTableInterface::DumpShard;
  using DumpIterator =
      monolith::hash_table::EmbeddingHashTableInterface::DumpIterator;
  using WriteFn = monolith::hash_table::EmbeddingHashTableInterface::WriteFn;

  // For the functor injected, it is ok to throw exceptions.
  Status Save(OpKernelContext* ctx, DumpShard shard, WriteFn write_fn,
              DumpIterator* iter) const;
  Status Restore(OpKernelContext* ctx, DumpShard shard,
                 std::function<bool(EntryDump*, int64_t*)> get_fn) const;
  void Clear() const { table_->Clear(); }
  int64_t Size() const { return table_->Size(); }

  int32 dim_size() const;
  int32 slice_size() const;
  int64 max_update_ts_sec() const;
  int64 last_evict_ts_sec() const;
  void set_last_evict_ts_sec(const int64_t last_evict_ts_sec);
  bool IsServingEntryType() const;
  std::string DebugString() const override;

  void SetHopscotchHashSet(
      HopscotchHashSet<std::pair<int64_t, const void*>>* hash_set);

  HopscotchHashSet<std::pair<int64_t, const void*>>* GetHashSet() const {
    return hash_set_;
  }

  std::vector<std::pair<int64_t, const void*>> TouchedKeySet() const;

  const monolith::hash_table::EmbeddingHashTableConfig& GetConfig() const;

 private:
  explicit EmbeddingHashTableTfBridge(HashFilterTfBridge* hash_filter)
      : hash_filter_(hash_filter) {}

  std::string name_;
  std::unique_ptr<monolith::hash_table::EmbeddingHashTableInterface> table_;
  monolith::hash_table::EmbeddingHashTableConfig config_;
  int64 dim_size_ = 0;
  std::unique_ptr<std::atomic<int64_t>> max_update_ts_sec_;
  HashFilterTfBridge* hash_filter_;

  std::unique_ptr<std::thread> evict_thread_;
  std::unique_ptr<std::atomic<int64_t>> last_evict_ts_sec_;
  mutex evict_mu_;
  bool evict_finished_ TF_GUARDED_BY(evict_mu_);

  HopscotchHashSet<std::pair<int64_t, const void*>>* hash_set_ = nullptr;
};

}  // namespace monolith_tf
}  // namespace tensorflow
#endif  // MONOLITH_NATIVE_TRAINING_RUNTIME_OPS_EMBEDDING_HASH_TABLE_TF_BRIDGE_H_
