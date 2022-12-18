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

#ifndef MONOLITH_NATIVE_TRAINING_RUNTIME_OPS_HASH_FILTER_TF_BRIDGE_H_
#define MONOLITH_NATIVE_TRAINING_RUNTIME_OPS_HASH_FILTER_TF_BRIDGE_H_
#include <cstdint>
#include <memory>

#include "absl/strings/str_format.h"
#include "monolith/native_training/runtime/hash_filter/filter.h"
#include "monolith/native_training/runtime/ops/file_utils.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"

namespace tensorflow {
namespace monolith_tf {

class HashFilterTfBridge : public ResourceBase {
 public:
  explicit HashFilterTfBridge(
      std::unique_ptr<monolith::hash_filter::Filter> filter,
      const monolith::hash_table::SlotOccurrenceThresholdConfig& config);

  bool ShouldBeFiltered(
      int64_t id, int64_t count,
      monolith::hash_table::EmbeddingHashTableInterface* table) {
    return filter_->ShouldBeFiltered(id, count, GetSlotOccurrenceThreshold(id),
                                     table);
  }

  bool ShouldBeFiltered(
      int64_t id,
      monolith::hash_table::EmbeddingHashTableInterface* table = nullptr) {
    return ShouldBeFiltered(id, 1, table);
  }

  int GetSplitNum() { return filter_->split_num(); }

  std::string DebugString() const override {
    return absl::StrFormat("Filter with capacity: %d", filter_->capacity());
  }

  // For the functor injected, it is ok to throw exceptions.
  Status Save(
      int split_idx,
      std::function<void(::monolith::hash_table::HashFilterSplitMetaDump)>
          write_meta_fn,
      std::function<void(::monolith::hash_table::HashFilterSplitDataDump)>
          write_data_fn) const;

  Status Restore(
      int split_idx,
      std::function<bool(::monolith::hash_table::HashFilterSplitMetaDump*)>
          get_meta_fn,
      std::function<bool(::monolith::hash_table::HashFilterSplitDataDump*)>
          get_data_fn) const;

 private:
  int GetSlotOccurrenceThreshold(int64_t fid) const;

  std::unique_ptr<monolith::hash_filter::Filter> filter_;
  std::vector<int> slot_to_occurrence_threshold_;
};

// Carries the data through async process.
// It will ref and unref |p_hash_filter|
struct HashFilterAsyncPack {
  HashFilterAsyncPack(OpKernelContext* p_ctx, HashFilterTfBridge* p_hash_filter,
                      std::string p_basename, std::function<void()> p_done,
                      int p_thread_num)
      : ctx(p_ctx),
        hash_filter(p_hash_filter),
        basename(p_basename),
        done(std::move(p_done)),
        thread_num(p_thread_num),
        finish_num(0),
        status(thread_num) {
    hash_filter->Ref();
  }

  ~HashFilterAsyncPack() { hash_filter->Unref(); }

  OpKernelContext* ctx;
  HashFilterTfBridge* hash_filter;
  std::string basename;
  std::function<void()> done;
  const int thread_num;
  std::atomic_int finish_num;
  std::vector<Status> status;
};

}  // namespace monolith_tf
}  // namespace tensorflow
#endif  // MONOLITH_NATIVE_TRAINING_RUNTIME_OPS_HASH_FILTER_TF_BRIDGE_H_
