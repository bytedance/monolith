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

#ifndef MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_FILTER_SLIDING_HASH_FILTER_H_
#define MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_FILTER_SLIDING_HASH_FILTER_H_

#include <functional>

#include "monolith/native_training/runtime/hash_filter/filter.h"
#include "monolith/native_training/runtime/hash_filter/hash_filter.h"

namespace monolith {
namespace hash_filter {

class SlidingHashFilter : public Filter {
 public:
  SlidingHashFilter(size_t capacity, int split_num);
  SlidingHashFilter(const SlidingHashFilter& other);
  uint32_t add(FID fid, uint32_t count);
  uint32_t get(FID fid) const;
  uint32_t size_mb() const { return filters_[0]->size_mb() * filters_.size(); }

  static size_t get_split_capacity(size_t capacity, int split_num) {
    return capacity / (split_num - max_forward_step_ + 1);
  }

  static uint32_t size_byte(size_t capacity, int split_num) {
    return HashFilter<uint16_t>::size_byte(
               get_split_capacity(capacity, split_num), 1.2) *
           split_num;
  }

  size_t estimated_total_element() const;
  size_t failure_count() const { return failure_count_; }
  size_t split_num() const { return split_num_; }

  bool ShouldBeFiltered(
      int64_t fid, int64_t count, int64_t slot_occurrence_threshold,
      const monolith::hash_table::EmbeddingHashTableInterface* table) override {
    if (slot_occurrence_threshold <= 0) {
      return false;
    }
    return this->add(fid, count) < slot_occurrence_threshold;
  }

  SlidingHashFilter& operator=(SlidingHashFilter const&) = delete;
  SlidingHashFilter* clone() const;
  bool operator==(SlidingHashFilter const&) const;

  // Saves the data.
  virtual void Save(
      int split_idx,
      std::function<void(::monolith::hash_table::HashFilterSplitMetaDump)>
          write_meta_fn,
      std::function<void(::monolith::hash_table::HashFilterSplitDataDump)>
          write_data_fn) const;

  // Restores the data from get_fn.
  virtual void Restore(
      int split_idx,
      std::function<bool(::monolith::hash_table::HashFilterSplitMetaDump*)>
          get_meta_fn,
      std::function<bool(::monolith::hash_table::HashFilterSplitDataDump*)>
          get_data_fn);

 private:
  size_t prev(size_t index) const {
    if (index == 0) return filters_.size() - 1;
    return index - 1;
  }
  size_t next(size_t index) const {
    if (index == filters_.size() - 1) return 0;
    return index + 1;
  }
  HashFilterIterator<uint16_t> bidirectional_find(
      size_t begin, int max_look, FID fid, bool exhaust,
      std::function<size_t(size_t)> go) const;

  ::monolith::hash_table::SlidingHashFilterMetaDump GetMetaDump() const;
  void RestoreMetaDump(
      const ::monolith::hash_table::HashFilterSplitMetaDump& dump);
  void ValidateData(uint32_t expect_value, uint32_t ckpt_value,
                    const char* msg);

  size_t split_num_;
  constexpr static int max_forward_step_ = 2;
  int max_backward_step_;
  constexpr static int MAX_STEP = 16;
  std::vector<std::unique_ptr<HashFilter<uint16_t>>> filters_;
  size_t head_;
  int head_increment_;
  size_t failure_count_;
};
}  // namespace hash_filter
}  // namespace monolith

#endif  // MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_FILTER_SLIDING_HASH_FILTER_H_
