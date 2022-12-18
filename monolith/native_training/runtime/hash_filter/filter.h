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

#ifndef MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_FILTER_FILTER_H_
#define MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_FILTER_FILTER_H_
#include <string>

#include "monolith/native_training/runtime/hash_filter/types.h"
#include "monolith/native_training/runtime/hash_table/embedding_hash_table.pb.h"
#include "monolith/native_training/runtime/hash_table/embedding_hash_table_interface.h"

namespace monolith {
namespace hash_filter {

class Filter {
 public:
  Filter() : capacity_(0) {}
  virtual uint32_t add(FID fid, uint32_t count) = 0;
  virtual uint32_t get(FID fid) const = 0;
  virtual uint32_t size_mb() const = 0;
  virtual size_t estimated_total_element() const = 0;
  virtual size_t failure_count() const = 0;
  virtual size_t capacity() const { return capacity_; }
  virtual size_t split_num() const = 0;
  virtual bool exceed_limit() const { return false; }
  virtual void set_name(const std::string& name) { name_ = name; }
  virtual Filter* clone() const = 0;
  virtual bool ShouldBeFiltered(
      int64_t fid, int64_t count, int64_t slot_occurrence_threshold,
      const monolith::hash_table::EmbeddingHashTableInterface* table) = 0;
  virtual void Save(
      int split_idx,
      std::function<void(::monolith::hash_table::HashFilterSplitMetaDump)>
          write_meta_fn,
      std::function<void(::monolith::hash_table::HashFilterSplitDataDump)>
          write_data_fn) const {}
  virtual void Restore(
      int split_idx,
      std::function<bool(::monolith::hash_table::HashFilterSplitMetaDump*)>
          get_meta_fn,
      std::function<bool(::monolith::hash_table::HashFilterSplitDataDump*)>
          get_data_fn) {}

  virtual ~Filter() {}
  constexpr static unsigned char count_bit = 4;
  constexpr static uint32_t max_count() { return (1 << count_bit) - 1; }

 protected:
  size_t capacity_;
  std::string name_;
};

}  // namespace hash_filter
}  // namespace monolith
#endif  // MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_FILTER_FILTER_H_
