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

/**
 * Implement a dummy hash filter which has no real hash filter logic inside.
 * It will always return HashFilter<uint16_t>::max_count() so that all FIDs can
 * pass the hash filter check.
 **/
#ifndef MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_FILTER_DUMMY_HASH_FILTER_H_
#define MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_FILTER_DUMMY_HASH_FILTER_H_

#include <functional>

#include "monolith/native_training/runtime/hash_filter/filter.h"
#include "monolith/native_training/runtime/hash_filter/hash_filter.h"

namespace monolith {
namespace hash_filter {

class DummyHashFilter : public Filter {
 public:
  DummyHashFilter() = default;
  DummyHashFilter(const DummyHashFilter& other) {}
  uint32_t add(FID fid, uint32_t count) {
    return HashFilter<uint16_t>::max_count();
  }
  uint32_t get(FID fid) const { return HashFilter<uint16_t>::max_count(); }
  uint32_t size_mb() const { return 0; }
  size_t estimated_total_element() const { return 0; }
  size_t failure_count() const { return 0; }
  size_t split_num() const { return 0; }

  bool ShouldBeFiltered(
      int64_t fid, int64_t count, int64_t slot_occurrence_threshold,
      const monolith::hash_table::EmbeddingHashTableInterface* table) override {
    return false;
  }

  DummyHashFilter& operator=(DummyHashFilter const&) = delete;
  DummyHashFilter* clone() const { return new DummyHashFilter(*this); }
};

}  // namespace hash_filter
}  // namespace monolith
#endif  // MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_FILTER_DUMMY_HASH_FILTER_H_
