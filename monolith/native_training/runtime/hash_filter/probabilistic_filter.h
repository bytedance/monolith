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

#ifndef MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_FILTER_PROBABILISTIC_FILTER_H_
#define MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_FILTER_PROBABILISTIC_FILTER_H_

#include "monolith/native_training/runtime/concurrency/xorshift.h"
#include "monolith/native_training/runtime/hash_filter/filter.h"
#include "monolith/native_training/runtime/hash_filter/hash_filter.h"

namespace monolith {
namespace hash_filter {

class ProbabilisticFilter : public Filter {
 public:
  explicit ProbabilisticFilter(bool equal_probability = false)
      : equal_probability_(equal_probability) {}

  uint32_t add(FID fid, uint32_t count) override {
    return HashFilter<uint16_t>::max_count();
  }

  uint32_t get(FID fid) const override {
    return HashFilter<uint16_t>::max_count();
  }

  uint32_t size_mb() const override { return 0; }

  size_t estimated_total_element() const override { return 0; }

  size_t failure_count() const override { return 0; }

  size_t split_num() const override { return 0; }

  Filter* clone() const override { return new ProbabilisticFilter(*this); }

  bool InsertedIntoHashTableUnequalProbability(
      int64_t count, int64_t slot_occurrence_threshold);

  bool InsertedIntoHashTableEqualProbability(int64_t count,
                                             int64_t slot_occurrence_threshold);

  bool ShouldBeFiltered(
      int64_t fid, int64_t count, int64_t slot_occurrence_threshold,
      const monolith::hash_table::EmbeddingHashTableInterface* table) override;

 private:
  bool equal_probability_;
};

}  // namespace hash_filter
}  // namespace monolith

#endif  // MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_FILTER_PROBABILISTIC_FILTER_H_
