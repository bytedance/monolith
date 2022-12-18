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

#include "monolith/native_training/runtime/hash_filter/probabilistic_filter.h"

namespace monolith {
namespace hash_filter {

using ::monolith::concurrency::XorShift;

bool ProbabilisticFilter::InsertedIntoHashTableUnequalProbability(
    int64_t count, int64_t slot_occurrence_threshold) {
  return XorShift::Rand32ThreadSafe() * slot_occurrence_threshold <
         std::numeric_limits<uint32_t>::max() * count;
}

bool ProbabilisticFilter::InsertedIntoHashTableEqualProbability(
    int64_t count, int64_t slot_occurrence_threshold) {
  float epsilon = 0.05;
  float p = 1 - std::pow(epsilon,
                         1.f / static_cast<float>(slot_occurrence_threshold));

  return XorShift::Rand32ThreadSafe() < std::numeric_limits<uint32_t>::max() *
                                            (1.f - std::pow(1.f - p, count));
}

bool ProbabilisticFilter::ShouldBeFiltered(
    int64_t fid, int64_t count, int64_t slot_occurrence_threshold,
    const monolith::hash_table::EmbeddingHashTableInterface* table) {
  if (table && !table->Contains(fid)) {
    if (equal_probability_) {
      return !InsertedIntoHashTableEqualProbability(count,
                                                    slot_occurrence_threshold);
    } else {
      return !InsertedIntoHashTableUnequalProbability(
          count, slot_occurrence_threshold);
    }
  }

  return false;
}

}  // namespace hash_filter
}  // namespace monolith
