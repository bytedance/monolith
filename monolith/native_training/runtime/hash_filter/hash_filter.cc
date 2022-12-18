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

#include "monolith/native_training/runtime/hash_filter/hash_filter.h"

namespace monolith {
namespace hash_filter {

namespace proto2 = google::protobuf;

using ::monolith::hash_table::SlidingHashFilterMetaDump;
using ::monolith::hash_table::HashFilterSplitMetaDump;
using ::monolith::hash_table::HashFilterSplitDataDump;

const int kMaxNumPerTfRecord = 10000;

template <>
void HashFilter<uint16_t>::Save(
    const SlidingHashFilterMetaDump& sliding_hash_filter_meta_dump,
    std::function<void(HashFilterSplitMetaDump)> write_meta_fn,
    std::function<void(HashFilterSplitDataDump)> write_data_fn) const {
  // Write meta part with one tf-record with HashFilterSplitMetaDump type.
  HashFilterSplitMetaDump meta_dump;
  meta_dump.set_failure_count(failure_count_);
  meta_dump.set_total_size(total_size_);
  meta_dump.set_num_elements(num_elements_);
  meta_dump.set_fill_rate(fill_rate_);
  *(meta_dump.mutable_sliding_hash_filter_meta()) =
      sliding_hash_filter_meta_dump;
  write_meta_fn(std::move(meta_dump));

  // Write data part with multiple tf-records of HashFilterSplitDataDump type.
  int tf_record_num =
      (map_.size() + kMaxNumPerTfRecord - 1) / kMaxNumPerTfRecord;

  for (int record_idx = 0; record_idx < tf_record_num; record_idx++) {
    int start = record_idx * kMaxNumPerTfRecord;
    int end =
        std::min(start + kMaxNumPerTfRecord, static_cast<int>(map_.size()));
    HashFilterSplitDataDump data_dump;
    data_dump.set_offset(start);
    for (int i = start; i < end; i++) {
      data_dump.add_data(map_[i]);
    }
    write_data_fn(std::move(data_dump));
  }
}

template <>
void HashFilter<uint16_t>::Restore(
    HashFilterSplitMetaDump meta_dump,
    std::function<bool(HashFilterSplitDataDump*)> get_data_fn) {
  // Restore hash filter meta.
  failure_count_ = meta_dump.failure_count();
  total_size_ = meta_dump.total_size();
  num_elements_ = meta_dump.num_elements();
  fill_rate_ = meta_dump.fill_rate();

  // Restore hash filter data.
  map_.resize(total_size_ + MAX_STEP, 0);
  HashFilterSplitDataDump data_dump;
  while (get_data_fn(&data_dump)) {
    int offset = data_dump.offset();
    for (int i = 0; i < data_dump.data_size(); i++) {
      map_[offset + i] = (uint16_t)(data_dump.data(i));
    }
  }
}

}  // namespace hash_filter
}  // namespace monolith
