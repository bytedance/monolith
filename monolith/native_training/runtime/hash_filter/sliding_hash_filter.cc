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

#include <iostream>

#include "absl/strings/str_format.h"
#include "glog/logging.h"

#include "monolith/native_training/runtime/hash_filter/sliding_hash_filter.h"

namespace monolith {
namespace hash_filter {

using ::monolith::hash_table::HashFilterSplitDataDump;
using ::monolith::hash_table::HashFilterSplitMetaDump;
using ::monolith::hash_table::SlidingHashFilterMetaDump;

SlidingHashFilter::SlidingHashFilter(size_t capacity, int split_num)
    : split_num_(split_num), head_(0), head_increment_(0), failure_count_(0) {
  capacity_ = capacity;
  if (capacity_ < 300) capacity_ = 300;
  if (split_num < 5) split_num = 5;
  filters_.resize(split_num);
  max_backward_step_ = split_num - max_forward_step_;
  // max_forward_step_ - 1 blocks are kept empty for looing forward
  size_t split_capacity = get_split_capacity(capacity_, split_num);
  for (auto& filter : filters_) {
    filter.reset(new HashFilter<uint16_t>(split_capacity, 1.2));
  }
}

SlidingHashFilter::SlidingHashFilter(const SlidingHashFilter& other)
    : max_backward_step_(other.max_backward_step_),
      filters_(other.filters_.size()),
      head_(other.head_),
      head_increment_(other.head_increment_),
      failure_count_(other.failure_count_) {
  capacity_ = other.capacity_;
  if (&other == this) {
    return;
  }
  for (size_t i = 0; i != filters_.size(); ++i) {
    filters_[i].reset(other.filters_[i]->clone());
  }
}

uint32_t SlidingHashFilter::add(FID fid, uint32_t count) {
  uint32_t old_count = 0;

  // Look forward to find current value
  HashFilterIterator<uint16_t> curr_iter = bidirectional_find(
      head_, max_forward_step_, fid, false,
      std::bind(&SlidingHashFilter::next, this, std::placeholders::_1));
  if (curr_iter.valid()) {
    if (!curr_iter.empty()) {
      return curr_iter.add(count);
    }
  } else {
    failure_count_ += 1;
    return HashFilter<uint16_t>::max_count();
  }

  // Look backward to find old value
  HashFilterIterator<uint16_t> old_iter = bidirectional_find(
      prev(head_), std::min(head_increment_, max_backward_step_), fid, true,
      std::bind(&SlidingHashFilter::prev, this, std::placeholders::_1));
  if (old_iter.valid()) {
    old_count = old_iter.get();
    curr_iter.add(old_count + count);
  } else {
    curr_iter.add(count);
  }
  if (filters_[head_]->full()) {
    head_ = next(head_);
    head_increment_ += 1;
    filters_[(head_ + max_forward_step_ - 1) % filters_.size()]->async_clear();
  }
  return old_count;
}

uint32_t SlidingHashFilter::get(FID fid) const {
  // Look forward to find current value
  HashFilterIterator<uint16_t> curr_iter = bidirectional_find(
      head_, max_forward_step_, fid, false,
      std::bind(&SlidingHashFilter::next, this, std::placeholders::_1));
  if (curr_iter.valid()) {
    if (!curr_iter.empty()) {
      return curr_iter.get();
    }
  } else {
    return HashFilter<uint16_t>::max_count();
  }

  // Look backward to find old value
  HashFilterIterator<uint16_t> iter = bidirectional_find(
      prev(head_), std::min(head_increment_, max_backward_step_), fid, true,
      std::bind(&SlidingHashFilter::prev, this, std::placeholders::_1));
  if (iter.valid()) {
    return iter.get();
  } else {
    return 0;
  }
}

HashFilterIterator<uint16_t> SlidingHashFilter::bidirectional_find(
    size_t begin, int max_look, FID fid, bool exhaust,
    std::function<size_t(size_t)> go) const {
  size_t index = begin;
  for (int i = 0; i != max_look; ++i) {
    HashFilterIterator<uint16_t> iter = filters_[index]->find(fid, MAX_STEP);
    // Looking forward only needs a valid position
    // Looking backward needs a non-empty position
    if (iter.valid() && (!exhaust || (exhaust && !iter.empty()))) return iter;
    index = go(index);
  }
  return HashFilterIterator<uint16_t>();
}

size_t SlidingHashFilter::estimated_total_element() const {
  size_t result = 0;
  for (auto& filter : filters_) {
    result += filter->estimated_total_element();
  }
  return result;
}

SlidingHashFilter* SlidingHashFilter::clone() const {
  return new SlidingHashFilter(*this);
}

bool SlidingHashFilter::operator==(const SlidingHashFilter& other) const {
  if (!(max_forward_step_ == other.max_forward_step_ && head_ == other.head_ &&
        head_increment_ % filters_.size() ==
            other.head_increment_ % filters_.size() &&
        capacity_ == other.capacity_ &&
        filters_.size() == other.filters_.size())) {
    return false;
  }
  for (size_t i = 0; i != filters_.size(); ++i) {
    if (!(*filters_[i] == *other.filters_[i])) {
      return false;
    }
  }
  return true;
}

SlidingHashFilterMetaDump SlidingHashFilter::GetMetaDump() const {
  SlidingHashFilterMetaDump dump;
  dump.set_split_num(split_num_);
  dump.set_max_forward_step(max_forward_step_);
  dump.set_max_backward_step(max_backward_step_);
  dump.set_max_step(MAX_STEP);
  dump.set_head(head_);
  dump.set_head_increment(head_increment_);
  dump.set_failure_count(failure_count_);
  return dump;
}

// Saves the data.
void SlidingHashFilter::Save(
    int split_idx, std::function<void(HashFilterSplitMetaDump)> write_meta_fn,
    std::function<void(HashFilterSplitDataDump)> write_data_fn) const {
  auto meta_dump = GetMetaDump();
  filters_[split_idx]->Save(meta_dump, std::move(write_meta_fn),
                            std::move(write_data_fn));
}

void SlidingHashFilter::ValidateData(uint32_t expect_value, uint32_t ckpt_value,
                                     const char* msg) {
  if (ckpt_value != expect_value) {
    throw std::runtime_error(
        absl::StrFormat("%s: %d does't match with : %d read from hash "
                        "filter checkpoint file.",
                        msg, expect_value, ckpt_value));
  }
}

void SlidingHashFilter::RestoreMetaDump(const HashFilterSplitMetaDump& dump) {
  auto& sliding_hash_filter_meta_dump = dump.sliding_hash_filter_meta();

  ValidateData(split_num_, sliding_hash_filter_meta_dump.split_num(),
               "split_num");
  ValidateData(max_forward_step_,
               sliding_hash_filter_meta_dump.max_forward_step(),
               "max_forward_step");
  ValidateData(max_backward_step_,
               sliding_hash_filter_meta_dump.max_backward_step(),
               "max_backward_step");
  ValidateData(MAX_STEP, sliding_hash_filter_meta_dump.max_step(), "max_step");
  head_ = sliding_hash_filter_meta_dump.head();
  head_increment_ = sliding_hash_filter_meta_dump.head_increment();
  failure_count_ = sliding_hash_filter_meta_dump.failure_count();
}

// Restores the data from get_fn.
void SlidingHashFilter::Restore(
    int split_idx, std::function<bool(HashFilterSplitMetaDump*)> get_meta_fn,
    std::function<bool(HashFilterSplitDataDump*)> get_data_fn) {
  HashFilterSplitMetaDump meta_dump;
  get_meta_fn(&meta_dump);
  RestoreMetaDump(meta_dump);
  filters_[split_idx]->Restore(meta_dump, std::move(get_data_fn));
}

}  // namespace hash_filter
}  // namespace monolith
