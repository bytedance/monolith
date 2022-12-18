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

#ifndef MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_FILTER_HASH_FILTER_H_
#define MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_FILTER_HASH_FILTER_H_
#include <cstdint>
#include <fstream>
#include <limits>

#include "absl/algorithm/container.h"
#include "absl/hash/hash.h"
#include "absl/types/span.h"

#include "monolith/native_training/runtime/hash_filter/filter.h"
#include "monolith/native_training/runtime/hash_table/embedding_hash_table.pb.h"

namespace monolith {
namespace hash_filter {

template <typename DATA>
class HashFilter;
template <typename DATA>
class HashFilterIterator {
  friend class HashFilter<DATA>;

 public:
  HashFilterIterator() : filter_(NULL), pvalue_(NULL), sign_(0) {}
  uint32_t add(uint32_t add_count) {
    assert(valid() && "check validation before add");
    if (add_count > HashFilter<DATA>::max_count())
      add_count = HashFilter<DATA>::max_count();
    if (*pvalue_ == 0) {
      ++filter_->num_elements_;
      if (filter_->num_elements_ > filter_->capacity_) {
        filter_->num_elements_ = filter_->capacity_;
        /*
        InvalidOperation io;
        io.why = "total elements exceeds filter capacity";
        throw io;
        */
      }
      *pvalue_ = (sign_ << HashFilter<DATA>::count_bit) + add_count;
      return 0;
    }
    unsigned char count = *pvalue_ & HashFilter<DATA>::max_count();
    if (count + add_count >= HashFilter<DATA>::max_count())
      *pvalue_ |= HashFilter<DATA>::max_count();
    else
      *pvalue_ += add_count;
    return count;
  }

  uint32_t get() const {
    if (!pvalue_) return HashFilter<DATA>::max_count();
    return *pvalue_ & HashFilter<DATA>::max_count();
  }

  bool valid() const { return pvalue_ != NULL; }

  bool empty() const {
    assert(valid() && "check validation before empty");
    return *pvalue_ == 0;
  }

 private:
  explicit HashFilterIterator(HashFilter<DATA>* filter, DATA* pvalue, DATA sign)
      : filter_(filter), pvalue_(pvalue), sign_(sign) {}
  HashFilter<DATA>* filter_;
  DATA* pvalue_;
  DATA sign_;
};

template <typename DATA>
class HashFilter : public Filter {
  friend class HashFilterIterator<DATA>;

 public:
  explicit HashFilter(size_t capacity, double fill_rate = 1.5)
      : failure_count_(0),
        total_size_(capacity * fill_rate),
        num_elements_(0),
        fill_rate_(fill_rate) {
    capacity_ = capacity;
    map_.resize(total_size_ + MAX_STEP, 0);
  }

  uint32_t add(FID fid, uint32_t count) override {
    HashFilterIterator<DATA> iter = find(fid, MAX_STEP);
    if (iter.valid()) {
      return iter.add(count);
    }
    failure_count_ += 1;
    return max_count();
  }

  uint32_t get(FID fid) const override {
    return const_cast<HashFilter*>(this)->find(fid, MAX_STEP).get();
  }

  HashFilterIterator<DATA> find(FID fid, int max_step) {
    assert(max_step <= MAX_STEP && "illegal max_step");
    DATA sign = signature(fid);
    int step = 0;
    size_t hash_value = hash(fid) % total_size_;
    DATA* pvalue = reinterpret_cast<DATA*>(&map_[hash_value]);
    do {
      if (*pvalue == 0 || (*pvalue >> count_bit) == sign) {
        return HashFilterIterator<DATA>(this, pvalue, sign);
      }
      ++pvalue;
      if (pvalue == &(*map_.end())) {
        pvalue = &map_[0];
      }
    } while (++step < max_step);
    return HashFilterIterator<DATA>(this, NULL, sign);
  }

  bool full() const { return num_elements_ >= capacity_ - 1; }

  // TODO make this async
  void async_clear() {
    fill(map_.begin(), map_.end(), 0);
    num_elements_ = 0;
    failure_count_ = 0;
  }

  uint32_t size_mb() const override {
    return map_.size() * sizeof(DATA) / 1024.0 / 1024.0;
  }

  static size_t size_byte(size_t capacity, double fill_rate = 1.5) {
    return capacity * sizeof(DATA) * fill_rate + MAX_STEP;
  }

  size_t failure_count() const override { return failure_count_; }

  size_t split_num() const override { return 0; }

  DATA signature(FID fid) const { return (fid >> 17 | fid << 15) & sign_mask; }

  size_t estimated_total_element() const override { return num_elements_; }

  bool exceed_limit() const override { return num_elements_ >= capacity_; }

  HashFilter* clone() const override { return new HashFilter(*this); }

  bool ShouldBeFiltered(
      int64_t fid, int64_t count, int64_t slot_occurrence_threshold,
      const monolith::hash_table::EmbeddingHashTableInterface* table) override {
    if (slot_occurrence_threshold <= 0) {
      return false;
    }
    return add(fid, count) < slot_occurrence_threshold;
  }

  bool operator==(const HashFilter& other) const {
    return total_size_ == other.total_size_ &&
           num_elements_ == other.num_elements_ &&
           capacity_ == other.capacity_ && fill_rate_ == other.fill_rate_ &&
           map_ == other.map_;
  }

  void Save(const ::monolith::hash_table::SlidingHashFilterMetaDump&
                sliding_hash_filter_meta_dump,
            std::function<void(::monolith::hash_table::HashFilterSplitMetaDump)>
                write_meta_fn,
            std::function<void(::monolith::hash_table::HashFilterSplitDataDump)>
                write_data_fn) const;

  void Restore(
      ::monolith::hash_table::HashFilterSplitMetaDump dump,
      std::function<bool(::monolith::hash_table::HashFilterSplitDataDump*)>
          get_data_fn);

  constexpr static DATA sign_mask =
      ((1 << (sizeof(DATA) * 8)) - 1) >> count_bit;

 private:
  constexpr static int DUMP_VALUE_SIZE = 1024 * 1024 * 20;  // 10-20MB
  constexpr static int MAX_STEP = 64;
  size_t hash(FID fid) const { return absl::Hash<FID>()(fid); }
  std::vector<DATA> map_;
  uint64_t failure_count_;
  uint64_t total_size_;
  uint64_t num_elements_;
  double fill_rate_;
};

}  // namespace hash_filter
}  // namespace monolith

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
#endif  // MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_FILTER_HASH_FILTER_H_
