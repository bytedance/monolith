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

#include "monolith/native_training/runtime/hash_table/cuckoohash/cuckoo_embedding_hash_table.h"

#include <cstdlib>
#include <utility>

#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "monolith/native_training/data/training_instance/cc/reader_util.h"
#include "monolith/native_training/runtime/allocator/block_allocator.h"
#include "monolith/native_training/runtime/hash_table/cuckoohash/cuckoohash_map.hpp"
#include "monolith/native_training/runtime/hash_table/entry_defs.h"

namespace monolith {
namespace hash_table {
namespace {

using allocator::EntryAddress;
using allocator::TSEmbeddingBlockAllocator;

const int64_t kSecPerDay = 24 * 60 * 60;

// A helper that wraps the object with a init_fn.
template <class T>
class WithInitFn : public T {
 public:
  template <typename... Args>
  explicit WithInitFn(const std::function<void(T&)>& init_fn, Args&&... args)
      : T(std::forward<Args>(args)...) {
    init_fn(*this);
  }
};

template <class TVal>
class EntryHelper {};

template <>
class EntryHelper<PackedEntry> {
 public:
  EntryHelper(size_t entry_size)
      : entry_size_(entry_size), alloc_(entry_size) {}

  template <typename Map, typename... Args>
  bool Upsert(Map* m, Args&&... args) {
    return m->upsert(std::forward<Args>(args)..., &alloc_);
  }

  void* Get(const PackedEntry& entry) const {
    return alloc_.GetEntryPointer(entry.get_entry_addr());
  }

  void DeallocateAll() { alloc_.DeallocateAll(); }

 private:
  size_t entry_size_;
  allocator::TSEmbeddingBlockAllocator alloc_;
};

template <>
class EntryHelper<RawEntry> {
 public:
  EntryHelper(size_t entry_size) : entry_size_(entry_size) {}

  template <typename Map, typename... Args>
  bool Upsert(Map* m, Args&&... args) {
    return m->upsert(std::forward<Args>(args)..., entry_size_);
  }

  void* Get(const RawEntry& entry) const { return entry.get(); }

  void DeallocateAll() {}

 private:
  size_t entry_size_;
};

template <int64_t length>
class EntryHelper<InlineEntry<length>> {
 public:
  template <typename Map, typename... Args>
  bool Upsert(Map* m, Args&&... args) {
    return m->upsert(std::forward<Args>(args)...);
  }

  const void* Get(const InlineEntry<length>& entry) const {
    return entry.get();
  }
  void* Get(InlineEntry<length>& entry) { return entry.get(); }

  void DeallocateAll() {}
};

struct Params {
  CuckooEmbeddingHashTableConfig config;
  std::unique_ptr<EntryAccessorInterface> accessor;
  uint64_t initial_capacity;
  SlotExpireTimeConfig slot_expire_time_config;
};

template <class EntryType>
class CuckooEmbeddingHashTable : public EmbeddingHashTableInterface {
 public:
  explicit CuckooEmbeddingHashTable(Params p,
                                    EntryHelper<EntryType> entry_helper)
      : config_(std::move(p.config)),
        accessor_(std::move(p.accessor)),
        entry_helper_(std::move(entry_helper)),
        default_expire_time_(p.slot_expire_time_config.default_expire_time()),
        m_(p.initial_capacity) {
    slot_to_expire_time_ =
        std::make_unique<absl::flat_hash_map<int64_t, int>>();
    for (const auto& slot_expire_time :
         p.slot_expire_time_config.slot_expire_times()) {
      (*slot_to_expire_time_)[slot_expire_time.slot()] =
          slot_expire_time.expire_time();
    }
  }

  // Returns the corresponding entry for |ids|.
  int64_t BatchLookup(absl::Span<int64_t> ids,
                      absl::Span<absl::Span<float>> embeddings) const override {
    int64_t found = 0;
    for (unsigned int index = 0; index < ids.size(); ++index) {
      int64_t id = ids[index];
      found += Lookup(id, embeddings[index]);
    }

    return found;
  }

  // Handles the corresponding entry for |ids|.
  void BatchLookupEntry(absl::Span<int64_t> ids,
                        absl::Span<EntryDump> entries) const override {
    for (unsigned int index = 0; index < ids.size(); ++index) {
      int64_t id = ids[index];
      LookupEntry(id, entries.subspan(index, index + 1));
    }
  }

  // Returns the corresponding entry for |id|.
  int64_t Lookup(int64_t id, absl::Span<float> embedding) const override {
    auto find_fn = [&](EntryType& entry) {
      accessor_->Fill(entry_helper_.Get(entry), embedding);
    };
    if (m_.find_fn(id, find_fn)) {
      return 1;
    }
    // By default, returns all zero.
    std::memset(embedding.data(), 0, sizeof(float) * embedding.size());
    return 0;
  }

  // Handles the corresponding entry for |id|.
  void LookupEntry(int64_t id, absl::Span<EntryDump> entry) const override {
    auto find_fn = [&](EntryType& raw_entry) {
      entry[0] = std::move(accessor_->Save(entry_helper_.Get(raw_entry),
                                           raw_entry.GetTimestamp()));
    };
    if (m_.find_fn(id, find_fn)) {
      return;
    }
  }

  // Update the hash table entry directly.
  void Assign(absl::Span<const int64_t> ids,
              absl::Span<const absl::Span<const float>> updates,
              int64_t update_time) override {
    for (size_t i = 0; i < ids.size(); ++i) {
      int64_t id = ids[i];
      auto update = updates[i];

      UpsertEntry(id, [&](EntryType& entry) {
        entry.SetTimestamp(update_time);
        accessor_->Assign(update, entry_helper_.Get(entry));
      });
    }
  }

  // Update the hash table entry directly.
  void AssignAdd(int64_t id, absl::Span<const float> update,
                 int64_t update_time) override {
    UpsertEntry(id, [&](EntryType& entry) {
      entry.SetTimestamp(update_time);
      accessor_->AssignAdd(update, entry_helper_.Get(entry));
    });
  }

  // Update the hash table based on optimizer.
  void BatchOptimize(absl::Span<int64_t> ids,
                     absl::Span<absl::Span<const float>> grads,
                     absl::Span<const float> learning_rates,
                     int64_t update_time, const int64_t global_step) override {
    for (size_t i = 0; i < ids.size(); ++i) {
      Optimize(ids[i], grads[i], learning_rates, update_time, global_step);
    }
  }

  // Update the hash table based on optimizer.
  void Optimize(int64_t id, absl::Span<const float> grad,
                absl::Span<const float> learning_rates, int64_t update_time,
                const int64_t global_step) override {
    UpsertEntry(id, [&](EntryType& entry) {
      entry.SetTimestamp(update_time);
      accessor_->Optimize(entry_helper_.Get(entry), grad, learning_rates,
                          global_step);
    });
  }

  // Evict the outdated hash table values based on the expire time and last
  // updated time.
  virtual void Evict(int64_t max_update_time) {
    auto should_be_evict_fn = [this, max_update_time](const int64_t& key,
                                                      const EntryType& entry) {
      const int64_t timestamp = entry.GetTimestamp();
      int expire_time = default_expire_time_;
      // TODO(zhen.li1): evict assumes the fid is v2 version.
      auto expire_time_iter = slot_to_expire_time_->find(slot_id_v2(key));
      if (expire_time_iter != slot_to_expire_time_->end()) {
        expire_time = expire_time_iter->second;
      }
      return max_update_time - timestamp >= expire_time * kSecPerDay;
    };
    m_.evict(should_be_evict_fn);
  }

  // Check if a given id exists in the hashtable
  bool Contains(const int64_t id) { return m_.contains(id); }

  // Saves the data. The implementation should guarantee that different shard
  // can be dumped in the parallel.
  void Save(DumpShard shard, WriteFn write_fn,
            DumpIterator* iter) const override {
    auto dump_fn = [&](const int64_t& key, const EntryType& entry) {
      EntryDump dump =
          accessor_->Save(entry_helper_.Get(entry), entry.GetTimestamp());
      dump.set_id(key);
      return write_fn(std::move(dump));
    };
    m_.partial_dump(shard, dump_fn, iter);
  }

  // Restores the data from get_fn. The implementation should guarantee that
  // different shard can be dumped in the parallel.
  // |get_fn| returns false if it is end of stream.
  int64_t Restore(DumpShard shard,
                  std::function<bool(EntryDump*, int64_t*)> get_fn) override {
    EntryDump dump;
    int64_t max_update_ts = 0;
    while (get_fn(&dump, &max_update_ts)) {
      UpsertEntry(dump.id(), [&](EntryType& entry) {
        uint32_t timestamp_sec = 0;
        accessor_->Restore(entry_helper_.Get(entry), &timestamp_sec,
                           std::move(dump));
        entry.SetTimestamp(timestamp_sec);
      });
    }
    return max_update_ts;
  }

  // Clears data of hash table.
  void Clear() override {
    auto fn = [this]() { entry_helper_.DeallocateAll(); };
    m_.clear_with_callback(fn);
  }

  int64_t Size() const override { return m_.size(); }

  int DimSize() const override { return accessor_->DimSize(); }

  int SliceSize() const override { return accessor_->SliceSize(); }

  bool Contains(int64_t id) const override { return m_.contains(id); }

 private:
  void UpsertEntry(int64_t id,
                   const std::function<void(EntryType&)>& upsert_fn) {
    auto init_fn = [&](EntryType& entry) {
      accessor_->Init(entry_helper_.Get(entry));
      upsert_fn(entry);
    };
    entry_helper_.Upsert(&m_, id, upsert_fn, init_fn);
  }

  CuckooEmbeddingHashTableConfig config_;
  std::unique_ptr<EntryAccessorInterface> accessor_;
  EntryHelper<EntryType> entry_helper_;
  std::unique_ptr<absl::flat_hash_map<int64_t, int>> slot_to_expire_time_;
  int64_t default_expire_time_;
  libcuckoo::cuckoohash_map<int64_t, WithInitFn<EntryType>> m_;
};

template <int64_t length>
std::unique_ptr<EmbeddingHashTableInterface> CreateInlineEntryTable(
    Params p, int64_t size_bytes) {
  if (size_bytes > InlineEntry<length>::capacity()) {
    std::abort();
  }
  return std::make_unique<CuckooEmbeddingHashTable<InlineEntry<length>>>(
      std::move(p), EntryHelper<InlineEntry<length>>());
}

}  // namespace

std::unique_ptr<EmbeddingHashTableInterface> NewCuckooEmbeddingHashTable(
    CuckooEmbeddingHashTableConfig config,
    std::unique_ptr<EntryAccessorInterface> accessor,
    EmbeddingHashTableConfig::EntryType type, uint64_t initial_capacity,
    const SlotExpireTimeConfig& slot_expire_time_config) {
  const int64_t size_bytes = accessor->SizeBytes();
  Params p = {
      std::move(config),
      std::move(accessor),
      initial_capacity,
      slot_expire_time_config,
  };
  if (type == EmbeddingHashTableConfig::PACKED) {
    EntryHelper<PackedEntry> helper(size_bytes);
    return std::make_unique<CuckooEmbeddingHashTable<PackedEntry>>(
        std::move(p), std::move(helper));
  } else if (type == EmbeddingHashTableConfig::RAW) {
    if (size_bytes <= 12) {
      return CreateInlineEntryTable<16>(std::move(p), size_bytes);
    } else if (size_bytes <= 20) {
      return CreateInlineEntryTable<24>(std::move(p), size_bytes);
    } else if (size_bytes <= 28) {
      return CreateInlineEntryTable<32>(std::move(p), size_bytes);
    } else {
      EntryHelper<RawEntry> helper(size_bytes);
      return std::make_unique<CuckooEmbeddingHashTable<RawEntry>>(
          std::move(p), std::move(helper));
    }
  }
  // Should not reach here.
  throw std::invalid_argument(
      absl::StrFormat("Unknonwn entry type table. %d", type));
  return nullptr;
}

}  // namespace hash_table
}  // namespace monolith
