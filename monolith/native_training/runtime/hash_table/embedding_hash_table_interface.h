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

#ifndef MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_EMBEDDING_HASH_TABLE_INTERFACE_H_
#define MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_EMBEDDING_HASH_TABLE_INTERFACE_H_
#include <cstdint>
#include <memory>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "monolith/native_training/runtime/hash_table/embedding_hash_table.pb.h"
using cudaStream_t = void*;
namespace monolith {
namespace hash_table {

// Hash table maps int64 to a float array with fixed length.
// Implemention of this interface should guarantee thread safety.
class EmbeddingHashTableInterface {
 public:
  virtual ~EmbeddingHashTableInterface() = default;

  // Returns the corresponding entry for |ids|.
  virtual int64_t BatchLookup(
      absl::Span<int64_t> ids,
      absl::Span<absl::Span<float>> embeddings) const = 0;

  // Handles the corresponding entry for |ids|.
  virtual void BatchLookupEntry(absl::Span<int64_t> ids,
                                absl::Span<EntryDump> entries) const = 0;

  // Returns the corresponding entry for |id|.
  virtual int64_t Lookup(int64_t id, absl::Span<float> embedding) const = 0;

  // Handles the corresponding entry for |id|.
  virtual void LookupEntry(int64_t id, absl::Span<EntryDump> entry) const = 0;

  // Update the hash table entry directly.
  virtual void Assign(absl::Span<const int64_t> ids,
                      absl::Span<const absl::Span<const float>> updates,
                      int64_t update_time) = 0;

  // Update the hash table entry directly.
  virtual void AssignAdd(int64_t id, absl::Span<const float> update,
                         int64_t update_time) = 0;

  // Update the hash table based on optimizer.
  virtual void BatchOptimize(absl::Span<int64_t> ids,
                             absl::Span<absl::Span<const float>> grads,
                             absl::Span<const float> learning_rates,
                             int64_t update_time,
                             const int64_t global_step = 0) = 0;

  // Update the hash table based on optimizer.
  virtual void Optimize(int64_t id, absl::Span<const float> grad,
                        absl::Span<const float> learning_rates,
                        int64_t update_time, const int64_t global_step = 0) = 0;

  // Evict the outdated hash table values based on the last updated time.
  virtual void Evict(int64_t max_update_time) = 0;

  // Check if a given id exists in the hashtable
  virtual bool Contains(const int64_t id) = 0;

  // To utilize multithread, we need to specify how many shard we will use.
  // Args:
  // offset - The offset of this shard, should be either 0, or return value from
  // Save.
  // limit - how many EntryDump will be fed into write_fn. Default to no limit.
  struct DumpShard {
    int idx;
    int total;
    int64_t limit = 1LL << 61;
  };

  struct DumpIterator {
    int64_t offset = 0;
  };

  using WriteFn = std::function<bool(EntryDump)>;

  // Saves the data. The implementation should guarantee that different shard
  // can be dumped in the parallel.
  virtual void Save(DumpShard shard, WriteFn write_fn,
                    DumpIterator* iter) const = 0;

  // Restores the data from get_fn. The implementation should guarantee that
  // different shard can be dumped in the parallel.
  // |get_fn| returns false if it is end of stream.
  // Returns max_update_ts in this shard.
  virtual int64_t Restore(DumpShard shard,
                          std::function<bool(EntryDump*, int64_t*)> get_fn) = 0;

  // Clears data of hash table.
  virtual void Clear() = 0;

  // Returns the size of the current table.
  virtual int64_t Size() const = 0;

  // Returns the dimension size of the current table.
  virtual int DimSize() const = 0;

  virtual int SliceSize() const = 0;

  // Returns true if the current table contains the given key.
  virtual bool Contains(int64_t id) const = 0;
};

// A decorator will default retdirect all method to base class.
class DefaultEmbeddingHashTableDecorator : public EmbeddingHashTableInterface {
 public:
  DefaultEmbeddingHashTableDecorator(
      std::unique_ptr<EmbeddingHashTableInterface> base)
      : base_(std::move(base)) {}

  EmbeddingHashTableInterface* base() const { return base_.get(); }

  EmbeddingHashTableInterface* base() { return base_.get(); }

  int64_t BatchLookup(absl::Span<int64_t> ids,
                      absl::Span<absl::Span<float>> embeddings) const override {
    return base_->BatchLookup(ids, embeddings);
  }

  void BatchLookupEntry(absl::Span<int64_t> ids,
                        absl::Span<EntryDump> entries) const override {
    return base_->BatchLookupEntry(ids, entries);
  }

  int64_t Lookup(int64_t id, absl::Span<float> embedding) const override {
    return base_->Lookup(id, embedding);
  }

  void LookupEntry(int64_t id, absl::Span<EntryDump> entry) const override {
    return base_->LookupEntry(id, entry);
  }

  void Assign(absl::Span<const int64_t> ids,
              absl::Span<const absl::Span<const float>> updates,
              int64_t update_time) override {
    return base_->Assign(ids, updates, update_time);
  }

  void AssignAdd(int64_t id, absl::Span<const float> update,
                 int64_t update_time) override {
    return base_->AssignAdd(id, update, update_time);
  }

  void BatchOptimize(absl::Span<int64_t> ids,
                     absl::Span<absl::Span<const float>> grads,
                     absl::Span<const float> learning_rates,
                     int64_t update_time,
                     const int64_t global_step = 0) override {
    return base_->BatchOptimize(ids, grads, learning_rates, update_time);
  }

  void Optimize(int64_t id, absl::Span<const float> grad,
                absl::Span<const float> learning_rates, int64_t update_time,
                const int64_t global_step = 0) override {
    return base_->Optimize(id, grad, learning_rates, update_time);
  }

  void Save(DumpShard shard, WriteFn write_fn,
            DumpIterator* iter) const override {
    return base_->Save(shard, std::move(write_fn), iter);
  }

  int64_t Restore(DumpShard shard,
                  std::function<bool(EntryDump*, int64_t*)> get_fn) override {
    return base_->Restore(std::move(shard), std::move(get_fn));
  }

  void Evict(int64_t max_update_time) { base_->Evict(max_update_time); }

  bool Contains(const int64_t id) { return base_->Contains(id); }

  void Clear() override { return base_->Clear(); }

  int64_t Size() const override { return base_->Size(); }

  int DimSize() const override { return base_->DimSize(); }

  int SliceSize() const override { return base_->SliceSize(); }

  bool Contains(int64_t id) const override { return base_->Contains(id); }

 private:
  std::unique_ptr<EmbeddingHashTableInterface> base_;
};

// A class that provides some usefult functionality. Like default values for
// some method.
class EmbeddingHashTableHelper : public DefaultEmbeddingHashTableDecorator {
 public:
  explicit EmbeddingHashTableHelper(
      std::unique_ptr<EmbeddingHashTableInterface> base)
      : DefaultEmbeddingHashTableDecorator(std::move(base)) {}

  using DefaultEmbeddingHashTableDecorator::Assign;
  // Provide default parameters.
  void Assign(absl::Span<const int64_t> ids,
              absl::Span<const absl::Span<const float>> updates) {
    return base()->Assign(ids, updates, 0);
  }

  using DefaultEmbeddingHashTableDecorator::AssignAdd;
  void AssignAdd(int64_t id, absl::Span<const float> update) {
    return base()->AssignAdd(id, update, 0);
  }

  using DefaultEmbeddingHashTableDecorator::BatchOptimize;
  void BatchOptimize(absl::Span<int64_t> ids,
                     absl::Span<absl::Span<const float>> grads,
                     absl::Span<const float> learning_rates) {
    return base()->BatchOptimize(ids, grads, learning_rates, 0, 0);
  }

  // Some wrapper for easy use.
  void AssignOne(int64_t id, absl::Span<const float> update,
                 int64_t update_time = 0) {
    return base()->Assign(absl::MakeConstSpan({id}),
                          absl::MakeConstSpan({update}), update_time);
  }

  using DefaultEmbeddingHashTableDecorator::Contains;
  using DefaultEmbeddingHashTableDecorator::Evict;

  using DefaultEmbeddingHashTableDecorator::Save;
  void Save(DumpShard shard, WriteFn write_fn) {
    DumpIterator iter;
    return base()->Save(std::move(shard), std::move(write_fn), &iter);
  }
};

}  // namespace hash_table
}  // namespace monolith
#endif  // MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_EMBEDDING_HASH_TABLE_INTERFACE_H_
