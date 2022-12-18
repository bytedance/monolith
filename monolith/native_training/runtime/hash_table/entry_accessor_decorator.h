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

#ifndef MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_ENTRY_ACCESSOR_DECORATOR_H_
#define MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_ENTRY_ACCESSOR_DECORATOR_H_

#include "monolith/native_training/runtime/hash_table/entry_accessor.h"

namespace monolith {
namespace hash_table {

// DEPRECATED: Prefer using retriever
//
// The base class of decorator. By default, it delegates all requests to the
// base entry accessor.
class EntryAccessorDecorator : public EntryAccessorInterface {
 public:
  explicit EntryAccessorDecorator(
      std::unique_ptr<EntryAccessorInterface> entry_accessor)
      : entry_accessor_(std::move(entry_accessor)) {}

  int64_t SizeBytes() const override { return entry_accessor_->SizeBytes(); }

  int DimSize() const override { return entry_accessor_->DimSize(); }

  int SliceSize() const override { return entry_accessor_->SliceSize(); }

  void Init(void* ctx) const override { entry_accessor_->Init(ctx); }

  void Fill(const void* ctx, absl::Span<float> num) const override {
    entry_accessor_->Fill(ctx, num);
  }

  void Assign(absl::Span<const float> num, void* ctx) const override {
    entry_accessor_->Assign(num, ctx);
  }

  void AssignAdd(absl::Span<const float> num, void* ctx) const override {
    entry_accessor_->AssignAdd(num, ctx);
  }

  void Optimize(void* ctx, absl::Span<const float> grad,
                absl::Span<const float> learning_rates,
                const int64_t global_step) const override {
    entry_accessor_->Optimize(ctx, grad, learning_rates, global_step);
  }

  EntryDump Save(const void* ctx, uint32_t timestamp_sec) const override {
    return entry_accessor_->Save(ctx, timestamp_sec);
  }

  void Restore(void* ctx, uint32_t* timestamp_sec,
               EntryDump dump) const override {
    entry_accessor_->Restore(ctx, timestamp_sec, dump);
  }

 protected:
  std::unique_ptr<EntryAccessorInterface> entry_accessor_;
};

}  // namespace hash_table
}  // namespace monolith

#endif  // MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_ENTRY_ACCESSOR_DECORATOR_H_
