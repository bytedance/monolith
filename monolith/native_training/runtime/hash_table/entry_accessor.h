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

#ifndef MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_ENTRY_ACCESSOR
#define MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_ENTRY_ACCESSOR

#include <cassert>
#include <cmath>
#include <cstdint>
#include <memory>

#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "glog/logging.h"
#include "monolith/native_training/runtime/hash_table/embedding_hash_table.pb.h"
#include "monolith/native_training/runtime/hash_table/initializer/initializer_interface.h"
#include "monolith/native_training/runtime/hash_table/optimizer/optimizer_interface.h"
#include "monolith/native_training/runtime/hash_table/utils.h"

namespace monolith {
namespace hash_table {

class EntryAccessorInterface {
 public:
  virtual ~EntryAccessorInterface() = default;

  // Size bytes need to be aloocated in this entry.
  virtual int64_t SizeBytes() const = 0;

  // The dim that this entry accessor can support
  virtual int DimSize() const = 0;

  // The number of slices in this entry.
  virtual int SliceSize() const = 0;

  // Initialize the given entry.
  virtual void Init(void* ctx) const = 0;

  // Fills the num based on entry.
  virtual void Fill(const void* ctx, absl::Span<float> num) const = 0;

  // Assign the entry using num
  virtual void Assign(absl::Span<const float> num, void* ctx) const = 0;

  // AssignAdd the entry using num
  virtual void AssignAdd(absl::Span<const float> num, void* ctx) const = 0;

  // Optimizes the entry with |grad|.
  virtual void Optimize(void* ctx, absl::Span<const float> grad,
                        absl::Span<const float> learning_rates,
                        const int64_t global_step = 0) const = 0;

  // Converts an entry to EntryDump.
  virtual EntryDump Save(const void* ctx, uint32_t timestamp_sec) const = 0;

  // Restores the entry from |dump|.
  virtual void Restore(void* ctx, uint32_t* timestamp_sec,
                       EntryDump dump) const = 0;
};

std::unique_ptr<EntryAccessorInterface> NewEntryAccessor(EntryConfig config);

}  // namespace hash_table
}  // namespace monolith
#endif  // MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_ENTRY_ACCESSOR
