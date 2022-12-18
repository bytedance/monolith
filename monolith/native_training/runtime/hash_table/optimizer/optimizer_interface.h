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

#ifndef MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_OPTIMIZER_OPTIMIZER_INTERFACE
#define MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_OPTIMIZER_OPTIMIZER_INTERFACE
#include <cstdint>

#include "absl/types/span.h"
#include "monolith/native_training/runtime/hash_table/optimizer/optimizer.pb.h"

namespace monolith {
namespace hash_table {

class OptimizerInterface {
 public:
  virtual ~OptimizerInterface() = default;

  // How many bytes are required for the optimizer
  virtual int64_t SizeBytes() const = 0;

  // The dim that this optimizer can support.
  virtual int DimSize() const = 0;

  // The slice size that this optimizer holds.
  virtual int SliceSize() const = 0;

  // Init optimizer ctx.
  // |num| is at least DimSize() long.
  virtual void Init(void* ctx) const = 0;

  // optimize the num based on gradients and the optimizer's data.
  // |num|, |grad| are float arrays whose length is at least DimSize().
  virtual void Optimize(void* ctx, absl::Span<float> num,
                        absl::Span<const float> grad,
                        absl::Span<const float> learning_rates,
                        const int64_t global_step = 0) const = 0;

  // Save and restore the entry.
  virtual OptimizerDump Save(const void* ctx) const = 0;
  virtual void Restore(void* ctx, OptimizerDump dump) const = 0;
};

}  // namespace hash_table
}  // namespace monolith
#endif  // MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_OPTIMIZER_OPTIMIZER_INTERFACE
