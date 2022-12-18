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

#ifndef MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_OPTIMIZER_OPTIMIZER_DECORATOR
#define MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_OPTIMIZER_OPTIMIZER_DECORATOR
#include <cstdint>

#include "absl/types/span.h"
#include "monolith/native_training/runtime/hash_table/optimizer/optimizer.pb.h"
#include "optimizer_interface.h"

namespace monolith {
namespace hash_table {

class OptimizerDecorator : public OptimizerInterface {
 public:
  virtual ~OptimizerDecorator() = default;

  explicit OptimizerDecorator(std::unique_ptr<OptimizerInterface> base_opt)
      : base_opt_(std::move(base_opt)) {}

  int64_t SizeBytes() const override {
    return base_opt_.get()->SizeBytes();
  }

  int DimSize() const override { return base_opt_.get()->DimSize(); }

  int SliceSize() const override { return base_opt_.get()->SliceSize(); }

  void Init(void* ctx) const override { return base_opt_.get()->Init(ctx); }

  void Optimize(void* ctx, absl::Span<float> num,
                absl::Span<const float> grad,
                absl::Span<const float> learning_rates,
                const int64_t global_step = 0) const override {
    return base_opt_.get()->Optimize(ctx, num, grad, learning_rates);
  }

  OptimizerDump Save(const void* ctx) const override { return base_opt_.get()->Save(ctx); }

  void Restore(void* ctx, OptimizerDump dump) const override { return base_opt_.get()->Restore(ctx, dump); }


  virtual void OptimizeWithLatestValue(void* ctx, absl::Span<float> num,
                               absl::Span<const float> grad,
                               absl::Span<const float> learning_rates,
                               absl::Span<float> latest_value,
                               const int64_t global_step = 0) const = 0;

 protected:
  std::unique_ptr<OptimizerInterface> base_opt_;
};

}  // namespace hash_table
}  // namespace monolith
#endif  // MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_OPTIMIZER_OPTIMIZER_DECORATOR
