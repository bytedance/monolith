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

#ifndef MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_RETRIEVER_RETRIEVER_BASE_H_
#define MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_RETRIEVER_RETRIEVER_BASE_H_

#include "monolith/native_training/runtime/hash_table/retriever/retriever_interface.h"

namespace monolith {
namespace hash_table {

class RetrieverBase : public RetrieverInterface {
 public:
  explicit RetrieverBase(int dim_size)
      : dim_size_(dim_size), size_bytes_(sizeof(float) * dim_size) {}

  int64_t SizeBytes() const override {
    return size_bytes_;
  }

  int DimSize() const override {
    return dim_size_;
  }

 protected:
  absl::Span<const float> GetNum(const void* ctx) const {
    const auto* ctx_float = static_cast<const float*>(ctx);
    return absl::MakeConstSpan(ctx_float, ctx_float + dim_size_);
  }

  int dim_size_;

  int64_t size_bytes_;
};

}  // namespace hash_table
}  // namespace monolith

#endif  // MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_RETRIEVER_RETRIEVER_BASE_H_
