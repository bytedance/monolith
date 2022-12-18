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

#ifndef MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_RETRIEVER_RETRIEVER_INTERFACE_H_
#define MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_RETRIEVER_RETRIEVER_INTERFACE_H_

#include "absl/types/span.h"

namespace monolith {
namespace hash_table {

class RetrieverInterface {
 public:
  virtual ~RetrieverInterface() = default;

  // How many bytes could be accessed by the retriever
  virtual int64_t SizeBytes() const = 0;

  // The dim that this retriever can support.
  virtual int DimSize() const = 0;

  // Retrieve the num data accessed by the retriever.
  // |num| is a float array whose length is DimSize().
  virtual void Retrieve(const void* ctx, absl::Span<float> num) const = 0;

  // Back propagation
  virtual void Backward(absl::Span<const float> num, absl::Span<float> grad,
                        int64_t global_step) const = 0;
};

}  // namespace hash_table
}  // namespace monolith

#endif  // MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_RETRIEVER_RETRIEVER_INTERFACE_H_
