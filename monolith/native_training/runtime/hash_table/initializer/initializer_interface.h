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

#ifndef MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_INITIALIZER_INTERFACE
#define MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_INITIALIZER_INTERFACE

#include "absl/types/span.h"

namespace monolith {
namespace hash_table {

class InitializerInterface {
 public:
  virtual ~InitializerInterface() = default;

  virtual int DimSize() const = 0;

  virtual void Initialize(absl::Span<float> nums) const = 0;
};

}  // namespace hash_table
}  // namespace monolith
#endif  // MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_INITIALIZER_INTERFACE
