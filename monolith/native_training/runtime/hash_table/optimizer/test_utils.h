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

#ifndef MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_OPTIMIZER_TEST_UTILS
#define MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_OPTIMIZER_TEST_UTILS

#include "absl/types/span.h"
#include "monolith/native_training/runtime/hash_table/optimizer/optimizer_interface.h"

namespace monolith {
namespace hash_table {

// The pre allocated memory for the optimizer.
class TestOptimizerEntry {
 public:
  explicit TestOptimizerEntry(OptimizerInterface* opt) : opt_(opt) {
    ctx_ = NewCtx();
    num_ = NewNum();
  }

  const void* ctx() { return ctx_.get(); }
  void* mutable_ctx() { return ctx_.get(); }

  std::vector<float>* mutable_num() { return &num_; }
  const std::vector<float>& num() { return num_; }
  absl::Span<float> mutable_num_span() { return absl::MakeSpan(num_); }

 private:
  std::unique_ptr<char[]> NewCtx() {
    return std::make_unique<char[]>(opt_->SizeBytes());
  }

  std::vector<float> NewNum() {
    return std::vector<float>(opt_->DimSize(), 0.0f);
  }

  OptimizerInterface* opt_;
  std::unique_ptr<char[]> ctx_;
  std::vector<float> num_;
};

}  // namespace hash_table
}  // namespace monolith

#endif  // MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_OPTIMIZER_TEST_UTILS
