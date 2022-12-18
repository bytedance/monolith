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

#include "monolith/native_training/runtime/hash_table/retriever/fake_quant_retriever.h"

#include <memory>
#include "absl/algorithm/container.h"

#include "monolith/native_training/runtime/hash_table/retriever/retriever_base.h"

namespace monolith {
namespace hash_table {
namespace {

class FakeQuantRetriever final : public RetrieverBase {
 public:
  FakeQuantRetriever(int dim_size, const FakeQuantizer& fake_quantizer)
      : RetrieverBase(dim_size), fake_quantizer_(fake_quantizer) {}

  void Retrieve(const void* ctx, absl::Span<float> num) const override {
    absl::c_copy(GetNum(ctx), num.begin());
    for (int i = 0; i < dim_size_; ++i) {
      num[i] = fake_quantizer_.Quantize(num[i]);
    }
  }

  void Backward(absl::Span<const float> num, absl::Span<float> grad,
                int64_t global_step) const override {}

 private:
  FakeQuantizer fake_quantizer_;
};

}  // namespace

std::unique_ptr<RetrieverInterface> NewFakeQuantRetriever(
    int dim_size, const FakeQuantizer& fake_quantizer) {
  return std::make_unique<FakeQuantRetriever>(dim_size, fake_quantizer);
}

}  // namespace hash_table
}  // namespace monolith
