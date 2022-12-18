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

#include "monolith/native_training/runtime/hash_table/retriever/retriever_combination.h"

#include <memory>

#include "monolith/native_training/runtime/hash_table/retriever/retriever_base.h"

namespace monolith {
namespace hash_table {
namespace {

class CombinedRetriever final : public RetrieverBase {
 public:
  CombinedRetriever(std::unique_ptr<RetrieverInterface> retriever1,
                    std::unique_ptr<RetrieverInterface> retriever2)
      : RetrieverBase(retriever1->DimSize() + retriever2->DimSize()),
        retriever1_(std::move(retriever1)),
        retriever2_(std::move(retriever2)) {}

  void Retrieve(const void* ctx, absl::Span<float> num) const override {
    const void* ctx2 = static_cast<const char*>(ctx) + retriever1_->SizeBytes();
    auto num2 = num.subspan(retriever1_->DimSize());
    retriever1_->Retrieve(ctx, num);
    retriever2_->Retrieve(ctx2, num2);
  }

  void Backward(absl::Span<const float> num, absl::Span<float> grad,
                int64_t global_step) const override {
    int dim_size1 = retriever1_->DimSize();
    retriever1_->Backward(num.subspan(0, dim_size1), grad.subspan(0, dim_size1),
                          global_step);
    retriever2_->Backward(num.subspan(dim_size1), grad.subspan(dim_size1),
                          global_step);
  }

 private:
  int dim_size_;
  std::unique_ptr<RetrieverInterface> retriever1_;
  std::unique_ptr<RetrieverInterface> retriever2_;
};

}  // namespace

std::unique_ptr<RetrieverInterface> CombineRetrievers(
    std::unique_ptr<RetrieverInterface> retriever1,
    std::unique_ptr<RetrieverInterface> retriever2) {
  return std::make_unique<CombinedRetriever>(std::move(retriever1),
                                             std::move(retriever2));
}

}  // namespace hash_table
}  // namespace monolith
