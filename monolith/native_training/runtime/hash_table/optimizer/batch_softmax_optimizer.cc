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

#include "monolith/native_training/runtime/hash_table/optimizer/batch_softmax_optimizer.h"

#include "absl/strings/str_format.h"
#include "glog/logging.h"

namespace monolith {
namespace hash_table {
namespace {

class BatchSoftmaxOptimizer : public OptimizerInterface {
 public:
  explicit BatchSoftmaxOptimizer(BatchSoftmaxOptimizerConfig config)
      : config_(std::move(config)) {
    DCHECK_EQ(config_.dim_size(), 1);
  }

  int64_t SizeBytes() const override { return sizeof(int64_t); }

  int DimSize() const override { return config_.dim_size(); }

  int SliceSize() const override { return 1; }

  void Init(void* ctx) const override {
    auto* A = reinterpret_cast<int64_t*>(ctx);
    *A = 0;
  }

  void Optimize(void* ctx, absl::Span<float> num, absl::Span<const float> grad,
                absl::Span<const float> learning_rates,
                const int64_t global_step) const override {
    float& B = num[0];
    int64_t& A = *reinterpret_cast<int64_t*>(ctx);
    float alpha = learning_rates[0];
    B = (1 - alpha) * B + alpha * static_cast<float>(global_step - A);
    if (global_step < 0) {
      LOG(FATAL) << absl::StrFormat(
          "global_step=%ld is negative, please investigate!", global_step);
    }
    A = global_step;
  }

  OptimizerDump Save(const void* ctx) const override {
    OptimizerDump dump;
    BatchSoftmaxOptimizerDump* batch_softmax_dump =
        dump.add_dump()->mutable_batch_softmax();
    int64_t A = *reinterpret_cast<const int64_t*>(ctx);
    batch_softmax_dump->set_global_step(A);
    return dump;
  }

  void Restore(void* ctx, OptimizerDump dump) const override {
    const BatchSoftmaxOptimizerDump& batch_softmax_dump =
        dump.dump(0).batch_softmax();
    int64_t& A = *reinterpret_cast<int64_t*>(ctx);
    A = batch_softmax_dump.global_step();
  }

 private:
  BatchSoftmaxOptimizerConfig config_;
};

}  // namespace

std::unique_ptr<OptimizerInterface> NewBatchSoftmaxOptimizer(
    BatchSoftmaxOptimizerConfig config) {
  return std::make_unique<BatchSoftmaxOptimizer>(std::move(config));
}

}  // namespace hash_table
}  // namespace monolith
