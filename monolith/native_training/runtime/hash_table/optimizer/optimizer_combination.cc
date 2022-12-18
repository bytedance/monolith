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

#include "monolith/native_training/runtime/hash_table/optimizer/optimizer_combination.h"

#include "absl/algorithm/container.h"
#include "absl/types/span.h"

namespace monolith {
namespace hash_table {
namespace {

namespace proto2 = google::protobuf;

class CombinedOptimizer : public OptimizerInterface {
 public:
  CombinedOptimizer(std::unique_ptr<OptimizerInterface> opt1,
                    std::unique_ptr<OptimizerInterface> opt2)
      : opt1_(std::move(opt1)),
        size_bytes1_(opt1_->SizeBytes()),
        dim_size1_(opt1_->DimSize()),
        dump_size1_(GetOptDumpSize(opt1_.get())),
        slice_size1_(opt1_->SliceSize()),
        opt2_(std::move(opt2)) {}

  int64_t SizeBytes() const override {
    return opt1_->SizeBytes() + opt2_->SizeBytes();
  }

  int DimSize() const override { return opt1_->DimSize() + opt2_->DimSize(); }

  int SliceSize() const override {
    return opt1_->SliceSize() + opt2_->SliceSize();
  }

  void Init(void* ctx) const override {
    void* ctx2 = static_cast<char*>(ctx) + size_bytes1_;
    opt1_->Init(ctx);
    opt2_->Init(ctx2);
  }

  void Optimize(void* ctx, absl::Span<float> num,
                absl::Span<const float> grad,
                absl::Span<const float> learning_rates,
                const int64_t global_step) const override {
    void* ctx2 = static_cast<char*>(ctx) + size_bytes1_;
    auto num2 = num.subspan(dim_size1_);
    auto grad2 = grad.subspan(dim_size1_);
    auto learning_rates2 = learning_rates.subspan(slice_size1_);
    opt1_->Optimize(ctx, num, grad, learning_rates, global_step);
    opt2_->Optimize(ctx2, num2, grad2, learning_rates2, global_step);
  }

  OptimizerDump Save(const void* ctx) const override {
    OptimizerDump combined_dump;
    OptimizerDump dump1 = opt1_->Save(ctx);
    const void* ctx2 = static_cast<const char*>(ctx) + size_bytes1_;
    OptimizerDump dump2 = opt2_->Save(ctx2);
    absl::c_move(*dump1.mutable_dump(), proto2::RepeatedFieldBackInserter(
                                            combined_dump.mutable_dump()));
    absl::c_move(*dump2.mutable_dump(), proto2::RepeatedFieldBackInserter(
                                            combined_dump.mutable_dump()));
    return combined_dump;
  }

  void Restore(void* ctx, OptimizerDump dump) const override {
    OptimizerDump dump1;
    for (int i = 0; i < dump_size1_; ++i) {
      *dump1.add_dump() = std::move(*dump.mutable_dump(i));
    }
    OptimizerDump dump2;
    for (int i = dump_size1_; i < dump.dump_size(); ++i) {
      *dump2.add_dump() = std::move(*dump.mutable_dump(i));
    }
    opt1_->Restore(ctx, std::move(dump1));
    void* ctx2 = static_cast<char*>(ctx) + size_bytes1_;
    opt2_->Restore(ctx2, std::move(dump2));
  }

 private:
  int GetOptDumpSize(OptimizerInterface* opt) {
    auto mem = std::make_unique<char[]>(opt->SizeBytes());
    opt->Init(mem.get());
    OptimizerDump dump = opt->Save(mem.get());
    return dump.dump_size();
  }

  std::unique_ptr<OptimizerInterface> opt1_;
  const int64_t size_bytes1_;
  const int dim_size1_;
  const int dump_size1_;
  const int slice_size1_;
  std::unique_ptr<OptimizerInterface> opt2_;
};

}  // namespace

std::unique_ptr<OptimizerInterface> CombineOptimizers(
    std::unique_ptr<OptimizerInterface> opt1,
    std::unique_ptr<OptimizerInterface> opt2) {
  return std::make_unique<CombinedOptimizer>(std::move(opt1), std::move(opt2));
}

}  // namespace hash_table
}  // namespace monolith
