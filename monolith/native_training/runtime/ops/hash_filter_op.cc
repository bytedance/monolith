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

#include <memory>

#include "monolith/native_training/runtime/hash_filter/dummy_hash_filter.h"
#include "monolith/native_training/runtime/hash_filter/probabilistic_filter.h"
#include "monolith/native_training/runtime/hash_filter/sliding_hash_filter.h"
#include "monolith/native_training/runtime/ops/hash_filter_tf_bridge.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_op_kernel.h"

namespace tensorflow {
namespace monolith_tf {

using ::monolith::hash_filter::DummyHashFilter;
using ::monolith::hash_filter::SlidingHashFilter;
using ::monolith::hash_filter::ProbabilisticFilter;

class DummyFilterOp : public ResourceOpKernel<HashFilterTfBridge> {
 public:
  explicit DummyFilterOp(OpKernelConstruction* ctx) : ResourceOpKernel(ctx) {}

  ~DummyFilterOp() override = default;

 private:
  Status CreateResource(HashFilterTfBridge** filter_bridge)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    auto filter = std::make_unique<DummyHashFilter>();
    *filter_bridge = new HashFilterTfBridge(std::move(filter), config_);
    return Status::OK();
  };

  monolith::hash_table::SlotOccurrenceThresholdConfig config_;
};

class HashFilterOp : public ResourceOpKernel<HashFilterTfBridge> {
 public:
  explicit HashFilterOp(OpKernelConstruction* ctx) : ResourceOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("capacity", &capacity_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("split_num", &split_num_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("config", &config_serialized_));
    if (!config_serialized_.empty()) {
      OP_REQUIRES(
          ctx, config_.ParseFromString(config_serialized_),
          errors::InvalidArgument("Unable to parse config. Make sure it "
                                  "is serialized version of "
                                  "SlotOccurrenceThresholdConfig."));
    }
  }

  ~HashFilterOp() override {}

 private:
  Status CreateResource(HashFilterTfBridge** filter_bridge)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    auto filter = std::make_unique<SlidingHashFilter>(capacity_, split_num_);
    // TODO(leqi.zou): We know this is NOT thread safe. But let's keep it as it
    // is because we may remove HashFilter in the future.
    *filter_bridge = new HashFilterTfBridge(std::move(filter), config_);
    return Status::OK();
  };

  int64 capacity_;
  int split_num_;
  std::string config_serialized_;
  monolith::hash_table::SlotOccurrenceThresholdConfig config_;
};

class ProbabilisticFilterOp : public ResourceOpKernel<HashFilterTfBridge> {
 public:
  explicit ProbabilisticFilterOp(OpKernelConstruction* ctx)
      : ResourceOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("equal_probability", &equal_probability_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("config", &config_serialized_));
    if (!config_serialized_.empty()) {
      OP_REQUIRES(
          ctx, config_.ParseFromString(config_serialized_),
          errors::InvalidArgument("Unable to parse config. Make sure it "
                                  "is serialized version of "
                                  "SlotOccurrenceThresholdConfig."));
    }
  }

  ~ProbabilisticFilterOp() override = default;

 private:
  Status CreateResource(HashFilterTfBridge** filter_bridge)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    auto filter = std::make_unique<ProbabilisticFilter>(equal_probability_);
    *filter_bridge = new HashFilterTfBridge(std::move(filter), config_);
    return Status::OK();
  };

  bool equal_probability_;
  std::string config_serialized_;
  monolith::hash_table::SlotOccurrenceThresholdConfig config_;
};

REGISTER_OP("MonolithHashFilter")
    .Output("handle: resource")
    .Attr("capacity: int = 300000000")
    .Attr("split_num: int = 7")
    // Config contains a string of pb message SlotOccurrenceThresholdConfig.
    .Attr("config: string = ''")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(Name("MonolithHashFilter").Device(DEVICE_CPU),
                        HashFilterOp);

REGISTER_OP("MonolithProbabilisticFilter")
    .Output("handle: resource")
    .Attr("equal_probability: bool = false")
    // Config contains a string of pb message SlotOccurrenceThresholdConfig.
    .Attr("config: string = ''")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(Name("MonolithProbabilisticFilter").Device(DEVICE_CPU),
                        ProbabilisticFilterOp);

REGISTER_OP("MonolithDummyHashFilter")
    .Output("handle: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);
REGISTER_KERNEL_BUILDER(Name("MonolithDummyHashFilter").Device(DEVICE_CPU),
                        DummyFilterOp);

}  // namespace monolith_tf
}  // namespace tensorflow
