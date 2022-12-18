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

#include <cstdlib>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "monolith/native_training/runtime/common/metrics.h"
#include "monolith/native_training/runtime/ops/logging_ops.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace monolith_tf {
namespace {

class TensorTimestampOp : public OpKernel {
 public:
  explicit TensorTimestampOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &types_));
  }

  void Compute(OpKernelContext* ctx) override {
    for (int i = 0; i < static_cast<int>(types_.size()); ++i) {
      ctx->set_output(i, ctx->input(i));
    }
    Tensor* ts;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(types_.size(), {}, &ts));
    auto ts_scalar = ts->scalar<int64>();
    ts_scalar() = absl::ToUnixMicros(absl::Now());
  }

 private:
  std::vector<DataType> types_;
};

REGISTER_OP("MonolithTensorsTimestamp")
    .Attr("T: list(type)")
    .Input("tensors_in: T")
    .Output("tensors_out: T")
    .Output("timestamp: int64")
    .SetShapeFn([](shape_inference::InferenceContext* ctx) {
      std::vector<DataType> types;
      TF_RETURN_IF_ERROR(ctx->GetAttr("T", &types));
      for (int i = 0; i < static_cast<int>(types.size()); ++i) {
        ctx->set_output(i, ctx->input(i));
      }
      ctx->set_output(types.size(), ctx->Scalar());
      return Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name("MonolithTensorsTimestamp").Device(DEVICE_CPU),
                        TensorTimestampOp);

// Deprecated.
class MetricOp : public OpKernel {
 public:
  explicit MetricOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("key", &key_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("tags", &tags_));
  }

  void Compute(OpKernelContext* ctx) override {
    for (int i = 0; i < static_cast<int>(types_.size()); ++i) {
      ctx->set_output(i, ctx->input(i));
    }
    const Tensor& value_tensor = ctx->input(types_.size());
    const float value = value_tensor.scalar<float>()();
    monolith::GetMetrics()->emit_timer(key_, value, tags_);
  }

 private:
  std::vector<DataType> types_;
  std::string key_;
  std::string tags_;
};

REGISTER_OP("MonolithMetric")
    .Attr("T: list(type)")
    .Attr("key: string")
    .Attr("tags: string")
    .Input("tensors_in: T")
    .Input("value: float")
    .Output("tensors_out: T")
    .SetShapeFn([](shape_inference::InferenceContext* ctx) {
      std::vector<DataType> types;
      TF_RETURN_IF_ERROR(ctx->GetAttr("T", &types));
      for (int i = 0; i < static_cast<int>(types.size()); ++i) {
        ctx->set_output(i, ctx->input(i));
      }
      return Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name("MonolithMetric").Device(DEVICE_CPU), MetricOp);

class MetricV2Op : public OpKernel {
 public:
  explicit MetricV2Op(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("key", &key_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("tags", &tags_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& value_tensor = ctx->input(0);
    const float value = value_tensor.scalar<float>()();
    monolith::GetMetrics()->emit_timer(key_, value, tags_);
  }

 private:
  std::string key_;
  std::string tags_;
};

REGISTER_OP("MonolithMetricV2")
    .Attr("key: string")
    .Attr("tags: string")
    .SetIsStateful()
    .Input("value: float")
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_KERNEL_BUILDER(Name("MonolithMetricV2").Device(DEVICE_CPU),
                        MetricV2Op);

struct MachineInfo : ResourceBase {
  int64 mem_limit = 0;

  std::string DebugString() const {
    return absl::StrFormat("mem_limit: %lld", mem_limit);
  }
};

class MachineInfoOp : public ResourceOpKernel<MachineInfo> {
 public:
  explicit MachineInfoOp(OpKernelConstruction* c) : ResourceOpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("mem_limit", &mem_limit_));
  }

 private:
  Status CreateResource(MachineInfo** info_out)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    auto* info = new MachineInfo();
    info->mem_limit = mem_limit_;
    *info_out = info;
    return Status::OK();
  }

  int64 mem_limit_;  // Unit bytes
};

REGISTER_OP("MonolithMachineInfo")
    .Output("handle: resource")
    .Attr("mem_limit: int")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(Name("MonolithMachineInfo").Device(DEVICE_CPU),
                        MachineInfoOp);

class MonolithCheckMachineHealthOp : public OpKernel {
 public:
  explicit MonolithCheckMachineHealthOp(OpKernelConstruction* c)
      : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    int64 current_mem = GetCurrentUsage();
    OP_REQUIRES(c, current_mem > 0,
                errors::Internal("Unable to get the current process usage."));
    MachineInfo* info = nullptr;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &info));
    core::ScopedUnref unref(info);
    Tensor* result_tensor;
    OP_REQUIRES_OK(c, c->allocate_output(0, {}, &result_tensor));
    auto result_scalar = result_tensor->scalar<tstring>();
    MachineHealthResult result;
    if (current_mem >= info->mem_limit) {
      result.set_status(MachineHealthResult::OUT_OF_MEMORY);
      result.set_message(
          absl::StrFormat("Memory limit exceeded. Current: %lld, Limit: %lld",
                          current_mem, info->mem_limit));
    }
    result_scalar() = result.SerializeAsString();
  }

  int64_t GetCurrentUsage() {
    FILE* file = fopen("/proc/self/status", "r");
    int64_t result = 0;
    char line[128];
    while (fgets(line, 128, file) != NULL) {
      if (std::strncmp(line, "VmRSS:", 6) == 0) {
        // The line is like `VmRSS:       708 kB`
        result = std::strtol(line + 6, nullptr, 10);
        break;
      }
    }
    fclose(file);
    return result * 1024;
  }
};

REGISTER_OP("MonolithCheckMachineHealth")
    .Input("machine_info_handle: resource")
    .Output("serialized_result: string")
    .SetShapeFn([](shape_inference::InferenceContext* ctx) {
      ctx->set_output(0, ctx->Scalar());
      return Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name("MonolithCheckMachineHealth").Device(DEVICE_CPU),
                        MonolithCheckMachineHealthOp);

}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow
