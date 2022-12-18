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

#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace monolith_tf {

class SwitchSlotOp : public OpKernel {
 public:
  using OpKernel::OpKernel;
  using ConstFlatSplits = typename TTypes<int64>::ConstFlat;

  explicit SwitchSlotOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("slot", &slot_));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("fid_version", &fid_version_));
  }

  void Compute(OpKernelContext *context) override {
    // Read the `rt_nested_splits` input & convert to Eigen tensors.
    OpInputList rt_nested_splits_in;
    OP_REQUIRES_OK(
        context, context->input_list("rt_nested_splits", &rt_nested_splits_in));
    const int rt_nested_splits_len = rt_nested_splits_in.size();

    OpOutputList rt_nested_splits_out;
    OP_REQUIRES_OK(context, context->output_list("nested_splits_out",
                                                 &rt_nested_splits_out));

    for (int i = 0; i < rt_nested_splits_len; ++i) {
      Tensor *out_splits;
      OP_REQUIRES_OK(context,
                     rt_nested_splits_out.allocate(
                         i, rt_nested_splits_in[i].shape(), &out_splits));
      std::memcpy(out_splits->data(), rt_nested_splits_in[i].data(),
                  sizeof(int64) * rt_nested_splits_in[i].NumElements());
    }

    const Tensor &rt_dense_values_in = context->input(rt_nested_splits_len);
    Tensor *dense_values_out;
    OP_REQUIRES_OK(context, context->allocate_output("dense_values_out",
                                                     rt_dense_values_in.shape(),
                                                     &dense_values_out));
    auto dense_values_int_ = rt_dense_values_in.flat<int64>();
    auto dense_values_out_ = dense_values_out->flat<int64>();

    for (int i = 0; i < dense_values_int_.size(); ++i) {
      if (fid_version_ == 1) {
        dense_values_out_(i) = convert_v1(dense_values_int_(i));
      } else {
        dense_values_out_(i) = convert_v2(dense_values_int_(i));
      }
    }
  }

 private:
  inline int64 convert_v1(int64 fid) {
    static int64 mask = (static_cast<int64>(1) << 55) - 1;
    return (static_cast<int64>(slot_) << 54) | (fid & mask);
  }

  inline int64 convert_v2(int64 fid) {
    static int64 mask = (static_cast<int64>(1) << 49) - 1;
    return (static_cast<int64>(slot_) << 48) | (fid & mask);
  }

  int slot_, fid_version_;
};

class FeatureCombineOp : public OpKernel {
 public:
  using OpKernel::OpKernel;
  using ConstFlatSplits = typename TTypes<int64>::ConstFlat;

  explicit FeatureCombineOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("slot", &slot_));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("fid_version", &fid_version_));
  }

  void Compute(OpKernelContext *context) override {
    // Read the `rt_nested_splits` input & convert to Eigen tensors.
    OpInputList rt_nested_splits_src1_in;
    OP_REQUIRES_OK(context, context->input_list("rt_nested_splits_src1",
                                                &rt_nested_splits_src1_in));
    int input_cnt = rt_nested_splits_src1_in.size();
    const Tensor &rt_dense_values_src1_in = context->input(input_cnt);
    input_cnt++;
    OpInputList rt_nested_splits_src2_in;
    OP_REQUIRES_OK(context, context->input_list("rt_nested_splits_src2",
                                                &rt_nested_splits_src2_in));
    input_cnt += rt_nested_splits_src2_in.size();
    const Tensor &rt_dense_values_src2_in = context->input(input_cnt);

    DCHECK_EQ(rt_nested_splits_src1_in.size(), rt_nested_splits_src2_in.size());

    OpOutputList nested_splits_sink;
    OP_REQUIRES_OK(context, context->output_list("nested_splits_sink",
                                                 &nested_splits_sink));

    int src_idx = 0;
    if (rt_nested_splits_src1_in.size() == 2) {
      auto batch_splits_src1 = rt_nested_splits_src1_in[src_idx].flat<int64>();
      auto batch_splits_src2 = rt_nested_splits_src2_in[src_idx].flat<int64>();
      DCHECK_EQ(batch_splits_src1.size(), batch_splits_src2.size());

      for (int i = 0; i < batch_splits_src1.size(); ++i) {
        DCHECK_EQ(batch_splits_src1(i), batch_splits_src2(i));
      }
      src_idx++;
    }

    auto ins_splits_src1 = rt_nested_splits_src1_in[src_idx].flat<int64>();
    auto rt_dense_values_src1 = rt_dense_values_src1_in.flat<int64>();
    auto ins_splits_src2 = rt_nested_splits_src2_in[src_idx].flat<int64>();
    auto rt_dense_values_src2 = rt_dense_values_src2_in.flat<int64>();

    int batch_size = ins_splits_src1.size() - 1;

    Tensor *ins_splits_sink;
    OP_REQUIRES_OK(context,
                   nested_splits_sink.allocate(
                       src_idx, rt_nested_splits_src2_in[src_idx].shape(),
                       &ins_splits_sink));
    auto ins_splits = ins_splits_sink->flat<int64>();
    ins_splits(0) = 0;
    for (int i = 0; i < batch_size; ++i) {
      int src1_start = ins_splits_src1(i);
      int src1_end = ins_splits_src1(i + 1);
      int src2_start = ins_splits_src2(i);
      int src2_end = ins_splits_src2(i + 1);
      ins_splits(i + 1) =
          ins_splits(i) + (src1_end - src1_start) * (src2_end - src2_start);
    }

    Tensor *dense_values_sink;
    OP_REQUIRES_OK(context, context->allocate_output("dense_values_sink",
                                                     {ins_splits(batch_size)},
                                                     &dense_values_sink));
    auto dense_values = dense_values_sink->flat<int64>();

    int idx = 0;
    for (int i = 0; i < batch_size; ++i) {
      int src1_start = ins_splits_src1(i);
      int src1_end = ins_splits_src1(i + 1);
      int src2_start = ins_splits_src2(i);
      int src2_end = ins_splits_src2(i + 1);

      for (int j = src1_start; j < src1_end; ++j) {
        int64 fid1 = rt_dense_values_src1(j);
        for (int k = src2_start; k < src2_end; ++k) {
          int64 fid2 = rt_dense_values_src2(k);
          if (fid_version_ == 1) {
            dense_values(idx++) = convert_v1(combine(fid1, fid2));
          } else {
            dense_values(idx++) = convert_v2(combine(fid1, fid2));
          }
        }
      }
    }
  }

 private:
  inline int64 convert_v1(int64 fid) {
    static int64 mask = (static_cast<int64>(1) << 55) - 1;
    return (static_cast<int64>(slot_) << 54) | (fid & mask);
  }

  inline int64 convert_v2(int64 fid) {
    static int64 mask = (static_cast<int64>(1) << 49) - 1;
    return (static_cast<int64>(slot_) << 48) | (fid & mask);
  }

  int64 combine(int64 fid1, int64 fid2) {
    auto mu = absl::int128(fid1) * absl::int128(fid2);

    uint64 hi = static_cast<uint64>((mu >> 64).operator long());  // NOLINT
    uint64 lo = static_cast<uint64>(mu.operator long());          // NOLINT

    return static_cast<int64>(hi ^ lo);
  }

  int slot_, fid_version_;
};

namespace {
REGISTER_KERNEL_BUILDER(Name("FeatureCombine").Device(DEVICE_CPU),
                        FeatureCombineOp);

REGISTER_KERNEL_BUILDER(Name("SwitchSlot").Device(DEVICE_CPU), SwitchSlotOp);
}

}  // namespace monolith_tf
}  // namespace tensorflow
