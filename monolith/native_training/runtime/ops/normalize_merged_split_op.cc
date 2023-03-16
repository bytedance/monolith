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

#include <cstring>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/threadpool.h"

namespace tensorflow {
namespace monolith_tf {
namespace {

class NormalizeMergedSplitOp : public OpKernel {
 public:
  explicit NormalizeMergedSplitOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor *row_split_input;
    OP_REQUIRES_OK(ctx, ctx->input("row_split", &row_split_input));

    const Tensor *row_split_size_input;
    OP_REQUIRES_OK(ctx, ctx->input("row_split_size", &row_split_size_input));

    const auto row_split_vec = row_split_input->flat<int64>();
    int split_num = row_split_input->dim_size(0);

    const auto row_split_size_vec = row_split_size_input->flat<int32>();
    int merge_num = row_split_size_input->dim_size(0);

    int output_size = split_num + 1 - merge_num;
    Tensor *normed_row_split_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("normed_row_split",
                                             TensorShape({
                                                 output_size,
                                             }),
                                             &normed_row_split_tensor));
    auto normed_row_split_flat = normed_row_split_tensor->flat<int64_t>();

    int offset = 0;
    int pre_size = 0;
    int output_idx = 0;
    /*
     row_split: 0, 2, 5, 5, 9, 0, 0, 3, 4
     row_split_size: 5, 4
     offset = 0, pre_size = 0
     output: 0, 2, 5, 5, 9
     offset = 5, pre_size = 9
     output: 0, 2, 5, 5, 9, 9, 12, 13
     offset = 9, pre_size = 13
     */
    for (size_t i = 0; i < merge_num; ++i) {
      if (i == 0) {
        for (int j = offset; j < offset + row_split_size_vec(i); ++j) {
          normed_row_split_flat(output_idx) = pre_size + row_split_vec(j);
          output_idx++;
        }
      } else {
        for (int j = offset + 1; j < offset + row_split_size_vec(i); ++j) {
          normed_row_split_flat(output_idx) = pre_size + row_split_vec(j);
          output_idx++;
        }
      }
      offset += row_split_size_vec(i);
      pre_size += row_split_vec(offset - 1);
    }
  }
};

REGISTER_OP("MonolithNormalizeMergedSplit")
    .Input("row_split: int64")
    .Input("row_split_size: int32")
    .Output("normed_row_split: int64")
    .SetShapeFn([](shape_inference::InferenceContext* ctx) {
      ctx->set_output(0, ctx->Vector(ctx->UnknownDim()));
      return Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name("MonolithNormalizeMergedSplit").Device(DEVICE_CPU),
                        NormalizeMergedSplitOp);

}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow
