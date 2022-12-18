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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace monolith_tf {

// The difference between this reduce sum op and tf.sparse.reduce_sum is that
// this supports sparse values which are vectors.
template <typename T>
class GenSeqMaskOp : public OpKernel {
 public:
  explicit GenSeqMaskOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_seq_length", &max_seq_length_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& splits = ctx->input(0);
    int64 batch_size = splits.dim_size(0) - 1;
    Tensor* mask = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, {batch_size, max_seq_length_}, &mask));
    std::memset(mask->data(), 0, mask->AllocatedBytes());

    auto splits_flat = splits.flat<T>();
    auto mask_mat = mask->matrix<T>();
    for (int64 i = 0; i < batch_size; ++i) {
      T size = splits_flat(i + 1) - splits_flat(i);
      size = size > max_seq_length_ ? max_seq_length_ : size;
      for (size_t j = 0; j < size; ++j) mask_mat(i, j) = 1;
    }
  }

 private:
  int max_seq_length_;
};

REGISTER_OP("GenSeqMask")
    .Input("splits: T")
    .Output("mask: T")
    .Attr("max_seq_length: int")
    .Attr("T: {int32, int64}")
    .SetShapeFn([](shape_inference::InferenceContext* ctx) {
      int max_seq_length;
      TF_RETURN_IF_ERROR(ctx->GetAttr("max_seq_length", &max_seq_length));
      if (ctx->FullyDefined(ctx->input(0))) {
        tensorflow::shape_inference::DimensionHandle batch_size;
        tensorflow::shape_inference::DimensionHandle input_dim =
            ctx->Dim(ctx->input(0), 0);
        TF_RETURN_IF_ERROR(
            ctx->Subtract(input_dim, ctx->MakeDim(1), &batch_size));
        ctx->set_output(
            0, ctx->MakeShape({batch_size, ctx->MakeDim(max_seq_length)}));
      } else {
        ctx->set_output(0, ctx->MakeShape({ctx->UnknownDim(),
                                           ctx->MakeDim(max_seq_length)}));
      }

      return Status::OK();
    });

REGISTER_KERNEL_BUILDER(
    Name("GenSeqMask").Device(DEVICE_CPU).TypeConstraint<int32>("T"),
    GenSeqMaskOp<int32>);

REGISTER_KERNEL_BUILDER(
    Name("GenSeqMask").Device(DEVICE_CPU).TypeConstraint<int64>("T"),
    GenSeqMaskOp<int64>);

}  // namespace monolith_tf
}  // namespace tensorflow
