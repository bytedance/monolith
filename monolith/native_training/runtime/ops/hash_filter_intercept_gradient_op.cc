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

#include "monolith/native_training/runtime/hash_filter/sliding_hash_filter.h"
#include "monolith/native_training/runtime/ops/hash_filter_tf_bridge.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace monolith_tf {

class HashFilterInterceptGradientOp : public OpKernel {
 public:
  explicit HashFilterInterceptGradientOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    ctx->set_output(0, ctx->input(2));
  }

  int threshold_;
};

class HashFilterInterceptGradientGradientOp : public OpKernel {
 public:
  explicit HashFilterInterceptGradientGradientOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    HashFilterTfBridge* filter = nullptr;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &filter));
    core::ScopedUnref unref(filter);
    const Tensor& ids = ctx->input(1);
    auto ids_vec = ids.vec<int64>();
    const Tensor& grad = ctx->input(2);
    TensorShape grad_shape(
        {ids.NumElements(), grad.NumElements() / ids.NumElements()});
    auto grad_mat = ctx->input(2).shaped<float, 2>(grad_shape.dim_sizes());
    Tensor* filtered_grad;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, grad_shape, &filtered_grad));
    auto filtered_grad_mat =
        filtered_grad->shaped<float, 2>(grad_shape.dim_sizes());
    for (int i = 0; i < ids_vec.dimension(0); ++i) {
      if (filter->ShouldBeFiltered(ids_vec(i))) {
        for (int j = 0; j < grad_shape.dim_size(1); ++j) {
          filtered_grad_mat(i, j) = 0;
        }
      } else {
        filtered_grad_mat.chip<0>(i) = grad_mat.chip<0>(i);
      }
    }
  }
};

REGISTER_OP("MonolithHashFilterInterceptGradient")
    .Input("filter_handle: resource")
    .Input("ids: int64")
    .Input("embeddings: float")
    .Output("same_embeddings: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(2));
      return Status::OK();
    });

REGISTER_KERNEL_BUILDER(
    Name("MonolithHashFilterInterceptGradient").Device(DEVICE_CPU),
    HashFilterInterceptGradientOp);

REGISTER_OP("MonolithHashFilterInterceptGradientGradient")
    .Input("filter_handle: resource")
    .Input("ids: int64")
    .Input("grad: float")
    .Output("filted_grad: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(2));
      return Status::OK();
    });

REGISTER_KERNEL_BUILDER(
    Name("MonolithHashFilterInterceptGradientGradient").Device(DEVICE_CPU),
    HashFilterInterceptGradientGradientOp);

}  // namespace monolith_tf
}  // namespace tensorflow