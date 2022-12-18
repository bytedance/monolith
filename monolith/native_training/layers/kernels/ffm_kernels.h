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

#ifndef MONOLITH_MONOLITH_NATIVE_TRAINING_LAYERS_KERNELS_FFM_KERNELS_H_
#define MONOLITH_MONOLITH_NATIVE_TRAINING_LAYERS_KERNELS_FFM_KERNELS_H_

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace monolith_tf {

template <typename Device>
struct FFMImpl {
  static void Compute(OpKernelContext *ctx, const std::string &int_type,
                      TTypes<float>::ConstMatrix left_matrix, int left_feat_num,
                      TTypes<float>::ConstMatrix right_matrix,
                      int right_feat_num, int batch_size, int dim_size,
                      TTypes<float>::Matrix output);
};

template <typename Device>
struct FFMGradImpl {
  static void Compute(OpKernelContext *ctx, const std::string &int_type,
                      TTypes<float>::ConstMatrix grad_matrix, int grad_feat_num,
                      TTypes<float>::ConstMatrix left_matrix, int left_feat_num,
                      TTypes<float>::ConstMatrix right_matrix,
                      int right_feat_num, int batch_size, int dim_size,
                      TTypes<float>::Matrix left_grad_matrix,
                      TTypes<float>::Matrix right_grad_matrix);
};

template <typename Device>
class FFMOp : public OpKernel {
 public:
  explicit FFMOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dim_size", &dim_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("int_type", &int_type_));
  }

  void Compute(OpKernelContext *ctx) override {
    const Tensor *left_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("left", &left_tensor));
    OP_REQUIRES(
        ctx, left_tensor->dims() == 2,
        errors::InvalidArgument("the left input tensor of ffm is not 2D"));
    int64 batch_size = left_tensor->dim_size(0);
    int64 left_feat_num = left_tensor->dim_size(1) / dim_size_;
    auto left_matrix = left_tensor->matrix<float>();

    const Tensor *right_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("right", &right_tensor));
    OP_REQUIRES(
        ctx, left_tensor->dims() == 2,
        errors::InvalidArgument("the right input tensor of ffm is not 2D"));
    OP_REQUIRES(ctx, batch_size == right_tensor->dim_size(0),
                errors::InvalidArgument(
                    "the batch size of left and right tensor are not match"));
    int64 right_feat_num = right_tensor->dim_size(1) / dim_size_;
    auto right_matrix = right_tensor->matrix<float>();

    Tensor *output_tensor = nullptr;
    int out_last_dim = 0;
    if (int_type_ == "dot") {
      out_last_dim = left_feat_num * right_feat_num;
    } else {
      out_last_dim = left_feat_num * right_feat_num * dim_size_;
    }

    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {batch_size, out_last_dim},
                                             &output_tensor));
    auto output_matrix = output_tensor->matrix<float>();
    FFMImpl<Device>::Compute(ctx, int_type_, left_matrix, left_feat_num,
                             right_matrix, right_feat_num, batch_size,
                             dim_size_, output_matrix);
  }

 private:
  int dim_size_;
  std::string int_type_;
};

template <typename Device>
class FFMGradOp : public OpKernel {
 public:
  explicit FFMGradOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dim_size", &dim_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("int_type", &int_type_));
  }

  void Compute(OpKernelContext *ctx) override {
    const Tensor *grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("grad", &grad_tensor));
    OP_REQUIRES(ctx, grad_tensor->dims() == 2,
                errors::InvalidArgument("the grad tensor of ffm is not 2D"));
    int batch_size = grad_tensor->dim_size(0);
    int grad_feat_num = 0;
    if (int_type_ == "dot") {
      grad_feat_num = grad_tensor->dim_size(1);
    } else {
      grad_feat_num = grad_tensor->dim_size(1) / dim_size_;
    }

    auto grad_matrix = grad_tensor->matrix<float>();

    const Tensor *left_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("left", &left_tensor));
    OP_REQUIRES(
        ctx, left_tensor->dims() == 2,
        errors::InvalidArgument("the left input tensor of ffm is not 2D"));
    int64 left_feat_num = left_tensor->dim_size(1) / dim_size_;
    auto left_matrix = left_tensor->matrix<float>();

    const Tensor *right_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("right", &right_tensor));
    OP_REQUIRES(
        ctx, left_tensor->dims() == 2,
        errors::InvalidArgument("the right input tensor of ffm is not 2D"));
    int64 right_feat_num = right_tensor->dim_size(1) / dim_size_;
    auto right_matrix = right_tensor->matrix<float>();

    OP_REQUIRES(ctx, grad_feat_num == left_feat_num * right_feat_num,
                errors::InvalidArgument("the in/out shape not match"));

    Tensor *left_grad_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, left_tensor->shape(), &left_grad_tensor));
    auto left_grad_matrix = left_grad_tensor->matrix<float>();

    Tensor *right_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, right_tensor->shape(),
                                             &right_grad_tensor));
    auto right_grad_matrix = right_grad_tensor->matrix<float>();

    FFMGradImpl<Device>::Compute(ctx, int_type_, grad_matrix, grad_feat_num,
                                 left_matrix, left_feat_num, right_matrix,
                                 right_feat_num, batch_size, dim_size_,
                                 left_grad_matrix, right_grad_matrix);
  }

 private:
  int dim_size_;
  std::string int_type_;
};

}  // namespace monolith_tf
}  // namespace tensorflow

#endif  // MONOLITH_MONOLITH_NATIVE_TRAINING_LAYERS_KERNELS_FFM_KERNELS_H_
