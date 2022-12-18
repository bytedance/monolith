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
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/util/work_sharder.h"

#include "monolith/native_training/runtime/hash_table/optimizer/avx_utils.h"

namespace tensorflow {
namespace monolith_tf {

// The difference between this reduce sum op and tf.sparse.reduce_sum is that
// this supports sparse values which are vectors.
class ReduceSumOp : public OpKernel {
 public:
  explicit ReduceSumOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& id_indices = ctx->input(0);
    auto id_indices_mat = id_indices.matrix<int64>();
    const Tensor& id_values = ctx->input(1);
    const int64 value_size = id_values.shape().dim_size(1);
    auto id_values_mat = id_values.matrix<float>();
    const Tensor& id_dense_shape = ctx->input(2);
    const int64 batch_size = id_dense_shape.flat<int64>()(0);
    Tensor* reduced;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, {batch_size, value_size}, &reduced));
    std::memset(reduced->data(), 0, reduced->AllocatedBytes());
    auto reduced_mat = reduced->matrix<float>();
    for (int64 i = 0; i < id_indices_mat.dimension(0); ++i) {
      int64 batch = id_indices_mat(i, 0);
      reduced_mat.chip<0>(batch) += id_values_mat.chip<0>(i);
    }
  }
};

// The difference between this reduce mean op and tf.sparse.reduce_mean is that
// this supports sparse values which are vectors.
class ReduceMeanOp : public OpKernel {
 public:
  explicit ReduceMeanOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& id_indices = ctx->input(0);
    auto id_indices_mat = id_indices.matrix<int64>();
    const Tensor& id_values = ctx->input(1);
    const int64 value_size = id_values.shape().dim_size(1);
    auto id_values_mat = id_values.matrix<float>();
    const Tensor& id_dense_shape = ctx->input(2);
    const int64 batch_size = id_dense_shape.flat<int64>()(0);

    Tensor* reduced;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, {batch_size, value_size}, &reduced));
    std::memset(reduced->data(), 0, reduced->AllocatedBytes());
    auto reduced_mat = reduced->matrix<float>();
    std::vector<size_t> counter(batch_size, 0);

    for (int64 i = 0; i < id_indices_mat.dimension(0); ++i) {
      int64 batch = id_indices_mat(i, 0);
      reduced_mat.chip<0>(batch) += id_values_mat.chip<0>(i);
      counter[batch] += 1;
    }
    for (int64 i = 0; i < batch_size; ++i) {
      float multiply = 1.0 / static_cast<float>(counter[i]);
      for (int64 j = 0; j < value_size; ++j) {
        reduced_mat(i, j) *= multiply;
      }
    }
  }
};

// The difference between this reduce square norm and tf.sparse.segment_sqrt_n
// is that this supports sparse values which are vectors.
class ReduceSquareNormOp : public OpKernel {
 public:
  explicit ReduceSquareNormOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& id_indices = ctx->input(0);
    auto id_indices_mat = id_indices.matrix<int64>();
    const Tensor& id_values = ctx->input(1);
    const int64 value_size = id_values.shape().dim_size(1);
    auto id_values_mat = id_values.matrix<float>();
    const Tensor& id_dense_shape = ctx->input(2);
    const int64 batch_size = id_dense_shape.flat<int64>()(0);

    Tensor* reduced;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, {batch_size, value_size}, &reduced));
    std::memset(reduced->data(), 0, reduced->AllocatedBytes());
    auto reduced_mat = reduced->matrix<float>();

    for (int64 i = 0; i < id_indices_mat.dimension(0); ++i) {
      int64 batch = id_indices_mat(i, 0);
      for (int64 j = 0; j < value_size; ++j) {
        reduced_mat(batch, j) += (id_values_mat(i, j) * id_values_mat(i, j));
      }
    }
    for (int64 i = 0; i < batch_size; ++i) {
      for (int64 j = 0; j < value_size; ++j) {
        reduced_mat(i, j) = std::sqrt(reduced_mat(i, j));
      }
    }
  }
};

class ReduceSumGradientOp : public OpKernel {
 public:
  explicit ReduceSumGradientOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& id_indices = ctx->input(0);
    auto id_indices_mat = id_indices.matrix<int64>();
    const int64 len_ids = id_indices_mat.dimension(0);
    const Tensor& grads = ctx->input(1);
    auto grads_mat = grads.matrix<float>();
    const int64 grad_size = grads_mat.dimension(1);
    Tensor* id_value_grads;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, {len_ids, grad_size}, &id_value_grads));
    auto id_value_grads_flat = id_value_grads->flat<float>();
    // Single thread is actually more efficient.
    for (int64 i = 0; i < len_ids; ++i) {
      int64 batch = id_indices_mat(i, 0);
      std::memcpy(
          static_cast<float*>(id_value_grads_flat.data()) + i * grad_size,
          const_cast<float*>(grads_mat.data()) + batch * grad_size,
          sizeof(float) * grad_size);
    }
  }
};

class ReduceMeanGradientOp : public OpKernel {
 public:
  explicit ReduceMeanGradientOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& id_indices = ctx->input(0);
    auto id_indices_mat = id_indices.matrix<int64>();
    const int64 len_ids = id_indices_mat.dimension(0);
    const Tensor& grads = ctx->input(1);
    auto grads_mat = grads.matrix<float>();
    const int64 grad_size = grads_mat.dimension(1);
    Tensor* id_value_grads;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, {len_ids, grad_size}, &id_value_grads));
    auto id_value_grads_mat = id_value_grads->matrix<float>();
    // grad_size equals to indices's batch size.
    std::vector<size_t> counter(grad_size, 0);
    for (int64 i = 0; i < len_ids; ++i) {
      int64 batch = id_indices_mat(i, 0);
      counter[batch] += 1;
    }

    for (int64 i = 0; i < len_ids; ++i) {
      int64 batch = id_indices_mat(i, 0);
      float multiply = 1.0 / static_cast<float>(counter[batch]);
      for (int64 j = 0; j < grad_size; ++j) {
        id_value_grads_mat(i, j) = grads_mat(batch, j) * multiply;
      }
    }
  }
};

class ReduceSquareNormGradientOp : public OpKernel {
 public:
  explicit ReduceSquareNormGradientOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& id_indices = ctx->input(0);
    auto id_indices_mat = id_indices.matrix<int64>();
    const int64 len_ids = id_indices_mat.dimension(0);
    const Tensor& id_values = ctx->input(1);
    auto id_values_mat = id_values.matrix<float>();
    const Tensor& grads = ctx->input(2);
    auto grads_mat = grads.matrix<float>();
    const int64 batch_size = grads_mat.dimension(0);
    const int64 grad_size = grads_mat.dimension(1);
    Tensor* id_value_grads;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, {len_ids, grad_size}, &id_value_grads));
    auto id_value_grads_mat = id_value_grads->matrix<float>();

    Tensor reduced_values(DT_FLOAT, TensorShape({batch_size, grad_size}));
    std::memset(reduced_values.data(), 0, reduced_values.AllocatedBytes());
    auto reduced_mat = reduced_values.matrix<float>();

    for (int64 i = 0; i < len_ids; ++i) {
      int64 batch = id_indices_mat(i, 0);
      for (int64 j = 0; j < grad_size; ++j) {
        reduced_mat(batch, j) += (id_values_mat(i, j) * id_values_mat(i, j));
      }
    }
    for (int64 i = 0; i < batch_size; ++i) {
      for (int64 j = 0; j < grad_size; ++j) {
        reduced_mat(i, j) = std::sqrt(reduced_mat(i, j));
      }
    }

    for (int64 i = 0; i < len_ids; ++i) {
      int64 batch = id_indices_mat(i, 0);
      for (int64 j = 0; j < grad_size; ++j) {
        // dl/dx = x/sqrt(sum(x)) * dl/dy
        float multiply = (reduced_mat(batch, j) == 0)
                             ? 0.0
                             : id_values_mat(i, j) / reduced_mat(batch, j);
        id_value_grads_mat(i, j) = grads_mat(batch, j) * multiply;
      }
    }
  }
};

// This is an extended op that supports reducesum and split in one fused op.
class ReduceSumAndSplitOp : public OpKernel {
 public:
  explicit ReduceSumAndSplitOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("M", &M_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("split_dims", &split_dims_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& id_indices = ctx->input(0);
    auto id_indices_mat = id_indices.matrix<int64>();
    const Tensor& id_values = ctx->input(1);
    const int64 value_size = id_values.shape().dim_size(1);
    auto id_values_mat = id_values.matrix<float>();
    const Tensor& id_dense_shape = ctx->input(2);
    const int64 batch_size = id_dense_shape.flat<int64>()(0);

    std::vector<Tensor*> reduced_list(M_);
    for (int i = 0; i < M_; ++i) {
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_output(i, {batch_size, split_dims_[i]},
                                          &reduced_list[i]));
      std::memset(reduced_list[i]->data(), 0,
                  reduced_list[i]->AllocatedBytes());
    }

    for (int64 i = 0; i < id_indices_mat.dimension(0); ++i) {
      int64 batch = id_indices_mat(i, 0);
      int emb_offset = 0;
      for (int j = 0; j < M_; ++j) {
        auto reduced_mat = reduced_list[j]->matrix<float>();
        int embedding_dim = split_dims_[j];
        float* input_a = const_cast<float*>(id_values_mat.data()) +
                         i * value_size + emb_offset;
        float* output_b =
            static_cast<float*>(reduced_mat.data()) + batch * split_dims_[j];
        ::monolith::hash_table::ReduceSum(input_a, output_b, output_b,
                                          split_dims_[j]);
        emb_offset += split_dims_[j];
      }
    }
  }

 private:
  int M_;
  std::vector<int> split_dims_;
};

class ReduceSumAndSplitGradientOp : public OpKernel {
 public:
  explicit ReduceSumAndSplitGradientOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("M", &M_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("split_dims", &split_dims_));
    grad_dim_ = 0;
    for (int i = 0; i < M_; i++) {
      grad_dim_ += split_dims_[i];
    }
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& id_indices = ctx->input(0);
    auto id_indices_mat = id_indices.matrix<int64>();
    const int64 len_ids = id_indices_mat.dimension(0);
    Tensor* id_value_grads;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, {len_ids, grad_dim_}, &id_value_grads));
    auto id_value_grads_flat = id_value_grads->flat<float>();
    int offset = 0;
    for (int i = 0; i < M_; ++i) {
      const Tensor& grads = ctx->input(i + 1);
      auto grads_mat = grads.matrix<float>();
      const int64 grad_size = grads_mat.dimension(1);
      CHECK(grad_size == split_dims_[i]);
      auto block_size = sizeof(float) * grad_size;
      for (int64 j = 0; j < len_ids; ++j) {
        int64 batch = id_indices_mat(j, 0);
        std::memcpy(static_cast<float*>(id_value_grads_flat.data()) +
                        j * grad_dim_ + offset,
                    const_cast<float*>(grads_mat.data()) + batch * grad_size,
                    block_size);
      }
      offset += grad_size;
    }
  }

 private:
  int M_;
  std::vector<int> split_dims_;
  int grad_dim_;
};

Status ReduceShape(shape_inference::InferenceContext* ctx) {
  shape_inference::ShapeHandle dense_handle;
  TF_RETURN_IF_ERROR(ctx->MakeShapeFromShapeTensor(2, &dense_handle));
  shape_inference::DimensionHandle dim0 = ctx->Dim(dense_handle, 0);
  shape_inference::DimensionHandle dim1 = ctx->Dim(ctx->input(1), 1);
  ctx->set_output(0, ctx->MakeShape({dim0, dim1}));
  return Status::OK();
}

Status GradientReduceShape(shape_inference::InferenceContext* ctx) {
  shape_inference::DimensionHandle len_id = ctx->Dim(ctx->input(0), 0);
  shape_inference::DimensionHandle grad_size = ctx->Dim(ctx->input(1), 1);
  ctx->set_output(0, ctx->MakeShape({len_id, grad_size}));
  return Status::OK();
}

Status FusedReduceShape(shape_inference::InferenceContext* ctx) {
  int M;
  TF_RETURN_IF_ERROR(ctx->GetAttr("M", &M));
  std::vector<int> split_dims;
  TF_RETURN_IF_ERROR(ctx->GetAttr("split_dims", &split_dims));
  CHECK_EQ(split_dims.size(), M);
  shape_inference::ShapeHandle dense_handle;
  TF_RETURN_IF_ERROR(ctx->MakeShapeFromShapeTensor(2, &dense_handle));
  for (int i = 0; i < M; i++) {
    shape_inference::DimensionHandle dim0 = ctx->Dim(dense_handle, 0);
    ctx->set_output(i, ctx->MakeShape({dim0, split_dims[i]}));
  }

  return Status::OK();
}

Status FusedGradientReduceShape(shape_inference::InferenceContext* ctx) {
  int M;
  TF_RETURN_IF_ERROR(ctx->GetAttr("M", &M));
  std::vector<int> split_dims;
  TF_RETURN_IF_ERROR(ctx->GetAttr("split_dims", &split_dims));
  CHECK_EQ(split_dims.size(), M);
  int grad_dim = 0;
  for (int i = 0; i < M; i++) {
    grad_dim += split_dims[i];
  }
  shape_inference::DimensionHandle len_id = ctx->Dim(ctx->input(0), 0);
  ctx->set_output(0, ctx->MakeShape({len_id, grad_dim}));
  return Status::OK();
}

REGISTER_OP("MonolithReduceSum")
    .Input("id_indices: int64")
    .Input("id_values: float")
    .Input("id_dense_shape: int64")
    .Output("reduced: float")
    .SetShapeFn(ReduceShape);

REGISTER_OP("MonolithReduceMean")
    .Input("id_indices: int64")
    .Input("id_values: float")
    .Input("id_dense_shape: int64")
    .Output("reduced: float")
    .SetShapeFn(ReduceShape);

REGISTER_OP("MonolithReduceSquareNorm")
    .Input("id_indices: int64")
    .Input("id_values: float")
    .Input("id_dense_shape: int64")
    .Output("reduced: float")
    .SetShapeFn(ReduceShape);

REGISTER_OP("MonolithReduceSumGradient")
    .Input("id_indices: int64")
    .Input("grads: float")
    .Output("id_values_grads: float")
    .SetShapeFn(GradientReduceShape);

REGISTER_OP("MonolithReduceMeanGradient")
    .Input("id_indices: int64")
    .Input("grads: float")
    .Output("id_values_grads: float")
    .SetShapeFn(GradientReduceShape);

REGISTER_OP("MonolithReduceSquareNormGradient")
    .Input("id_indices: int64")
    .Input("id_values: float")
    .Input("grads: float")
    .Output("id_values_grads: float")
    .SetShapeFn(GradientReduceShape);

REGISTER_OP("MonolithFusedReduceSumAndSplit")
    .Input("id_indices: int64")
    .Input("id_values: float")
    .Input("id_dense_shape: int64")
    .Output("reduced: M * float")
    .Attr("M: int")
    .Attr("split_dims: list(int)")
    .SetShapeFn(FusedReduceShape);

REGISTER_OP("MonolithFusedReduceSumAndSplitGradient")
    .Input("id_indices: int64")
    .Input("grads: M * float")
    .Output("output_grad: float")
    .Attr("M: int")
    .Attr("split_dims: list(int)")
    .SetShapeFn(FusedGradientReduceShape);

REGISTER_KERNEL_BUILDER(Name("MonolithReduceSum").Device(DEVICE_CPU),
                        ReduceSumOp);
REGISTER_KERNEL_BUILDER(Name("MonolithReduceMean").Device(DEVICE_CPU),
                        ReduceMeanOp);
REGISTER_KERNEL_BUILDER(Name("MonolithReduceSquareNorm").Device(DEVICE_CPU),
                        ReduceSquareNormOp);

REGISTER_KERNEL_BUILDER(Name("MonolithReduceSumGradient").Device(DEVICE_CPU),
                        ReduceSumGradientOp);
REGISTER_KERNEL_BUILDER(Name("MonolithReduceMeanGradient").Device(DEVICE_CPU),
                        ReduceMeanGradientOp);
REGISTER_KERNEL_BUILDER(
    Name("MonolithReduceSquareNormGradient").Device(DEVICE_CPU),
    ReduceSquareNormGradientOp);
REGISTER_KERNEL_BUILDER(
    Name("MonolithFusedReduceSumAndSplit").Device(DEVICE_CPU),
    ReduceSumAndSplitOp);
REGISTER_KERNEL_BUILDER(
    Name("MonolithFusedReduceSumAndSplitGradient").Device(DEVICE_CPU),
    ReduceSumAndSplitGradientOp);

}  // namespace monolith_tf
}  // namespace tensorflow