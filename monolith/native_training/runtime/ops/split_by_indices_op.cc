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
#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/threadpool.h"

namespace tensorflow {
namespace monolith_tf {
namespace {

// Given an int64 tensor, split it into multiple tensors based on the value.
template <class T>
class SplitByIndicesOp : public OpKernel {
 public:
  explicit SplitByIndicesOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_splits", &num_splits_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& indices = ctx->input(0);
    auto indices_vec = indices.vec<int64>();
    const int64 num_elements = indices.NumElements();
    const Tensor& input = ctx->input(1);
    const int64 element_size =
        num_elements == 0 ? 0 : input.NumElements() / num_elements;
    auto input_mat = input.shaped<T, 2>({num_elements, element_size});

    // Here we use a naive implementation (No parallel)
    std::vector<int64> split_sizes(num_splits_, 0);
    for (int i = 0; i < num_elements; ++i) {
      ++split_sizes[indices_vec(i)];
    }

    std::vector<Tensor*> splitted(num_splits_);
    for (int i = 0; i < num_splits_; ++i) {
      TensorShape output_shape;
      output_shape.AddDim(split_sizes[i]);
      const TensorShape& shape = input.shape();
      for (int j = 1; j < shape.dims(); ++j) {
        output_shape.AddDim(shape.dim_size(j));
      }
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, output_shape, &splitted[i]));
    }
    std::vector<typename TTypes<T>::Matrix> splitted_mat;
    splitted_mat.reserve(num_splits_);
    for (int i = 0; i < num_splits_; ++i) {
      splitted_mat.emplace_back(
          splitted[i]->shaped<T, 2>({split_sizes[i], element_size}));
    }
    for (int64 i = num_elements - 1; i >= 0; --i) {
      int64 index = indices_vec(i);
      splitted_mat[index].template chip<0>(--split_sizes[index]) =
          input_mat.template chip<0>(i);
    }
  }

 private:
  int num_splits_;
};

class SplitByIndicesGradientOp : public OpKernel {
 public:
  explicit SplitByIndicesGradientOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_splits", &num_splits_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& indices = ctx->input(0);
    auto indices_vec = indices.vec<int64>();
    const int64 num_elements = indices.NumElements();
    std::vector<int64> split_sizes(num_splits_, 0);
    for (int64 i = 0; i < num_elements; ++i) {
      ++split_sizes[indices_vec(i)];
    }
    const Tensor& input = ctx->input(1);
    const int64 element_size =
        num_elements == 0 ? 0 : input.NumElements() / num_elements;
    const int grads_offset = 2;
    std::vector<TTypes<const float>::Matrix> grads_mat;
    grads_mat.reserve(num_splits_);
    for (int i = 0; i < num_splits_; ++i) {
      const Tensor& grad = ctx->input(grads_offset + i);
      grads_mat.emplace_back(
          grad.shaped<float, 2>({split_sizes[i], element_size}));
    }
    Tensor* input_grads;
    TensorShape input_grads_shape = ctx->input(grads_offset).shape();
    input_grads_shape.set_dim(0, num_elements);
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, input_grads_shape, &input_grads));
    auto input_grads_mat =
        input_grads->shaped<float, 2>({num_elements, element_size});
    for (int64 i = num_elements - 1; i >= 0; --i) {
      int64 index = indices_vec(i);
      input_grads_mat.chip<0>(i) =
          grads_mat[index].chip<0>(--split_sizes[index]);
    }
  }

 private:
  int num_splits_;
};

template <class T>
class ReorderByIndicesOp : public OpKernel {
 public:
  explicit ReorderByIndicesOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_of_shards", &num_of_shards_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const Tensor& shard_ids = ctx->input(1);
    auto shard_id_vec = shard_ids.vec<int32>();
    const int64 num_elements = shard_ids.NumElements();
    // NOTE: element_size should always be 1 except for the test where we use
    // assign-add for initial assignment.
    const int64 element_size =
        num_elements == 0 ? 0 : input.NumElements() / num_elements;
    auto input_mat = input.shaped<T, 2>({num_elements, element_size});

    // We first count the number of unique FIDs, and also calculate the size for
    // each shard.
    typename absl::flat_hash_set<T> id_set;
    std::vector<int64> splits_offsets(num_of_shards_, 0);
    std::vector<std::vector<int64>> ids_for_splits(num_of_shards_,
                                                   std::vector<int64>{});
    int64 uniq_id_size = 0;
    for (int i = 0; i < num_elements; ++i) {
      // First insertion if never sees it before.
      if (id_set.insert(input_mat(i, 0)).second) {
        auto index = shard_id_vec(i);
        ids_for_splits[index].emplace_back(i);
        ++(splits_offsets[index]);
        ++uniq_id_size;
      }
    }

    // We allocate the buffer here.
    TensorShape output_shape;
    output_shape.AddDim(uniq_id_size);
    const TensorShape& shape = input.shape();
    for (int j = 1; j < shape.dims(); ++j) {
      output_shape.AddDim(shape.dim_size(j));
    }
    Tensor *outputs, *output_sizes;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &outputs));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape{num_of_shards_},
                                             &output_sizes));

    // We assign the split sizes here.
    auto output_shape_vec = output_sizes->vec<int32>();
    for (int i = 0; i < num_of_shards_; i++) {
      output_shape_vec(i) = splits_offsets[i];
      if (i > 0) splits_offsets[i] += splits_offsets[i - 1];
    }
    // We assign the reordered IDs here.
    typename TTypes<T>::Matrix splitted_mat =
        outputs->shaped<T, 2>({uniq_id_size, element_size});
    for (int i = num_of_shards_ - 1; i >= 0; --i) {
      for (const int64& j : ids_for_splits[i]) {
        splitted_mat.template chip<0>(--splits_offsets[i]) =
            input_mat.template chip<0>(j);
      }
    }
  }

 private:
  int num_of_shards_;
};

class RaggedSplitByIndicesOp : public OpKernel {
 public:
  explicit RaggedSplitByIndicesOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("num_splits", &num_splits_));
  }

  void Compute(OpKernelContext* c) override {
    const auto indices = c->input(0).vec<int64>();
    const auto num = c->input(1).vec<int64>();
    const auto num_split = c->input(2).vec<int64>();
    OP_REQUIRES(c, num.size() == indices.size(),
                errors::InvalidArgument(
                    "Ragged tensor values size must match indices size. Got ",
                    num.size(), " v.s. ", indices.size()));
    std::vector<int64> split_sizes(num_splits_, 0);
    for (int64 i = 0; i < indices.size(); ++i) {
      ++split_sizes[indices(i)];
    }
    std::vector<int64> splitted_offsets(num_splits_, 0);
    std::vector<TTypes<int64>::Vec> splitted_nums;
    std::vector<TTypes<int64>::Vec> splitted_num_splits;
    std::vector<TTypes<int64>::Vec> splitted_pos;
    for (int i = 0; i < num_splits_; ++i) {
      Tensor* t;
      OP_REQUIRES_OK(c, c->allocate_output(i, {split_sizes[i]}, &t));
      splitted_nums.push_back(t->vec<int64>());
      OP_REQUIRES_OK(
          c, c->allocate_output(i + num_splits_, {num_split.size()}, &t));
      auto splitted_num_split = t->vec<int64>();
      splitted_num_split(0) = 0;
      splitted_num_splits.push_back(splitted_num_split);
      OP_REQUIRES_OK(
          c, c->allocate_output(i + 2 * num_splits_, {split_sizes[i]}, &t));
      splitted_pos.push_back(t->vec<int64>());
    }
    int split_offset = 1;
    for (int64 i = 0;; ++i) {
      while (split_offset < num_split.size() && i == num_split(split_offset)) {
        for (int index = 0; index < num_splits_; ++index) {
          splitted_num_splits[index](split_offset) = splitted_offsets[index];
        }
        ++split_offset;
      }
      if (i >= indices.size()) break;
      int64 index = indices(i);
      splitted_pos[index](splitted_offsets[index]) = i;
      splitted_nums[index](splitted_offsets[index]) = num(i);
      ++splitted_offsets[index];
    }
    OP_REQUIRES(c, split_offset == num_split.size(),
                errors::InvalidArgument("The input ragged tensor is invalid."));
  }

 private:
  int num_splits_;
};

REGISTER_OP("MonolithRaggedSplitByIndices")
    .Input("indices: int64")
    .Input("num: int64")
    .Input("num_split: int64")
    .Output("splitted_nums: num_splits * int64")
    .Output("splitted_num_splits: num_splits * int64")
    .Output("splitted_pos: num_splits * int64")
    .Attr("num_splits: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int num_splits;
      TF_RETURN_IF_ERROR(c->GetAttr("num_splits", &num_splits));
      int offset = 0;
      for (int i = 0; i < num_splits; ++i) {
        c->set_output(i, c->Vector(c->UnknownDim()));
      }
      offset += num_splits;
      for (int i = 0; i < num_splits; ++i) {
        c->set_output(offset + i, c->input(2));
      }
      offset += num_splits;
      for (int i = 0; i < num_splits; ++i) {
        c->set_output(i, c->Vector(c->UnknownDim()));
      }
      return Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name("MonolithRaggedSplitByIndices").Device(DEVICE_CPU),
                        RaggedSplitByIndicesOp);

REGISTER_OP("MonolithSplitByIndices")
    .Input("indices: int64")
    .Input("input: T")
    .Output("splitted: num_splits * T")
    .Attr("num_splits: int")
    .Attr("T: type")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int num_splits;
      TF_RETURN_IF_ERROR(c->GetAttr("num_splits", &num_splits));
      shape_inference::ShapeHandle shape_handle = c->input(1);
      int rank = c->Rank(shape_handle);
      std::vector<shape_inference::DimensionHandle> dim_handles;
      dim_handles.push_back(c->UnknownDim());
      for (int i = 1; i < rank; ++i) {
        dim_handles.push_back(c->Dim(shape_handle, i));
      }
      for (int i = 0; i < num_splits; ++i) {
        c->set_output(i, c->MakeShape(dim_handles));
      }
      return Status::OK();
    });

REGISTER_OP("MonolithSplitByIndicesGradient")
    .Input("indices: int64")
    .Input("input: float")
    .Input("grads: num_splits * float")
    .Output("input_grads: float")
    .Attr("num_splits: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int num_splits;
      TF_RETURN_IF_ERROR(c->GetAttr("num_splits", &num_splits));
      shape_inference::DimensionHandle num_elements = c->Dim(c->input(0), 0);
      shape_inference::ShapeHandle shape_handle = c->input(1);
      int rank = c->Rank(shape_handle);
      std::vector<shape_inference::DimensionHandle> dim_handles;
      dim_handles.push_back(num_elements);
      for (int i = 1; i < rank; ++i) {
        dim_handles.push_back(c->Dim(shape_handle, i));
      }
      c->set_output(0, c->MakeShape(dim_handles));
      return Status::OK();
    });

REGISTER_OP("MonolithReorderByIndices")
    .Input("input: T")
    .Input("shard_ids: int32")
    .Output("reordered_tensor: T")
    .Output("shard_sizes: int32")
    .Attr("num_of_shards: int")
    .Attr("T: type")
    .SetDoNotOptimize()  // Crash with grappler.
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int num_of_shards;
      TF_RETURN_IF_ERROR(c->GetAttr("num_of_shards", &num_of_shards));
      shape_inference::ShapeHandle shape_handle = c->input(0);
      int rank = c->Rank(shape_handle);
      std::vector<shape_inference::DimensionHandle> dim_handles;
      dim_handles.push_back(c->UnknownDim());
      for (int i = 1; i < rank; ++i) {
        dim_handles.push_back(c->Dim(shape_handle, i));
      }
      // The first output is for the reshuffled first element.
      c->set_output(0, c->MakeShape(dim_handles));
      // The second output is for the all2all sizes.
      c->set_output(1, c->MakeShape({num_of_shards}));
      return Status::OK();
    });

#define REGISTER_KERNEL(type)                             \
  REGISTER_KERNEL_BUILDER(Name("MonolithSplitByIndices")  \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<type>("T"), \
                          SplitByIndicesOp<type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(int64);

#undef REGISTER_KERNEL

REGISTER_KERNEL_BUILDER(
    Name("MonolithSplitByIndicesGradient").Device(DEVICE_CPU),
    SplitByIndicesGradientOp);

#define REGISTER_KERNEL_REORDER_BY_INDICES(type)           \
  REGISTER_KERNEL_BUILDER(Name("MonolithReorderByIndices") \
                              .Device(DEVICE_CPU)          \
                              .TypeConstraint<type>("T"),  \
                          ReorderByIndicesOp<type>)

REGISTER_KERNEL_REORDER_BY_INDICES(float);
REGISTER_KERNEL_REORDER_BY_INDICES(int64);

#undef REGISTER_KERNEL_REORDER_BY_INDICES

}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow
