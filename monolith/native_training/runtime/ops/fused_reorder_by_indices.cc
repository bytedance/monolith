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
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {
namespace monolith_tf {

template <class T>
class FusedReorderByIndicesOp : public OpKernel {
 public:
  explicit FusedReorderByIndicesOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_of_shards", &num_of_shards_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("slot_embedding_dims", &slot_embedding_dims_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("M", &M_));
  }

  void Compute(OpKernelContext* ctx) override {
    std::vector<typename TTypes<const T>::Vec> input_vecs;
    for (int m = 0; m < M_; ++m) {
      input_vecs.emplace_back(ctx->input(m).vec<T>());
    }

    Tensor *output, *shard_sizes, *sharded_slot_sizes;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape{num_of_shards_},
                                             &shard_sizes));
    auto shard_sizes_vec = shard_sizes->vec<int32>();
    for (int i = 0; i < num_of_shards_; ++i) shard_sizes_vec(i) = 0;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(2, TensorShape{num_of_shards_ * M_},
                                        &sharded_slot_sizes));
    auto sharded_slot_sizes_vec = sharded_slot_sizes->vec<int32>();
    for (int i = 0; i < num_of_shards_ * M_; ++i) sharded_slot_sizes_vec(i) = 0;

    std::vector<typename absl::flat_hash_map<T, int>> ids_sets(
        M_, absl::flat_hash_map<T, int>{});
    std::vector<std::vector<std::vector<T>>> ids_for_splits(
        num_of_shards_, std::vector<std::vector<T>>(M_, std::vector<T>{}));

    for (int m = 0; m < M_; ++m) {
      auto shard_indices_vec = ctx->input(M_ + m).vec<int32>();
      for (int i = 0; i < ctx->input(m).NumElements(); ++i) {
        auto val = input_vecs[m](i);
        // ids_sets to dedup per merged slot m in M_
        if (ids_sets[m].insert({val, 0}).second) {
          auto n = shard_indices_vec(i);
          ids_for_splits[n][m].emplace_back(val);
          sharded_slot_sizes_vec(n * M_ + m)++;
        }
      }
    }

    int64 uniq_id_size = 0;
    for (int m = 0; m < M_; ++m) uniq_id_size += ids_sets[m].size();

    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, TensorShape{uniq_id_size}, &output));
    auto output_vec = output->vec<T>();
    auto output_ptr = output_vec.data() + uniq_id_size;
    for (int n = num_of_shards_ - 1; n >= 0; --n) {
      for (int m = M_ - 1; m >= 0; --m) {
        auto v = ids_for_splits[n][m];
        auto s = sharded_slot_sizes_vec(n * M_ + m);
        output_ptr -= s;
        std::memcpy(output_ptr, v.data(), s * sizeof(T));
        shard_sizes_vec(n) += s;
      }
    }

    // Below we use absl::flat_hash_map ids_sets to track the fused embedding
    // offsets,
    //   and create output embedding_offset tensors as size of input fids
    //   tensors.
    int embedding_offset = 0;
    for (int n = 0; n < num_of_shards_; ++n) {
      for (int m = 0; m < M_; ++m) {
        int dim = slot_embedding_dims_[m];
        for (auto& val : ids_for_splits[n][m]) {
          ids_sets[m][val] = embedding_offset;
          embedding_offset += dim;
        }
      }
    }
    for (int m = 0; m < M_; ++m) {
      Tensor* output_embedding_offset;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(3 + m, ctx->input(m).shape(),
                                               &output_embedding_offset));
      auto output_embedding_offset_vec = output_embedding_offset->vec<int32>();
      for (int i = 0; i < ctx->input(m).NumElements(); ++i) {
        auto val = input_vecs[m](i);
        output_embedding_offset_vec(i) = ids_sets[m][val];
      }
    }
  }

 private:
  int num_of_shards_;
  std::vector<int32> slot_embedding_dims_;
  int M_;
};

REGISTER_OP("FusedReorderByIndices")
    .Input("input: M * T")
    .Input("shard_indices: M * int32")
    .Output("output: T")
    .Output("shard_sizes: int32")
    .Output("sharded_slot_sizes: int32")
    .Output("output_embedding_offset: M * int32")
    .Attr("num_of_shards: int")
    .Attr("slot_embedding_dims: list(int)")
    .Attr("M: int")
    .Attr("T: type")
    .SetDoNotOptimize()  // No grappler.
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle input;
      std::vector<shape_inference::DimensionHandle> dim_handles;
      dim_handles.push_back(c->UnknownDim());

      // The WithRank call validates that the input shape c->input(0)
      //  has a shape with exactly one dimension.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input));

      // The first output is the tensor in shape (?,)
      // It contains deduped reordered ids.
      c->set_output(0, c->MakeShape(dim_handles));

      // The second output is the tensor in shape (num_of_shards,)
      // It contains shard sizes.
      int num_of_shards;
      TF_RETURN_IF_ERROR(c->GetAttr("num_of_shards", &num_of_shards));
      c->set_output(1, c->MakeShape({num_of_shards}));

      // The third output is the tensor in shape (num_of_shards*M,)
      //  where M is the number of type T input.
      // It contains sharded (merged) slot sizes.
      int M;
      TF_RETURN_IF_ERROR(c->GetAttr("M", &M));
      c->set_output(2, c->MakeShape({num_of_shards * M}));

      // The fourth output is a list of tensors of the input shapes.
      for (int m = 0; m < M; ++m) {
        c->set_output(3 + m, c->input(m));
      }
      return Status::OK();
    });

#define REGISTER_KERNEL_FUSED_REORDER_BY_INDICES(type)    \
  REGISTER_KERNEL_BUILDER(Name("FusedReorderByIndices")   \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<type>("T"), \
                          FusedReorderByIndicesOp<type>)

REGISTER_KERNEL_FUSED_REORDER_BY_INDICES(int64);

#undef REGISTER_KERNEL_FUSED_REORDER_BY_INDICES

}  // namespace monolith_tf
}  // namespace tensorflow
