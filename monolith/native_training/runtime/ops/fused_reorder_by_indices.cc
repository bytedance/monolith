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

#include <chrono>
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
    OP_REQUIRES_OK(ctx, ctx->GetAttr("rank0_empty", &rank0_empty_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_of_shards", &num_shards_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("M", &num_tables_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("slot_embedding_dims", &slot_embedding_dims_));
  }

  void Compute(OpKernelContext* ctx) override {
    // auto start = std::chrono::steady_clock::now();
    std::vector<typename absl::flat_hash_map<T, int>> ids_sets(num_tables_);
    std::vector<std::vector<T>> ids_for_splits(num_tables_ * num_shards_);
    int total_fids = 0;
    for (int m = 0; m < num_tables_; ++m) {
      auto data = ctx->input(m).vec<T>().data();
      auto sz = ctx->input(m).NumElements();
      total_fids += sz;
      // Performance critical: reserve enough space so ids_sets won't rehash
      ids_sets[m].reserve(sz);
      for (int n = 0; n < num_shards_; n++)
        // reserve so ids_for_splits will **most likely** not reallocate
        ids_for_splits[n * num_tables_ + m].reserve((sz + sz / 4) /
                                                    num_shards_);

      int dim = slot_embedding_dims_[m];
      for (int i = 0; i < sz; ++i) {
        auto val = data[i];
        auto& vec = ids_for_splits[shard_func(val) * num_tables_ + m];
        if (ids_sets[m].insert({val, vec.size() * dim}).second)
          vec.push_back(val);
      }
    }

    Tensor *output, *shard_sizes, *sharded_slot_sizes;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, {num_shards_}, &shard_sizes));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, {num_shards_ * num_tables_},
                                            &sharded_slot_sizes));
    auto shard_sizes_vec = shard_sizes->vec<int32>();
    auto sharded_slot_sizes_vec = sharded_slot_sizes->vec<int32>();
    shard_sizes_vec.setZero();

    int uniq_id_size = 0;
    int emb_offset = 0;
    // compute a column major order emb_offsets for better cache
    std::vector<int> emb_offsets_cm(num_tables_ * num_shards_);
    for (int n = 0; n < num_shards_; n++) {
      for (int m = 0; m < num_tables_; ++m) {
        auto idx = n * num_tables_ + m;
        auto sz = ids_for_splits[idx].size();
        sharded_slot_sizes_vec(idx) = sz;
        shard_sizes_vec(n) += sz;
        uniq_id_size += sz;
        emb_offsets_cm[m * num_shards_ + n] = emb_offset;
        emb_offset += sz * slot_embedding_dims_[m];
      }
    }

    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {uniq_id_size}, &output));
    auto output_ptr = output->vec<T>().data();
    for (const auto& vec : ids_for_splits) {
      std::memcpy(output_ptr, vec.data(), sizeof(T) * vec.size());
      output_ptr += vec.size();
    }

    Tensor *emb_offset_sz, *fused_emb_offset;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(3, {num_tables_}, &emb_offset_sz));
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(4, {total_fids}, &fused_emb_offset));
    auto emb_offset_sz_vec = emb_offset_sz->vec<int32>().data();
    auto fused_emb_offset_vec = fused_emb_offset->vec<int32>().data();

    total_fids = 0;
    for (int m = 0; m < num_tables_; ++m) {
      auto sz = ctx->input(m).NumElements();
      auto data = ctx->input(m).vec<T>().data();
      emb_offset_sz_vec[m] = sz;
      for (int i = 0; i < sz; ++i) {
        auto val = data[i];
        fused_emb_offset_vec[total_fids + i] =
            ids_sets[m][val] +
            emb_offsets_cm[shard_func(val) + m * num_shards_];
      }
      total_fids += sz;
    }
    // std::cout << "fused reorder took "
    //           << (std::chrono::steady_clock::now() - start).count() * 1e-9
    //           << std::endl;
  }
  // TODO(hanzhizhou): consider precompute this or add specialization for
  // rank0_empty=false and num_shards=power of 2. Currently 10% of the total
  // time is spent on this function during the computation of this OP
  inline int shard_func(int64 val) {
    return val % (num_shards_ - rank0_empty_) + rank0_empty_;
  }

 private:
  bool rank0_empty_;
  int num_shards_;
  int num_tables_;
  std::vector<int32> slot_embedding_dims_;
};

REGISTER_OP("FusedReorderByIndices")
    .Input("input: M * T")
    .Output("output: T")
    .Output("shard_sizes: int32")
    .Output("sharded_slot_sizes: int32")
    .Output("emb_offset_sz: int32")
    .Output("fused_emb_offset: int32")
    .Attr("num_of_shards: int")
    .Attr("slot_embedding_dims: list(int)")
    .Attr("rank0_empty: bool")
    .Attr("M: int")
    .Attr("T: type")
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

      // The fourth output is an array of offsets to the fifth output
      c->set_output(3, c->Vector(M));
      c->set_output(4, c->Vector(c->UnknownDim()));
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
