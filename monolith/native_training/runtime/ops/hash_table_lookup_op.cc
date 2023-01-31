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

#include "absl/strings/str_format.h"
#include "monolith/native_training/runtime/hash_table/utils.h"
#include "monolith/native_training/runtime/ops/embedding_hash_table_tf_bridge.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/default/integral_types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace monolith_tf {

using CPUDevice = Eigen::ThreadPoolDevice;

class HashTableLookupOp : public OpKernel {
 public:
  explicit HashTableLookupOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dim_size", &dim_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_multi_threads", &use_multi_threads_));
  }

  void Compute(OpKernelContext* ctx) override {
    EmbeddingHashTableTfBridge* hash_table = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &hash_table));
    core::ScopedUnref unref(hash_table);
    const Tensor& ids = ctx->input(1);
    const int64 len_ids = ids.NumElements();
    OP_REQUIRES(
        ctx, dim_size_ == hash_table->dim_size(),
        errors::InvalidArgument(absl::StrFormat(
            "dim_size should match hash table size. %d vs %d. Node name: %s",
            dim_size_, hash_table->dim_size(), def().name())));
    Tensor* embeddings;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, {len_ids, dim_size_}, &embeddings));
    auto embeddings_mat = embeddings->matrix<float>();
    auto ids_flat = ids.flat<int64_t>();

    if (use_multi_threads_) {
      auto lookup = [&](const int64 begin, const int64 end) {
        int64_t hit_fid_count = 0;
        hash_table->BatchLookup(
            ctx, (end - begin), const_cast<int64_t*>(ids_flat.data() + begin),
            embeddings_mat.data() + begin * dim_size_, &hit_fid_count);
      };

      // TODO(zhangbiao.david, youlong.cheng): tweak this number for
      // optimization.
      const int64 kCostPerUnit = 8 * dim_size_;
      const DeviceBase::CpuWorkerThreads& worker_threads =
          *ctx->device()->tensorflow_cpu_worker_threads();
      Shard(worker_threads.num_threads, worker_threads.workers, len_ids,
            kCostPerUnit, lookup);
    } else {
      int64_t hit_fid_count = 0;
      hash_table->BatchLookup(ctx, len_ids,
                              const_cast<int64_t*>(ids_flat.data()),
                              embeddings_mat.data(), &hit_fid_count);
    }
  }

  int64 dim_size_;
  bool use_multi_threads_;
};

class HashTableLookupEntryOp : public OpKernel {
 public:
  explicit HashTableLookupEntryOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    EmbeddingHashTableTfBridge* hash_table = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &hash_table));
    core::ScopedUnref unref(hash_table);
    const Tensor& ids = ctx->input(1);
    const int64 len_ids = ids.NumElements();
    auto ids_flat = ids.flat<int64_t>();

    std::vector<EmbeddingHashTableTfBridge::EntryDump> entries(len_ids);
    if (entries.size() > 0) {
      hash_table->BatchLookupEntry(
          ctx, len_ids, const_cast<int64_t*>(ids_flat.data()), &entries[0]);
    }

    Tensor* entry_strs;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {len_ids}, &entry_strs));
    auto entry_str_vec = entry_strs->vec<tstring>();
    for (int i = 0; i < entries.size(); ++i) {
      entry_str_vec(i) = entries[i].SerializeAsString();
    }
  }
};

class HashTableLookupGradientOp : public OpKernel {
 public:
  explicit HashTableLookupGradientOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& id_indices = ctx->input(0);
    const Tensor& id_values = ctx->input(1);
    const Tensor& input_grads = ctx->input(2);

    OP_REQUIRES(
        ctx, id_indices.dim_size(0) == id_values.dim_size(0),
        errors::InvalidArgument(
            "id_indices's first dim and id_values dim should be same. Got ",
            id_indices.dim_size(0), "v.s. ", id_values.dim_size(0)));
    const int64 batch_size = id_indices.dim_size(0);
    const int64 embedding_dim = input_grads.dim_size(1);

    Tensor* output_ids;
    ctx->allocate_output(0, {batch_size}, &output_ids);
    Tensor* output_grads;
    ctx->allocate_output(1, {batch_size, embedding_dim}, &output_grads);

    auto id_indices_mat = id_indices.matrix<int64>();
    auto id_values_vec = id_values.vec<int64>();
    auto input_grads_mat = input_grads.matrix<float>();
    auto output_ids_vec = output_ids->vec<int64>();
    auto output_grads_mat = output_grads->matrix<float>();
    for (int64 i = 0; i < batch_size; ++i) {
      const int64 batch = id_indices_mat(i, 0);
      const int64 id = id_values_vec(i);
      output_ids_vec(i) = id;
      for (int64 j = 0; j < embedding_dim; ++j) {
        output_grads_mat(i, j) = input_grads_mat(batch, j);
      }
    }
  }
};

template <typename Device>
class HashTableFusedLookupOp : public OpKernel {
 public:
  explicit HashTableFusedLookupOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &num_tables_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_of_shards", &num_shards_));
  }

  void ComputeH(OpKernelContext* ctx);
  void Compute(OpKernelContext* ctx) override { ComputeH(ctx); }

 private:
  int num_shards_;
  int num_tables_;
};

template <>
void HashTableFusedLookupOp<CPUDevice>::ComputeH(OpKernelContext* ctx) {
  auto ids_flat = ctx->input(num_tables_ + 0).flat<int64_t>().data();
  auto slot_size_vec = ctx->input(num_tables_ + 1).vec<int>().data();
  auto slot_size_cnt = num_tables_ * num_shards_;

  Tensor *embeddings_ts, *emb_splits_ts, *key_offsets_ts, *emb_offsets_ts;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(1, {num_shards_}, &emb_splits_ts));
  OP_REQUIRES_OK(ctx,
                 ctx->allocate_output(2, {slot_size_cnt + 1}, &key_offsets_ts));
  OP_REQUIRES_OK(ctx,
                 ctx->allocate_output(3, {slot_size_cnt + 1}, &emb_offsets_ts));
  ctx->set_output(4, ctx->input(num_tables_ + 0));
  auto key_offsets = key_offsets_ts->vec<int>().data();
  auto emb_offsets = emb_offsets_ts->vec<int>().data();
  auto emb_splits = emb_splits_ts->vec<int>().data();

  std::vector<EmbeddingHashTableTfBridge*> hash_tables(num_tables_, nullptr);
  std::vector<int> hash_table_dims(num_tables_, 0);
  for (int table_id = 0; table_id < num_tables_; table_id++) {
    EmbeddingHashTableTfBridge* hash_table = nullptr;
    OP_REQUIRES_OK(
        ctx, LookupResource(ctx, HandleFromInput(ctx, table_id), &hash_table));
    core::ScopedUnref unref(hash_table);
    hash_tables[table_id] = hash_table;
    hash_table_dims[table_id] = hash_table->dim_size();
  }

  int total_keys, total_embs;
  std::tie(total_keys, total_embs) =
      monolith::hash_table::ComputeFusedOffsets<false>(
          slot_size_vec, hash_table_dims.data(), num_tables_, num_shards_,
          key_offsets, emb_offsets, nullptr, emb_splits);

  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {total_embs}, &embeddings_ts));
  auto embeddings = embeddings_ts->vec<float>().data();

  auto lookup = [&](const int begin, const int end) {
    for (int shard_id = begin; shard_id < end; shard_id++) {
      for (int table_id = 0; table_id < num_tables_; table_id++) {
        int curr_idx = shard_id * num_tables_ + table_id;
        int64_t hit_fid_count = 0;
        hash_tables[table_id]->BatchLookup(
            ctx, slot_size_vec[curr_idx],
            const_cast<int64_t*>(ids_flat) + key_offsets[curr_idx],
            embeddings + emb_offsets[curr_idx], &hit_fid_count);
      }
    }
  };

  // TODO(zouxuan): tweak this number for optimization.
  const int64 kCostPerUnit = 1000000;
  const DeviceBase::CpuWorkerThreads& worker_threads =
      *ctx->device()->tensorflow_cpu_worker_threads();
  Shard(worker_threads.num_threads, worker_threads.workers, num_shards_,
        kCostPerUnit, lookup);
}


REGISTER_OP("MonolithHashTableLookup")
    .Input("table_handle: resource")
    .Input("ids: int64")
    .Output("embeddings: float32")
    .Attr("dim_size: int")
    .Attr("use_multi_threads: bool = false")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int64 dim_size;
      TF_RETURN_IF_ERROR(c->GetAttr("dim_size", &dim_size));
      shape_inference::DimensionHandle len_ids = c->Dim(c->input(1), 0);
      c->set_output(0, c->Matrix(len_ids, dim_size));
      return Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name("MonolithHashTableLookup").Device(DEVICE_CPU),
                        HashTableLookupOp);

REGISTER_OP("MonolithHashTableLookupEntry")
    .Input("table_handle: resource")
    .Input("ids: int64")
    .Output("entry_str: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::DimensionHandle len_ids = c->Dim(c->input(1), 0);
      c->set_output(0, c->Vector(len_ids));
      return Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name("MonolithHashTableLookupEntry").Device(DEVICE_CPU),
                        HashTableLookupEntryOp);

REGISTER_OP("MonolithHashTableLookupGradient")
    .Input("id_indices: int64")
    .Input("id_values: int64")
    .Input("input_grads : float")
    .Output("ids: int64")
    .Output("output_grads: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::DimensionHandle batch_size = c->Dim(c->input(0), 0);
      shape_inference::DimensionHandle embedding_size = c->Dim(c->input(2), 1);
      c->set_output(0, c->MakeShape({batch_size}));
      c->set_output(1, c->MakeShape({batch_size, embedding_size}));
      return Status::OK();
    });

REGISTER_KERNEL_BUILDER(
    Name("MonolithHashTableLookupGradient").Device(DEVICE_CPU),
    HashTableLookupGradientOp);

REGISTER_OP("MonolithHashTableFusedLookup")
    .Input("table_handles: N * resource")
    .Input("ids: int64")
    .Input("fused_slot_size: int32")
    .Input("req_time: int64")
    .Output("embeddings: float32")
    .Output("embedding_splits: int32")
    .Output("id_offsets: int32")
    .Output("embedding_offsets: int32")
    .Output("indices: int64")
    .Attr("N: int")
    .Attr("num_of_shards: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int num_tables, num_shards;
      TF_RETURN_IF_ERROR(c->GetAttr("num_of_shards", &num_shards));
      TF_RETURN_IF_ERROR(c->GetAttr("N", &num_tables));
      c->set_output(0, c->Vector(c->UnknownDim()));
      c->set_output(1, c->Vector(num_shards));
      c->set_output(2, c->Vector(num_tables * num_shards + 1));
      c->set_output(3, c->Vector(num_tables * num_shards + 1));
      c->set_output(4, c->input(num_tables));
      auto shape = c->input(num_tables + 1);
      TF_RETURN_IF_ERROR(c->WithRank(shape, 1, &shape));
      auto dim = c->Dim(shape, 0);
      TF_RETURN_IF_ERROR(c->WithValue(dim, num_tables * num_shards, &dim));
      return Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name("MonolithHashTableFusedLookup").Device(DEVICE_CPU),
                        HashTableFusedLookupOp<CPUDevice>);

}  // namespace monolith_tf
}  // namespace tensorflow
