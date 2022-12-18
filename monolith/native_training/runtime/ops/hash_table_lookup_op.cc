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
#include "monolith/native_training/runtime/ops/embedding_hash_table_tf_bridge.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/default/integral_types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace monolith_tf {

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

class HashTableFusedLookupOp : public OpKernel {
 public:
  explicit HashTableFusedLookupOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &num_tables_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_of_shards", &num_of_shards_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& ids = ctx->input(num_tables_ + 0);
    // [num_of_tables*num_of_shards_]
    const Tensor& fused_slot_size = ctx->input(num_tables_ + 1);
    const int64 slot_size_cnt = fused_slot_size.NumElements();
    const int64 num_of_tables = slot_size_cnt / num_of_shards_;
    OP_REQUIRES(
        ctx, num_of_tables == num_tables_,
        errors::InvalidArgument(
            "len(fused_slot_size) / num_of_shards != len(table_handles)"));
    const auto& fused_slot_size_vec = fused_slot_size.vec<int>();

    std::vector<EmbeddingHashTableTfBridge*> hash_tables(num_of_tables,
                                                         nullptr);
    std::vector<int> hash_table_dims(num_of_tables, 0);

    for (int table_id = 0; table_id < num_of_tables; table_id++) {
      EmbeddingHashTableTfBridge* hash_table = nullptr;
      OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, table_id),
                                         &hash_table));
      core::ScopedUnref unref(hash_table);
      hash_tables[table_id] = hash_table;
      hash_table_dims[table_id] = hash_table->dim_size();
    }

    Tensor *embeddings, *embedding_splits, *id_offsets, *embedding_offsets,
        *embedding_sizes;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(1, {num_of_shards_}, &embedding_splits));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, {slot_size_cnt}, &id_offsets));
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(3, {slot_size_cnt}, &embedding_offsets));
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(4, {slot_size_cnt}, &embedding_sizes));

    int output_dim = 0;
    int prev_output_dim = 0;
    // [num_of_tables*num_of_shards_] for offsets for different shards and
    // tables.
    std::vector<int> input_offsets(slot_size_cnt, 0);
    std::vector<int> output_offsets(slot_size_cnt, 0);
    auto segment_embedding_dims = embedding_sizes->flat<int32>().data();
    // [num_of_shards_] for splits
    // embedding_splits and embedding_offsets are used as a metadata for later
    // stages.
    auto output_splits = embedding_splits->flat<int32>().data();
    for (int shard_id = 0; shard_id < num_of_shards_; shard_id++) {
      for (int table_id = 0; table_id < num_of_tables; table_id++) {
        int curr_idx = num_of_tables * shard_id + table_id;
        int segment_dim =
            hash_table_dims[table_id] * fused_slot_size_vec(curr_idx);
        output_dim += segment_dim;
        segment_embedding_dims[curr_idx] = segment_dim;
        if (curr_idx > 0) {
          output_offsets[curr_idx] = output_offsets[curr_idx - 1] +
                                     segment_embedding_dims[curr_idx - 1];
          input_offsets[curr_idx] =
              input_offsets[curr_idx - 1] + fused_slot_size_vec(curr_idx - 1);
        }
      }
      output_splits[shard_id] = output_dim - prev_output_dim;
      prev_output_dim = output_dim;
    }
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {output_dim}, &embeddings));
    auto embeddings_vec = embeddings->vec<float>();
    auto ids_flat = ids.flat<int64_t>();

    std::memcpy(id_offsets->flat<int32>().data(), input_offsets.data(),
                sizeof(int32) * slot_size_cnt);
    std::memcpy(embedding_offsets->flat<int32>().data(), output_offsets.data(),
                sizeof(int32) * slot_size_cnt);
    auto lookup = [&](const int begin, const int end) {
      for (int shard_id = begin; shard_id < end; shard_id++) {
        for (int table_id = 0; table_id < num_of_tables; table_id++) {
          int curr_idx = shard_id * num_of_tables + table_id;
          int index_begin = input_offsets[curr_idx];
          int embedding_offset = output_offsets[curr_idx];
          int64_t hit_fid_count = 0;
          hash_tables[table_id]->BatchLookup(
              ctx, fused_slot_size_vec(curr_idx),
              const_cast<int64_t*>(ids_flat.data()) + index_begin,
              static_cast<float*>(embeddings_vec.data()) + embedding_offset,
              &hit_fid_count);
        }
      }
    };

    // TODO(zouxuan): tweak this number for optimization.
    const int64 kCostPerUnit = 1000000;
    const DeviceBase::CpuWorkerThreads& worker_threads =
        *ctx->device()->tensorflow_cpu_worker_threads();
    Shard(worker_threads.num_threads, worker_threads.workers, num_of_shards_,
          kCostPerUnit, lookup);
  }

  int32 num_of_shards_;
  int32 num_tables_;
};

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
    .Output("embeddings: float32")
    .Output("embedding_splits: int32")
    .Output("id_offsets: int32")
    .Output("embedding_offsets: int32")
    .Output("embedding_sizes: int32")
    .Attr("N: int")
    .Attr("num_of_shards: int")
    .SetDoNotOptimize()  // Crash with grappler.
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int64 num_of_shards;
      int num_tables_;
      TF_RETURN_IF_ERROR(c->GetAttr("num_of_shards", &num_of_shards));
      TF_RETURN_IF_ERROR(c->GetAttr("N", &num_tables_));
      std::vector<shape_inference::DimensionHandle> dim_handles;
      dim_handles.push_back(c->UnknownDim());
      c->set_output(0, c->MakeShape(dim_handles));
      c->set_output(1, c->Vector(num_of_shards));
      c->set_output(2, c->input(num_tables_ + 1));
      c->set_output(3, c->input(num_tables_ + 1));
      c->set_output(4, c->input(num_tables_ + 1));
      return Status::OK();
    });
REGISTER_KERNEL_BUILDER(Name("MonolithHashTableFusedLookup").Device(DEVICE_CPU),
                        HashTableFusedLookupOp);

}  // namespace monolith_tf
}  // namespace tensorflow
