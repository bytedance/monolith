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

#include <vector>

#include "absl/strings/str_format.h"
#include "monolith/native_training/runtime/common/metrics.h"
#include "monolith/native_training/runtime/hash_table/utils.h"
#include "monolith/native_training/runtime/ops/embedding_hash_table_tf_bridge.h"
#include "monolith/native_training/runtime/ops/multi_hash_table.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/default/integral_types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace monolith_tf {

using CPUDevice = Eigen::ThreadPoolDevice;
class MultiHashTableLookupOp : public OpKernel {
 public:
  explicit MultiHashTableLookupOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    core::RefCountPtr<MultiHashTable> mtable;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &mtable));
    auto id_vec = ctx->input(1).flat<int64_t>();
    auto id_split_vec = ctx->input(2).flat<int64>();
    OP_REQUIRES(ctx, id_split_vec.size() == mtable->size() + 1,
                errors::InvalidArgument("id_split must be ", mtable->size() + 1,
                                        ". Current: ", id_split_vec.size()));
    int emb_size = 0;
    for (int i = 0; i < mtable->size(); ++i) {
      const int num_ids = id_split_vec(i + 1) - id_split_vec(i);
      emb_size += num_ids * mtable->table(i)->dim_size();
    }
    Tensor* emb_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {emb_size}, &emb_tensor));
    int64_t* id_ptr = const_cast<int64_t*>(id_vec.data());
    int emb_offset = 0;
    float* emb_data = reinterpret_cast<float*>(emb_tensor->data());
    int64_t total_hit_fid_count = 0, total_num_ids = 0;
    for (int i = 0; i < mtable->size(); ++i) {
      EmbeddingHashTableTfBridge* table = mtable->table(i);
      const int num_ids = id_split_vec(i + 1) - id_split_vec(i);
      total_num_ids += num_ids;
      int64_t hit_fid_count = 0;
      OP_REQUIRES_OK(ctx,
                     table->BatchLookup(ctx, num_ids, id_ptr + id_split_vec(i),
                                        emb_data + emb_offset, &hit_fid_count));
      total_hit_fid_count += hit_fid_count;
      emb_offset += num_ids * table->dim_size();
    }

    if (mtable->size() && mtable->table(0)->IsServingEntryType() &&
        total_num_ids) {
      const std::string tagkv = "name=all";
      float hit_rate = total_hit_fid_count / static_cast<float>(total_num_ids);
      monolith::GetMetrics()->emit_timer("lookup_fid_hit_rate", hit_rate,
                                         tagkv);
    }
  }
};

class MultiHashTableLookupEntryOp : public OpKernel {
 public:
  explicit MultiHashTableLookupEntryOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    core::RefCountPtr<MultiHashTable> mtable;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &mtable));
    auto id_vec = ctx->input(1).flat<int64_t>();
    auto id_split_vec = ctx->input(2).flat<int64>();
    OP_REQUIRES(ctx, id_split_vec.size() == mtable->size() + 1,
                errors::InvalidArgument("id_split must be ", mtable->size() + 1,
                                        ". Current: ", id_split_vec.size()));
    Tensor* entry_tensor;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, {id_vec.size()}, &entry_tensor));
    auto entry = entry_tensor->vec<tstring>();
    int64_t* id_ptr = const_cast<int64_t*>(id_vec.data());
    for (int i = 0; i < mtable->size(); ++i) {
      EmbeddingHashTableTfBridge* table = mtable->table(i);
      const int num_ids = id_split_vec(i + 1) - id_split_vec(i);
      std::vector<EmbeddingHashTableTfBridge::EntryDump> entries(num_ids);
      OP_REQUIRES_OK(
          ctx, table->BatchLookupEntry(ctx, num_ids, id_ptr + id_split_vec(i),
                                       entries.data()));
      for (int j = 0; j < num_ids; ++j) {
        entry(j + id_split_vec(i)) = entries[j].SerializeAsString();
      }
    }
  }

 private:
  int num_tables_;
  bool enable_inter_table_parallelism_;
  int64 cost_per_table_;
};

template <typename Device>
class MultiHashTableFusedLookupOp : public OpKernel {
 public:
  explicit MultiHashTableFusedLookupOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_of_shards", &num_shards_));
  }

  void ComputeH(OpKernelContext* ctx);
  void Compute(OpKernelContext* ctx) override { ComputeH(ctx); }

 private:
  int num_shards_;
};

template <>
void MultiHashTableFusedLookupOp<CPUDevice>::ComputeH(OpKernelContext* ctx) {
  auto ids_flat = ctx->input(1).flat<int64_t>().data();
  auto slot_size_vec = ctx->input(2).vec<int>().data();
  auto slot_size_cnt = ctx->input(2).NumElements();

  Tensor *embeddings_ts, *emb_splits_ts, *key_offsets_ts, *emb_offsets_ts;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(1, {num_shards_}, &emb_splits_ts));
  OP_REQUIRES_OK(ctx,
                 ctx->allocate_output(2, {slot_size_cnt + 1}, &key_offsets_ts));
  OP_REQUIRES_OK(ctx,
                 ctx->allocate_output(3, {slot_size_cnt + 1}, &emb_offsets_ts));
  ctx->set_output(4, ctx->input(1));
  auto key_offsets = key_offsets_ts->vec<int>().data();
  auto emb_offsets = emb_offsets_ts->vec<int>().data();
  auto emb_splits = emb_splits_ts->vec<int>().data();

  core::RefCountPtr<MultiHashTable> mtable;
  OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &mtable));

  int num_tables_ = mtable->size();
  std::vector<int> hash_table_dims(num_tables_, 0);
  for (int table_id = 0; table_id < num_tables_; table_id++) {
    hash_table_dims[table_id] = mtable->table(table_id)->dim_size();
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
        mtable->table(table_id)->BatchLookup(
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


REGISTER_OP("MonolithMultiHashTableLookup")
    .Input("mtable: resource")
    .Input("id: int64")
    .Input("id_split: int64")
    .Output("embedding: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Vector(c->UnknownDim()));

      return Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name("MonolithMultiHashTableLookup").Device(DEVICE_CPU),
                        MultiHashTableLookupOp);

REGISTER_OP("MonolithMultiHashTableLookupEntry")
    .Input("mtable: resource")
    .Input("id: int64")
    .Input("id_split: int64")
    .Output("serialized_entries: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));

      return Status::OK();
    });

REGISTER_KERNEL_BUILDER(
    Name("MonolithMultiHashTableLookupEntry").Device(DEVICE_CPU),
    MultiHashTableLookupEntryOp);

REGISTER_OP("MonolithMultiHashTableFusedLookup")
    .Input("mtable: resource")
    .Input("ids: int64")
    .Input("fused_slot_size: int32")
    .Input("req_time: int64")
    .Output("embeddings: float32")
    .Output("embedding_splits: int32")
    .Output("id_offsets: int32")
    .Output("embedding_offsets: int32")
    .Output("indices: int64")
    .Attr("num_of_shards: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int num_shards;
      TF_RETURN_IF_ERROR(c->GetAttr("num_of_shards", &num_shards));
      c->set_output(0, c->Vector(c->UnknownDim()));
      c->set_output(1, c->Vector(num_shards));

      auto shape = c->input(2);
      TF_RETURN_IF_ERROR(c->WithRank(shape, 1, &shape));
      auto dim = c->Dim(shape, 0);
      shape_inference::DimensionHandle out;
      TF_RETURN_IF_ERROR(c->Add(dim, 1, &out));
      c->set_output(2, c->Vector(out));
      c->set_output(3, c->Vector(out));
      c->set_output(4, c->input(1));
      return Status::OK();
    });

REGISTER_KERNEL_BUILDER(
    Name("MonolithMultiHashTableFusedLookup").Device(DEVICE_CPU),
    MultiHashTableFusedLookupOp<CPUDevice>);

}  // namespace monolith_tf
}  // namespace tensorflow
