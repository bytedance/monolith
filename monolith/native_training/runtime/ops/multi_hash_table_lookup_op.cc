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

}  // namespace monolith_tf
}  // namespace tensorflow
