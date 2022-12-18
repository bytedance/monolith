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

#include "monolith/native_training/data/training_instance/cc/reader_util.h"
#include "monolith/native_training/runtime/hash_table/embedding_hash_table_interface.h"
#include "monolith/native_training/runtime/ops/embedding_hash_table_tf_bridge.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace monolith_tf {
namespace {

class HashTableSizeOp : public OpKernel {
 public:
  explicit HashTableSizeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    EmbeddingHashTableTfBridge* table = nullptr;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &table));
    Tensor* size_tensor;

    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {}, &size_tensor));
    size_tensor->scalar<int64>()() = table->Size();
  }
};

REGISTER_OP("MonolithHashTableSize")
    .Input("table_handle: resource")
    .Output("size: int64")
    .SetShapeFn(shape_inference::ScalarShape);
REGISTER_KERNEL_BUILDER(Name("MonolithHashTableSize").Device(DEVICE_CPU),
                        HashTableSizeOp);
class SaveAsTensorOp : public OpKernel {
 public:
  explicit SaveAsTensorOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* c) override {
    EmbeddingHashTableTfBridge* table = nullptr;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &table));
    auto shard = GetShard(c);
    auto iter = GetIter(c);
    std::vector<EmbeddingHashTableTfBridge::EntryDump> dumps;
    dumps.reserve(shard.limit);
    auto write_fn = [&dumps](EmbeddingHashTableTfBridge::EntryDump dump) {
      dumps.push_back(std::move(dump));
      return true;
    };
    OP_REQUIRES_OK(c, table->Save(c, shard, write_fn, &iter));

    Tensor* new_offset;
    OP_REQUIRES_OK(c, c->allocate_output(0, {}, &new_offset));
    new_offset->scalar<int64>()() = iter.offset;
    Tensor* out_dump;
    OP_REQUIRES_OK(c, c->allocate_output(1, {(int64)dumps.size()}, &out_dump));
    auto out_dump_vec = out_dump->vec<tstring>();
    for (int i = 0; i < dumps.size(); ++i) {
      out_dump_vec(i) = dumps[i].SerializeAsString();
    }
  }

 private:
  EmbeddingHashTableTfBridge::DumpShard GetShard(OpKernelContext* c) {
    EmbeddingHashTableTfBridge::DumpShard shard;
    shard.idx = c->input(1).scalar<int32>()();
    shard.total = c->input(2).scalar<int32>()();
    shard.limit = c->input(3).scalar<int64>()();
    return shard;
  }

  EmbeddingHashTableTfBridge::DumpIterator GetIter(OpKernelContext* c) {
    EmbeddingHashTableTfBridge::DumpIterator iter;
    iter.offset = c->input(4).scalar<int64>()();
    return iter;
  }

 private:
  int num_shard_;
  int shard_id_;
};

REGISTER_OP("MonolithHashTableSaveAsTensor")
    .Input("table_handle: resource")
    .Input("shard_idx: int32")
    .Input("num_shards: int32")
    .Input("limit: int64")
    .Input("offset: int64")
    .Output("new_offset: int64")
    .Output("entry: string")
    .SetShapeFn([](shape_inference::InferenceContext* ctx) {
      ctx->set_output(0, ctx->Scalar());
      ctx->set_output(1, ctx->Vector({ctx->UnknownDim()}));
      return Status::OK();
    });

REGISTER_KERNEL_BUILDER(
    Name("MonolithHashTableSaveAsTensor").Device(DEVICE_CPU), SaveAsTensorOp);

// In the future, probably we want to convert EntryDump to a set of tensors
// before we extract useful information.
class ExtractSlotFromEntryOp : public OpKernel {
 public:
  explicit ExtractSlotFromEntryOp(OpKernelConstruction* c) : OpKernel(c) {
    c->GetAttr("fid_v2", &fid_v2_);
  }

  void Compute(OpKernelContext* c) {
    const Tensor& dump = c->input(0);
    auto dump_vec = dump.vec<tstring>();
    const int len = dump.NumElements();
    Tensor* slot;
    OP_REQUIRES_OK(c, c->allocate_output(0, {len}, &slot));
    auto slot_vec = slot->vec<int32>();
    for (int i = 0; i < len; ++i) {
      const tstring& s = dump_vec(i);
      monolith::hash_table::EntryDump d;
      if (!d.ParseFromArray(s.data(), s.size())) {
        LOG_EVERY_N_SEC(WARNING, 10) << "Fail to parse entry dump.";
      }
      if (fid_v2_) {
        slot_vec(i) = slot_id_v2(d.id());
      } else {
        slot_vec(i) = slot_id_v1(d.id());
      }
    }
  }

 private:
  bool fid_v2_;
};

REGISTER_OP("MonolithExtractSlotFromEntry")
    .Input("entry: string")
    .Output("slot: int32")
    .Attr("fid_v2: bool")
    .SetShapeFn([](shape_inference::InferenceContext* ctx) {
      ctx->set_output(0, ctx->input(0));
      return Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name("MonolithExtractSlotFromEntry").Device(DEVICE_CPU),
                        ExtractSlotFromEntryOp);

}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow
