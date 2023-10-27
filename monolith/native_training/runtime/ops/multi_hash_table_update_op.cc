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

#include "absl/types/span.h"
#include "monolith/native_training/runtime/concurrency/queue.h"
#include "monolith/native_training/runtime/ops/embedding_hash_table_tf_bridge.h"
#include "monolith/native_training/runtime/ops/hash_filter_tf_bridge.h"
#include "monolith/native_training/runtime/ops/multi_hash_table.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/work_sharder.h"
namespace tensorflow {
namespace monolith_tf {

using CPUDevice = Eigen::ThreadPoolDevice;

namespace {

using monolith::concurrency::Queue;

Status MismatchLength(absl::string_view tensor_name, int tensor_size,
                      int expected_size) {
  return errors::InvalidArgument("The length of tensor `", tensor_name,
                                 "` doesn't equal to table num. ", tensor_size,
                                 "v.s.", expected_size);
}

Status LengthTooShort(absl::string_view tensor_name, int tensor_size) {
  return errors::InvalidArgument("The length of tensor `", tensor_name,
                                 "` is too short. Currently value",
                                 tensor_size);
}

class MultiHashTableOptimizeOp : public OpKernel {
 public:
  explicit MultiHashTableOptimizeOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* c) override {
    core::RefCountPtr<MultiHashTable> mtable;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &mtable));
    auto id_vec = c->input(1).flat<int64>();
    auto id_split = c->input(2).flat<int64>();
    OP_REQUIRES(c, id_split.size() - 1 == mtable->size(),
                MismatchLength("id", id_split.size() - 1, mtable->size()));
    auto value_vec = c->input(3).flat<float>();
    auto learning_rate_vec = c->input(4).flat<float>();
    int64 update_time = c->input(5).scalar<int64>()();
    int64 global_step = c->input(6).scalar<int64>()();
    int n = mtable->size();
    int value_offset = 0;
    int learning_rate_offset = 0;
    for (int i = 0; i < n; ++i) {
      EmbeddingHashTableTfBridge* table = mtable->table(i);
      const int num_ids = id_split(i + 1) - id_split(i);
      const int value_size =
          (id_split(i + 1) - id_split(i)) * table->dim_size();
      OP_REQUIRES(c, value_offset + value_size <= value_vec.size(),
                  LengthTooShort("value", value_vec.size()));
      auto learning_rate = absl::MakeConstSpan(
          learning_rate_vec.data() + learning_rate_offset, table->slice_size());
      learning_rate_offset += table->slice_size();
      OP_REQUIRES(c, learning_rate_offset <= learning_rate_vec.size(),
                  LengthTooShort("learning_rate", learning_rate_vec.size()));

      OP_REQUIRES_OK(
          c, table->BatchOptimize(
                 c, num_ids,
                 reinterpret_cast<const int64_t*>(id_vec.data() + id_split(i)),
                 value_vec.data() + value_offset, learning_rate, update_time,
                 false, global_step));
      value_offset += value_size;
    }
    c->set_output(0, c->input(0));
  }
};

REGISTER_OP("MonolithMultiHashTableOptimize")
    .Input("mtable: resource")
    .Input("id: int64")
    .Input("id_split: int64")
    .Input("value: float")
    .Input("learning_rate: float")
    .Input("update_time: int64")
    .Input("global_step: int64")
    .Output("updated_table: resource")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(
    Name("MonolithMultiHashTableOptimize").Device(DEVICE_CPU),
    MultiHashTableOptimizeOp);

class MultiHashTableAssignOp : public OpKernel {
 public:
  explicit MultiHashTableAssignOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* c) override {
    core::RefCountPtr<MultiHashTable> mtable;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &mtable));
    auto id_vec = c->input(1).flat<int64>();
    auto id_split = c->input(2).flat<int64>();
    OP_REQUIRES(c, id_split.size() - 1 == mtable->size(),
                MismatchLength("id", id_split.size() - 1, mtable->size()));
    auto value_vec = c->input(3).flat<float>();
    int64 update_time = c->input(4).scalar<int64>()();
    int n = mtable->size();
    int value_offset = 0;
    for (int i = 0; i < n; ++i) {
      EmbeddingHashTableTfBridge* table = mtable->table(i);
      const int num_ids = id_split(i + 1) - id_split(i);
      const int value_size = num_ids * table->dim_size();
      OP_REQUIRES(c, value_offset + value_size <= value_vec.size(),
                  LengthTooShort("value", value_vec.size()));
      OP_REQUIRES_OK(
          c, table->Assign(
                 c, num_ids,
                 reinterpret_cast<const int64_t*>(id_vec.data() + id_split(i)),
                 value_vec.data() + value_offset, update_time));
      value_offset += value_size;
    }
    c->set_output(0, c->input(0));
  }
};

REGISTER_OP("MonolithMultiHashTableAssign")
    .Input("mtable: resource")
    .Input("id: int64")
    .Input("id_split: int64")
    .Input("value: float")
    .Input("update_time: int64")
    .Output("updated_table: resource")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(Name("MonolithMultiHashTableAssign").Device(DEVICE_CPU),
                        MultiHashTableAssignOp);

class MultiHashTableAssignAddOp : public OpKernel {
 public:
  explicit MultiHashTableAssignAddOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* c) override {
    core::RefCountPtr<MultiHashTable> mtable;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &mtable));
    auto id_vec = c->input(1).flat<int64>();
    auto id_split = c->input(2).flat<int64>();
    OP_REQUIRES(c, id_split.size() - 1 == mtable->size(),
                MismatchLength("id", id_split.size() - 1, mtable->size()));
    auto value_vec = c->input(3).flat<float>();
    int64 update_time = c->input(4).scalar<int64>()();
    int n = mtable->size();
    int value_offset = 0;
    for (int i = 0; i < n; ++i) {
      EmbeddingHashTableTfBridge* table = mtable->table(i);
      const int num_ids = id_split(i + 1) - id_split(i);
      const int value_size = num_ids * table->dim_size();
      OP_REQUIRES(c, value_offset + value_size <= value_vec.size(),
                  LengthTooShort("value", value_vec.size()));
      for (int j = id_split(i); j < id_split(i + 1); ++j) {
        auto value = absl::MakeConstSpan(value_vec.data() + value_offset,
                                         table->dim_size());
        OP_REQUIRES_OK(c, table->AssignAdd2(id_vec(j), value, update_time));
        value_offset += table->dim_size();
      }
    }
    c->set_output(0, c->input(0));
  }
};

REGISTER_OP("MonolithMultiHashTableAssignAdd")
    .Input("mtable: resource")
    .Input("id: int64")
    .Input("id_split: int64")
    .Input("value: float")
    .Input("update_time: int64")
    .Output("updated_table: resource")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(
    Name("MonolithMultiHashTableAssignAdd").Device(DEVICE_CPU),
    MultiHashTableAssignAddOp);

class MultiHashTableReinitializeOp : public OpKernel {
 public:
  explicit MultiHashTableReinitializeOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* c) override {
    core::RefCountPtr<MultiHashTable> mtable;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &mtable));
    auto table_name = c->input(1).scalar<tstring>()();
    auto id_vec = c->input(2).flat<int64>();
    Tensor* status_tensor;
    OP_REQUIRES_OK(c, c->allocate_output(1, {id_vec.size()}, &status_tensor));
    auto status_vec = status_tensor->vec<int32>();
    // -1: table_name does not exist, and the id will not be processed
    //  0: the id was inserted and is initialized
    //  1: the id was already in the table and is reinitialized
    status_vec.setConstant(-1);
    int* status = reinterpret_cast<int*>(status_tensor->data());
    std::vector<std::string> names = mtable->names();
    auto it = std::find_if(
        names.begin(), names.end(),
        [&table_name](const std::string& name) { return name == table_name; });
    if (it == names.end()) {
      LOG(ERROR) << "table " << table_name << " does not exist!";
    } else {
      int index = std::distance(names.begin(), it);
      EmbeddingHashTableTfBridge* table = mtable->table(index);
      OP_REQUIRES_OK(c, table->Reinitialize(
                            reinterpret_cast<const int64_t*>(id_vec.data()),
                            id_vec.size(), status));
    }
    c->set_output(0, c->input(0));
  }
};

REGISTER_OP("MonolithMultiHashTableReinitialize")
    .Input("mtable: resource")
    .Input("table_name: string")
    .Input("id: int64")
    .Output("updated_table: resource")
    .Output("id_status: int32")
    .SetShapeFn([](shape_inference::InferenceContext* ctx) {
      ctx->set_output(0, ctx->Scalar());
      ctx->set_output(1, ctx->input(2));
      return Status::OK();
    });

REGISTER_KERNEL_BUILDER(
    Name("MonolithMultiHashTableReinitialize").Device(DEVICE_CPU),
    MultiHashTableReinitializeOp);

template <typename Device>
class MultiHashTableFusedOptimizeOp : public OpKernel {
 public:
  explicit MultiHashTableFusedOptimizeOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_of_shards", &num_shards_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("enable_grad_accumulation",
                                     &enable_grad_accumulation_));
  }

  void ComputeH(OpKernelContext* ctx);
  void Compute(OpKernelContext* ctx) override {
    ComputeH(ctx);
    ctx->set_output(0, ctx->input(0));
  }

 private:
  bool enable_grad_accumulation_;
  int num_shards_;
};

template <>
void MultiHashTableFusedOptimizeOp<CPUDevice>::ComputeH(OpKernelContext* ctx) {
  auto ids = ctx->input(1).vec<int64_t>().data();
  auto num_ids = ctx->input(1).NumElements();
  auto indices = ctx->input(2).vec<int64_t>().data();
  auto slot_size_vec = ctx->input(3).vec<int32>().data();
  auto id_grads = ctx->input(4).vec<float>().data();
  auto num_grads = ctx->input(4).NumElements();
  auto key_offsets = ctx->input(5).vec<int32>().data();
  auto emb_offsets = ctx->input(6).vec<int32>().data();
  auto learning_rates = ctx->input(7).vec<float>().data();
  auto req_time = ctx->input(8).scalar<int64_t>()();
  auto global_step = ctx->input(9).scalar<int64_t>()();

  core::RefCountPtr<MultiHashTable> mtable;
  OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &mtable));

  int num_tables_ = mtable->size();
  auto optimize = [&](const int begin, const int end) {
    for (int shard_id = begin; shard_id < end; shard_id++) {
      int learning_rate_offset = 0;
      for (int table_id = 0; table_id < num_tables_; table_id++) {
        int curr_idx = shard_id * num_tables_ + table_id;
        auto table = mtable->table(table_id);
        auto learning_rate = absl::MakeConstSpan(
            learning_rates + learning_rate_offset, table->slice_size());
        learning_rate_offset += table->slice_size();
        table->BatchOptimize(ctx, slot_size_vec[curr_idx],
                             ids + key_offsets[curr_idx],
                             id_grads + emb_offsets[curr_idx], learning_rate,
                             req_time, enable_grad_accumulation_, global_step);
      }
    }
  };
  // TODO(zouxuan): tweak this number for optimization.
  const int64 kCostPerUnit = 10000000;
  const DeviceBase::CpuWorkerThreads& worker_threads =
      *ctx->device()->tensorflow_cpu_worker_threads();
  Shard(worker_threads.num_threads, worker_threads.workers, num_shards_,
        kCostPerUnit, optimize);
}


REGISTER_OP("MonolithMultiHashTableFusedOptimize")
    .Input("mtable: resource")
    .Input("ids: int64")
    .Input("indices: int64")
    .Input("fused_slot_size: int32")
    .Input("id_grads: float")
    .Input("id_offsets: int32")
    .Input("grad_offsets: int32")
    .Input("learning_rate_tensors: float")
    .Input("req_time: int64")
    .Input("global_step: int64")
    .Output("mtable_out: resource")
    .Attr("num_of_shards: int")
    .Attr("enable_grad_accumulation: bool = false")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(
    Name("MonolithMultiHashTableFusedOptimize").Device(DEVICE_CPU),
    MultiHashTableFusedOptimizeOp<CPUDevice>);

}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow
