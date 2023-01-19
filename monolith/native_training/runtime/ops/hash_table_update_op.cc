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

#include "monolith/native_training/runtime/concurrency/queue.h"
#include "monolith/native_training/runtime/ops/embedding_hash_table_tf_bridge.h"
#include "monolith/native_training/runtime/ops/hash_filter_tf_bridge.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace monolith_tf {

using monolith::concurrency::Queue;
using CPUDevice = Eigen::ThreadPoolDevice;
class HashTableAssignOp : public OpKernel {
 public:
  explicit HashTableAssignOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    EmbeddingHashTableTfBridge* hash_table = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &hash_table));
    core::ScopedUnref unref(hash_table);
    const Tensor& id_values = ctx->input(1);
    const Tensor& id_updates = ctx->input(2);
    const Tensor& update_time_tensor = ctx->input(3);
    int64_t update_time = update_time_tensor.scalar<int64_t>()();
    auto id_values_vec = id_values.vec<int64>();
    const int num_updates = id_values_vec.dimension(0);
    OP_REQUIRES_OK(
        ctx, hash_table->Assign(
                 ctx, num_updates, static_cast<int64_t*>(id_values.data()),
                 static_cast<float*>(id_updates.data()), update_time));

    ctx->set_output(0, ctx->input(0));
  }
};

class HashTableAssignAddOp : public OpKernel {
 public:
  explicit HashTableAssignAddOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    EmbeddingHashTableTfBridge* hash_table = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &hash_table));
    core::ScopedUnref unref(hash_table);
    const Tensor& id_values = ctx->input(1);
    const Tensor& id_updates = ctx->input(2);
    const Tensor& update_time_tensor = ctx->input(3);
    int64_t update_time = update_time_tensor.scalar<int64_t>()();
    auto id_values_vec = id_values.vec<int64>();
    const int64 num_updates = id_values_vec.dimension(0);
    for (int64 i = 0; i < num_updates; ++i) {
      OP_REQUIRES_OK(
          ctx, hash_table->AssignAdd(ctx, id_values_vec(i),
                                     id_updates.SubSlice(i), update_time));
    }

    ctx->set_output(0, ctx->input(0));
  }
};

class HashTableOptimizeOp : public OpKernel {
 public:
  explicit HashTableOptimizeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_multi_threads", &use_multi_threads_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("enable_dedup", &enable_dedup_));

    int queue_size = 0;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("queue_size", &queue_size));
    CHECK_GE(queue_size, 0);

    queue_ =
        queue_size > 0
            ? std::make_unique<
                  Queue<std::tuple<Tensor, Tensor, absl::Span<const float>>>>(
                  queue_size)
            : nullptr;
  }

  void Compute(OpKernelContext* ctx) override {
    EmbeddingHashTableTfBridge* hash_table = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &hash_table));
    core::ScopedUnref unref(hash_table);
    const Tensor& id_values = ctx->input(1);
    const Tensor& id_updates = ctx->input(2);
    const Tensor& learning_rate_tensor = ctx->input(3);
    const Tensor& update_time_tensor = ctx->input(4);
    const Tensor& global_step = ctx->input(5);

    int64_t update_time = update_time_tensor.scalar<int64_t>()();
    size_t num_updates = id_values.NumElements();
    auto ids_flat = id_values.flat<int64_t>();
    auto* ids = const_cast<int64_t*>(ids_flat.data());
    absl::Span<const float> learning_rate_values =
        absl::MakeSpan(static_cast<float*>(learning_rate_tensor.data()),
                       learning_rate_tensor.NumElements());
    int64_t global_step_value = global_step.scalar<int64_t>()();
    if (use_multi_threads_) {
      auto dim_size = hash_table->dim_size();
      auto update = [&](const int64 begin, const int64 end) {
        OP_REQUIRES_OK(
            ctx, hash_table->BatchOptimize(
                     ctx, (end - begin), (ids + begin),
                     static_cast<float*>(id_updates.data()) + begin * dim_size,
                     learning_rate_values, update_time, enable_dedup_,
                     global_step_value));
      };

      // TODO(zhangbiao.david, youlong.cheng): tweak this number for
      // optimization.
      const int64 kCostPerUnit = 20 * dim_size;
      const DeviceBase::CpuWorkerThreads& worker_threads =
          *ctx->device()->tensorflow_cpu_worker_threads();
      Shard(worker_threads.num_threads, worker_threads.workers, num_updates,
            kCostPerUnit, update);
    } else {
      std::chrono::milliseconds timeout(1);

      // Optimize using this thread if operation timing out
      if (queue_ &&
          queue_->try_push({id_values, id_updates, learning_rate_values},
                           timeout)) {
        auto thread_pool =
            ctx->device()->tensorflow_cpu_worker_threads()->workers;
        thread_pool->Schedule(
            [this, ctx, update_time, hash_table, global_step_value]() {
              auto ids_and_grads = queue_->pop();
              const auto& id_values = std::get<0>(ids_and_grads);
              const auto& tensor = std::get<1>(ids_and_grads);
              auto& learning_rate_values = std::get<2>(ids_and_grads);
              size_t num_updates = id_values.NumElements();
              auto ids_flat = id_values.flat<int64_t>();
              hash_table->BatchOptimize(
                  ctx, num_updates, const_cast<int64_t*>(ids_flat.data()),
                  static_cast<float*>(tensor.data()), learning_rate_values,
                  update_time, enable_dedup_, global_step_value);
            });
      } else {
        OP_REQUIRES_OK(ctx, hash_table->BatchOptimize(
                                ctx, num_updates, ids,
                                static_cast<float*>(id_updates.data()),
                                learning_rate_values, update_time,
                                enable_dedup_, global_step_value));
      }
    }

    ctx->set_output(0, ctx->input(0));
  }

 private:
  bool use_multi_threads_;
  bool enable_dedup_;
  mutable std::unique_ptr<
      Queue<std::tuple<Tensor, Tensor, absl::Span<const float>>>>
      queue_;
};

template <typename Device>
class HashTableFusedOptimizeOp : public OpKernel {
 public:
  explicit HashTableFusedOptimizeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &num_tables_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_of_shards", &num_shards_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("enable_grad_accumulation",
                                     &enable_grad_accumulation_));
  }

  void ComputeH(OpKernelContext* ctx);
  void Compute(OpKernelContext* ctx) override {
    ComputeH(ctx);
    for (int table_id = 0; table_id < num_tables_; table_id++) {
      ctx->set_output(table_id, ctx->input(table_id));
    }
  }

 private:
  bool enable_grad_accumulation_;
  int num_tables_;
  int num_shards_;
};

template <>
void HashTableFusedOptimizeOp<CPUDevice>::ComputeH(OpKernelContext* ctx) {
  auto ids = ctx->input(num_tables_).vec<int64_t>().data();
  auto slot_size_vec = ctx->input(num_tables_ + 1).vec<int32>().data();
  auto id_grads = ctx->input(num_tables_ + 2).vec<float>().data();
  auto key_offsets = ctx->input(num_tables_ + 3).vec<int32>().data();
  auto emb_offsets = ctx->input(num_tables_ + 4).vec<int32>().data();
  auto learning_rates = ctx->input(num_tables_ + 5).vec<float>().data();
  auto update_time = ctx->input(num_tables_ + 6).scalar<int64_t>()();
  auto global_step = ctx->input(num_tables_ + 7).scalar<int64_t>()();

  std::vector<EmbeddingHashTableTfBridge*> hash_tables(num_tables_, nullptr);
  for (int table_id = 0; table_id < num_tables_; table_id++) {
    EmbeddingHashTableTfBridge* hash_table = nullptr;
    OP_REQUIRES_OK(
        ctx, LookupResource(ctx, HandleFromInput(ctx, table_id), &hash_table));
    core::ScopedUnref unref(hash_table);
    hash_tables[table_id] = hash_table;
  }
  auto optimize = [&](const int begin, const int end) {
    for (int shard_id = begin; shard_id < end; shard_id++) {
      int learning_rate_offset = 0;
      for (int table_id = 0; table_id < num_tables_; table_id++) {
        int curr_idx = shard_id * num_tables_ + table_id;
        auto table = hash_tables[table_id];
        auto learning_rate = absl::MakeConstSpan(
            learning_rates + learning_rate_offset, table->slice_size());
        learning_rate_offset += table->slice_size();
        table->BatchOptimize(
            ctx, slot_size_vec[curr_idx], ids + key_offsets[curr_idx],
            id_grads + emb_offsets[curr_idx], learning_rate, update_time,
            enable_grad_accumulation_, global_step);
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


REGISTER_OP("MonolithHashTableAssign")
    .Input("table_handle: resource")
    .Input("id_values: int64")
    .Input("id_updates: float")
    .Input("req_time: int64")
    .Output("table_handle_output: resource")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(Name("MonolithHashTableAssign").Device(DEVICE_CPU),
                        HashTableAssignOp);

REGISTER_OP("MonolithHashTableAssignAdd")
    .Input("table_handle: resource")
    .Input("id_values: int64")
    .Input("id_updates: float")
    .Input("req_time: int64")
    .Output("table_handle_output: resource")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(Name("MonolithHashTableAssignAdd").Device(DEVICE_CPU),
                        HashTableAssignAddOp);

REGISTER_OP("MonolithHashTableOptimize")
    .Input("table_handle: resource")
    .Input("id_values: int64")
    .Input("id_updates: float")
    .Input("learning_rate_tensor: float")
    .Input("req_time: int64")
    .Input("global_step: int64")
    .Output("table_handle_output: resource")
    .Attr("use_multi_threads: bool = false")
    .Attr("queue_size: int = 0")
    .Attr("enable_dedup: bool = false")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(Name("MonolithHashTableOptimize").Device(DEVICE_CPU),
                        HashTableOptimizeOp);

REGISTER_OP("MonolithHashTableFusedOptimize")
    .Input("table_handles: N * resource")
    .Input("ids: int64")
    .Input("fused_slot_size: int32")
    .Input("id_grads: float")
    .Input("id_offsets: int32")
    .Input("grad_offsets: int32")
    .Input("learning_rate_tensors: float")
    .Input("req_time: int64")
    .Input("global_step: int64")
    .Output("table_handles_output: N * resource")
    .Attr("N: int")
    .Attr("num_of_shards: int")
    .Attr("enable_grad_accumulation: bool = false")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int num_tables, num_shards;
      TF_RETURN_IF_ERROR(c->GetAttr("N", &num_tables));
      TF_RETURN_IF_ERROR(c->GetAttr("num_of_shards", &num_shards));
      for (int i = 0; i < num_tables; ++i) {
        c->set_output(i, c->Scalar());
      }
      auto shape = c->input(num_tables + 1);
      TF_RETURN_IF_ERROR(c->WithRank(shape, 1, &shape));
      auto dim = c->Dim(shape, 0);
      TF_RETURN_IF_ERROR(c->WithValue(dim, num_tables * num_shards, &dim));
      return Status::OK();
    });

REGISTER_KERNEL_BUILDER(
    Name("MonolithHashTableFusedOptimize").Device(DEVICE_CPU),
    HashTableFusedOptimizeOp<CPUDevice>);

}  // namespace monolith_tf
}  // namespace tensorflow
