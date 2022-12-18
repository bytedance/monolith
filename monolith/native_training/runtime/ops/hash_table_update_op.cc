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

class HashTableFusedOptimizeOp : public OpKernel {
 public:
  explicit HashTableFusedOptimizeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &num_tables_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_of_shards", &num_of_shards_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("enable_grad_accumulation",
                                     &enable_grad_accumulation_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& ids = ctx->input(num_tables_);
    const Tensor& fused_slot_size = ctx->input(num_tables_ + 1);
    const int64 slot_size_cnt = fused_slot_size.NumElements();
    const int64 num_of_tables = slot_size_cnt / num_of_shards_;
    OP_REQUIRES(
        ctx, num_of_tables == num_tables_,
        errors::InvalidArgument(
            "len(fused_slot_size) / num_of_shards != len(table_handles)"));
    const auto& fused_slot_size_vec = fused_slot_size.vec<int32>();
    const Tensor& id_grads = ctx->input(num_tables_ + 2);
    const auto& id_offsets_flat = ctx->input(num_tables_ + 3).vec<int32>();
    const auto& grad_offsets_flat = ctx->input(num_tables_ + 4).vec<int32>();
    const Tensor& learning_rate_tensors = ctx->input(num_tables_ + 5);
    absl::Span<const float> learning_rate_values =
        absl::MakeSpan(static_cast<float*>(learning_rate_tensors.data()),
                       learning_rate_tensors.NumElements());
    const auto& learning_rate_lengths_flat =
        ctx->input(num_tables_ + 6).vec<int32>();
    int64_t update_time = ctx->input(num_tables_ + 7).scalar<int64_t>()();
    int64_t global_step_value = ctx->input(num_tables_ + 8).scalar<int64_t>()();

    std::vector<EmbeddingHashTableTfBridge*> hash_tables(num_of_tables,
                                                         nullptr);
    std::vector<int> hash_table_dims(num_of_tables, 0);
    std::vector<int> learning_rate_offsets_flat(num_of_tables, 0);

    for (int table_id = 0; table_id < num_of_tables; table_id++) {
      EmbeddingHashTableTfBridge* hash_table = nullptr;
      OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, table_id),
                                         &hash_table));
      core::ScopedUnref unref(hash_table);
      hash_tables[table_id] = hash_table;
      hash_table_dims[table_id] = hash_table->dim_size();
      if (table_id > 0) {
        learning_rate_offsets_flat[table_id] =
            learning_rate_offsets_flat[table_id - 1] +
            learning_rate_lengths_flat(table_id - 1);
      }
    }
    auto optimize = [&](const int begin, const int end) {
      for (int shard_id = begin; shard_id < end; shard_id++) {
        for (int table_id = 0; table_id < num_of_tables; table_id++) {
          int curr_idx = shard_id * num_of_tables + table_id;
          int index_offset = id_offsets_flat(curr_idx);
          int gradient_offset = grad_offsets_flat(curr_idx);
          int learning_rate_offset = learning_rate_offsets_flat[table_id];
          int learning_rate_length = learning_rate_lengths_flat(table_id);
          hash_tables[table_id]->BatchOptimize(
              ctx, fused_slot_size_vec(curr_idx),
              static_cast<int64_t*>(ids.data()) + index_offset,
              static_cast<float*>(id_grads.data()) + gradient_offset,
              learning_rate_values.subspan(learning_rate_offset,
                                           learning_rate_length),
              update_time, enable_grad_accumulation_, global_step_value);
        }
      }
    };
    // TODO(zouxuan): tweak this number for optimization.
    const int64 kCostPerUnit = 10000000;
    const DeviceBase::CpuWorkerThreads& worker_threads =
        *ctx->device()->tensorflow_cpu_worker_threads();
    Shard(worker_threads.num_threads, worker_threads.workers, num_of_shards_,
          kCostPerUnit, optimize);
    for (int table_id = 0; table_id < num_tables_; table_id++) {
      ctx->set_output(table_id, ctx->input(table_id));
    }
  }

 private:
  bool enable_grad_accumulation_;
  int32 num_tables_;
  int32 num_of_shards_;
};

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
    .Input("learning_rate_lengths: int32")
    .Input("req_time: int64")
    .Input("global_step: int64")
    .Output("table_handles_output: N * resource")
    .Attr("N: int")
    .Attr("num_of_shards: int")
    .Attr("enable_grad_accumulation: bool = false")
    .SetDoNotOptimize()  // Crash with grappler.
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int N = 0;
      TF_RETURN_IF_ERROR(c->GetAttr("N", &N));
      for (int i = 0; i < N; ++i) {
        c->set_output(i, c->Scalar());
      }
      return Status::OK();
    });
REGISTER_KERNEL_BUILDER(
    Name("MonolithHashTableFusedOptimize").Device(DEVICE_CPU),
    HashTableFusedOptimizeOp);

}  // namespace monolith_tf
}  // namespace tensorflow
