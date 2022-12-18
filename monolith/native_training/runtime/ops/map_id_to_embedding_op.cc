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

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/util/work_sharder.h"

#include "monolith/native_training/runtime/hash_table/optimizer/avx_utils.h"

namespace tensorflow {
namespace monolith_tf {
namespace {

// The input embeddings are a list of 2D tensors.
// This represents the embedding: embeddings[index].chip<0>(pos)
struct EmbeddingLocation {
  int64 index;
  int64 pos;
};

}  // namespace

// Maps input ids into embeddings.
class MapIdToEmbeddingOp : public OpKernel {
 public:
  explicit MapIdToEmbeddingOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_splits", &num_splits_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_multi_threads", &use_multi_threads_));
  }

  void Compute(OpKernelContext* ctx) override {
    absl::flat_hash_map<int64, EmbeddingLocation> id_to_loc;
    int64 total_split_ids = 0;
    for (int i = 0; i < num_splits_; ++i) {
      total_split_ids += ctx->input(i).flat<int64>().dimension(0);
    }
    id_to_loc.reserve(total_split_ids);

    for (int i = 0; i < num_splits_; ++i) {
      auto ids = ctx->input(i).flat<int64>();
      for (int64 j = 0; j < ids.dimension(0); ++j) {
        id_to_loc.insert({ids(j), {i, j}});
      }
    }

    std::vector<TTypes<const float>::Matrix> embeddings;
    embeddings.reserve(num_splits_);
    for (int i = 0; i < num_splits_; ++i) {
      embeddings.emplace_back(ctx->input(num_splits_ + i).matrix<float>());
    }
    int64 embedding_dim = embeddings[0].dimension(1);

    const Tensor& input = ctx->input(2 * num_splits_);
    auto input_flat = input.flat<int64>();
    Tensor* output;
    TensorShape output_shape = input.shape();
    output_shape.AddDim(embedding_dim);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    auto output_mat =
        output->shaped<float, 2>({input.NumElements(), embedding_dim});
    auto map_fn = [&](const int64 begin, const int64 end) {
      for (int64 k = begin; k < end; ++k) {
        auto iter = id_to_loc.find(input_flat(k));
        if (iter == id_to_loc.end()) {
          return ctx->SetStatus(
              errors::InvalidArgument("Unable to map id ", input_flat(k)));
        }
        const EmbeddingLocation& loc = iter->second;
        output_mat.chip<0>(k) = embeddings[loc.index].chip<0>(loc.pos);
      }
    };

    int64 total = input_flat.dimension(0);
    if (use_multi_threads_) {
      auto worker_threads = ctx->device()->tensorflow_cpu_worker_threads();
      auto workers = worker_threads->workers;
      int num_shards = std::min(5LL, std::max(1LL, total / 10000));
      int64 block_size = total / num_shards;
      workers->ParallelFor(
          total, thread::ThreadPool::SchedulingParams(
                     thread::ThreadPool::SchedulingStrategy::kFixedBlockSize,
                     absl::nullopt, block_size),
          map_fn);
    } else {
      map_fn(0, total);
    }
  }

 private:
  int num_splits_;
  bool use_multi_threads_;
};

class MapIdToEmbeddingGradientOp : public OpKernel {
 public:
  explicit MapIdToEmbeddingGradientOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_splits", &num_splits_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(num_splits_);
    const Tensor& grads = ctx->input(num_splits_ + 1);
    const int64 embedding_size = grads.dim_size(grads.dims() - 1);
    absl::flat_hash_map<int64, EmbeddingLocation> id_to_loc;
    std::vector<TTypes<float>::Matrix> embedding_grads_mats;
    embedding_grads_mats.reserve(num_splits_);
    for (int i = 0; i < num_splits_; ++i) {
      auto ids = ctx->input(i).flat<int64>();
      int64 len_ids = ids.dimension(0);
      for (int64 j = 0; j < ids.dimension(0); ++j) {
        id_to_loc.insert({ids(j), {i, j}});
      }
      Tensor* output;
      OP_REQUIRES_OK(
          ctx, ctx->allocate_output(i, {len_ids, embedding_size}, &output));
      std::memset(output->data(), 0, output->AllocatedBytes());
      embedding_grads_mats.emplace_back(output->matrix<float>());
    }
    auto input_flat = input.flat<int64>();
    auto grads_mat =
        grads.shaped<float, 2>({input.NumElements(), embedding_size});
    for (int64 k = 0; k < input_flat.dimension(0); ++k) {
      const EmbeddingLocation& loc = id_to_loc.find(input_flat(k))->second;
      embedding_grads_mats[loc.index].chip<0>(loc.pos) += grads_mat.chip<0>(k);
    }
  }

 private:
  int num_splits_;
};

// Maps input ids into embeddings with only 1 tensor.
class GatherEmbeddingsByInputOp : public OpKernel {
 public:
  explicit GatherEmbeddingsByInputOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_multi_threads", &use_multi_threads_));
  }

  void Compute(OpKernelContext* ctx) override {
    absl::flat_hash_map<int64, int64> id_to_loc;
    auto ids = ctx->input(0).flat<int64>();
    for (int i = 0; i < ids.dimension(0); ++i) {
      id_to_loc.insert({ids(i), i});
    }

    TTypes<const float>::Matrix embeddings = ctx->input(1).matrix<float>();
    OP_REQUIRES(ctx, embeddings.dimension(0) == ids.dimension(0),
                errors::InvalidArgument("See unmatched embedding dim ",
                                        embeddings.dimension(0), " and id dim ",
                                        ids.dimension(0)));
    int64 embedding_dim = embeddings.dimension(1);

    const Tensor& input = ctx->input(2);
    auto input_flat = input.flat<int64>();
    Tensor *output, *output_index_mapping;
    TensorShape output_shape = input.shape();
    output_shape.AddDim(embedding_dim);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(1, input.shape(), &output_index_mapping));
    auto output_mat =
        output->shaped<float, 2>({input.NumElements(), embedding_dim});
    auto output_index_mapping_flat = output_index_mapping->flat<int64>();

    auto fill_fn = [&](const int64 begin, const int64 end) {
      for (int64 k = begin; k < end; ++k) {
        auto iter = id_to_loc.find(input_flat(k));
        if (iter == id_to_loc.end()) {
          return ctx->SetStatus(
              errors::InvalidArgument("Unable to map id ", input_flat(k)));
        }
        const int64& pos = iter->second;
        output_mat.chip<0>(k) = embeddings.chip<0>(pos);
        output_index_mapping_flat(k) = pos;
      }
    };
    if (use_multi_threads_) {
      // TODO(zouxuan): tune this for performance.
      const int64 kCostPerUnit = 4 * embedding_dim;
      const DeviceBase::CpuWorkerThreads& worker_threads =
          *ctx->device()->tensorflow_cpu_worker_threads();
      Shard(worker_threads.num_threads, worker_threads.workers,
            input_flat.dimension(0), kCostPerUnit, fill_fn);
    } else {
      fill_fn(0, input_flat.dimension(0));
    }
  }

 private:
  bool use_multi_threads_;
};

class GatherEmbeddingsByInputGradientOp : public OpKernel {
 public:
  explicit GatherEmbeddingsByInputGradientOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& ids = ctx->input(0);
    const Tensor& grads = ctx->input(1);
    auto index_mapping_flat = ctx->input(2).flat<int64>();
    auto ids_flat = ids.flat<int64>();
    // Reshape it to len(input):embedding_size shape.
    const int64 embedding_size = grads.dim_size(grads.dims() - 1);
    const int64 input_size = index_mapping_flat.dimension(0);
    auto grads_mat = grads.shaped<float, 2>({input_size, embedding_size});
    const int64 len_ids = ids_flat.dimension(0);

    Tensor* output;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, {len_ids, embedding_size}, &output));
    std::memset(output->data(), 0, output->AllocatedBytes());
    TTypes<float>::Matrix embedding_grads_mats = output->matrix<float>();

    for (int64 k = 0; k < input_size; ++k) {
      const int64 loc = index_mapping_flat(k);
      embedding_grads_mats.chip<0>(loc) += grads_mat.chip<0>(k);
    }
  }
};

class FusedGatherEmbeddingsByInputOp : public OpKernel {
 public:
  explicit FusedGatherEmbeddingsByInputOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("M", &num_of_inputs_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("embedding_dims", &embedding_dims_));
  }

  void Compute(OpKernelContext* ctx) override {
    auto fused_embeddings_flat = ctx->input(0).flat<float>();

    OpInputList inputs;
    OP_REQUIRES_OK(ctx, ctx->input_list("embedding_offsets", &inputs));
    OpOutputList outputs;
    OP_REQUIRES_OK(ctx, ctx->output_list("output", &outputs));

    std::vector<float*> output_ptrs(outputs.size());
    DCHECK_EQ(num_of_inputs_, outputs.size());
    for (int i = 0; i < num_of_inputs_; ++i) {
      TensorShape output_shape = inputs[i].shape();
      output_shape.AddDim(embedding_dims_[i]);
      Tensor* out;
      OP_REQUIRES_OK(ctx, outputs.allocate(i, output_shape, &out));
      output_ptrs[i] = out->flat<float>().data();
    }

    auto fill_fn = [&](const int64 begin, const int64 end) {
      for (int i = begin; i < end; ++i) {
        auto embedding_offset_vec = inputs[i].vec<int32>();
        int embedding_dim = embedding_dims_[i];

        for (int j = 0; j < embedding_offset_vec.size(); ++j) {
          auto offset = embedding_offset_vec(j);
          std::memcpy(output_ptrs[i] + j * embedding_dim,
                      fused_embeddings_flat.data() + offset,
                      embedding_dim * sizeof(float));
        }
      }
    };

    auto worker_threads = ctx->device()->tensorflow_cpu_worker_threads();
    worker_threads->workers->ParallelFor(
        num_of_inputs_,
        thread::ThreadPool::SchedulingParams(
            thread::ThreadPool::SchedulingStrategy::kFixedBlockSize,
            absl::nullopt,
            1),  // block_size
        fill_fn);
  }

 private:
  int num_of_inputs_;
  std::vector<int> embedding_dims_;
};

class FusedGatherEmbeddingsByInputGradientOp : public OpKernel {
 public:
  explicit FusedGatherEmbeddingsByInputGradientOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("embedding_dims", &embedding_dims_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("M", &num_of_inputs_));
  }

  void Compute(OpKernelContext* ctx) override {
    int32 fused_embeddings_size = ctx->input(0).flat<int32>()(0);
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            0, TensorShape({fused_embeddings_size}), &output));
    auto output_flat = output->flat<float>();
    std::memset(output->data(), 0, output->AllocatedBytes());
    // By design, different inputs from num_of_inputs_ are sharded into
    // different positions in the flattened gradients, and thus simply do a
    // parallel fill function.
    auto fill_fn = [&](const int64 begin, const int64 end) {
      for (int i = begin; i < end; ++i) {
        auto input_flat = ctx->input(1 + i).flat<float>();
        auto embedding_offset_vec =
            ctx->input(num_of_inputs_ + 1 + i).vec<int32>();
        int embedding_dim = embedding_dims_[i];
        for (int j = 0; j < embedding_offset_vec.dimension(0); ++j) {
          int32 offset = embedding_offset_vec(j);
          const float* input_a =
              const_cast<float*>(input_flat.data()) + j * embedding_dim;
          float* output_b = static_cast<float*>(output_flat.data()) + offset;
          // Use AVX acceleration for reducesum.
          ::monolith::hash_table::ReduceSum(input_a, output_b, output_b,
                                            embedding_dim);
        }
      }
    };

    auto worker_threads = ctx->device()->tensorflow_cpu_worker_threads();
    worker_threads->workers->ParallelFor(
        num_of_inputs_,
        thread::ThreadPool::SchedulingParams(
            thread::ThreadPool::SchedulingStrategy::kFixedBlockSize,
            absl::nullopt,
            1),  // block_size
        fill_fn);
  }

 private:
  std::vector<int> embedding_dims_;
  int num_of_inputs_;
};

REGISTER_OP("MonolithMapIdToEmbedding")
    .Input("ids: num_splits * int64")
    .Input("embeddings: num_splits * float")
    .Input("input: int64")
    .Output("output: float")
    .Attr("num_splits: int")
    .Attr("use_multi_threads: bool = true")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int num_splits;
      TF_RETURN_IF_ERROR(c->GetAttr("num_splits", &num_splits));
      shape_inference::ShapeHandle embedding_shape =
          c->MakeShape({c->Dim(c->input(num_splits), -1)});
      shape_inference::ShapeHandle input_shape = c->input(2 * num_splits);
      shape_inference::ShapeHandle output_shape;
      TF_RETURN_IF_ERROR(
          c->Concatenate(input_shape, embedding_shape, &output_shape));
      c->set_output(0, output_shape);
      return Status::OK();
    });

REGISTER_OP("MonolithMapIdToEmbeddingGradient")
    .Input("ids: num_splits * int64")
    .Input("input: int64")
    .Input("grads: float")
    .Output("embedding_grads: num_splits * float")
    .Attr("num_splits: int")
    .SetShapeFn([](shape_inference::InferenceContext* ctx) {
      int num_splits;
      TF_RETURN_IF_ERROR(ctx->GetAttr("num_splits", &num_splits));
      shape_inference::DimensionHandle embedding_size =
          ctx->Dim(ctx->input(num_splits + 1), -1);
      for (int i = 0; i < num_splits; ++i) {
        shape_inference::DimensionHandle len_ids = ctx->Dim(ctx->input(i), 0);
        ctx->set_output(i, ctx->MakeShape({len_ids, embedding_size}));
      }
      return Status::OK();
    });

REGISTER_OP("MonolithGatherEmbeddingsByInput")
    .Input("ids: int64")
    .Input("embeddings: float")
    .Input("input: int64")
    .Output("output: float")
    .Output("output_index_mapping: int64")
    .SetDoNotOptimize()  // Crash with grappler.
    .Attr("use_multi_threads: bool = false")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle embedding_shape =
          c->MakeShape({c->Dim(c->input(1), -1)});
      shape_inference::ShapeHandle input_shape = c->input(2);
      shape_inference::ShapeHandle output_shape;
      TF_RETURN_IF_ERROR(
          c->Concatenate(input_shape, embedding_shape, &output_shape));
      c->set_output(0, output_shape);
      c->set_output(1, input_shape);
      return Status::OK();
    });

REGISTER_OP("MonolithGatherEmbeddingsByInputGradient")
    .Input("ids: int64")
    .Input("grads: float")
    .Input("index_mapping: int64")
    .Output("embedding_grads: float")
    .SetDoNotOptimize()  // Crash with grappler.
    .SetShapeFn([](shape_inference::InferenceContext* ctx) {
      shape_inference::DimensionHandle embedding_size =
          ctx->Dim(ctx->input(1), -1);
      shape_inference::DimensionHandle len_ids = ctx->Dim(ctx->input(0), 0);
      ctx->set_output(0, ctx->MakeShape({len_ids, embedding_size}));
      return Status::OK();
    });

REGISTER_OP("MonolithFusedGatherEmbeddingsByInput")
    .Input("fused_embeddings: float")
    .Input("embedding_offsets: M * int32")
    .Output("output: M * float")
    .Attr("embedding_dims: list(int)")
    .Attr("M: int")
    .SetDoNotOptimize()  // Crash with grappler.
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int M;
      std::vector<int> embedding_dims;
      TF_RETURN_IF_ERROR(c->GetAttr("embedding_dims", &embedding_dims));
      TF_RETURN_IF_ERROR(c->GetAttr("M", &M));
      for (int i = 0; i < M; ++i) {
        shape_inference::DimensionHandle dim = c->Dim(c->input(1 + i), 0);
        c->set_output(i, c->MakeShape({dim, embedding_dims[i]}));
      }
      return Status::OK();
    });

REGISTER_OP("MonolithFusedGatherEmbeddingsByInputGradient")
    .Input("fused_embeddings_size: int32")
    .Input("grads: M * float")
    .Input("embedding_offsets: M * int32")
    .Output("output: float")
    .Attr("embedding_dims: list(int)")
    .Attr("M: int")
    .SetDoNotOptimize()  // Crash with grappler.
    .SetShapeFn([](shape_inference::InferenceContext* ctx) {
      shape_inference::DimensionHandle output_dim;
      TF_RETURN_IF_ERROR(ctx->MakeDimForScalarInput(0, &output_dim));
      ctx->set_output(0, ctx->Vector(output_dim));
      return Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name("MonolithMapIdToEmbedding").Device(DEVICE_CPU),
                        MapIdToEmbeddingOp);

REGISTER_KERNEL_BUILDER(
    Name("MonolithMapIdToEmbeddingGradient").Device(DEVICE_CPU),
    MapIdToEmbeddingGradientOp);

REGISTER_KERNEL_BUILDER(
    Name("MonolithGatherEmbeddingsByInput").Device(DEVICE_CPU),
    GatherEmbeddingsByInputOp);

REGISTER_KERNEL_BUILDER(
    Name("MonolithGatherEmbeddingsByInputGradient").Device(DEVICE_CPU),
    GatherEmbeddingsByInputGradientOp);

REGISTER_KERNEL_BUILDER(
    Name("MonolithFusedGatherEmbeddingsByInput").Device(DEVICE_CPU),
    FusedGatherEmbeddingsByInputOp);

REGISTER_KERNEL_BUILDER(
    Name("MonolithFusedGatherEmbeddingsByInputGradient").Device(DEVICE_CPU),
    FusedGatherEmbeddingsByInputGradientOp);

}  // namespace monolith_tf
}  // namespace tensorflow
