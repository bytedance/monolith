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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "monolith/native_training/runtime/ops/alloc_utils.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/gpu_device_array.h"
#include "tensorflow/core/kernels/gpu_device_array_gpu.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace monolith_tf {

typedef Eigen::GpuDevice GPUDevice;
// To run mnay segment_sum ops on various input lengths and emb dims,
// in one single GPU kernel. We define input group i:
//  * indices with n_i length and s_i segments,
//    s_i <= n_i as input_outer_dim_size;
//    s_i <= output_outer_dim_size also;
//    For example, [1,1,1,2,2,4] with n_i = 5, s_i = 3
//    where output_outer_dim_size >= 4 >= s_i
//  * values with n_i input_outer_dim_size and d_i dims
// The total computation workload is sum n_i * d_i on i.
// For all n_i, we stride with a fixed length k_n, so that
// the same stride can have chance to reduce in local thread.
// The total gpu workload is now the sum on i of
//  [(n_i // k_n) + 1] * d_i
template <typename T, typename Index, int OuterDimTileSize>
__global__ void FusedSortedSegmentSumCustomKernel(
    GpuDeviceArrayStruct<Index> input_outer_dim_sizes_data,
    GpuDeviceArrayStruct<Index> inner_dim_sizes_data,
    GpuDeviceArrayStruct<Index> output_outer_dim_sizes_data,
    GpuDeviceArrayStruct<const Index*> segment_idss_data,  // __restrict__
    GpuDeviceArrayStruct<const T*> inputs_data,            // __restrict__
    GpuDeviceArrayStruct<T*> outputs_data,                 // __restrict__
    GpuDeviceArrayStruct<Index> stripe_offsets_data,
    const Index total_stripe_count) {
  Index* input_outer_dim_sizes =
      GetGpuDeviceArrayOnDevice(&input_outer_dim_sizes_data);
  Index* inner_dim_sizes = GetGpuDeviceArrayOnDevice(&inner_dim_sizes_data);
  Index* output_outer_dim_sizes =
      GetGpuDeviceArrayOnDevice(&output_outer_dim_sizes_data);

  const Index* __restrict__* segment_idss =
      GetGpuDeviceArrayOnDevice(&segment_idss_data);
  const T* __restrict__* inputs = GetGpuDeviceArrayOnDevice(&inputs_data);
  T* __restrict__* outputs = GetGpuDeviceArrayOnDevice(&outputs_data);
  Index* stripe_offsets = GetGpuDeviceArrayOnDevice(&stripe_offsets_data);

  // if using shared memory
  // Ref:
  // https://github.com/tensorflow/tensorflow/blob/v2.4.0/tensorflow/core/kernels/split_lib_gpu.cu.cc#L124
  GPU_DYNAMIC_SHARED_MEM_DECL(sizeof(Index), unsigned char, smem);
  Index N = input_outer_dim_sizes_data.size;
  Index* ptr = reinterpret_cast<Index*>(smem);
  Index* smem_input_outer_dim_sizes = ptr;
  ptr += N;
  Index* smem_inner_dim_sizes = ptr;
  ptr += N;
  Index* smem_output_outer_dim_sizes = ptr;
  ptr += N;
  Index* smem_stripe_offsets = ptr;
  for (int x = threadIdx.x; x < N; x += blockDim.x) {
    smem_input_outer_dim_sizes[x] = input_outer_dim_sizes[x];
    smem_inner_dim_sizes[x] = inner_dim_sizes[x];
    smem_output_outer_dim_sizes[x] = output_outer_dim_sizes[x];
  }
  for (int x = threadIdx.x; x < N + 1 /*stripe_offsets_data.size*/;
       x += blockDim.x) {
    smem_stripe_offsets[x] = stripe_offsets[x];
  }
  __syncthreads();
  stripe_offsets = smem_stripe_offsets;
  input_outer_dim_sizes = smem_input_outer_dim_sizes;
  inner_dim_sizes = smem_inner_dim_sizes;
  output_outer_dim_sizes = smem_output_outer_dim_sizes;

  Index i = 0;
  for (Index stripe_index : GpuGridRangeX(total_stripe_count)) {
    // Determine the abstract computation unit amd local_stripe_index
    while (stripe_offsets[i + 1] <= stripe_index) ++i;
    Index local_stripe_index = stripe_index - stripe_offsets[i];

    auto input_outer_dim_size = input_outer_dim_sizes[i];
    auto inner_dim_size = inner_dim_sizes[i];
    auto output_outer_dim_size = output_outer_dim_sizes[i];
    if (input_outer_dim_size == 0 || inner_dim_size == 0 ||
        output_outer_dim_size == 0)
      continue;
    auto segment_ids = segment_idss[i];
    auto input = inputs[i];
    auto output = outputs[i];

    // Start computation: segment sum
    const Index segment_offset = local_stripe_index % inner_dim_size;
    const Index input_outer_dim_index_base =
        local_stripe_index / inner_dim_size * Index(OuterDimTileSize);

    T sum = T(0);
    Index first_segment_id = segment_ids[input_outer_dim_index_base];
    Index last_output_segment_id = output_outer_dim_size;

    const Index actual_stripe_height =
        min(Index(OuterDimTileSize),
            input_outer_dim_size - input_outer_dim_index_base);
    // #pragma unroll
    for (Index j = 0; j < actual_stripe_height; j++) {
      Index current_output_segment_id =
          segment_ids[input_outer_dim_index_base + j];
      // Decide whether to write result to global memory.
      // Result is only written to global memory if we move
      // to another segment. Otherwise we can keep accumulating
      // locally.
      if (current_output_segment_id > last_output_segment_id) {
        const Index output_index =
            last_output_segment_id * inner_dim_size + segment_offset;
        // decide whether to write result to global memory using atomic
        // operations
        if (last_output_segment_id == first_segment_id) {
          GpuAtomicAdd(output + output_index, sum);
        } else {
          *(output + output_index) = sum;
        }
        sum = T(0);
      }
      sum += ldg(input + (input_outer_dim_index_base + j) * inner_dim_size +
                 segment_offset);
      last_output_segment_id = current_output_segment_id;
    }
    // For the last result in a strip, always write using atomic operations
    // due to possible race conditions with threads computing
    // the following strip.
    const Index output_index =
        last_output_segment_id * inner_dim_size + segment_offset;
    GpuAtomicAdd(output + output_index, sum);
  }
}

// Returns true if the three tensors have valid number of elements
// If shape_input has 0 elements, then we need to have indices and updates with
// exactly 0 elements too, otherwise we should error. If indices has 0 elements
// then updates should also have 0 elements, otherwise we should error.
bool ValidEmptyOutputShape(int64 num_inputs, int64 num_indices,
                           int64 num_updates) {
  if (num_indices == 0 && num_updates == 0) {
    return true;  // regardless of num_inputs ?= 0, covers both cases
  }
  // now we want all 3 tensors to have values
  return (num_inputs != 0 && num_indices != 0 && num_updates != 0);
}

template <typename T, typename Index>
class FusedSegmentSumGPU : public OpKernel {
 public:
  explicit FusedSegmentSumGPU(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &N_));
  }

  void Compute(OpKernelContext* ctx) override {
    GPUDevice gpu_device = ctx->eigen_device<GPUDevice>();
    const int OuterDimTileSize = 8;
    Index stripe_offset = 0;  // max as total_stripe_count
    GpuDeviceArrayOnHost<Index> stripe_offsets(ctx, N_ + 1);
    OP_REQUIRES_OK(ctx, stripe_offsets.Init());

    OpInputList indices_list;
    OP_REQUIRES_OK(ctx, ctx->input_list("indices", &indices_list));
    OpInputList updates_list;
    OP_REQUIRES_OK(ctx, ctx->input_list("updates", &updates_list));
    OpInputList shape_list;
    OP_REQUIRES_OK(ctx, ctx->input_list("shape", &shape_list));
    OpOutputList outputs;
    OP_REQUIRES_OK(ctx, ctx->output_list("outputs", &outputs));

    GpuDeviceArrayOnHost<const Index*> indices_ptrs(ctx, N_);
    // TODO(peng): concat then memcpy if necessary
    OP_REQUIRES_OK(ctx, indices_ptrs.Init());
    GpuDeviceArrayOnHost<const T*> updates_ptrs(ctx, N_);
    OP_REQUIRES_OK(ctx, updates_ptrs.Init());
    GpuDeviceArrayOnHost<T*> output_ptrs(ctx, N_);
    OP_REQUIRES_OK(ctx, output_ptrs.Init());

    GpuDeviceArrayOnHost<Index> input_outer_dim_sizes(ctx, N_);
    OP_REQUIRES_OK(ctx, input_outer_dim_sizes.Init());
    GpuDeviceArrayOnHost<Index> inner_dim_sizes(ctx, N_);
    OP_REQUIRES_OK(ctx, inner_dim_sizes.Init());
    GpuDeviceArrayOnHost<Index> output_outer_dim_sizes(ctx, N_);
    OP_REQUIRES_OK(ctx, output_outer_dim_sizes.Init());
    // Shared memory used by four <Index> typed Device array.
    int smem_usage = sizeof(Index) * (4 * N_ + 1);

    for (int i = 0; i < N_; ++i) {
      const Tensor& indices = indices_list[i];
      const Tensor& updates = updates_list[i];
      const Tensor& shape_input = shape_list[i];

      OP_REQUIRES(ctx, indices.shape().dims() >= 1,
                  errors::InvalidArgument(
                      "Indices shape must have rank at least one. Found:",
                      indices.shape().DebugString()));
      OP_REQUIRES(ctx, updates.shape().dims() >= 1,
                  errors::InvalidArgument(
                      "Updates shape must have rank at least one. Found:",
                      updates.shape().DebugString()));

      auto vec = shape_input.flat<Index>();
      TensorShape output_shape;
      OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(vec.data(), vec.size(),
                                                      &output_shape));

      OP_REQUIRES(ctx,
                  ValidEmptyOutputShape(shape_input.NumElements(),
                                        indices.shape().num_elements(),
                                        updates.shape().num_elements()),
                  errors::InvalidArgument(
                      "Indices and updates specified for empty output shape"));

      OP_REQUIRES(ctx, shape_input.dims() == 1,
                  errors::InvalidArgument("Shape must be a vector"));

      //
      Index input_total_size = updates.NumElements();
      auto input_shape = updates.shape();
      Index input_outer_dim_size = input_shape.dim_size(0);
      Index inner_dim_size = 1;
      for (int j = 1; j < input_shape.dims(); ++j)
        inner_dim_size *= input_shape.dim_size(j);
      input_outer_dim_sizes.Set(i, input_outer_dim_size);
      inner_dim_sizes.Set(i, inner_dim_size);
      output_outer_dim_sizes.Set(i, output_shape.dim_size(0));

      stripe_offsets.Set(i, stripe_offset);
      Index input_outer_dim_num_stripe =
          Eigen::divup(input_outer_dim_size, Index(OuterDimTileSize));
      stripe_offset += input_outer_dim_num_stripe * inner_dim_size;

      //
      Tensor* out;
      OP_REQUIRES_OK(ctx, outputs.allocate(i, output_shape, &out));
      gpu_device.memset(out->flat<T>().data(), T(0),
                        sizeof(T) * out->NumElements());
      output_ptrs.Set(i, out->flat<T>().data());
      updates_ptrs.Set(i, updates.flat<T>().data());
      indices_ptrs.Set(i, indices.flat<Index>().data());
    }
    const Index total_stripe_count = stripe_offset;
    stripe_offsets.Set(N_, stripe_offset);
    OP_REQUIRES_OK(ctx, stripe_offsets.Finalize());

    OP_REQUIRES_OK(ctx, input_outer_dim_sizes.Finalize());
    OP_REQUIRES_OK(ctx, inner_dim_sizes.Finalize());
    OP_REQUIRES_OK(ctx, output_outer_dim_sizes.Finalize());

    OP_REQUIRES_OK(ctx, indices_ptrs.Finalize());
    OP_REQUIRES_OK(ctx, updates_ptrs.Finalize());
    OP_REQUIRES_OK(ctx, output_ptrs.Finalize());

    auto config = GetGpuLaunchConfig(total_stripe_count, gpu_device);
    GpuLaunchKernel(
        FusedSortedSegmentSumCustomKernel<T, Index, OuterDimTileSize>,
        config.block_count, config.thread_per_block,
        /*shared_memory_size_bytes=*/smem_usage, gpu_device.stream(),
        input_outer_dim_sizes.data(), inner_dim_sizes.data(),
        output_outer_dim_sizes.data(), indices_ptrs.data(), updates_ptrs.data(),
        output_ptrs.data(), stripe_offsets.data(), total_stripe_count);
  }

 private:
  int N_;
};

// a struct that stores a mapping from the current column to the base address in
// the tensor after split
template <typename T>
struct __align__(16) ColumnInfo {
  T* __restrict__ base_address;  // this includes the current slice index
  int slice_len;
};

template <typename T>
struct __align__(16) TableInfo {
  int c_offset;
  int r_offset;
  T* __restrict__ embs;
};

// given the row and column index in output,
// sum up the corresponding column of the rows that need to be reduced
template <typename T>
__device__ __forceinline__ void fused_reduce_and_split(
    int r_group_id, const int* __restrict__ row_prefix, ColumnInfo<T> col_info,
    const float* __restrict__ emb, int emb_len) {
  int row_end = ldg(row_prefix + r_group_id + 1);
  T sum = T(0);
  for (int r = ldg(row_prefix + r_group_id); r < row_end; r++)
    sum += ldg(emb + r * emb_len);
  col_info.base_address[r_group_id * col_info.slice_len] = sum;
}

// initialize shared memory so that
// first (N + 1) * sizeof(TableInfo<T>) byte is the array of TableInfo<T>
// and the following (N + 1) * sizeof(int) byte is the array of int (table
// splits)
template <typename T>
__device__ __forceinline__ const int* init_shared_mem(
    GpuDeviceArrayStruct<int, 0>& _table_splits,          // N + 1
    GpuDeviceArrayStruct<TableInfo<T>, 0>& _table_infos,  // N + 1
    int* shared_mem) {
  int table_info_sz = _table_splits.size * (sizeof(TableInfo<T>) / sizeof(int));
  auto s_table_infos = shared_mem;
  auto s_table_splits = shared_mem + table_info_sz;
  auto g_table_splits = GetGpuDeviceArrayOnDevice(&_table_splits);
  auto g_table_infos =
      reinterpret_cast<int*>(GetGpuDeviceArrayOnDevice(&_table_infos));
  int total_shared_sz = table_info_sz + _table_splits.size;
  for (int i = threadIdx.x; i < total_shared_sz; i += blockDim.x) {
    if (i < table_info_sz) {
      s_table_infos[i] = g_table_infos[i];
    } else {
      int j = i - table_info_sz;
      s_table_splits[j] = g_table_splits[j];
    }
  }
  __syncthreads();
  return s_table_splits;
}

// GpuDeviceArrayStruct<int, 0> guarantees stroing address in global memory
// so __ldg can works properly
// this kernel works the best when the number of rows to be reduced is
// relatively even across the batch dimension, so that each threads' workload is
// about the same
template <typename T>
__global__ void FusedReduceSumAndSplitKernel(
    GpuDeviceArrayStruct<int, 0> _table_splits,                // N + 1
    GpuDeviceArrayStruct<TableInfo<const T>, 0> _table_infos,  // N + 1
    const ColumnInfo<T>* __restrict__ col_infos,
    const int* __restrict__ row_splits,
    int total  // =total number of elements in output (tables * n_rows after
               // reduction * emb_len)
) {
  extern __shared__ int shared_mem[];
  auto table_infos = reinterpret_cast<const TableInfo<const T>*>(shared_mem);
  auto table_splits = init_shared_mem(_table_splits, _table_infos, shared_mem);

  int table_idx = 1;
  for (int outer_tid = threadIdx.x + blockIdx.x * blockDim.x; outer_tid < total;
       outer_tid += blockDim.x * gridDim.x) {
    while (outer_tid >= table_splits[table_idx]) table_idx++;
    table_idx -= 1;
    int idx = outer_tid - table_splits[table_idx];
    auto table = table_infos[table_idx];
    auto next_table = table_infos[table_idx + 1];
    int emb_len = next_table.c_offset - table.c_offset;
    int row_idx = idx / emb_len;
    int col_idx = idx % emb_len;
    fused_reduce_and_split(row_idx, row_splits + table.r_offset,
                           col_infos[table.c_offset + col_idx],  // 16-byte load
                           table.embs + col_idx, emb_len);
  }
}

template <typename ValueType, int MaxInlineValues = 8>
const ValueType* GetGpuDeviceArrayOnHost(
    const GpuDeviceArrayStruct<ValueType, MaxInlineValues>* data) {
  if (data->size <= MaxInlineValues) {
    return data->inline_values;
  } else {
    return data->out_of_line_values;
  }
}

template <typename T>
class FusedReduceAndSplitGPU : public OpKernel {
 public:
  explicit FusedReduceAndSplitGPU(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &N_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("slice_dims", &slice_dims_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("row_split_splits", &row_split_splits_));
  }

  void Compute(OpKernelContext* ctx) override {
    profiler::TraceMe activity(
        []() { return "FusedReduceAndSplitGPUPreprocessing"; });
    const auto& gpu_device = ctx->eigen_gpu_device();
    OpInputList updates_list;
    OpOutputList outputs;
    OP_REQUIRES_OK(ctx, ctx->input_list("updates", &updates_list));
    OP_REQUIRES_OK(ctx, ctx->output_list("outputs", &outputs));

    GpuDeviceArrayOnHost<int, 0> table_splits(ctx, N_ + 1);
    GpuDeviceArrayOnHost<TableInfo<const T>, 0> table_infos(ctx, N_ + 1);
    OP_REQUIRES_OK(ctx, table_splits.Init());
    OP_REQUIRES_OK(ctx, table_infos.Init());

    // precalculate the size we need for storing row splits and column infos
    int col_info_size = 0;
    // number of rows after reduction = batch_size + 1
    int batch_size = row_split_splits_[1] - row_split_splits_[0] - 1;
    FusedAlignedOutputAllocator<EIGEN_MAX_ALIGN_BYTES / sizeof(T)> fao_alloc(
        ctx);
    for (int i = 0; i < N_; i++) {
      const Tensor& updates = updates_list[i];
      table_splits.Set(i, fao_alloc.get_unaligned_total());
      table_infos.Set(
          i, {col_info_size, row_split_splits_[i], updates.flat<T>().data()});
      int emb_len = updates.dim_size(1);
      // number of rows after reduction * embedding length
      fao_alloc.add_slice(batch_size * emb_len);
      col_info_size += emb_len;
    }
    table_splits.Set(N_, fao_alloc.get_unaligned_total());
    table_infos.Set(N_, {col_info_size, row_split_splits_[N_], nullptr});
    OP_REQUIRES_OK(ctx, table_splits.Finalize());
    OP_REQUIRES_OK(ctx, table_infos.Finalize());

    // since these arrays are used as the backing storage,
    // we don't allow them to store values inline
    // because that will be on the host's stack and will not be
    // device-accessible
    GpuDeviceArrayOnHost<ColumnInfo<T>, 0> col_infos(ctx, col_info_size);
    OP_REQUIRES_OK(ctx, col_infos.Init());

    // hide some latency by the h2ds above
    fao_alloc.allocate(outputs.expected_output_dtype(0));

    col_info_size = 0;
    for (int i = 0; i < slice_dims_.size(); i++) {
      int slice_len = slice_dims_[i];
      Tensor out = fao_alloc.get_slice({batch_size, slice_len});
      auto data = out.flat<T>().data();
      outputs.set(i, std::move(out));
      for (int k = 0; k < slice_len; k++) {
        col_infos.Set(col_info_size++, {data + k, slice_len});
      }
    }
    OP_REQUIRES_OK(ctx, col_infos.Finalize());
    auto smem_sz = (sizeof(TableInfo<const T>) + sizeof(int)) * (N_ + 1);
    auto config =
        GetGpuLaunchConfig(fao_alloc.get_unaligned_total(), gpu_device,
                           FusedReduceSumAndSplitKernel<T>, smem_sz, 0);
    GpuLaunchKernel(FusedReduceSumAndSplitKernel<T>, config.block_count,
                    config.thread_per_block, smem_sz, gpu_device.stream(),
                    table_splits.data(), table_infos.data(),
                    GetGpuDeviceArrayOnHost(&col_infos.data()),
                    ctx->input(0).vec<int32>().data(),  // row_splits base ptr
                    fao_alloc.get_unaligned_total());
  }

 private:
  int N_;
  std::vector<int> slice_dims_, row_split_splits_;
};

template <typename T>
__device__ __forceinline__ void fused_reduce_and_split_grad(
    int r_group_id, const int* __restrict__ row_splits, int count,
    ColumnInfo<const T> col_info, float* __restrict__ emb, int emb_len) {
  // from https://en.cppreference.com/w/cpp/algorithm/upper_bound
  int it, step, first = 0;
  while (count > 0) {
    it = first;
    step = count / 2;
    it += step;

    // ldg will not be helpful here due to irregular pattern
    if (!(r_group_id < row_splits[it])) {
      first = ++it;
      count -= step + 1;
    } else {
      count = step;
    }
  }
  // find the largest first element in row_splits >= r_group_id
  emb[r_group_id * emb_len] =
      col_info.base_address[(first - 1) * col_info.slice_len];
}

// workload of threads are perfectly balanced: (1 load + 1 store) * (total /
// num_threads) the bottleneck of this kernel is the binary search in the device
// function above
template <typename T>
__global__ void FusedReduceSumAndSplitKernelGrad(
    GpuDeviceArrayStruct<int, 0> _table_splits,          // N + 1
    GpuDeviceArrayStruct<TableInfo<T>, 0> _table_infos,  // N + 1
    const ColumnInfo<const T>* __restrict__ col_infos,
    const int* __restrict__ row_splits,
    int total  // =total number of elements in output (tables * n_rows before
               // reduction * emb_len)
) {
  extern __shared__ int shared_mem[];
  auto table_infos = reinterpret_cast<const TableInfo<T>*>(shared_mem);
  auto table_splits = init_shared_mem(_table_splits, _table_infos, shared_mem);

  int table_idx = 1;
  for (int outer_tid = threadIdx.x + blockIdx.x * blockDim.x; outer_tid < total;
       outer_tid += blockDim.x * gridDim.x) {
    while (outer_tid >= table_splits[table_idx]) table_idx++;
    table_idx -= 1;
    int idx = outer_tid - table_splits[table_idx];
    auto table = table_infos[table_idx];
    auto next_table = table_infos[table_idx + 1];
    int emb_len = next_table.c_offset - table.c_offset;
    int row_len = next_table.r_offset - table.r_offset;
    int row_idx = idx / emb_len;
    int col_idx = idx % emb_len;
    fused_reduce_and_split_grad(
        row_idx, row_splits + table.r_offset, row_len,
        col_infos[table.c_offset + col_idx],  // 16-byte load
        table.embs + col_idx, emb_len);
  }
}

template <typename T>
class FusedReduceAndSplitGPUGrad : public OpKernel {
 public:
  explicit FusedReduceAndSplitGPUGrad(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("slice_dims", &slice_dims_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("row_split_splits", &row_split_splits_));
    N_ = row_split_splits_.size() - 1;  // =num_tables
  }

  void Compute(OpKernelContext* ctx) override {
    profiler::TraceMe activity(
        []() { return "FusedReduceAndSplitGPUGradPreprocessing"; });
    const auto& gpu_device = ctx->eigen_gpu_device();
    OpInputList slice_list, updates_list;
    OpOutputList outputs;
    OP_REQUIRES_OK(ctx, ctx->input_list("updates", &updates_list));
    OP_REQUIRES_OK(ctx, ctx->input_list("slices", &slice_list));
    OP_REQUIRES_OK(ctx, ctx->output_list("outputs", &outputs));

    GpuDeviceArrayOnHost<int, 0> table_splits(ctx, N_ + 1);
    GpuDeviceArrayOnHost<TableInfo<T>, 0> table_infos(ctx, N_ + 1);
    OP_REQUIRES_OK(ctx, table_splits.Init());
    OP_REQUIRES_OK(ctx, table_infos.Init());

    FusedAlignedOutputAllocator<EIGEN_MAX_ALIGN_BYTES / sizeof(T)> fao_alloc(
        ctx);
    int col_info_size = 0;
    for (int i = 0; i < N_; i++) {
      // number of rows before reduction
      int num_rows = updates_list[i].dim_size(0);
      int emb_len = updates_list[i].dim_size(1);

      table_splits.Set(i, fao_alloc.get_unaligned_total());
      fao_alloc.add_slice(num_rows * emb_len);
      col_info_size += emb_len;
    }
    // check for overflow
    OP_REQUIRES(ctx, fao_alloc.get_unaligned_total() <= INT_MAX,
                errors::InvalidArgument("There are too many elements to be "
                                        "processed by fused reduce and split"));
    table_splits.Set(N_, fao_alloc.get_unaligned_total());
    OP_REQUIRES_OK(ctx, table_splits.Finalize());

    // col_info stores the base src address of each slice
    GpuDeviceArrayOnHost<ColumnInfo<const T>, 0> col_infos(ctx, col_info_size);
    OP_REQUIRES_OK(ctx, col_infos.Init());

    col_info_size = 0;
    for (int i = 0; i < slice_dims_.size(); i++) {
      int slice_len = slice_dims_[i];
      auto data = slice_list[i].flat<T>().data();
      for (int j = 0; j < slice_len; j++) {
        col_infos.Set(col_info_size++, {data + j, slice_len});
      }
    }
    OP_REQUIRES_OK(ctx, col_infos.Finalize());

    // hide some latency by the h2ds above
    fao_alloc.allocate(outputs.expected_output_dtype(0));

    col_info_size = 0;
    for (int i = 0; i < N_; i++) {
      int num_rows = updates_list[i].dim_size(0);
      int emb_len = updates_list[i].dim_size(1);
      Tensor out = fao_alloc.get_slice({num_rows, emb_len});
      table_infos.Set(
          i, {col_info_size, row_split_splits_[i], out.flat<T>().data()});
      outputs.set(i, std::move(out));
      col_info_size += emb_len;
    }
    table_infos.Set(N_, {col_info_size, row_split_splits_[N_], nullptr});
    OP_REQUIRES_OK(ctx, table_infos.Finalize());

    auto smem_sz = (sizeof(TableInfo<const T>) + sizeof(int)) * (N_ + 1);
    auto config =
        GetGpuLaunchConfig(fao_alloc.get_unaligned_total(), gpu_device,
                           FusedReduceSumAndSplitKernelGrad<T>, smem_sz, 0);
    GpuLaunchKernel(FusedReduceSumAndSplitKernelGrad<T>, config.block_count,
                    config.thread_per_block, smem_sz, gpu_device.stream(),
                    table_splits.data(), table_infos.data(),
                    GetGpuDeviceArrayOnHost(&col_infos.data()),
                    ctx->input(0).vec<int32>().data(),
                    fao_alloc.get_unaligned_total());
  }

 private:
  int N_;
  std::vector<int> slice_dims_;  // fused array of slice dimensions
  std::vector<int> row_split_splits_;
};

#define REGISTER_FUSED_REDUCE_AND_SPLIT_GRAD(type)                   \
  REGISTER_KERNEL_BUILDER(Name("MonolithFusedReduceAndSplitGPUGrad") \
                              .Device(DEVICE_GPU)                    \
                              .TypeConstraint<type>("T"),            \
                          FusedReduceAndSplitGPUGrad<type>)
TF_CALL_float(REGISTER_FUSED_REDUCE_AND_SPLIT_GRAD);

REGISTER_OP("MonolithFusedReduceAndSplitGPUGrad")
    .Input("splits: int32")   // input of the forward op
    .Input("updates: M * T")  // input of the forward op, needed to do shape
                              // inference
    .Input("slices: N * T")   // output of the forward op
    .Output("outputs: M * T")
    .Attr("slice_dims: list(int)")        // from the forward op
    .Attr("row_split_splits: list(int)")  // from the forward op
    .Attr("T: type")
    .Attr("N: int")
    .Attr("M: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int N, M;
      std::vector<int> slice_dims, row_split_splits;

      TF_RETURN_IF_ERROR(c->GetAttr("N", &N));
      TF_RETURN_IF_ERROR(c->GetAttr("M", &M));
      TF_RETURN_IF_ERROR(c->GetAttr("slice_dims", &slice_dims));
      TF_RETURN_IF_ERROR(c->GetAttr("row_split_splits", &row_split_splits));

      // simple sanity checks
      if (slice_dims.size() != N)
        return errors::InvalidArgument(
            "len(slice_dims) must equal to the number of input slices");
      if (row_split_splits.size() != M + 1)
        return errors::InvalidArgument(
            "len(row_split_splits) must equal to M + 1");

      for (int i = 0; i < M; i++) {
        c->set_output(i, c->input(1 + i));
      }
      return Status::OK();
    });

#define REGISTER_FUSED_REDUCE_AND_SPLIT(type)                    \
  REGISTER_KERNEL_BUILDER(Name("MonolithFusedReduceAndSplitGPU") \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("T"),        \
                          FusedReduceAndSplitGPU<type>)
TF_CALL_float(REGISTER_FUSED_REDUCE_AND_SPLIT);

REGISTER_OP("MonolithFusedReduceAndSplitGPU")
    .Input("splits: int32")
    .Input("updates: N * T")
    .Output("outputs: num_slices * T")
    .Attr("num_slices: int >= 1")
    .Attr("slice_dims: list(int)")
    .Attr("row_split_splits: list(int)")
    .Attr("T: type")
    .Attr("N: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int N;
      std::vector<int> slice_dims, row_split_splits;

      TF_RETURN_IF_ERROR(c->GetAttr("N", &N));
      TF_RETURN_IF_ERROR(c->GetAttr("slice_dims", &slice_dims));
      TF_RETURN_IF_ERROR(c->GetAttr("row_split_splits", &row_split_splits));

      // simple sanity checks
      int num_outputs = c->num_outputs();
      if (slice_dims.size() != num_outputs)
        return errors::InvalidArgument(
            "len(slice_dims) must equal to num_slices");

      int batch_size = row_split_splits[1] - row_split_splits[0] - 1;
      for (int i = 0; i < num_outputs; i++) {
        auto output_shape = c->MakeShape({batch_size, slice_dims[i]});
        c->set_output(i, output_shape);
      }
      return Status::OK();
    });

#define REGISTER_FUSED_SCATTER_ND_KERNEL_INDEX(type, index_type)      \
  REGISTER_KERNEL_BUILDER(Name("MonolithFusedSegmentSum")             \
                              .Device(DEVICE_GPU)                     \
                              .TypeConstraint<type>("T")              \
                              .TypeConstraint<index_type>("Tindices") \
                              .HostMemory("shape"),                   \
                          FusedSegmentSumGPU<type, index_type>)

#define REGISTER_FUSED_SCATTER_ND_KERNEL(type)         \
  REGISTER_FUSED_SCATTER_ND_KERNEL_INDEX(type, int32); \
  REGISTER_FUSED_SCATTER_ND_KERNEL_INDEX(type, int64);

TF_CALL_float(REGISTER_FUSED_SCATTER_ND_KERNEL);
// TF_CALL_GPU_NUMBER_TYPES(REGISTER_FUSED_SCATTER_ND_KERNEL);

#undef REGISTER_FUSED_SCATTER_ND_KERNEL
#undef REGISTER_FUSED_SCATTER_ND_KERNEL_INDEX

REGISTER_OP("MonolithFusedSegmentSum")
    .Input("indices: N * Tindices")
    .Input("updates: N * T")
    .Input("shape: N * Tindices")
    .Output("outputs: N * T")
    .Attr("T: type")
    .Attr("Tindices: {int32, int64}")
    .Attr("N: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int N;
      TF_RETURN_IF_ERROR(c->GetAttr("N", &N));
      for (int i = N - 1; i >= 0; --i) {
        shape_inference::ShapeHandle indices_shape;
        TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(i), 1, &indices_shape));
        shape_inference::ShapeHandle updates_shape;
        TF_RETURN_IF_ERROR(
            c->WithRankAtLeast(c->input(N + i), 1, &updates_shape));
        shape_inference::ShapeHandle output_shape;
        TF_RETURN_IF_ERROR(
            c->MakeShapeFromShapeTensor(2 * N + i, &output_shape));
        shape_inference::ShapeHandle
            expanded_indices_shape;  // mimic expand_dims(indices, -1)
        TF_RETURN_IF_ERROR(c->Concatenate(indices_shape, c->Vector(1),
                                          &expanded_indices_shape));
        TF_RETURN_IF_ERROR(shape_inference::ScatterNdShapeHelper(
            c, expanded_indices_shape, updates_shape,
            output_shape));  // set shape to output 0
        if (c->input_handle_shapes_and_types(0) == nullptr &&
            c->num_outputs() > 0) {
          c->set_output(i, c->output(0));
        }
      }
      return Status::OK();
    });

}  // namespace monolith_tf
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
