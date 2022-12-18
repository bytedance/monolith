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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/gpu_device_array.h"
#include "tensorflow/core/kernels/gpu_device_array_gpu.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace monolith_tf {

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
__global__ void FusedGatherKernel(
    const T* __restrict__ fused_embeddings,
    GpuDeviceArrayStruct<const int32*> input_ptr_data,
    GpuDeviceArrayStruct<T*> output_ptr_data,
    GpuDeviceArrayStruct<int32> embedding_dims_data,
    GpuDeviceArrayStruct<int32> offsets_data, int32 size) {
  const int32** input_ptrs = GetGpuDeviceArrayOnDevice(&input_ptr_data);
  T** output_ptrs = GetGpuDeviceArrayOnDevice(&output_ptr_data);

  int32* offsets = GetGpuDeviceArrayOnDevice(&offsets_data);
  int32* embedding_dims = GetGpuDeviceArrayOnDevice(&embedding_dims_data);

  // if using shared memory
  // Ref:
  // https://github.com/tensorflow/tensorflow/blob/v2.4.0/tensorflow/core/kernels/split_lib_gpu.cu.cc#L124
  GPU_DYNAMIC_SHARED_MEM_DECL(sizeof(int32), unsigned char, smem);
  int32* smem_offsets = reinterpret_cast<int32*>(smem);
  int32* smem_embedding_dims = smem_offsets + offsets_data.size;
  for (int x = threadIdx.x; x < offsets_data.size; x += blockDim.x) {
    smem_offsets[x] = offsets[x];
  }
  for (int x = threadIdx.x; x < embedding_dims_data.size; x += blockDim.x) {
    smem_embedding_dims[x] = embedding_dims[x];
  }
  __syncthreads();
  offsets = smem_offsets;
  embedding_dims = smem_embedding_dims;

  int i = 0;
  for (int32 idx : GpuGridRangeX<int32>(size)) {
    // safe offsets read: when idx == size - 1, i+1 == num_inputs
    // since num_inputs := number of merged slot < 100,
    // linear search would be sufficient here
    while (offsets[i + 1] <= idx) ++i;
    int32 local_idx = idx - offsets[i];
    int32 dim = embedding_dims[i];
    int j = local_idx / dim;
    int k = local_idx % dim;

    int32 emb_offset = input_ptrs[i][j];
    T* output_ptr = output_ptrs[i];
    *(output_ptr + j * dim + k) = ldg(fused_embeddings + emb_offset + k);
  }
}

template <typename T>
__global__ void FusedGatherGradKernel(
    T* output_ptr, GpuDeviceArrayStruct<const T*> input_ptr_data,
    GpuDeviceArrayStruct<const int32*> offset_ptr_data,
    GpuDeviceArrayStruct<int32> embedding_dims_data,
    GpuDeviceArrayStruct<int32> offsets_data, int32 size) {
  const T** input_ptrs = GetGpuDeviceArrayOnDevice(&input_ptr_data);
  const int32** offset_ptrs = GetGpuDeviceArrayOnDevice(&offset_ptr_data);

  int32* offsets = GetGpuDeviceArrayOnDevice(&offsets_data);
  int32* embedding_dims = GetGpuDeviceArrayOnDevice(&embedding_dims_data);

  // if using shared memory
  // Ref:
  // https://github.com/tensorflow/tensorflow/blob/v2.4.0/tensorflow/core/kernels/split_lib_gpu.cu.cc#L124
  GPU_DYNAMIC_SHARED_MEM_DECL(sizeof(int32), unsigned char, smem);
  int32* smem_offsets = reinterpret_cast<int32*>(smem);
  int32* smem_embedding_dims = smem_offsets + offsets_data.size;
  for (int x = threadIdx.x; x < offsets_data.size; x += blockDim.x) {
    smem_offsets[x] = offsets[x];
  }
  for (int x = threadIdx.x; x < embedding_dims_data.size; x += blockDim.x) {
    smem_embedding_dims[x] = embedding_dims[x];
  }
  __syncthreads();
  offsets = smem_offsets;
  embedding_dims = smem_embedding_dims;

  int i = 0;
  for (int32 idx : GpuGridRangeX<int32>(size)) {
    // safe offsets read: when idx == size - 1, i+1 == num_inputs
    // since num_inputs := number of merged slot < 100,
    // linear search would be sufficient here
    while (offsets[i + 1] <= idx) ++i;
    int32 local_idx = idx - offsets[i];
    int32 dim = embedding_dims[i];
    int j = local_idx / dim;
    int k = local_idx % dim;

    const int32 emb_offset = offset_ptrs[i][j];
    const T* input_ptr = input_ptrs[i];
    GpuAtomicAdd(output_ptr + emb_offset + k, *(input_ptr + j * dim + k));
  }
}

template <typename T>
struct SetZeroFunctor {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat out) {
    To32Bit(out).device(d) = To32Bit(out).constant(T(0));
  }
};

template <typename T>
class FusedGatherEmbeddingsByInputOpGPU : public OpKernel {
 public:
  explicit FusedGatherEmbeddingsByInputOpGPU(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("M", &num_of_inputs_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("embedding_dims", &embedding_dims_));
  }

  void Compute(OpKernelContext* ctx) override {
    auto fused_embeddings_flat = ctx->input(0).flat<T>();

    OpInputList inputs;
    OP_REQUIRES_OK(ctx, ctx->input_list("embedding_offsets", &inputs));
    OpOutputList outputs;
    OP_REQUIRES_OK(ctx, ctx->output_list("output", &outputs));

    DCHECK_EQ(num_of_inputs_, outputs.size());
    GpuDeviceArrayOnHost<const int32*> input_ptrs(ctx, num_of_inputs_);
    GpuDeviceArrayOnHost<T*> output_ptrs(ctx, outputs.size());
    OP_REQUIRES_OK(ctx, input_ptrs.Init());
    OP_REQUIRES_OK(ctx, output_ptrs.Init());

    GpuDeviceArrayOnHost<int32> embedding_dims(ctx, num_of_inputs_);
    OP_REQUIRES_OK(ctx, embedding_dims.Init());
    int32 offset = 0;
    GpuDeviceArrayOnHost<int32> offsets(ctx, num_of_inputs_ + 1);
    OP_REQUIRES_OK(ctx, offsets.Init());
    int smem_usage = sizeof(int32) * (num_of_inputs_ + 1 + num_of_inputs_);
    // smem: offsets + embedding_dims
    for (int i = 0; i < num_of_inputs_; ++i) {
      auto dim = embedding_dims_[i];
      embedding_dims.Set(i, dim);
      auto s = inputs[i].NumElements();  // == input[i].shape().dim_size(0)
      offsets.Set(i, offset);
      offset += s * dim;
      input_ptrs.Set(i, inputs[i].flat<int32>().data());
      TensorShape output_shape = inputs[i].shape();
      output_shape.AddDim(dim);
      Tensor* out;
      OP_REQUIRES_OK(ctx, outputs.allocate(i, output_shape, &out));
      output_ptrs.Set(i, out->flat<T>().data());
    }
    offsets.Set(num_of_inputs_, offset);  // offset val here is total workload
    OP_REQUIRES_OK(ctx, offsets.Finalize());
    OP_REQUIRES_OK(ctx, input_ptrs.Finalize());
    OP_REQUIRES_OK(ctx, output_ptrs.Finalize());
    OP_REQUIRES_OK(ctx, embedding_dims.Finalize());

    GPUDevice gpu_device = ctx->eigen_device<GPUDevice>();

    // We use a 2D LaunchConfig here to make thread (x, y) of every
    // input tensor y better benefit from the ldg local cache read
    // for multiple x of x + n * grid_stride.
    // >>> auto config = GetGpu2DLaunchConfig(max_input_size, num_of_inputs_,
    //                                        gpu_device);
    //
    // However, across inputs the distribution of elements thus thread workload
    // can be imbalanced in this implementation.
    //
    /// One alternative implmentation for this Op is based on ComputeAsync +
    // Multiple Kernel Calls, for example similar to
    // https://github.com/tensorflow/tensorflow/blob/v2.4.0/tensorflow/core/kernels/dynamic_partition_op_gpu.cu.cc#L454-L469
    //
    // The chosen implementation is to distribute the output workload balanced
    // on threads,
    // while searching the idx input bucket to which the output val belongs to.
    auto config = GetGpuLaunchConfig(offset, gpu_device);
    GpuLaunchKernel(
        FusedGatherKernel<T>, config.block_count, config.thread_per_block,
        /*shared_memory_size_bytes=*/smem_usage, gpu_device.stream(),
        fused_embeddings_flat.data(), input_ptrs.data(), output_ptrs.data(),
        embedding_dims.data(), offsets.data(), offset);
  }

 private:
  int num_of_inputs_;
  std::vector<int32> embedding_dims_;
};

template <typename T>
class FusedGatherEmbeddingsByInputGradientOpGPU : public OpKernel {
 public:
  explicit FusedGatherEmbeddingsByInputGradientOpGPU(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("M", &num_of_inputs_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("embedding_dims", &embedding_dims_));
  }

  void Compute(OpKernelContext* ctx) override {
    GPUDevice gpu_device = ctx->eigen_device<GPUDevice>();

    OpInputList inputs;
    OP_REQUIRES_OK(ctx, ctx->input_list("grads", &inputs));
    OpInputList embedding_offsets;
    OP_REQUIRES_OK(ctx,
                   ctx->input_list("embedding_offsets", &embedding_offsets));

    GpuDeviceArrayOnHost<const T*> input_ptrs(ctx, num_of_inputs_);
    GpuDeviceArrayOnHost<const int32*> emb_offset_ptrs(ctx, num_of_inputs_);
    OP_REQUIRES_OK(ctx, input_ptrs.Init());
    OP_REQUIRES_OK(ctx, emb_offset_ptrs.Init());

    GpuDeviceArrayOnHost<int32> embedding_dims(ctx, num_of_inputs_);
    OP_REQUIRES_OK(ctx, embedding_dims.Init());

    int32 offset = 0;
    GpuDeviceArrayOnHost<int32> offsets(ctx,
                                        num_of_inputs_ + 1);  // input_offsets
    OP_REQUIRES_OK(ctx, offsets.Init());
    int smem_usage = sizeof(int32) * (num_of_inputs_ + 1 + num_of_inputs_);
    // smem: offsets + embedding_dims
    for (int i = 0; i < num_of_inputs_; ++i) {
      input_ptrs.Set(i, inputs[i].flat<T>().data());
      emb_offset_ptrs.Set(i, embedding_offsets[i].flat<int32>().data());
      auto s = embedding_offsets[i].NumElements();
      auto dim = embedding_dims_[i];
      embedding_dims.Set(i, dim);
      offsets.Set(i, offset);
      offset += s * dim;
    }
    offsets.Set(num_of_inputs_, offset);  // offset val here is total workload
    OP_REQUIRES_OK(ctx, offsets.Finalize());
    OP_REQUIRES_OK(ctx, input_ptrs.Finalize());
    OP_REQUIRES_OK(ctx, emb_offset_ptrs.Finalize());
    OP_REQUIRES_OK(ctx, embedding_dims.Finalize());

    int32 fused_embeddings_size = ctx->input(0).scalar<int32>().data()[0];
    Tensor* output_tensor;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, TensorShape({fused_embeddings_size}),
                                        &output_tensor));

    SetZeroFunctor<T> zero_functor;
    auto output = output_tensor->flat<T>();
    zero_functor(gpu_device, output);

    auto config = GetGpuLaunchConfig(offset, gpu_device);
    GpuLaunchKernel(
        FusedGatherGradKernel<T>, config.block_count, config.thread_per_block,
        /*shared_memory_size_bytes=*/smem_usage, gpu_device.stream(),
        output.data(), input_ptrs.data(), emb_offset_ptrs.data(),
        embedding_dims.data(), offsets.data(), offset);
  }

 private:
  int num_of_inputs_;
  std::vector<int32> embedding_dims_;
};

REGISTER_KERNEL_BUILDER(
    Name("MonolithFusedGatherEmbeddingsByInput").Device(DEVICE_GPU),
    FusedGatherEmbeddingsByInputOpGPU<float>);

REGISTER_KERNEL_BUILDER(Name("MonolithFusedGatherEmbeddingsByInputGradient")
                            .Device(DEVICE_GPU)
                            .HostMemory("fused_embeddings_size"),
                        FusedGatherEmbeddingsByInputGradientOpGPU<float>);

}  // namespace monolith_tf
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
