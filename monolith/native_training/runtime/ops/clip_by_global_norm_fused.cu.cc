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

#ifndef MONOLITH_NATIVE_TRAINING_RUNTIME_OPS_CLIP_BY_GLOBAL_NORM_IN_PLACE_CU
#define MONOLITH_NATIVE_TRAINING_RUNTIME_OPS_CLIP_BY_GLOBAL_NORM_IN_PLACE_CU
#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "monolith/native_training/runtime/ops/alloc_utils.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/gpu_device_array.h"
#include "tensorflow/core/kernels/gpu_device_array_gpu.h"
#include "tensorflow/core/kernels/gpu_prim.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace monolith_tf {
namespace {
template <int BLOCK_THREADS>
__global__ void globalReduceSum(
    GpuDeviceArrayStruct<const float*> input_ptrs_da,
    GpuDeviceArrayStruct<int> offsets_da, float* out, int size) {
  const float** input_ptrs = GetGpuDeviceArrayOnDevice(&input_ptrs_da);
  int* offsets = GetGpuDeviceArrayOnDevice(&offsets_da);

  extern __shared__ int smem[];
  for (int x = threadIdx.x; x < offsets_da.size; x += blockDim.x) {
    smem[x] = offsets[x];
  }
  __syncthreads();
  offsets = smem;

  float thread_sum = 0;
  int i = 0;
  GPU_1D_KERNEL_LOOP(idx, size) {
    // safe offsets read: when idx == size - 1, i+1 == N_
    while (offsets[i + 1] <= idx) ++i;
    int j = idx - offsets[i];
    float v = ldg(input_ptrs[i] + j);
    thread_sum += v * v;  // l2
  }
  // thread reduce sum to block reduce sum
  typedef gpuprim::BlockReduce<float, BLOCK_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  float block_sum = BlockReduce(temp_storage).Sum(thread_sum);
  if (threadIdx.x == 0)
    // block reduce sum to global reduce sum
    atomicAdd(out, block_sum);
}

__global__ void element_wise_mul(
    GpuDeviceArrayStruct<const float*> input_ptrs_da,
    GpuDeviceArrayStruct<float*> output_ptrs_da,
    GpuDeviceArrayStruct<int> offsets_da, int size, float scale) {
  const float** input_ptrs = GetGpuDeviceArrayOnDevice(&input_ptrs_da);
  int* offsets = GetGpuDeviceArrayOnDevice(&offsets_da);
  float** output_ptrs = GetGpuDeviceArrayOnDevice(&output_ptrs_da);

  extern __shared__ int smem[];
  for (int x = threadIdx.x; x < offsets_da.size; x += blockDim.x) {
    smem[x] = offsets[x];
  }
  __syncthreads();
  offsets = smem;

  int i = 0;
  GPU_1D_KERNEL_LOOP(idx, size) {
    // safe offsets read: when idx == size - 1, i+1 == num_inputs
    while (offsets[i + 1] <= idx) ++i;
    int j = idx - offsets[i];
    output_ptrs[i][j] = ldg(input_ptrs[i] + j) * scale;
  }
}
}  // namespace

class ClipByGlobalNormFused : public OpKernel {
 public:
  explicit ClipByGlobalNormFused(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("N", &N_));
  }

  void Compute(OpKernelContext* context) override {
    const auto& gpu_device = context->eigen_gpu_device();
    GpuDeviceArrayOnHost<const float*> input_ptrs_da(context, N_);
    GpuDeviceArrayOnHost<int> offsets(context, N_ + 1);
    OP_REQUIRES_OK(context, input_ptrs_da.Init());
    OP_REQUIRES_OK(context, offsets.Init());
    FusedAlignedOutputAllocator<EIGEN_MAX_ALIGN_BYTES / sizeof(float)>
        fao_alloc(context);
    for (int i = 0; i < N_; ++i) {
      input_ptrs_da.Set(i, context->input(i).flat<float>().data());
      offsets.Set(i, fao_alloc.get_unaligned_total());
      fao_alloc.add_slice(context->input(i).NumElements());
    }
    int total = fao_alloc.get_unaligned_total();
    offsets.Set(N_, total);

    OP_REQUIRES_OK(context, input_ptrs_da.Finalize());
    OP_REQUIRES_OK(context, offsets.Finalize());

    Tensor d_norm;
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, {}, &d_norm));
    gpu_device.memset(d_norm.data(), 0, sizeof(float));

    constexpr int block_sz = 1024;
    const int smem_usage = sizeof(int) * (N_ + 1);
    TF_CHECK_OK(GpuLaunchKernel(
        globalReduceSum<block_sz>,
        std::min(gpu_device.maxGpuThreadsPerMultiProcessor() / block_sz, 1) *
            gpu_device.getNumGpuMultiProcessors(),
        block_sz, smem_usage, gpu_device.stream(), input_ptrs_da.data(),
        offsets.data(), d_norm.flat<float>().data(), total));

    // async kernel launch above can hide some latency of the code below until
    // synchronize
    Tensor* h_norm_out;
    OP_REQUIRES_OK(context, context->allocate_output(N_, {}, &h_norm_out));
    gpu_device.memcpyDeviceToHost(h_norm_out->data(), d_norm.data(),
                                  sizeof(float));

    GpuDeviceArrayOnHost<float*> output_ptrs_da(context, N_);
    OP_REQUIRES_OK(context, output_ptrs_da.Init());

    float clip_norm = context->input(N_).scalar<float>()();
    fao_alloc.allocate(DT_FLOAT);
    for (int i = 0; i < N_; ++i) {
      auto t = fao_alloc.get_slice(context->input(i).shape());
      output_ptrs_da.Set(i, t.flat<float>().data());
      // if this ends up unused, it will be overwritten
      context->set_output(i, std::move(t));
    }
    OP_REQUIRES_OK(context, output_ptrs_da.Finalize());
    auto config = GetGpuLaunchConfig(total, gpu_device);

    gpu_device.synchronize();

    float global_norm = std::sqrt(h_norm_out->scalar<float>()());
    if (global_norm > clip_norm) {
      TF_CHECK_OK(GpuLaunchKernel(element_wise_mul, config.block_count,
                                  config.thread_per_block, smem_usage,
                                  gpu_device.stream(), input_ptrs_da.data(),
                                  output_ptrs_da.data(), offsets.data(), total,
                                  clip_norm / global_norm));
    } else {
      for (int i = 0; i < N_; ++i) {
        *context->mutable_output(i) = context->input(i);
      }
    }
  }

 private:
  int N_;
};

REGISTER_OP("MonolithClipByGlobalNormFused")
    .Input("grad_list: N * float")
    .Input("clip_norm: float")
    .Output("clipped: N * float")
    .Output("global_norm: float")
    .Attr("N: int")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      for (int i = 0; i < c->num_inputs(); ++i) {
        c->set_output(i, c->input(i));
      }
      return tensorflow::Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name("MonolithClipByGlobalNormFused")
                            .Device(DEVICE_GPU)
                            .HostMemory("global_norm")
                            .HostMemory("clip_norm"),
                        ClipByGlobalNormFused);

}  // namespace monolith_tf
}  // namespace tensorflow

#endif
#endif  // MONOLITH_NATIVE_TRAINING_RUNTIME_OPS_CLIP_BY_GLOBAL_NORM_IN_PLACE_CU
