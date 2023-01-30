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

#include "monolith/native_training/runtime/ops/clip_by_global_norm.h"

#include "monolith/native_training/runtime/ops/alloc_utils.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/gpu_device_array.h"
#include "tensorflow/core/kernels/gpu_device_array_gpu.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace monolith {

namespace {
__global__ void element_wise_mul(
    GpuDeviceArrayStruct<const float*> input_ptrs_da,
    GpuDeviceArrayStruct<float*> output_ptrs_da,
    GpuDeviceArrayStruct<int> offsets_da, int size, float scale) {
  const float** input_ptrs = GetGpuDeviceArrayOnDevice(&input_ptrs_da);
  int* offsets = GetGpuDeviceArrayOnDevice(&offsets_da);
  float** output_ptrs = GetGpuDeviceArrayOnDevice(&output_ptrs_da);

  // if using shared memory
  // Ref:
  // https://github.com/tensorflow/tensorflow/blob/v2.4.0/tensorflow/core/kernels/split_lib_gpu.cu.cc#L124
  GPU_DYNAMIC_SHARED_MEM_DECL(sizeof(int), unsigned char, smem);
  int* smem_offsets = reinterpret_cast<int*>(smem);
  for (int x = threadIdx.x; x < offsets_da.size; x += blockDim.x) {
    smem_offsets[x] = offsets[x];
  }
  __syncthreads();
  offsets = smem_offsets;

  int i = 0;
  GPU_1D_KERNEL_LOOP(idx, size) {
    // safe offsets read: when idx == size - 1, i+1 == num_inputs
    while (offsets[i + 1] <= idx) ++i;
    int j = idx - offsets[i];
    output_ptrs[i][j] = ldg(input_ptrs[i] + j) * scale;
  }
}
}  // namespace

typedef Eigen::GpuDevice GPUDevice;

template <>
struct ClipByGlobalNormImpl<GPUDevice> {
  static void Compute(OpKernelContext* context, float scale) {
    const auto& gpu_device = context->eigen_gpu_device();
    auto N_ = context->num_inputs() - 2;
    GpuDeviceArrayOnHost<const float*> input_ptrs_da(context, N_);
    GpuDeviceArrayOnHost<int> offsets(context, N_ + 1);
    OP_REQUIRES_OK(context, input_ptrs_da.Init());
    OP_REQUIRES_OK(context, offsets.Init());
    monolith_tf::FusedAlignedOutputAllocator<EIGEN_MAX_ALIGN_BYTES /
                                             sizeof(float)>
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

    GpuDeviceArrayOnHost<float*> output_ptrs_da(context, N_);
    OP_REQUIRES_OK(context, output_ptrs_da.Init());
    fao_alloc.allocate(DT_FLOAT);
    for (int i = 0; i < N_; ++i) {
      auto t = fao_alloc.get_slice(context->input(i).shape());
      output_ptrs_da.Set(i, t.flat<float>().data());
      context->set_output(i, std::move(t));
    }
    OP_REQUIRES_OK(context, output_ptrs_da.Finalize());
    auto config = GetGpuLaunchConfig(total, gpu_device);

    const int smem_usage = sizeof(int) * (N_ + 1);
    TF_CHECK_OK(GpuLaunchKernel(
        element_wise_mul, config.block_count, config.thread_per_block,
        smem_usage, gpu_device.stream(), input_ptrs_da.data(),
        output_ptrs_da.data(), offsets.data(), total, scale));
  }
};

REGISTER_KERNEL_BUILDER(Name("MonolithClipByGlobalNorm")
                            .Device(DEVICE_GPU)
                            .HostMemory("global_norm")
                            .HostMemory("clip_norm"),
                        ClipByGlobalNorm<GPUDevice>);
}  // namespace monolith
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
