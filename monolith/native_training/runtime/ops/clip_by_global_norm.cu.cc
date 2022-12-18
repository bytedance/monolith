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
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/gpu_device_array.h"
#include "tensorflow/core/kernels/gpu_device_array_gpu.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

#include "monolith/native_training/runtime/ops/clip_by_global_norm.h"

namespace tensorflow {
namespace monolith {

namespace {
__global__ void element_wise_mul(
    GpuDeviceArrayStruct<const float*> input_ptrs_da,
    GpuDeviceArrayStruct<int> offsets_da,
    GpuDeviceArrayStruct<float*> output_ptrs_da, int size, float scale) {
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
  static void Compute(OpKernelContext* context,
                      const std::vector<const float*>& input_ptrs,
                      const std::vector<int>& input_lens,
                      const std::vector<float*>& output_ptrs, float scale) {
    GPUDevice gpu_device = context->eigen_device<GPUDevice>();
    int num_inputs = input_ptrs.size();  // #inputs == #outputs

    GpuDeviceArrayOnHost<const float*> input_ptrs_da(context, num_inputs);
    OP_REQUIRES_OK(context, input_ptrs_da.Init());
    for (int i = 0; i < num_inputs; ++i) {
      input_ptrs_da.Set(i, input_ptrs[i]);
    }
    OP_REQUIRES_OK(context, input_ptrs_da.Finalize());

    int offset = 0;
    GpuDeviceArrayOnHost<int> offsets(context, num_inputs + 1);
    int smem_usage = sizeof(int) * (num_inputs + 1);
    OP_REQUIRES_OK(context, offsets.Init());
    for (int i = 0; i < num_inputs; ++i) {
      offsets.Set(i, offset);
      offset += input_lens[i];  // * suffix_dim_size(1)
    }
    offsets.Set(num_inputs, offset);  // offset val here is total workload
    OP_REQUIRES_OK(context, offsets.Finalize());

    GpuDeviceArrayOnHost<float*> output_ptrs_da(context, num_inputs);
    OP_REQUIRES_OK(context, output_ptrs_da.Init());
    for (int i = 0; i < num_inputs; ++i) {
      output_ptrs_da.Set(i, output_ptrs[i]);
    }
    OP_REQUIRES_OK(context, output_ptrs_da.Finalize());

    auto config = GetGpuLaunchConfig(offset, gpu_device);
    TF_CHECK_OK(GpuLaunchKernel(
        element_wise_mul, config.block_count, config.thread_per_block,
        smem_usage, gpu_device.stream(), input_ptrs_da.data(), offsets.data(),
        output_ptrs_da.data(), offset, scale));
  }
};

REGISTER_KERNEL_BUILDER(Name("MonolithClipByGlobalNorm")
                            .Device(DEVICE_GPU)
                            .HostMemory("global_norm"),
                        ClipByGlobalNorm<GPUDevice>);
}  // namespace monolith
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
