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
namespace monolith {

namespace {  // Seperate for CUDA Kernel Def

template <int BLOCK_THREADS>
__global__ void globalReduceSum(
    GpuDeviceArrayStruct<const float*> input_ptrs_da,
    GpuDeviceArrayStruct<int> offsets_da, float* out, int size) {
  const float** input_ptrs = GetGpuDeviceArrayOnDevice(&input_ptrs_da);
  int* offsets = GetGpuDeviceArrayOnDevice(&offsets_da);

  // if using shared memory
  // Ref:
  // https://github.com/tensorflow/tensorflow/blob/v2.4.0/tensorflow/core/kernels/concat_lib_gpu_impl.cu.cc#L73
  GPU_DYNAMIC_SHARED_MEM_DECL(sizeof(int), unsigned char, smem);
  int* smem_offsets = reinterpret_cast<int*>(smem);
  for (int x = threadIdx.x; x < offsets_da.size; x += blockDim.x) {
    smem_offsets[x] = offsets[x];
  }
  __syncthreads();
  offsets = smem_offsets;

  float thread_sum = 0;
  int i = 0;
  GPU_1D_KERNEL_LOOP(idx, size) {
    // safe offsets read: when idx == size - 1, i+1 == num_inputs
    while (offsets[i + 1] <= idx) ++i;
    int j = idx - offsets[i];
    float v = ldg(input_ptrs[i] + j);
    thread_sum += v * v;  // l2
  }
  // thread reduce sum to block reduce sum
  typedef gpuprim::BlockReduce<float, BLOCK_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  float block_sum = BlockReduce(temp_storage).Sum(thread_sum);
  __syncthreads();
  if (threadIdx.x == 0)
    // block reduce sum to global reduce sum
    atomicAdd(out, block_sum);
}

}  // namespace

template <typename Device>
struct GlobalReduceImpl {
  static void Compute(OpKernelContext* context,
                      const std::vector<const float*>& input_ptrs,
                      const std::vector<int>& input_lens,
                      const std::vector<float*>& output_ptrs, float global_norm,
                      float clip_norm);
};

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
struct SetZeroFunctor {
  void operator()(const GPUDevice& d, typename TTypes<T>::Scalar out) {
    To32Bit(out).device(d) = To32Bit(out).constant(T(0));
  }
};

template <>
struct GlobalReduceImpl<GPUDevice> {
  static void Compute(OpKernelContext* context,
                      const std::vector<const float*>& input_ptrs,
                      const std::vector<int>& input_lens,
                      TTypes<float>::Scalar output) {
    GPUDevice gpu_device = context->eigen_device<GPUDevice>();
    int num_inputs = input_ptrs.size();

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
      offset += input_lens[i];
    }
    offsets.Set(num_inputs, offset);  // offset val here is total workload
    OP_REQUIRES_OK(context, offsets.Finalize());

    SetZeroFunctor<float> zero_functor;
    zero_functor(gpu_device, output);

    const int thread_per_block = 1024;  // const int for globalReduceSum
    const int physical_thread_count =
        std::min(gpu_device.getNumGpuMultiProcessors() *
                     gpu_device.maxGpuThreadsPerMultiProcessor(),
                 offset);
    const int block_count =
        std::min(DivUp(physical_thread_count, thread_per_block),
                 gpu_device.getNumGpuMultiProcessors());
    TF_CHECK_OK(GpuLaunchKernel(globalReduceSum<thread_per_block>, block_count,
                                thread_per_block, smem_usage,
                                gpu_device.stream(), input_ptrs_da.data(),
                                offsets.data(), output.data(), offset));
  }
};

template <typename Device>
class GlobalReduce : public OpKernel {
 public:
  explicit GlobalReduce(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("N", &num_inputs_));
  }

  void Compute(OpKernelContext* context) override {
    VLOG(1) << "In GlobalReduce Computation";

    auto num_inputs = context->num_inputs();
    std::vector<const float*> input_ptrs(num_inputs);
    std::vector<int> input_lens(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      input_ptrs[i] = context->input(i).flat<float>().data();
      input_lens[i] = context->input(i).NumElements();
    }

    Tensor* out;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &out));
    GlobalReduceImpl<Device>::Compute(context, input_ptrs, input_lens,
                                      out->scalar<float>().data());
  }

 private:
  int num_inputs_;
};

REGISTER_OP("GlobalL2Reduce")
    .Input("grad_list: N * float")
    .Output("global_norm: float")
    .Attr("N: int")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(Name("GlobalL2Reduce").Device(DEVICE_GPU),
                        GlobalReduce<GPUDevice>);

}  // namespace monolith
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
