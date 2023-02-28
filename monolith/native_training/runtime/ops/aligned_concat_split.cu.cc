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
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/gpu_device_array.h"
#include "tensorflow/core/kernels/gpu_device_array_gpu.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace monolith_tf {

__global__ void flat_concat(
    GpuDeviceArrayStruct<const float*> input_ptrs_da,  // length = 2N+1
    int total, const float* _scale, float* out) {
  float scale = *_scale;
  auto _input_ptrs = GetGpuDeviceArrayOnDevice(&input_ptrs_da);
  extern __shared__ const float* input_ptrs[];
  for (int i = threadIdx.x; i < input_ptrs_da.size; i += blockDim.x)
    input_ptrs[i] = _input_ptrs[i];
  __syncthreads();
  auto N = (input_ptrs_da.size - 1) / 2;
  auto sizes = reinterpret_cast<const int*>(input_ptrs + N);
  auto offsets = sizes + N;
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto stride = blockDim.x * gridDim.x;
  int work_id = 0;
  for (int id = tid; id < total; id += stride) {
    while (offsets[work_id + 1] <= id) work_id++;

    int i = id - offsets[work_id];
    if (i < sizes[work_id]) {
      out[id] = input_ptrs[work_id][i] * scale;
    } else {
      out[id] = 0.0f;
    }
  }
}

// Flatten each input and then concatenate them. This op also ensures that the
// start position of each input in the concat output is suitably aligned (as per
// Tensorflow/Eigen's requirement), so that we can perform a split without
// copying the underlying memory
class AlignedFlatConcat : public OpKernel {
 public:
  explicit AlignedFlatConcat(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("N", &N_));
  }

  void Compute(OpKernelContext* context) override {
    const auto& gpu_device = context->eigen_gpu_device();
    static_assert(sizeof(int) * 2 == sizeof(const float*));
    GpuDeviceArrayOnHost<const float*> input_ptrs(context, 2 * N_ + 1);
    OP_REQUIRES_OK(context, input_ptrs.Init());

    FusedAlignedOutputAllocator<EIGEN_MAX_ALIGN_BYTES / sizeof(float)>
        fao_alloc(context);
    std::vector<int> offsets_sizes(2 * N_ + 2);
    for (int i = 0; i < N_; ++i) {
      auto sz = context->input(i).NumElements();
      input_ptrs.Set(i, context->input(i).flat<float>().data());
      offsets_sizes[i] = sz;
      offsets_sizes[N_ + i] = fao_alloc.get_aligned_total();
      fao_alloc.add_slice(sz);
    }
    int total = fao_alloc.get_aligned_total();
    offsets_sizes[2 * N_] = total;
    auto data = reinterpret_cast<const float**>(offsets_sizes.data());
    for (int i = 0; i <= N_; ++i) input_ptrs.Set(N_ + i, data[i]);

    OP_REQUIRES_OK(context, input_ptrs.Finalize());
    OP_REQUIRES(context, 2 * N_ + 1 <= 2048,
                errors::Unknown("Total size of ", 2 * N_ + 1,
                                " is greater than 2048 so is not supported. "
                                "Please contact the developers."));

    Tensor* out;
    OP_REQUIRES_OK(context, context->allocate_output(0, {total}, &out));
    auto config = GetGpuLaunchConfig(total, gpu_device);
    TF_CHECK_OK(GpuLaunchKernel(
        flat_concat, config.block_count, config.thread_per_block,
        sizeof(const float*) * (2 * N_ + 1), gpu_device.stream(),
        input_ptrs.data(), total, context->input(N_).flat<float>().data(),
        out->flat<float>().data()));
  }

 private:
  int N_;
};

class AlignedFlatSplit : public OpKernel {
 public:
  explicit AlignedFlatSplit(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("N", &N_));
  }
  void Compute(OpKernelContext* context) override {
    FusedAlignedOutputAllocator<EIGEN_MAX_ALIGN_BYTES / sizeof(float)>
        fao_alloc(context);
    const auto& flat = context->input(N_);
    for (int i = 0; i < N_; ++i) {
      context->set_output(i,
                          fao_alloc.get_slice(context->input(i).shape(), flat));
    }
  }

 private:
  int N_;
};
REGISTER_OP("MonolithAlignedFlatConcat")
    .Input("inputs: N * float")
    .Input("scale: float")
    .Output("concat: float")
    .Attr("N: int")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->Vector(c->UnknownDim()));
      return tensorflow::Status::OK();
    });

REGISTER_KERNEL_BUILDER(
    Name("MonolithAlignedFlatConcat").Device(DEVICE_GPU),
    AlignedFlatConcat);

REGISTER_OP("MonolithAlignedFlatSplit")
    .Input("inputs: N * float")  // for shape inference only, data not used
    .Input("flat: float")
    .Output("concat: N * float")
    .Attr("N: int")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      for (int i = 0; i < c->num_inputs() - 1; ++i) {
        c->set_output(i, c->input(i));
      }
      return tensorflow::Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name("MonolithAlignedFlatSplit").Device(DEVICE_GPU),
                        AlignedFlatSplit);

}  // namespace monolith_tf
}  // namespace tensorflow

#endif
