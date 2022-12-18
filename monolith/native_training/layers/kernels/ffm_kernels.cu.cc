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
#include "monolith/native_training/layers/kernels/ffm_kernels.h"
#include <string>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/gpu_device_array.h"
#include "tensorflow/core/kernels/gpu_device_array_gpu.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"

namespace tensorflow {
namespace monolith_tf {

using GPUDevice = Eigen::GpuDevice;

__global__ void FFMKernelMultiply(TTypes<float>::ConstMatrix left_matrix,
                                  int left_feat_num,
                                  TTypes<float>::ConstMatrix right_matrix,
                                  int right_feat_num, int batch_size,
                                  int dim_size, TTypes<float>::Matrix output) {
  GPU_1D_KERNEL_LOOP(b, batch_size) {
    for (int l = 0; l < left_feat_num; ++l) {
      int l_idx = l * dim_size;
      for (int r = 0; r < right_feat_num; ++r) {
        int r_idx = r * dim_size;
        int o_idx = (l * right_feat_num + r) * dim_size;
        for (int k = 0; k < dim_size; ++k) {
          output(b, o_idx + k) =
              left_matrix(b, l_idx + k) * right_matrix(b, r_idx + k);
        }
      }
    }
  }
}

__global__ void FFMKernelDot(TTypes<float>::ConstMatrix left_matrix,
                             int left_feat_num,
                             TTypes<float>::ConstMatrix right_matrix,
                             int right_feat_num, int batch_size, int dim_size,
                             TTypes<float>::Matrix output) {
  GPU_1D_KERNEL_LOOP(b, batch_size) {
    for (int j = 0; j < output.dimension(1); ++j) {
      output(b, j) = 0;
    }
  }
  __syncthreads();

  GPU_1D_KERNEL_LOOP(b, batch_size) {
    for (int l = 0; l < left_feat_num; ++l) {
      int l_idx = l * dim_size;
      for (int r = 0; r < right_feat_num; ++r) {
        int r_idx = r * dim_size;
        int o_idx = l * right_feat_num + r;
        for (int k = 0; k < dim_size; ++k) {
          output(b, o_idx) +=
              left_matrix(b, l_idx + k) * right_matrix(b, r_idx + k);
        }
      }
    }
  }
}

template <>
struct FFMImpl<GPUDevice> {
  static void Compute(OpKernelContext *ctx, const std::string &int_type,
                      TTypes<float>::ConstMatrix left_matrix, int left_feat_num,
                      TTypes<float>::ConstMatrix right_matrix,
                      int right_feat_num, int batch_size, int dim_size,
                      TTypes<float>::Matrix output) {
    Eigen::GpuDevice gpu_device = ctx->eigen_device<Eigen::GpuDevice>();
    auto config = GetGpuLaunchConfig(batch_size, gpu_device);

    if (int_type == "dot") {
      TF_CHECK_OK(GpuLaunchKernel(
          FFMKernelDot, config.block_count, config.thread_per_block, 0,
          gpu_device.stream(), left_matrix, left_feat_num, right_matrix,
          right_feat_num, batch_size, dim_size, output));
    } else {
      TF_CHECK_OK(GpuLaunchKernel(
          FFMKernelMultiply, config.block_count, config.thread_per_block, 0,
          gpu_device.stream(), left_matrix, left_feat_num, right_matrix,
          right_feat_num, batch_size, dim_size, output));
    }
  }
};

__global__ void FFMGradKernelMultiply(
    TTypes<float>::ConstMatrix grad_matrix, int grad_feat_num,
    TTypes<float>::ConstMatrix left_matrix, int left_feat_num,
    TTypes<float>::ConstMatrix right_matrix, int right_feat_num, int batch_size,
    int dim_size, TTypes<float>::Matrix left_grad_matrix,
    TTypes<float>::Matrix right_grad_matrix) {
  GPU_1D_KERNEL_LOOP(b, batch_size) {
    for (int g = 0; g < left_feat_num * dim_size; ++g) {
      left_grad_matrix(b, g) = 0;
    }
    for (int g = 0; g < right_feat_num * dim_size; ++g) {
      right_grad_matrix(b, g) = 0;
    }
  }
  __syncthreads();

  GPU_1D_KERNEL_LOOP(b, batch_size) {
    for (int g = 0; g < grad_feat_num; ++g) {
      int l_idx = (g / right_feat_num) * dim_size;
      int r_idx = (g % right_feat_num) * dim_size;

      int g_idx = g * dim_size;
      for (int k = 0; k < dim_size; ++k) {
        left_grad_matrix(b, l_idx + k) +=
            grad_matrix(b, g_idx + k) * right_matrix(b, r_idx + k);

        right_grad_matrix(b, r_idx + k) +=
            grad_matrix(b, g_idx + k) * left_matrix(b, l_idx + k);
      }
    }
  }
}

__global__ void FFMGradKernelDot(
    TTypes<float>::ConstMatrix grad_matrix, int grad_feat_num,
    TTypes<float>::ConstMatrix left_matrix, int left_feat_num,
    TTypes<float>::ConstMatrix right_matrix, int right_feat_num, int batch_size,
    int dim_size, TTypes<float>::Matrix left_grad_matrix,
    TTypes<float>::Matrix right_grad_matrix) {
  GPU_1D_KERNEL_LOOP(b, batch_size) {
    for (int g = 0; g < left_feat_num * dim_size; ++g) {
      left_grad_matrix(b, g) = 0;
    }
    for (int g = 0; g < right_feat_num * dim_size; ++g) {
      right_grad_matrix(b, g) = 0;
    }
  }
  __syncthreads();

  GPU_1D_KERNEL_LOOP(b, batch_size) {
    for (int g = 0; g < grad_feat_num; ++g) {
      int l_idx = (g / right_feat_num) * dim_size;
      int r_idx = (g % right_feat_num) * dim_size;

      for (int k = 0; k < dim_size; ++k) {
        left_grad_matrix(b, l_idx + k) +=
            grad_matrix(b, g) * right_matrix(b, r_idx + k);

        right_grad_matrix(b, r_idx + k) +=
            grad_matrix(b, g) * left_matrix(b, l_idx + k);
      }
    }
  }
}

template <>
struct FFMGradImpl<GPUDevice> {
  static void Compute(OpKernelContext *ctx, const std::string &int_type,
                      TTypes<float>::ConstMatrix grad_matrix, int grad_feat_num,
                      TTypes<float>::ConstMatrix left_matrix, int left_feat_num,
                      TTypes<float>::ConstMatrix right_matrix,
                      int right_feat_num, int batch_size, int dim_size,
                      TTypes<float>::Matrix left_grad_matrix,
                      TTypes<float>::Matrix right_grad_matrix) {
    Eigen::GpuDevice gpu_device = ctx->eigen_device<Eigen::GpuDevice>();
    auto config = GetGpuLaunchConfig(batch_size, gpu_device);

    if (int_type == "dot") {
      TF_CHECK_OK(GpuLaunchKernel(
          FFMGradKernelDot, config.block_count, config.thread_per_block, 0,
          gpu_device.stream(), grad_matrix, grad_feat_num, left_matrix,
          left_feat_num, right_matrix, right_feat_num, batch_size, dim_size,
          left_grad_matrix, right_grad_matrix));
    } else {
      TF_CHECK_OK(GpuLaunchKernel(
          FFMGradKernelMultiply, config.block_count, config.thread_per_block, 0,
          gpu_device.stream(), grad_matrix, grad_feat_num, left_matrix,
          left_feat_num, right_matrix, right_feat_num, batch_size, dim_size,
          left_grad_matrix, right_grad_matrix));
    }
  }
};

namespace {

REGISTER_KERNEL_BUILDER(Name("FFM").Device(DEVICE_GPU), FFMOp<GPUDevice>)

REGISTER_KERNEL_BUILDER(Name("FFMGrad").Device(DEVICE_GPU),
                        FFMGradOp<GPUDevice>)
}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
