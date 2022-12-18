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

#include "monolith/native_training/layers/kernels/ffm_kernels.h"
#include <string>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace monolith_tf {

using CPUDevice = Eigen::ThreadPoolDevice;

template <>
struct FFMImpl<CPUDevice> {
  static void Compute(OpKernelContext *ctx, const std::string &int_type,
                      TTypes<float>::ConstMatrix left_matrix, int left_feat_num,
                      TTypes<float>::ConstMatrix right_matrix,
                      int right_feat_num, int batch_size, int dim_size,
                      TTypes<float>::Matrix output) {
    output.setZero();

    for (int l = 0; l < left_feat_num; ++l) {
      int l_idx = l * dim_size;
      for (int r = 0; r < right_feat_num; ++r) {
        int r_idx = r * dim_size;
        if (int_type == "dot") {
          int o_idx = l * right_feat_num + r;
          for (int b = 0; b < batch_size; ++b) {
            for (int k = 0; k < dim_size; ++k) {
              output(b, o_idx) +=
                  left_matrix(b, l_idx + k) * right_matrix(b, r_idx + k);
            }
          }
        } else {
          int o_idx = (l * right_feat_num + r) * dim_size;
          for (int b = 0; b < batch_size; ++b) {
            for (int k = 0; k < dim_size; ++k) {
              output(b, o_idx + k) =
                  left_matrix(b, l_idx + k) * right_matrix(b, r_idx + k);
            }
          }
        }
      }
    }
  }
};

template <>
struct FFMGradImpl<CPUDevice> {
  static void Compute(OpKernelContext *ctx, const std::string &int_type,
                      TTypes<float>::ConstMatrix grad_matrix, int grad_feat_num,
                      TTypes<float>::ConstMatrix left_matrix, int left_feat_num,
                      TTypes<float>::ConstMatrix right_matrix,
                      int right_feat_num, int batch_size, int dim_size,
                      TTypes<float>::Matrix left_grad_matrix,
                      TTypes<float>::Matrix right_grad_matrix) {
    left_grad_matrix.setZero();
    right_grad_matrix.setZero();

    for (int g = 0; g < grad_feat_num; ++g) {
      int l_idx = (g / right_feat_num) * dim_size;
      int r_idx = (g % right_feat_num) * dim_size;

      if (int_type == "dot") {
        for (int b = 0; b < batch_size; ++b) {
          for (int k = 0; k < dim_size; ++k) {
            left_grad_matrix(b, l_idx + k) +=
                grad_matrix(b, g) * right_matrix(b, r_idx + k);

            right_grad_matrix(b, r_idx + k) +=
                grad_matrix(b, g) * left_matrix(b, l_idx + k);
          }
        }
      } else {
        int g_idx = g * dim_size;
        for (int b = 0; b < batch_size; ++b) {
          for (int k = 0; k < dim_size; ++k) {
            left_grad_matrix(b, l_idx + k) +=
                grad_matrix(b, g_idx + k) * right_matrix(b, r_idx + k);

            right_grad_matrix(b, r_idx + k) +=
                grad_matrix(b, g_idx + k) * left_matrix(b, l_idx + k);
          }
        }
      }
    }
  }
};

namespace {

REGISTER_KERNEL_BUILDER(Name("FFM").Device(DEVICE_CPU), FFMOp<CPUDevice>)

REGISTER_KERNEL_BUILDER(Name("FFMGrad").Device(DEVICE_CPU),
                        FFMGradOp<CPUDevice>)

}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow
