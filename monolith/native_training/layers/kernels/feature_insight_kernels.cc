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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <random>
#include <string>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace monolith_tf {

class FeatureInsightOp : public OpKernel {
 public:
  explicit FeatureInsightOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    std::vector<int32> segment_sizes;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("segment_sizes", &segment_sizes));

    int32 idx = 0;
    num_feature_ = segment_sizes.size();
    for (int32 size : segment_sizes) {
      for (int i = 0; i < size; ++i) {
        segment_id_map_.push_back(idx);
      }
      idx++;
    }
  }

  void Compute(OpKernelContext *ctx) override {
    const Tensor *input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
    auto input_mat = input_tensor->matrix<float>();
    const Tensor *weight_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("weight", &weight_tensor));
    auto weight_mat = weight_tensor->matrix<float>();

    int64 batch_size = input_tensor->dim_size(0);
    int64 out_dim = weight_tensor->dim_size(1);

    Tensor *out_tensor;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(
                 "output", {batch_size, num_feature_ * out_dim}, &out_tensor));
    auto out_mat = out_tensor->matrix<float>();
    out_mat.setZero();

    Tensor tmp_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(out_tensor->dtype(), {num_feature_},
                                           &tmp_tensor));
    auto tmp_mat = tmp_tensor.flat<float>();

    for (size_t i = 0; i < batch_size; ++i) {  // batch_size
      for (size_t k = 0; k < out_dim; ++k) {   // out_size
        tmp_mat.setZero();
        for (size_t j = 0; j < input_tensor->dim_size(1);
             ++j) {  // total_embedding_size
          int32 idx = segment_id_map_[j];
          tmp_mat(idx) += input_mat(i, j) * weight_mat(j, k);
        }

        for (size_t idx = 0; idx < num_feature_; ++idx) {
          out_mat(i, idx * out_dim + k) += tmp_mat(idx);
        }
      }
    }
  }

 private:
  int64 num_feature_;
  std::vector<int32> segment_id_map_;
};

class FeatureInsightGradOp : public OpKernel {
 public:
  explicit FeatureInsightGradOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("segment_sizes", &segment_sizes_));

    int K;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("K", &K));

    num_feature_ = segment_sizes_.size();
    int grad_dim = num_feature_ * K;
    grad_dim_to_k_.reserve(grad_dim);
    grad_dim_to_feature_idx_.reserve(grad_dim);
    feature_idx_to_embedding_start_.reserve(num_feature_);
    for (int i = 0; i < num_feature_; ++i) {
      for (int j = 0; j < K; ++j) {
        grad_dim_to_feature_idx_.push_back(i);
        grad_dim_to_k_.push_back(j);
      }

      if (i == 0) {
        feature_idx_to_embedding_start_.push_back(0);
      } else {
        feature_idx_to_embedding_start_.push_back(
            feature_idx_to_embedding_start_[i - 1] + segment_sizes_[i - 1]);
      }
    }
  }

  void Compute(OpKernelContext *ctx) override {
    const Tensor *grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("grad", &grad_tensor));
    auto grad_mat = grad_tensor->matrix<float>();
    const Tensor *input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
    auto input_mat = input_tensor->matrix<float>();
    const Tensor *weight_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("weight", &weight_tensor));
    auto weight_mat = weight_tensor->matrix<float>();

    Tensor *input_grad_tensor;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("input_grad", input_tensor->shape(),
                                        &input_grad_tensor));
    auto input_grad_mat = input_grad_tensor->matrix<float>();
    input_grad_mat.setZero();

    Tensor *weight_grad_tensor;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("weight_grad", weight_tensor->shape(),
                                        &weight_grad_tensor));
    auto weight_grad_mat = weight_grad_tensor->matrix<float>();
    weight_grad_mat.setZero();

    int64 batch_size = input_tensor->dim_size(0);
    int64 grad_dim = grad_tensor->dim_size(1);
    LOG(INFO) << "get data done! batch_size=" << batch_size
              << ", grad_dim=" << grad_dim;
    for (size_t i = 0; i < batch_size; ++i) {  // batch_size
      for (size_t g = 0; g < grad_dim; ++g) {
        int k = grad_dim_to_k_[g];
        int feature_idx = grad_dim_to_feature_idx_[g];
        int start = feature_idx_to_embedding_start_[feature_idx];
        int end = start + segment_sizes_[feature_idx];
        float grad_val = grad_mat(i, g);
        for (int j = start; j < end; ++j) {
          weight_grad_mat(j, k) += grad_val * input_mat(i, j);
          input_grad_mat(i, j) += grad_val * weight_mat(j, k);
        }
      }
    }
  }

 private:
  int64 num_feature_;
  std::vector<int32> segment_sizes_;
  std::vector<int32> grad_dim_to_k_;
  std::vector<int32> grad_dim_to_feature_idx_;
  std::vector<int32> feature_idx_to_embedding_start_;
};

namespace {

REGISTER_KERNEL_BUILDER(Name("FeatureInsight").Device(DEVICE_CPU),
                        FeatureInsightOp)

REGISTER_KERNEL_BUILDER(Name("FeatureInsightGrad").Device(DEVICE_CPU),
                        FeatureInsightGradOp)
}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow
