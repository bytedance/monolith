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

class InbatchAucLossOp : public OpKernel {
 public:
  explicit InbatchAucLossOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    float neg_weight;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("neg_weight", &neg_weight));
    CHECK_GT(neg_weight, 0);
    CHECK_LE(neg_weight, 1.0);
  }

  void Compute(OpKernelContext *ctx) override {
    const Tensor *label_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("label", &label_tensor));
    const Tensor *logit_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("logit", &logit_tensor));
    OP_REQUIRES(ctx, label_tensor->NumElements() == logit_tensor->NumElements(),
                errors::InvalidArgument("the label and logit not match"));

    std::vector<size_t> positive, negative;
    auto label_flat = label_tensor->flat<float>();
    for (size_t i = 0; i < label_flat.size(); ++i) {
      if (label_flat(i) > 0) {
        positive.push_back(i);
      } else if (label_flat(i) > -10000) {
        negative.push_back(i);
      }
    }

    float loss = 0;
    auto logit_flat = logit_tensor->flat<float>();
    for (const size_t &i : positive) {
      float pos_logit = logit_flat(i);
      for (const size_t &j : negative) {
        float diff = pos_logit - logit_flat(j);
        if (diff > -87 && diff < 88) {
          loss += diff - log(1.0 + exp(diff));
        } else if (diff <= -87) {
          loss += diff;
        }
      }
    }

    Tensor *loss_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {}, &loss_tensor));
    loss_tensor->scalar<float>()() = loss;
  }
};

class InbatchAucLossGradOp : public OpKernel {
 public:
  explicit InbatchAucLossGradOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("neg_weight", &neg_weight_));
  }

  void Compute(OpKernelContext *ctx) override {
    const Tensor *label_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("label", &label_tensor));
    const Tensor *logit_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("logit", &logit_tensor));
    OP_REQUIRES(ctx, label_tensor->NumElements() == logit_tensor->NumElements(),
                errors::InvalidArgument("the label and logit not match"));
    const Tensor *grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("grad", &grad_tensor));
    float grad = grad_tensor->scalar<float>()();

    Tensor *logit_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, logit_tensor->shape(),
                                             &logit_grad_tensor));
    auto logit_grad_float = logit_grad_tensor->flat<float>();
    logit_grad_float.setZero();

    std::vector<size_t> positive, negative;
    auto label_flat = label_tensor->flat<float>();
    for (size_t i = 0; i < label_flat.size(); ++i) {
      if (label_flat(i) > 0) {
        positive.push_back(i);
      } else if (label_flat(i) > -10000) {
        negative.push_back(i);
      }
    }

    auto logit_flat = logit_tensor->flat<float>();
    for (const size_t &i : positive) {
      float pos_logit = logit_flat(i);
      for (const size_t &j : negative) {
        float diff = pos_logit - logit_flat(j);
        float grad_ij;
        if (diff > -87 && diff < 88) {
          grad_ij = 1.0 - 1.0 / (1.0 + exp(-diff));
        } else if (diff <= -87) {
          grad_ij = 1;
        } else {
          grad_ij = 0;
        }

        logit_grad_float(i) += grad_ij;
        logit_grad_float(j) -= neg_weight_ * grad_ij;
      }
    }

    if (grad != 1) {
      for (size_t i = 0; i < logit_grad_float.size(); ++i) {
        logit_grad_float(i) *= grad;
      }
    }
  }

 private:
  float neg_weight_;
};

namespace {

REGISTER_KERNEL_BUILDER(Name("InbatchAucLoss").Device(DEVICE_CPU),
                        InbatchAucLossOp)

REGISTER_KERNEL_BUILDER(Name("InbatchAucLossGrad").Device(DEVICE_CPU),
                        InbatchAucLossGradOp)

REGISTER_OP("InbatchAucLoss")
    .Input("label: float")
    .Input("logit: float")
    .Attr("neg_weight: float")
    .Output("loss: float")
    .SetDoNotOptimize()
    .SetShapeFn([](shape_inference::InferenceContext *ctx) {
      ctx->set_output(0, ctx->Scalar());
      return Status::OK();
    });

REGISTER_OP("InbatchAucLossGrad")
    .Input("label: float")
    .Input("logit: float")
    .Input("grad: float")
    .Attr("neg_weight: float")
    .Output("logit_grad: float")
    .SetDoNotOptimize()
    .SetShapeFn([](shape_inference::InferenceContext *ctx) {
      ctx->set_output(0, ctx->input(1));
      return Status::OK();
    });

}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow
