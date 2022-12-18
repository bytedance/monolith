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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

REGISTER_OP("BernoulliGate")
    .Input("alpha: float")
    .Output("sampled: float")
    .Output("proba: float")
    .Attr("ste_type: string")
    .Attr("use_logistic: bool")
    .Attr("temperature: float")
    .SetShapeFn([](shape_inference::InferenceContext *ctx) {
      ctx->set_output(0, ctx->input(0));
      return Status::OK();
    });

REGISTER_OP("BernoulliGateGrad")
    .Input("grad: float")
    .Input("alpha: float")
    .Input("proba: float")
    .Output("output: float")
    .Attr("ste_type: string")
    .Attr("use_logistic: bool")
    .Attr("temperature: float")
    .SetShapeFn([](shape_inference::InferenceContext *ctx) {
      ctx->set_output(0, ctx->input(0));
      return Status::OK();
    });

REGISTER_OP("DiscreteGate")
    .Input("alpha: float")
    .Output("sampled: float")
    .Output("proba: float")
    .Attr("is_one_hot: bool")
    .Attr("use_gumbel: bool")
    .Attr("temperature: float")
    .SetShapeFn([](shape_inference::InferenceContext *ctx) {
      ctx->set_output(0, ctx->input(0));
      ctx->set_output(1, ctx->input(0));
      return Status::OK();
    });

REGISTER_OP("DiscreteGateGrad")
    .Input("grad: float")
    .Input("sampled: float")
    .Input("proba: float")
    .Output("output: float")
    .Attr("is_one_hot: bool")
    .Attr("temperature: float")
    .SetShapeFn([](shape_inference::InferenceContext *ctx) {
      ctx->set_output(0, ctx->input(0));
      return Status::OK();
    });

REGISTER_OP("DiscreteTruncatedGate")
    .Input("alpha: float")
    .Output("sampled: float")
    .Output("proba: float")
    .Attr("threshold: float")
    .Attr("drop_first_dim: bool")
    .Attr("use_gumbel: bool")
    .Attr("temperature: float")
    .SetShapeFn([](shape_inference::InferenceContext *ctx) {
      bool drop_first_dim;
      TF_RETURN_IF_ERROR(ctx->GetAttr("drop_first_dim", &drop_first_dim));

      if (drop_first_dim) {
        shape_inference::DimensionHandle alpha_dim = ctx->Dim(ctx->input(0), 0);
        shape_inference::DimensionHandle sampled_dim;
        ctx->Subtract(alpha_dim,  1, &sampled_dim);

        ctx->set_output(0, ctx->Vector(sampled_dim));
      } else {
        ctx->set_output(0, ctx->input(0));
      }

      ctx->set_output(1, ctx->input(0));
      return Status::OK();
    });

REGISTER_OP("DiscreteTruncatedGateGrad")
    .Input("grad: float")
    .Input("sampled: float")
    .Input("proba: float")
    .Output("output: float")
    .Attr("threshold: float")
    .Attr("drop_first_dim: bool")
    .Attr("temperature: float")
    .SetShapeFn([](shape_inference::InferenceContext *ctx) {
      ctx->set_output(0, ctx->input(2));
      return Status::OK();
    });

}  // namespace tensorflow
