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

REGISTER_OP("FeatureInsight")
    .Input("input: float")
    .Input("weight: float")
    .Output("output: float")
    .Attr("segment_sizes: list(int)")
    .SetShapeFn([](shape_inference::InferenceContext *ctx) {
      std::vector<int32> segment_sizes;
      TF_RETURN_IF_ERROR(ctx->GetAttr("segment_sizes", &segment_sizes));
      auto batch_size = ctx->Dim(ctx->input(0), 0);
      shape_inference::DimensionHandle out_dims;
      TF_RETURN_IF_ERROR(ctx->Multiply(ctx->MakeDim(segment_sizes.size()),
                                       ctx->Dim(ctx->input(1), 1), &out_dims));

      ctx->set_output(0, ctx->Matrix(batch_size, out_dims));
      return Status::OK();
    });

REGISTER_OP("FeatureInsightGrad")
    .Input("grad: float")
    .Input("input: float")
    .Input("weight: float")
    .Output("input_grad: float")
    .Output("weight_grad: float")
    .Attr("segment_sizes: list(int)")
    .Attr("K: int")
    .SetShapeFn([](shape_inference::InferenceContext *ctx) {
      ctx->set_output(0, ctx->input(1));
      ctx->set_output(1, ctx->input(2));
      return Status::OK();
    });

}  // namespace tensorflow
