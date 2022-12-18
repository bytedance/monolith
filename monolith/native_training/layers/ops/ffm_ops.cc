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

REGISTER_OP("FFM")
    .Input("left: float")
    .Input("right: float")
    .Output("output: float")
    .Attr("dim_size: int")
    .Attr("int_type: string")
    .SetShapeFn([](shape_inference::InferenceContext *ctx) {
      int dim_size;
      TF_RETURN_IF_ERROR(ctx->GetAttr("dim_size", &dim_size));
      auto batch_size = ctx->Dim(ctx->input(0), 0);

      std::string int_type;
      TF_RETURN_IF_ERROR(ctx->GetAttr("int_type", &int_type));

      shape_inference::DimensionHandle tmp_dims;
      ctx->Multiply(ctx->DimKnownRank(ctx->input(0), 1),
                    ctx->DimKnownRank(ctx->input(1), 1), &tmp_dims);

      shape_inference::DimensionHandle out_dims;
      if (int_type == "dot") {
        ctx->Divide(tmp_dims, dim_size * dim_size, true, &out_dims);
      } else {
        ctx->Divide(tmp_dims, dim_size, true, &out_dims);
      }

      ctx->set_output(0, ctx->Matrix(batch_size, out_dims));
      return Status::OK();
    });

REGISTER_OP("FFMGrad")
    .Input("grad: float")
    .Input("left: float")
    .Input("right: float")
    .Output("left_grad: float")
    .Output("right_grad: float")
    .Attr("dim_size: int")
    .Attr("int_type: string")
    .SetShapeFn([](shape_inference::InferenceContext *ctx) {
      ctx->set_output(0, ctx->input(1));
      ctx->set_output(1, ctx->input(2));
      return Status::OK();
    });

}  // namespace tensorflow
