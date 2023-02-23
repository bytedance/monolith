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

#include "idl/matrix/proto/example.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace monolith_tf {
namespace {

Status ShapeFn(shape_inference::InferenceContext *ctx) {
  int batch_size = ctx->Value(ctx->Dim(ctx->input(0), 0));
  std::vector<int> shapes;
  std::vector<DataType> dtypes;
  TF_RETURN_IF_ERROR(ctx->GetAttr("shapes", &shapes));
  TF_RETURN_IF_ERROR(ctx->GetAttr("dtypes", &dtypes));

  if (batch_size > 0) {  // know batch_size
    for (size_t i = 0; i < dtypes.size(); ++i) {
      if (i >= shapes.size()) {
        ctx->set_output(i, ctx->Vector(ctx->UnknownDim()));
      } else {
        DataType dtype = dtypes[i];
        int shape = shapes[i];
        if (shape == -1) {
          if (dtype != DataType::DT_INT64) {
            return errors::InvalidArgument(
                "If shape is -1, then dtype must be int64");
          }
          ctx->set_output(i, ctx->Vector(batch_size + 1));
        } else {
          ctx->set_output(i, ctx->Matrix(batch_size, shape));
        }
      }
    }
  } else {  // batch_size unknown
    for (size_t i = 0; i < dtypes.size(); ++i) {
      if (i >= shapes.size()) {
        ctx->set_output(i, ctx->Vector(ctx->UnknownDim()));
      } else {
        int shape = shapes[i];
        if (shape > 0) {
          ctx->set_output(i, ctx->Matrix(ctx->UnknownDim(), shape));
        } else {
          ctx->set_output(i, ctx->Vector(ctx->UnknownDim()));
        }
      }
    }
  }

  return Status::OK();
}

REGISTER_OP("ParseInstances")
    .Input("pb_input: T")
    .Output("tensors: dtypes")
    .Attr("fidv1_features: list(int)")
    .Attr("fidv2_features: list(string)")
    .Attr("names: list(string)")
    .Attr("shapes: list(int)")
    .Attr("dtypes: list(type)")
    .Attr("extra_names: list(string)")
    .Attr("T: {variant, string}")
    .SetDoNotOptimize()  // Source dataset ops must disable constant folding.
    .SetShapeFn(ShapeFn);

REGISTER_OP("ParseInstancesV2")
    .Input("pb_input: T")
    .Output("tensors: dtypes")
    .Output("sparse_features: variant")
    .Attr("fidv1_features: list(int)")
    .Attr("fidv2_features: list(string)")
    .Attr("names: list(string)")
    .Attr("shapes: list(int)")
    .Attr("dtypes: list(type)")
    .Attr("extra_names: list(string)")
    .Attr("T: {variant, string}")
    .SetDoNotOptimize()  // Source dataset ops must disable constant folding.
    .SetShapeFn([](shape_inference::InferenceContext *ctx) {
      auto status = ShapeFn(ctx);
      std::vector<DataType> dtypes;
      TF_RETURN_IF_ERROR(ctx->GetAttr("dtypes", &dtypes));
      ctx->set_output(dtypes.size(), ctx->input(0));
      return status;
    });

REGISTER_OP("ParseExamples")
    .Input("pb_input: T")
    .Output("tensors: dtypes")
    .Attr("names: list(string)")
    .Attr("shapes: list(int)")
    .Attr("dtypes: list(type)")
    .Attr("extra_names: list(string)")
    .Attr("T: {variant, string}")
    .SetDoNotOptimize()  // Source dataset ops must disable constant folding.
    .SetShapeFn(ShapeFn);

REGISTER_OP("ParseExamplesV2")
    .Input("pb_input: T")
    .Output("tensors: dtypes")
    .Output("sparse_features: variant")
    .Attr("names: list(string)")
    .Attr("shapes: list(int)")
    .Attr("dtypes: list(type)")
    .Attr("extra_names: list(string)")
    .Attr("T: {variant, string}")
    .SetDoNotOptimize()  // Source dataset ops must disable constant folding.
    .SetShapeFn([](shape_inference::InferenceContext *ctx) {
      auto status = ShapeFn(ctx);
      std::vector<DataType> dtypes;
      TF_RETURN_IF_ERROR(ctx->GetAttr("dtypes", &dtypes));
      ctx->set_output(dtypes.size(), ctx->input(0));
      return status;
    });

REGISTER_OP("ParseExampleBatch")
    .Input("pb_input: T")
    .Output("tensors: dtypes")
    .Attr("names: list(string)")
    .Attr("shapes: list(int)")
    .Attr("dtypes: list(type)")
    .Attr("extra_names: list(string)")
    .Attr("T: {variant, string}")
    .SetDoNotOptimize()  // Source dataset ops must disable constant folding.
    .SetShapeFn([](shape_inference::InferenceContext *ctx) {
      std::vector<int> shapes;
      std::vector<DataType> dtypes;
      TF_RETURN_IF_ERROR(ctx->GetAttr("shapes", &shapes));
      TF_RETURN_IF_ERROR(ctx->GetAttr("dtypes", &dtypes));

      for (size_t i = 0; i < dtypes.size(); ++i) {
        if (i >= shapes.size()) {
          ctx->set_output(i, ctx->Vector(ctx->UnknownDim()));
        } else {
          int shape = shapes[i];
          if (shape > 0) {
            ctx->set_output(i, ctx->Matrix(ctx->UnknownDim(), shape));
          } else {
            ctx->set_output(i, ctx->Vector(ctx->UnknownDim()));
          }
        }
      }
      return Status::OK();
    });

REGISTER_OP("ParseExampleBatchV2")
    .Input("pb_input: T")
    .Output("tensors: dtypes")
    .Output("sparse_features: variant")
    .Attr("names: list(string)")
    .Attr("shapes: list(int)")
    .Attr("dtypes: list(type)")
    .Attr("extra_names: list(string)")
    .Attr("T: {variant, string}")
    .SetDoNotOptimize()  // Source dataset ops must disable constant folding.
    .SetShapeFn([](shape_inference::InferenceContext *ctx) {
      // same as ParseExampleBatch
      std::vector<int> shapes;
      std::vector<DataType> dtypes;
      TF_RETURN_IF_ERROR(ctx->GetAttr("shapes", &shapes));
      TF_RETURN_IF_ERROR(ctx->GetAttr("dtypes", &dtypes));

      for (size_t i = 0; i < dtypes.size(); ++i) {
        if (i >= shapes.size()) {
          ctx->set_output(i, ctx->Vector(ctx->UnknownDim()));
        } else {
          int shape = shapes[i];
          if (shape > 0) {
            ctx->set_output(i, ctx->Matrix(ctx->UnknownDim(), shape));
          } else {
            ctx->set_output(i, ctx->Vector(ctx->UnknownDim()));
          }
        }
      }

      // add sparse_features
      ctx->set_output(dtypes.size(), ctx->Scalar());
      return Status::OK();
    });

REGISTER_OP("ParseExampleBatchList")
    .Input("inputs: N * variant")
    .Output("tensors: dtypes")
    .Attr("label_config: string")
    .Attr("names: list(string)")
    .Attr("shapes: list(int)")
    .Attr("dtypes: list(type)")
    .Attr("extra_names: list(string)")
    .Attr("positive_label: float")
    .Attr("negative_label: float")
    .Attr("N: int")
    .SetDoNotOptimize()  // Source dataset ops must disable constant folding.
    .SetShapeFn([](shape_inference::InferenceContext *ctx) {
      std::vector<int> shapes;
      std::vector<DataType> dtypes;
      TF_RETURN_IF_ERROR(ctx->GetAttr("shapes", &shapes));
      TF_RETURN_IF_ERROR(ctx->GetAttr("dtypes", &dtypes));

      for (size_t i = 0; i < dtypes.size(); ++i) {
        if (i >= shapes.size()) {
          ctx->set_output(i, ctx->Vector(ctx->UnknownDim()));
        } else {
          int shape = shapes[i];
          if (shape > 0) {
            ctx->set_output(i, ctx->Matrix(ctx->UnknownDim(), shape));
          } else {
            ctx->set_output(i, ctx->Vector(ctx->UnknownDim()));
          }
        }
      }
      return Status::OK();
    });
}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow
