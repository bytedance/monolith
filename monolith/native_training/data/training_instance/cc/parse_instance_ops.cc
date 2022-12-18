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
namespace monolith_tf {
namespace {

bool IsUnsetHandle(shape_inference::DimensionHandle handle) {
  return handle.Handle() == 0;
}

shape_inference::ShapeHandle GetBatched1D(
    shape_inference::InferenceContext *ctx,
    shape_inference::DimensionHandle batch_size, int dim) {
  if (IsUnsetHandle(batch_size)) {
    return ctx->Vector(dim);
  } else {
    return ctx->Matrix(batch_size, dim);
  }
}

Status SetParseInstanceShape(shape_inference::InferenceContext *ctx,
                             shape_inference::DimensionHandle batch_size) {
  int offset = 0;

  // Ragged tensor
  int n;
  TF_RETURN_IF_ERROR(ctx->GetAttr("N", &n));
  for (int i = 0; i < n; ++i) {
    if (IsUnsetHandle(batch_size)) {
      ctx->set_output(offset + i, ctx->Vector(2));
      continue;
    }
    int batch_size_value = ctx->Value(batch_size);
    if (batch_size_value == shape_inference::InferenceContext::kUnknownDim) {
      ctx->set_output(offset + i, ctx->Vector(ctx->UnknownDim()));
    } else {
      ctx->set_output(offset + i, ctx->Vector(batch_size_value + 1));
    }
  }
  offset += n;

  for (int i = 0; i < n; ++i) {
    ctx->set_output(offset + i, ctx->Vector(ctx->UnknownDim()));
  }
  offset += n;

  // float tensor
  int m;
  TF_RETURN_IF_ERROR(ctx->GetAttr("M", &m));
  std::vector<int> float_feature_dims;
  TF_RETURN_IF_ERROR(ctx->GetAttr("float_feature_dims", &float_feature_dims));
  for (int i = 0; i < m; ++i) {
    ctx->set_output(offset + i,
                    GetBatched1D(ctx, batch_size, float_feature_dims[i]));
  }
  offset += m;

  // int64 tensor
  int o;
  TF_RETURN_IF_ERROR(ctx->GetAttr("O", &o));
  std::vector<int> int64_feature_dims;
  TF_RETURN_IF_ERROR(ctx->GetAttr("int64_feature_dims", &int64_feature_dims));
  for (int i = 0; i < o; ++i) {
    ctx->set_output(offset + i,
                    GetBatched1D(ctx, batch_size, int64_feature_dims[i]));
  }
  offset += o;

  // string tensor
  int p;
  TF_RETURN_IF_ERROR(ctx->GetAttr("P", &p));
  std::vector<int> string_feature_dims;
  TF_RETURN_IF_ERROR(ctx->GetAttr("string_feature_dims", &string_feature_dims));
  for (int i = 0; i < p; ++i) {
    ctx->set_output(offset + i,
                    GetBatched1D(ctx, batch_size, string_feature_dims[i]));
  }
  offset += p;

  // misc_feature_float
  int q;
  TF_RETURN_IF_ERROR(ctx->GetAttr("Q", &q));
  std::vector<int> misc_float_dims;
  TF_RETURN_IF_ERROR(ctx->GetAttr("misc_float_dims", &misc_float_dims));
  for (int i = 0; i < q; ++i) {
    ctx->set_output(offset + i,
                    GetBatched1D(ctx, batch_size, misc_float_dims[i]));
  }
  offset += q;

  // misc_feature_int64
  int r;
  TF_RETURN_IF_ERROR(ctx->GetAttr("R", &r));
  std::vector<int> misc_int64_dims;
  TF_RETURN_IF_ERROR(ctx->GetAttr("misc_int64_dims", &misc_int64_dims));
  for (int i = 0; i < r; ++i) {
    ctx->set_output(offset + i,
                    GetBatched1D(ctx, batch_size, misc_int64_dims[i]));
  }
  offset += r;

  // misc_feature_string
  int s;
  TF_RETURN_IF_ERROR(ctx->GetAttr("S", &s));
  std::vector<int> misc_string_dims;
  TF_RETURN_IF_ERROR(ctx->GetAttr("misc_string_dims", &misc_string_dims));
  for (int i = 0; i < s; ++i) {
    ctx->set_output(offset + i,
                    GetBatched1D(ctx, batch_size, misc_string_dims[i]));
  }
  offset += s;

  return Status::OK();
}

// We use fid_features for FIDV1 key and str_features for FIDv2 keys.
// In fid v1, we use the slot whitelist, and in fid v2, we use the
// feature_name
// whitelist.
REGISTER_OP("MonolithParseInstances")
    .Input("serialized: string")
    .Output("ragged_feature_splits: N * int64")
    .Output("ragged_feature_values: N * int64")
    .Output("float_feature_values: M * float32")
    .Output("int64_feature_values: O * int64")
    .Output("string_feature_values: P * string")
    .Output("misc_float_feature_values: Q * float32")
    .Output("misc_int64_feature_values: R * int64")
    .Output("misc_string_feature_values: S * string")
    .Attr("N: int >= 0")
    .Attr("M: int >= 0")
    .Attr("O: int >= 0")
    .Attr("P: int >= 0")
    .Attr("Q: int >= 0")
    .Attr("R: int >= 0")
    .Attr("S: int >= 0")
    .Attr("fidv1_features: list(int)")
    .Attr("fidv2_features: list(string)")
    .Attr("float_features: list(string)")
    .Attr("float_feature_dims: list(int)")
    .Attr("int64_features: list(string)")
    .Attr("int64_feature_dims: list(int)")
    .Attr("string_features: list(string)")
    .Attr("string_feature_dims: list(int)")
    .Attr("misc_float_features: list(string)")
    .Attr("misc_float_dims: list(int)")
    .Attr("misc_int64_features: list(string)")
    .Attr("misc_int64_dims: list(int)")
    .Attr("misc_string_features: list(string)")
    .Attr("misc_string_dims: list(int)")
    .SetDoNotOptimize()  // Source dataset ops must disable constant folding.
    .SetShapeFn([](shape_inference::InferenceContext *ctx) {
      shape_inference::DimensionHandle batch_size = ctx->Dim(ctx->input(0), 0);
      return SetParseInstanceShape(ctx, batch_size);
    });

REGISTER_OP("MonolithRawParseInstance")
    .Attr("T: list(type)")
    .Input("serialized: string")
    .Output("tensors : T")
    .Attr("fidv1_features: list(int) = []")
    .Attr("fidv2_features: list(string) = []")
    .Attr("float_features: list(string) = []")
    .Attr("float_feature_dims: list(int) = []")
    .Attr("int64_features: list(string) = []")
    .Attr("int64_feature_dims: list(int) = []")
    .Attr("string_features: list(string) = []")
    .Attr("string_feature_dims: list(int) = []")
    .Attr("misc_float_features: list(string) = []")
    .Attr("misc_float_dims: list(int) = []")
    .Attr("misc_int64_features: list(string) = []")
    .Attr("misc_int64_dims: list(int) = []")
    .Attr("misc_string_features: list(string) = []")
    .Attr("misc_string_dims: list(int) = []")
    .Attr("collapse_batch_dim: bool = false")
    .Attr("fid_output_type: {'REGULAR', 'CONCAT'} = 'REGULAR'")
    .SetDoNotOptimize()  // Source dataset ops must disable constant folding.
    .SetShapeFn([](shape_inference::InferenceContext *ctx) {
      return Status::OK();
    });

}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow
