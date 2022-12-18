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

/*
该函数包含6个输出
fid_list:             shape(table_count*ps_num, 若干fid),
                      将不同feature样本的全部fid聚合，并按照ps_num分shard，按照feature->table的映射聚合填充
fid_list_row_splits   shape(table_count*ps_num, table内feature个数+1),
                      与fid_list组成ragged_tensor，主要作用是将相同table内不同feature区分开
fid_offset            shape(特征数*batch_size*fid数),
                      样本的一维平铺，最后不是存储的fid，而是在fid_list中的偏移，用于在fid_list寻址
                      高32位为fid_list 第一维度与feature在table内index 的组合
                      低32位为fid在当前feature fid_list的第几位
feature_offset        shape(特征数*batch_size),
                      对fid_offset一维平铺的拆解，标识每个样本的分界点
nfl_offset            shape(特征数),
                      对fid_offset/feature_offset一维平铺的拆解，标识每个特征的分界点
batch_size
*/

REGISTER_OP("ShardingSparseFids")
    .Input("pb_input: variant")
    .Output("fid_list: N * int64")
    .Output("fid_offset: uint64")
    .Output("feature_offset: int32")
    .Output("nfl_offset: uint32")
    .Output("batch_size: int32")
    .Attr("ps_num: int")
    .Attr("feature_cfgs: string")
    .Attr("N: int")
    .Attr("unique: bool")
    .Attr("input_type: string")
    .Attr("parallel_flag: int")
    .SetDoNotOptimize()  // Source dataset ops must disable constant folding.
    .SetShapeFn([](shape_inference::InferenceContext *ctx) {
      // fid_list
      int N = 0;
      TF_RETURN_IF_ERROR(ctx->GetAttr("N", &N));
      for (int i = 0; i < N; ++i) {
        ctx->set_output(i, ctx->Vector(ctx->UnknownDim()));  // fid_list
      }
      ctx->set_output(N,
                      ctx->Vector(ctx->UnknownDim()));  // fid_offset
      ctx->set_output(N + 1,
                      ctx->Vector(ctx->UnknownDim()));  // feature_offset
      ctx->set_output(N + 2,
                      ctx->Vector(ctx->UnknownDim()));  // nfl_offset
      ctx->set_output(N + 3, ctx->Scalar());            // batch_size

      return Status::OK();
    });

REGISTER_OP("ShardingSparseFidsV2")
    .Input("pb_input: variant")
    .Output("fid_list: N * int64")
    .Output("fid_list_row_splits: N * int64")
    .Output("fid_offset: uint64")
    .Output("feature_offset: int32")
    .Output("nfl_offset: uint32")
    .Output("batch_size: int32")
    .Attr("ps_num: int")
    .Attr("feature_cfgs: string")
    .Attr("N: int")
    .Attr("unique: bool")
    .Attr("input_type: string")
    .Attr("parallel_flag: int")
    .SetDoNotOptimize()  // Source dataset ops must disable constant folding.
    .SetShapeFn([](shape_inference::InferenceContext *ctx) {
      // fid_list
      int N = 0;
      TF_RETURN_IF_ERROR(ctx->GetAttr("N", &N));
      for (int i = 0; i < N; ++i) {
        ctx->set_output(i, ctx->Vector(ctx->UnknownDim()));  // fid_list
        ctx->set_output(N + i,
                        ctx->Vector(ctx->UnknownDim()));  // fid_list_row_splits
      }
      N *= 2;
      ctx->set_output(N,
                      ctx->Vector(ctx->UnknownDim()));  // fid_offset
      ctx->set_output(N + 1,
                      ctx->Vector(ctx->UnknownDim()));  // feature_offset
      ctx->set_output(N + 2,
                      ctx->Vector(ctx->UnknownDim()));  // nfl_offset
      ctx->set_output(N + 3, ctx->Scalar());            // batch_size

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
