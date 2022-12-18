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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

REGISTER_OP("ExtractFid")
    .Input("input: int64")
    .Attr("slot: int")
    .Output("output: int64");

REGISTER_OP("FeatureHash")
    .Input("input: variant")
    .Attr("names: list(string)")
    .Output("output: variant")
    .SetShapeFn([](shape_inference::InferenceContext *ctx) {
      ctx->set_output(0, ctx->input(0));
      return Status::OK();
    });

REGISTER_OP("SetFilter")
    .Input("input: variant")
    .Attr("filter_fids: list(int)")
    .Attr("has_fids: list(int)")
    .Attr("select_fids: list(int)")
    .Attr("has_actions: list(int)")
    .Attr("req_time_min: int")
    .Attr("select_slots: list(int)")
    .Attr("variant_type: string")
    .Output("output: bool");

REGISTER_OP("ValueFilter")
    .Input("input: variant")
    .Attr("field_name: string")
    .Attr("op: string")
    .Attr("float_operand: list(float)")
    .Attr("int_operand: list(int)")
    .Attr("string_operand: list(string)")
    .Attr("operand_filepath: string")
    .Attr("keep_empty: bool = false")
    .Attr("variant_type: string")
    .Output("output: bool");

REGISTER_OP("AddAction")
    .Input("input: variant")
    .Attr("field_name: string")
    .Attr("op: string")
    .Attr("float_operand: list(float)")
    .Attr("int_operand: list(int)")
    .Attr("string_operand: list(string)")
    .Attr("variant_type: string")
    .Attr("actions: list(int)")
    .Output("output: variant");

REGISTER_OP("AddLabel")
    .Input("input: variant")
    .Attr("config: string")
    .Attr("negative_value: float")
    .Attr("sample_rate: float")
    .Attr("variant_type: string")
    .Output("output: variant");

REGISTER_OP("ScatterLabel")
    .Input("input: variant")
    .Attr("config: string")
    .Attr("variant_type: string")
    .Output("output: variant");

REGISTER_OP("FilterByLabel")
    .Input("input: variant")
    .Attr("label_threshold: list(float)")
    .Attr("filter_equal: bool")
    .Attr("variant_type: string")
    .Output("valid: bool");

REGISTER_OP("SpecialStrategy")
    .Input("input: variant")
    .Attr("special_strategies: list(int)")
    .Attr("sample_rates: list(float)")
    .Attr("labels: list(float)")
    .Attr("strategy_list: list(int)")
    .Attr("keep_empty_strategy: bool = true")
    .Attr("variant_type: string")
    .Output("output: bool");

REGISTER_OP("NegativeSample")
    .Input("input: variant")
    .Attr("drop_rate: float")
    .Attr("label_index: int = 0")
    .Attr("threshold: float = 0.0")
    .Attr("variant_type: string")
    .Output("output: bool");

REGISTER_OP("LabelUpperBound")
    .Input("input: variant")
    .Attr("label_upper_bounds: list(float)")
    .Attr("variant_type: string")
    .Output("output: variant");

REGISTER_OP("LabelNormalization")
    .Input("input: variant")
    .Attr("norm_methods: list(string)")
    .Attr("norm_values: list(float)")
    .Attr("variant_type: string")
    .Output("output: variant");

REGISTER_OP("UseFieldAsLabel")
    .Input("input: variant")
    .Attr("field_name: string")
    .Attr("overwrite_invalid_value: bool")
    .Attr("label_threshold: float")
    .Attr("variant_type: string")
    .Output("output: variant");

REGISTER_OP("SwitchSlot")
    .Input("rt_nested_splits: RAGGED_RANK * int64")
    .Input("rt_dense_values: int64")
    .Output("nested_splits_out: RAGGED_RANK * int64")
    .Output("dense_values_out: int64")
    .Attr("slot: int >=1")
    .Attr("fid_version: int")
    .Attr("RAGGED_RANK: int >= 1")
    .SetShapeFn([](shape_inference::InferenceContext *ctx) {
      int rank;
      TF_RETURN_IF_ERROR(ctx->GetAttr("RAGGED_RANK", &rank));
      for (int i = 0; i < rank; ++i) {
        ctx->set_output(i, ctx->input(i));
      }
      ctx->set_output(rank, ctx->input(rank));

      return Status::OK();
    });

REGISTER_OP("FeatureCombine")
    .Input("rt_nested_splits_src1: RAGGED_RANK * int64")
    .Input("rt_dense_values_src1: int64")
    .Input("rt_nested_splits_src2: RAGGED_RANK * int64")
    .Input("rt_dense_values_src2: int64")
    .Output("nested_splits_sink: RAGGED_RANK * int64")
    .Output("dense_values_sink: int64")
    .Attr("slot: int >=1")
    .Attr("fid_version: int")
    .Attr("RAGGED_RANK: int >= 1")
    .SetShapeFn([](shape_inference::InferenceContext *ctx) {
      int rank;
      TF_RETURN_IF_ERROR(ctx->GetAttr("RAGGED_RANK", &rank));
      for (int i = 0; i < rank; ++i) {
        ctx->set_output(i, ctx->input(i));
      }
      ctx->set_output(rank, ctx->input(rank));

      return Status::OK();
    });

REGISTER_OP("ItemPoolCreate")
    .Output("pool: resource")
    .Attr("start_num: int")
    .Attr("max_item_num_per_channel: int")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    // .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ItemPoolRandomFill")
    .Input("ipool: resource")
    .Output("opool: resource")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ItemPoolCheck")
    .Input("ipool: resource")
    .Input("global_step: int64")
    .Output("opool: resource")
    .Attr("model_path: string")
    .Attr("nshards: int")
    .Attr("buffer_size: int")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ItemPoolSave")
    .Input("ipool: resource")
    .Input("global_step: int64")
    .Output("opool: resource")
    .Attr("model_path: string")
    .Attr("nshards: int")
    .Attr("random_sleep_ms: int=0")
    // .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ItemPoolRestore")
    .Input("ipool: resource")
    .Input("global_step: int64")
    .Output("opool: resource")
    .Attr("model_path: string")
    .Attr("buffer_size: int")
    .Attr("nshards: int")
    .Attr("random_sleep_ms: int=0")
    // .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("FillMultiRankOutput")
    .Input("input: variant")
    .Attr("variant_type: string")
    .Attr("enable_draw_as_rank: bool = false")
    .Attr("enable_chnid_as_rank: bool = false")
    .Attr("enable_lineid_rank_as_rank: bool = false")
    .Attr("rank_num: int = 18")
    .Output("output: variant")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("UseF100MultiHead")
    .Input("input: variant")
    .Attr("variant_type: string")
    .Output("output: variant")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("MapId")
    .Input("input: T")
    .Attr("from_value: list(int)")
    .Attr("to_value: list(int)")
    .Attr("default_value: int")
    .Output("output: T")
    .Attr("T: {int32, int64}")
    .SetShapeFn([](shape_inference::InferenceContext *ctx) {
      ctx->set_output(0, ctx->input(0));
      return Status::OK();
    });

REGISTER_OP("MultiLabelGen")
    .Input("input: variant")
    .Attr("task_num: int")
    .Attr("head_to_index: string")
    .Attr("head_field: string")
    .Attr("action_priority: string")
    .Attr("pos_actions: list(int)")
    .Attr("neg_actions: list(int)")
    .Attr("use_origin_label: bool")
    .Attr("pos_label: float")
    .Attr("neg_label: float")
    .Attr("variant_type: string")
    .Output("output: variant")
    .SetShapeFn([](shape_inference::InferenceContext *ctx) {
      ctx->set_output(0, ctx->input(0));
      return Status::OK();
    });

REGISTER_OP("StringToVariant")
    .Input("input: string")
    .Attr("input_type: string")
    .Attr("has_header: bool")
    .Attr("has_sort_id: bool")
    .Attr("lagrangex_header: bool")
    .Attr("kafka_dump_prefix: bool")
    .Attr("kafka_dump: bool")
    .Attr("chnids: list(int)")
    .Attr("datasources: list(string)")
    .Attr("default_datasource: string")
    .Output("output: variant")
    .SetDoNotOptimize()
    .SetShapeFn([](shape_inference::InferenceContext *ctx) {
      ctx->set_output(0, ctx->input(0));
      return Status::OK();
    });

REGISTER_OP("VariantToZeros")
    .Input("input: variant")
    .Output("output: int64")
    .SetDoNotOptimize()
    .SetShapeFn([](shape_inference::InferenceContext *ctx) {
      ctx->set_output(0, ctx->input(0));
      return Status::OK();
    });

REGISTER_OP("HasVariant")
    .Input("input: variant")
    .Output("output: bool")
    .Attr("variant_type: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("KafkaGroupReadableInit")
    .Input("topics: string")
    .Input("metadata: string")
    .Output("resource: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("KafkaGroupReadableNext")
    .Input("input: resource")
    .Input("index: int64")
    .Input("message_poll_timeout: int64")
    .Input("stream_timeout: int64")
    .Output("message: string")
    .Output("key: string")
    .Output("continue_fetch: int64")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->UnknownDim()}));
      c->set_output(1, c->MakeShape({c->UnknownDim()}));
      c->set_output(2, c->Scalar());
      return Status::OK();
    });
}  // namespace tensorflow
