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
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

REGISTER_OP("PBDataset")
    .Input("file_name: string")
    .Input("use_snappy: bool")
    .Input("has_sort_id: bool")
    .Input("kafka_dump: bool")
    .Input("kafka_dump_prefix: bool")
    .Input("buffer_size: int64")
    .Input("lagrangex_header: bool")
    .Input("input_pb_type: string")
    .Input("output_pb_type: string")
    .Input("feature_pruning_type: int32")
    .Input("feature_name_list: string")
    .Input("feature_id_list: int32")
    .Attr("out_type: {variant, string}")
    .Output("handle: variant")
    .SetDoNotOptimize()  // Source dataset ops must disable constant folding.
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 1, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("InstanceReweightDataset")
    .Input("input: variant")
    .Attr("method: int")
    .Attr("actions: list(int)")
    .Attr("weights: list(int)")
    .Attr("labels: list(int)")
    .Attr("priorities: list(int)")
    .Attr("variant_type: string")
    .Output("handle: variant")
    .SetDoNotOptimize()  // Source dataset ops must disable constant folding.
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 0, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("InstanceNegativeGenDataset")
    .Input("input: variant")
    .Input("pool: resource")
    .Attr("neg_num: int >= 1")
    .Attr("per_channel: bool")
    .Attr("channel_feature: string")
    .Attr("item_features: list(string)")
    .Attr("label_index: int >= 0")
    .Attr("positive_label: int")
    .Attr("negative_label: int")
    .Attr("negative_action: int")
    .Attr("action_priority: string")
    .Attr("positive_actions: list(int)")
    .Attr("index_feature: string")
    .Attr("throw_origin: bool")
    .Attr("throw_origin_neg: bool")
    .Attr("cache_only_pos: bool")
    .Attr("real_neg_instance_weight: float")
    .Attr("sampled_neg_instance_weight: float")
    .Attr("unbias_sampled_neg: bool")
    .Attr("origin_neg_in_pool_proba: float")
    .Attr("neg_sample_declay_factor: float")
    .Attr("hard_easy_ratio: float")
    .Attr("variant_type: string")
    .Output("handle: variant")
    .SetDoNotOptimize()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 0, &unused));
      return shape_inference::ScalarShape(c);
    });


REGISTER_OP("SplitFlowDataset")
    .Input("input: variant")
    .Attr("data_flow: list(string)")
    .Attr("index: int")
    .Attr("max_queue_size: int")
    .Attr("variant_type: string")
    .Output("handle: variant")
    .SetDoNotOptimize()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 0, &unused));
      return shape_inference::ScalarShape(c);
    });


REGISTER_OP("MergeFlowDataset")
    .Input("inputs:  N * variant")
    .Attr("data_flow: list(string)")
    .Attr("max_queue_size: int")
    .Attr("variant_type: string")
    .Attr("N: int >= 1")
    .Output("handle: variant")
    .SetDoNotOptimize()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 0, &unused));
      return shape_inference::ScalarShape(c);
    });


REGISTER_OP("DynamicMatchingFilesDataset")
    .Input("patterns: string")
    .Output("handle: variant")
    .SetDoNotOptimize()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      return shape_inference::ScalarShape(c);
    });

}  // namespace tensorflow
