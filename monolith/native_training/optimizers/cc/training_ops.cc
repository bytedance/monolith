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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#define EIGEN_USE_THREADS
#include "monolith/native_training/optimizers/cc/kernels/training_op_helpers.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace monolith_tf {

template <bool is_resource>
ShapeHandle ShapeOrHandleShape(InferenceContext* c, int input) {
  auto* handle_data = c->input_handle_shapes_and_types(input);
  if (handle_data != nullptr && !handle_data->empty() &&
      (*handle_data)[0].dtype != DT_INVALID) {
    return (*handle_data)[0].shape;
  }
  return c->input(input);
}

template <>
ShapeHandle ShapeOrHandleShape<true>(InferenceContext* c, int input) {
  auto* handle_data = c->input_handle_shapes_and_types(input);
  if (handle_data != nullptr && !handle_data->empty() &&
      (*handle_data)[0].dtype != DT_INVALID) {
    return (*handle_data)[0].shape;
  }
  // If a resource input is missing shape information, we should return
  // UnknownShape rather than the shape of the input, which is a scalar
  // resource handle.
  return c->UnknownShape();
}

// Handle the gradient and, if <is_sparse>, indices inputs.
// <s> is an input+output parameter, containing the current known input shape to
// the gradient.
template <bool is_sparse, bool is_resource>
static Status HandleGradAndIndicesInputs(InferenceContext* c, int grad_idx,
                                         ShapeHandle* s) {
  ShapeHandle grad = ShapeOrHandleShape<is_resource>(c, grad_idx);
  if (!is_sparse) {
    TF_RETURN_IF_ERROR(c->Merge(*s, grad, s));
    return Status::OK();
  }
  // Indices is a vector where indices.dim[0].rank == grad[0].rank.
  ShapeHandle indices;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(grad_idx + 1), 1, &indices));
  DimensionHandle unused;
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(indices, 0), c->Dim(grad, 0), &unused));
  // Trailing part of grad matches trailing part of *s.
  ShapeHandle grad_unknown_first;
  TF_RETURN_IF_ERROR(
      c->ReplaceDim(grad, 0, c->UnknownDim(), &grad_unknown_first));
  TF_RETURN_IF_ERROR(c->Merge(*s, grad_unknown_first, s));

  return Status::OK();
}

static Status ApplyAdamomShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape</*is_resource=*/true>(c, 0);  // var
  TF_RETURN_IF_ERROR(
      c->Merge(s, ShapeOrHandleShape</*is_resource=*/true>(c, 1), &s));  // m
  TF_RETURN_IF_ERROR(
      c->Merge(s, ShapeOrHandleShape</*is_resource=*/true>(c, 2), &s));  // v
  TF_RETURN_IF_ERROR(
      c->Merge(s, ShapeOrHandleShape</*is_resource=*/true>(c, 3), &s));  // c
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));              // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));  // ada_decay
  TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused));  // mom_decay
  TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &unused));  // epsilon
  TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 0, &unused));  // weight_decay
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs</*is_sparse=*/false, /*is_resource=*/true>(
          c, 9 /* grad_idx */, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("ResourceApplyAdamom")
    .Input("var: resource")
    .Input("m: resource")
    .Input("v: resource")
    .Input("c: resource")
    .Input("learning_rate: float")
    .Input("ada_decay: float")
    .Input("mom_decay: float")
    .Input("epsilon: float")
    .Input("weight_decay: float")
    .Input("grad: float")
    .Attr("use_locking: bool = false")
    .Attr("update_slots: bool = true")
    .Attr("use_v2: bool = false")
    .SetShapeFn(ApplyAdamomShapeFn);

static Status ApplyRmspropShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape</*is_resource=*/true>(c, 0);  // var
  TF_RETURN_IF_ERROR(
      c->Merge(s, ShapeOrHandleShape</*is_resource=*/true>(c, 1), &s));  // m
  TF_RETURN_IF_ERROR(
      c->Merge(s, ShapeOrHandleShape</*is_resource=*/true>(c, 2), &s));  // v
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));              // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));  // beta1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));  // beta2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused));  // epsilon
  TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &unused));  // weight_decay
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs</*is_sparse=*/false, /*is_resource=*/true>(
          c, 8 /* grad_idx */, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("ResourceApplyRmsprop")
    .Input("var: resource")
    .Input("m: resource")
    .Input("v: resource")
    .Input("learning_rate: float")
    .Input("beta1: float")
    .Input("beta2: float")
    .Input("epsilon: float")
    .Input("weight_decay: float")
    .Input("grad: float")
    .Attr("use_locking: bool = false")
    .Attr("update_slots: bool = true")
    .Attr("use_v2: bool = false")
    .SetShapeFn(ApplyRmspropShapeFn);
}  // namespace monolith_tf
}  // namespace tensorflow
