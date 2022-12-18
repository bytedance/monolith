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

#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace monolith_tf {

class StaticReshapeNOp : public OpKernel {
 public:
  explicit StaticReshapeNOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shapes", &shapes_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("enable_parallelism", &enable_parallelism_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("cost_per_tensor", &cost_per_tensor_));
  }

  void Compute(OpKernelContext* ctx) override {
    int num_inputs = ctx->num_inputs();
    OP_REQUIRES(ctx, shapes_.size() == num_inputs,
                errors::InvalidArgument(
                    "`shapes` size must equal to `inputs`, got shapes (",
                    shapes_.size(), ") vs `inputs` (", num_inputs, ")"));

    Tensor* tensor_sizes = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(num_inputs, {num_inputs}, &tensor_sizes));
    auto sizes_vec = tensor_sizes->vec<int64_t>();

    auto thread_pool = ctx->device()->tensorflow_cpu_worker_threads()->workers;
    auto reshape = [&](int64_t begin, int64_t end) {
      for (int idx = static_cast<int>(begin); idx < static_cast<int>(end);
           idx++) {
        const Tensor& input = ctx->input(idx);
        int tensor_size = input.NumElements();
        TensorShape shape;
        const PartialTensorShape& partial = shapes_.at(idx);

        // Maybe infer unk dim.
        int64_t product = 1;
        int unknown_index = -1;
        OP_REQUIRES(ctx, partial.dims() > 0,
                    errors::InvalidArgument(
                        "Shape cannot be unknown rank for input [", idx, "]!"));
        for (int d = 0; d < partial.dims(); d++) {
          int dim = partial.dim_size(d);
          if (dim == -1) {
            OP_REQUIRES(
                ctx, unknown_index == -1,
                errors::InvalidArgument(
                    "Only one input size may be -1, not both ", unknown_index,
                    " and ", d, "for input [", idx, "]!"));
            unknown_index = d;
            shape.AddDim(1);
          } else {
            shape.AddDim(dim);
            product *= dim;
          }
        }
        if (unknown_index != -1) {
          if (product == 0) {
            // In this case, tensor_size should be 0.
            // Check will perform later.
            shape.set_dim(unknown_index, 0);
          } else {
            OP_REQUIRES(ctx, tensor_size % product == 0,
                        errors::InvalidArgument(
                            "Input[", idx, "] of size ", tensor_size,
                            " cannot be reshaped as ", shape.DebugString()));
            shape.set_dim(unknown_index, tensor_size / product);
          }
        }
        OP_REQUIRES(
            ctx, input.NumElements() == shape.num_elements(),
            errors::InvalidArgument(
                "Input[", idx, "] to reshape is a tensor with ",
                input.NumElements(), " values, but the requested shape has ",
                shape.num_elements()));
        Tensor output(input.dtype());
        CHECK(output.CopyFrom(input, shape));
        ctx->set_output(idx, output);
        sizes_vec(idx) = shape.num_elements();
      }
    };

    if (enable_parallelism_) {
      thread_pool->ParallelFor(num_inputs, cost_per_tensor_, reshape);
    } else {
      reshape(0, num_inputs);
    }
  }

 private:
  std::vector<PartialTensorShape> shapes_;
  bool enable_parallelism_;
  int64 cost_per_tensor_;
};

REGISTER_OP("MonolithStaticReshapeN")
    .Input("inputs: dtypes")
    .Output("outputs: dtypes")
    .Output("sizes: int64")
    .Attr("dtypes: list(type)")
    .Attr("shapes: list(shape)")
    .Attr("enable_parallelism: bool = true")
    .Attr("cost_per_tensor: int = 10000000")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      std::vector<PartialTensorShape> shapes;
      TF_RETURN_IF_ERROR(c->GetAttr("shapes", &shapes));
      for (int i = 0; i < shapes.size(); i++) {
        shape_inference::ShapeHandle shape;
        TF_RETURN_IF_ERROR(
            c->MakeShapeFromPartialTensorShape(shapes[i], &shape));
        c->set_output(i, shape);
      }
      c->set_output(shapes.size(), c->Vector(shapes.size()));
      return Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name("MonolithStaticReshapeN").Device(DEVICE_CPU),
                        StaticReshapeNOp);

}  // namespace monolith_tf
}  // namespace tensorflow
