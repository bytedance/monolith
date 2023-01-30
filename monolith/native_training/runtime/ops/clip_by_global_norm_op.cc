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

#include "clip_by_global_norm.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace monolith {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <>
struct ClipByGlobalNormImpl<CPUDevice> {
  static void Compute(OpKernelContext* context, float scale) {
    int num_inputs = context->num_inputs() - 2;
    bool user_parallel = num_inputs > 4;
    auto func = [context, scale](int64 start, int64 end) {
      for (int64 i = start; i < end; ++i) {
        Tensor* temp;
        context->allocate_output(i, context->input(i).shape(), &temp);
        temp->flat<float>() = context->input(i).flat<float>() * scale;
      }
    };
    if (user_parallel) {
      auto worker_threads =
          *(context->device()->tensorflow_cpu_worker_threads());
      Shard(worker_threads.num_threads, worker_threads.workers, num_inputs,
            (num_inputs + worker_threads.num_threads - 1) /
                worker_threads.num_threads,
            func);
    } else {
      func(0, num_inputs);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("MonolithClipByGlobalNorm").Device(DEVICE_CPU),
                        ClipByGlobalNorm<CPUDevice>);

// End: Kernel Definition
REGISTER_OP("MonolithClipByGlobalNorm")
    .Input("grad_list: N * float")
    .Input("global_norm: float")
    .Input("clip_norm: float")
    .Output("clip_grad_list: N * float")
    .Attr("N: int")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      int input_n = c->num_inputs() - 2;
      for (int i = 0; i < input_n; ++i) {
        c->set_output(i, c->input(i));
      }
      return tensorflow::Status::OK();
    });

}  // namespace monolith
}  // namespace tensorflow
