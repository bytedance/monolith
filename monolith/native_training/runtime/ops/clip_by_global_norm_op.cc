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
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/work_sharder.h"

#include "clip_by_global_norm.h"

namespace tensorflow {
namespace monolith {

typedef Eigen::ThreadPoolDevice CPUDevice;

void clip_by_global_norm_fp32(const float* input, int len, float* output,
                              float scale) {
  __m256 scale_vec = {scale, scale, scale, scale, scale, scale, scale, scale};
  int i = 0;
  __m256 reg;
  for (i = 0; i + 8 <= len; input += 8, output += 8, i += 8) {
    reg = _mm256_load_ps(input);
    reg = _mm256_mul_ps(reg, scale_vec);
    _mm256_store_ps(output, reg);
  }
  while (i++ < len) {
    *output = *input * scale;
    output++;
    input++;
  }
}

template <>
struct ClipByGlobalNormImpl<CPUDevice> {
  static void Compute(OpKernelContext* context,
                      const std::vector<const float*>& input_ptrs,
                      const std::vector<int>& input_lens,
                      const std::vector<float*>& output_ptrs, float scale) {
    int num_inputs = input_ptrs.size();
    bool user_parallel = num_inputs > 4;
    auto func = [&input_ptrs, &input_lens, &output_ptrs, scale](int64 start,
                                                                int64 end) {
      for (int64 i = start; i < end; ++i) {
        clip_by_global_norm_fp32(input_ptrs[i], input_lens[i], output_ptrs[i],
                                 scale);
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
    .Output("clip_grad_list: N * float")
    .Attr("clip_norm: float")
    .Attr("N: int")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      int input_n = c->num_inputs() - 1;
      for (int i = 0; i < input_n; ++i) {
        c->set_output(i, c->input(i));
      }
      return tensorflow::Status::OK();
    });

}  // namespace monolith
}  // namespace tensorflow
