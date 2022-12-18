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

#ifndef MONOLITH_NATIVE_TRAINING_RUNTIME_OPS_CLIP_BY_GLOBAL_NORM
#define MONOLITH_NATIVE_TRAINING_RUNTIME_OPS_CLIP_BY_GLOBAL_NORM

#if defined(_ENABLE_AVX) && defined(__AVX__)
#include <immintrin.h>
#endif

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace monolith {

template <typename Device>
struct ClipByGlobalNormImpl {
  static void Compute(OpKernelContext* context,
                      const std::vector<const float*>& input_ptrs,
                      const std::vector<int>& input_lens,
                      const std::vector<float*>& output_ptrs, float global_norm,
                      float clip_norm);
};

template <typename Device>
class ClipByGlobalNorm : public OpKernel {
 public:
  explicit ClipByGlobalNorm(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("clip_norm", &clip_norm_));
  }

  void Compute(OpKernelContext* context) override {
    VLOG(1) << "In ClipByGlobalNorm Computation";

    auto num_inputs = context->num_inputs() - 1;
    std::vector<const float*> input_ptrs(num_inputs);
    std::vector<int> input_lens(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      input_ptrs[i] = context->input(i).flat<float>().data();
      input_lens[i] = context->input(i).NumElements();
    }

    std::vector<float*> output_ptrs(num_inputs);
    float global_norm = context->input(num_inputs).scalar<float>().data()[0];
    if (global_norm > clip_norm_) {
      for (int i = 0; i < num_inputs; ++i) {
        Tensor* temp;
        OP_REQUIRES_OK(context, context->allocate_output(
                                    i, context->input(i).shape(), &temp));
        output_ptrs[i] = temp->flat<float>().data();
      }
      ClipByGlobalNormImpl<Device>::Compute(context, input_ptrs, input_lens,
                                            output_ptrs,
                                            clip_norm_ / global_norm);
    } else {
      // If no clip, output as input.
      for (int i = 0; i < num_inputs; ++i) {
        context->set_output(i, context->input(i));
      }
    }
  }

 private:
  float clip_norm_;
};

}  // namespace monolith
}  // namespace tensorflow

#endif  // MONOLITH_NATIVE_TRAINING_RUNTIME_OPS_CLIP_BY_GLOBAL_NORM
