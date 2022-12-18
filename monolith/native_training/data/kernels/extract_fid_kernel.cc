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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

#include "absl/hash/internal/city.h"

namespace tensorflow {
namespace monolith_tf {

class ExtractFidOp : public OpKernel {
 public:
  using OpKernel::OpKernel;
  using ConstFlatSplits = typename TTypes<int64>::ConstFlat;

  explicit ExtractFidOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("slot", &slot_));
    slot_ = slot_ << 48;
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int64>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<int64>();

    // Set all to its fid.
    const int N = input.size();
    int64 bits_left = (1ll << 49) - 1;
    for (int i = 0; i < N; i++) {
      uint64_t tmp = input(i);
      int64 hash_val =
          absl::hash_internal::CityHash64(reinterpret_cast<char*>(&tmp), 8);
      output_flat(i) = (hash_val & bits_left | slot_);
    }
  }

 private:
  int64 slot_;
};

namespace {
REGISTER_KERNEL_BUILDER(Name("ExtractFid").Device(DEVICE_CPU), ExtractFidOp);

}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow
