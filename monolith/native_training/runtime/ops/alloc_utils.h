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

#pragma once
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
namespace tensorflow {
namespace monolith_tf {
/**
 * A tiny, fast allocator that allocates aligned output tensors,
 * each having different shape.
 * 
 * Useful when you have > 500 output tensors for your Op
 * and calls allocate_output become the bottleneck
 * 
 * How to use:
 * First, initialize your allocator with the desired alignment.
 * Note that the alignment is specified in terms of the number of elements of 
 * the corresponding dtype, not in bytes.
 * FusedAlignedOutputAllocator<EIGEN_MAX_ALIGN_BYTES / sizeof(your_dtype)> alloc;
 * 
 * Then, tell the allocator the total size of your output by calling add_slice in a loop
 * for (int i = 0; i < num_outputs; i++) {
 *   alloc.add_slice(num_elements_in_this_output);
 * }
 * 
 * Then, call .allocate. This will be a single call to ctx->allocate_temp
 * alloc.allocate(YOUR_DTYPE);
 * 
 * Finally, get each output in the same order as you call add_slice.
 * You need to specify the shape for each output. 
 * for (int i = 0; i < num_outputs; i++) {
 *   ctx->set_output(i, alloc.get_slice({DIM_SIZE1, ...}));
 * }
 * 
 * There's also a get_unaligned_total that may come in handy
 * if you want to get the total size of your output without padding
*/
template <size_t alignment>
class FusedAlignedOutputAllocator {
 public:
  explicit FusedAlignedOutputAllocator(OpKernelContext* ctx): ctx_(ctx) {
  }
  void add_slice(int64 num_elements) {
    total_ += num_elements;
    aligned_total_ += round_up_to_align(num_elements);
  }
  void allocate(DataType dtype) {
    // allocate_temp may seem suspicious here, but it's properly reference counted
    // (including its slice), so we don't need to worry about its lifetime problem
    OP_REQUIRES_OK(ctx_, ctx_->allocate_temp(dtype, {aligned_total_}, &flat_out_));
    aligned_total_ = 0;
  }
  Tensor get_slice(std::initializer_list<int64> shape) {
    int64 num_elements = 1;
#pragma unroll
    for (auto dim : shape) {
      num_elements *= dim;
    }
    Tensor reshaped;
    // note: CopyFrom and Slice doesn't copy the underlying memory
    (void)reshaped.CopyFrom(flat_out_.Slice(aligned_total_, aligned_total_ + num_elements), shape);
    aligned_total_ += round_up_to_align(num_elements);
    return reshaped;
  }
  int64 get_unaligned_total() const {
    return total_;
  }

 private:
  OpKernelContext* ctx_;
  int64 aligned_total_ = 0;
  int64 total_ = 0;
  Tensor flat_out_;
  static constexpr int64 round_up_to_align(int64 a) {
    if constexpr (alignment == 0)
      return a;
    constexpr int64 temp = alignment - 1;
    constexpr int64 temp2 = ~temp;
    return (a + temp) & temp2;
  }
};
}  // namespace monolith_tf
}  // namespace tensorflow
