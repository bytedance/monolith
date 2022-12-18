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

#include "monolith/native_training/runtime/ops/multi_hash_table.h"
#include "absl/types/span.h"
#include "monolith/native_training/runtime/concurrency/queue.h"
#include "monolith/native_training/runtime/ops/embedding_hash_table_tf_bridge.h"
#include "monolith/native_training/runtime/ops/hash_filter_tf_bridge.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace monolith_tf {
namespace {

using monolith::concurrency::Queue;

Status MismatchLength(absl::string_view tensor_name, int tensor_size,
                      int expected_size) {
  return errors::InvalidArgument("The length of tensor `", tensor_name,
                                 "` doesn't equal to table num. ", tensor_size,
                                 "v.s.", expected_size);
}

Status LengthTooShort(absl::string_view tensor_name, int tensor_size) {
  return errors::InvalidArgument("The length of tensor `", tensor_name,
                                 "` is too short. Currently value",
                                 tensor_size);
}

class MultiHashTableOptimizeOp : public OpKernel {
 public:
  explicit MultiHashTableOptimizeOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* c) override {
    core::RefCountPtr<MultiHashTable> mtable;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &mtable));
    auto id_vec = c->input(1).flat<int64>();
    auto id_split = c->input(2).flat<int64>();
    OP_REQUIRES(c, id_split.size() - 1 == mtable->size(),
                MismatchLength("id", id_split.size() - 1, mtable->size()));
    auto value_vec = c->input(3).flat<float>();
    auto learning_rate_vec = c->input(4).flat<float>();
    int64 update_time = c->input(5).scalar<int64>()();
    int64 global_step = c->input(6).scalar<int64>()();
    int n = mtable->size();
    int value_offset = 0;
    int learning_rate_offset = 0;
    for (int i = 0; i < n; ++i) {
      EmbeddingHashTableTfBridge* table = mtable->table(i);
      const int num_ids = id_split(i + 1) - id_split(i);
      const int value_size =
          (id_split(i + 1) - id_split(i)) * table->dim_size();
      OP_REQUIRES(c, value_offset + value_size <= value_vec.size(),
                  LengthTooShort("value", value_vec.size()));
      auto learning_rate = absl::MakeConstSpan(
          learning_rate_vec.data() + learning_rate_offset, table->slice_size());
      learning_rate_offset += table->slice_size();
      OP_REQUIRES(c, learning_rate_offset <= learning_rate_vec.size(),
                  LengthTooShort("learning_rate", learning_rate_vec.size()));

      OP_REQUIRES_OK(c, table->BatchOptimize(
                            c, num_ids, reinterpret_cast<const int64_t*>(
                                            id_vec.data() + id_split(i)),
                            value_vec.data() + value_offset, learning_rate,
                            update_time, false, global_step));
      value_offset += value_size;
    }
    c->set_output(0, c->input(0));
  }
};

REGISTER_OP("MonolithMultiHashTableOptimize")
    .Input("mtable: resource")
    .Input("id: int64")
    .Input("id_split: int64")
    .Input("value: float")
    .Input("learning_rate: float")
    .Input("update_time: int64")
    .Input("global_step: int64")
    .Output("updated_table: resource")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(
    Name("MonolithMultiHashTableOptimize").Device(DEVICE_CPU),
    MultiHashTableOptimizeOp);

class MultiHashTableAssignOp : public OpKernel {
 public:
  explicit MultiHashTableAssignOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* c) override {
    core::RefCountPtr<MultiHashTable> mtable;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &mtable));
    auto id_vec = c->input(1).flat<int64>();
    auto id_split = c->input(2).flat<int64>();
    OP_REQUIRES(c, id_split.size() - 1 == mtable->size(),
                MismatchLength("id", id_split.size() - 1, mtable->size()));
    auto value_vec = c->input(3).flat<float>();
    int64 update_time = c->input(4).scalar<int64>()();
    int n = mtable->size();
    int value_offset = 0;
    for (int i = 0; i < n; ++i) {
      EmbeddingHashTableTfBridge* table = mtable->table(i);
      const int num_ids = id_split(i + 1) - id_split(i);
      const int value_size = num_ids * table->dim_size();
      OP_REQUIRES(c, value_offset + value_size <= value_vec.size(),
                  LengthTooShort("value", value_vec.size()));
      OP_REQUIRES_OK(
          c, table->Assign(c, num_ids, reinterpret_cast<const int64_t*>(
                                           id_vec.data() + id_split(i)),
                           value_vec.data() + value_offset, update_time));
      value_offset += value_size;
    }
    c->set_output(0, c->input(0));
  }
};

REGISTER_OP("MonolithMultiHashTableAssign")
    .Input("mtable: resource")
    .Input("id: int64")
    .Input("id_split: int64")
    .Input("value: float")
    .Input("update_time: int64")
    .Output("updated_table: resource")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(Name("MonolithMultiHashTableAssign").Device(DEVICE_CPU),
                        MultiHashTableAssignOp);

class MultiHashTableAssignAddOp : public OpKernel {
 public:
  explicit MultiHashTableAssignAddOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* c) override {
    core::RefCountPtr<MultiHashTable> mtable;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &mtable));
    auto id_vec = c->input(1).flat<int64>();
    auto id_split = c->input(2).flat<int64>();
    OP_REQUIRES(c, id_split.size() - 1 == mtable->size(),
                MismatchLength("id", id_split.size() - 1, mtable->size()));
    auto value_vec = c->input(3).flat<float>();
    int64 update_time = c->input(4).scalar<int64>()();
    int n = mtable->size();
    int value_offset = 0;
    for (int i = 0; i < n; ++i) {
      EmbeddingHashTableTfBridge* table = mtable->table(i);
      const int num_ids = id_split(i + 1) - id_split(i);
      const int value_size = num_ids * table->dim_size();
      OP_REQUIRES(c, value_offset + value_size <= value_vec.size(),
                  LengthTooShort("value", value_vec.size()));
      for (int j = id_split(i); j < id_split(i + 1); ++j) {
        auto value = absl::MakeConstSpan(value_vec.data() + value_offset,
                                         table->dim_size());
        OP_REQUIRES_OK(c, table->AssignAdd2(id_vec(j), value, update_time));
        value_offset += table->dim_size();
      }
    }
    c->set_output(0, c->input(0));
  }
};

REGISTER_OP("MonolithMultiHashTableAssignAdd")
    .Input("mtable: resource")
    .Input("id: int64")
    .Input("id_split: int64")
    .Input("value: float")
    .Input("update_time: int64")
    .Output("updated_table: resource")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(
    Name("MonolithMultiHashTableAssignAdd").Device(DEVICE_CPU),
    MultiHashTableAssignAddOp);

}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow
