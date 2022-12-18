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

#include <cstring>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/threadpool.h"

namespace tensorflow {
namespace monolith_tf {
namespace {

// We want to create a tensor, while its lifecycle going with the step.
// So the idea here is to use step_container.
//
// The following code is borrowed from variable_ops.cc in Tensorflow.
string SharedTensorName(const string& tensor_name,
                        const FrameAndIter& control_frame) {
  if (control_frame.frame_id != kIllegalFrameId &&
      control_frame.iter_id != kIllegalIterId) {
    return strings::StrCat(tensor_name, "/frame:", control_frame.frame_id,
                           "/iter:", control_frame.iter_id);
  }
  return tensor_name;
}

struct SharedTensor : public ResourceBase {
  // Maybe we can add a mutex here if needed.
  std::string name;
  Tensor val;
  string DebugString() const override { return name; }
  int64 MemoryUsed() const override { return val.AllocatedBytes(); }
};

class UniqueKeyWitValueAndOffsetOp : public OpKernel {
 public:
  explicit UniqueKeyWitValueAndOffsetOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("dims", &dims_));
    OP_REQUIRES_OK(c, c->GetAttr("generate_buffer", &generate_buffer_));
    dims_size_ = dims_.size();
  }

  void Compute(OpKernelContext* c) override {
    auto key = c->input(0).vec<int64>();
    auto key_split = c->input(1).vec<int64>();
    OP_REQUIRES(c, key_split.size() == dims_size_ + 1,
                errors::InvalidArgument("RaggedKey should have ", dims_size_,
                                        " but got ", key_split.size() - 1));
    Tensor* t;

    std::vector<int64> unique_key;
    unique_key.reserve(key.size());

    OP_REQUIRES_OK(c, c->allocate_output(1, {dims_size_ + 1}, &t));
    auto unique_key_split_vec = t->vec<int64>();
    unique_key_split_vec(0) = 0;

    OP_REQUIRES_OK(c, c->allocate_output(2, {key.size()}, &t));
    auto value_offset_vec = t->vec<int64>();
    int64 value_offset_vec_offset = 0;

    std::vector<int64> value_offset_split;
    value_offset_split.reserve(key.size());
    value_offset_split.push_back(0);

    int64 value_offset = 0;
    absl::flat_hash_map<int64, absl::InlinedVector<int64, 4>> m;
    int j = 0;
    m.reserve(2 * (key_split(1) - key_split(0)));
    for (int i = 0;; ++i) {
      while (i == key_split(j + 1)) {
        unique_key_split_vec(j + 1) = unique_key.size();
        for (int k = unique_key_split_vec(j); k < unique_key_split_vec(j + 1);
             ++k) {
          auto it = m.find(unique_key[k]);
          for (int64 value_offset_for_key : it->second) {
            value_offset_vec(value_offset_vec_offset++) = value_offset_for_key;
          }
          value_offset_split.push_back(value_offset_vec_offset);
        }
        ++j;
        if (j < dims_size_) {
          m.clear();
          m.reserve(2 * (key_split(j + 1) - key_split(j)));
        } else {
          break;
        }
      }
      if (i == key.size()) break;
      auto it = m.find(key(i));
      if (it == m.end()) {
        m.insert({key(i), {value_offset}});
        unique_key.push_back(key(i));
      } else {
        it->second.push_back(value_offset);
      }
      value_offset += dims_[j];
    }
    OP_REQUIRES_OK(c, c->allocate_output(0, {unique_key.size()}, &t));
    auto unique_key_vec = t->vec<int64>();
    std::memcpy(unique_key_vec.data(), unique_key.data(),
                sizeof(int64) * unique_key.size());
    OP_REQUIRES_OK(c, c->allocate_output(3, {value_offset_split.size()}, &t));
    auto value_offset_split_vec = t->vec<int64>();
    std::memcpy(value_offset_split_vec.data(), value_offset_split.data(),
                sizeof(int64) * value_offset_split.size());
    OP_REQUIRES_OK(c, CreateSharedTensor(c, {value_offset}));
  }

  Status CreateSharedTensor(OpKernelContext* c, TensorShape shape) {
    if (!generate_buffer_) {
      Tensor* handle;
      TF_RETURN_IF_ERROR(c->allocate_output(4, TensorShape({}), &handle));
      handle->scalar<ResourceHandle>()() = ResourceHandle();
      return Status::OK();
    }
    const std::string unique_name =
        SharedTensorName(def().name(), c->frame_iter());
    Tensor t;
    TF_RETURN_IF_ERROR(c->allocate_temp(DataType::DT_FLOAT, shape, &t));
    SharedTensor* st = new SharedTensor();
    st->name = unique_name;
    st->val = std::move(t);
    auto* container = c->step_container();

    TF_RETURN_IF_ERROR(
        container->Create(c->resource_manager(), unique_name, st));
    Tensor* handle;
    TF_RETURN_IF_ERROR(c->allocate_output(4, TensorShape({}), &handle));
    handle->scalar<ResourceHandle>()() =
        container->MakeResourceHandle<SharedTensor>(unique_name, *c->device());
    return Status::OK();
  }

 private:
  std::vector<int> dims_;
  int dims_size_;
  bool generate_buffer_;
};

REGISTER_OP("MonolithUniqueKeyWithValueAndOffset")
    .Input("key: int64")
    .Input("key_split: int64")
    .Output("unique_key: int64")
    .Output("unique_key_split: int64")
    .Output("value_offset: int64")
    .Output("value_offset_split: int64")
    .Output("value_buffer: resource")
    .Attr("dims: list(int)")
    .Attr("generate_buffer: bool")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Vector(c->UnknownDim()));
      c->set_output(1, c->input(1));
      c->set_output(2, c->input(0));
      c->set_output(3, c->Vector(c->UnknownDim()));
      c->set_output(4, c->Scalar());
      return Status::OK();
    });

REGISTER_KERNEL_BUILDER(
    Name("MonolithUniqueKeyWithValueAndOffset").Device(DEVICE_CPU),
    UniqueKeyWitValueAndOffsetOp);

class FinallizeSharedTensorOp : public OpKernel {
 public:
  explicit FinallizeSharedTensorOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    core::RefCountPtr<SharedTensor> st = nullptr;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &st));
    c->set_output(0, st->val);
    OP_REQUIRES_OK(c, DeleteResource(c, HandleFromInput(c, 0)));
  }
};

REGISTER_OP("MonolithFinalizeSharedTensor")
    .Input("handle: num_tensors * resource")
    .Output("t: dtype")
    .Attr("shape: shape")
    .Attr("dtype: type")
    .Attr("num_tensors: int")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ExplicitShape);

REGISTER_KERNEL_BUILDER(Name("MonolithFinalizeSharedTensor").Device(DEVICE_CPU),
                        FinallizeSharedTensorOp);

class FillWithOffsetMapOp : public OpKernel {
 public:
  explicit FillWithOffsetMapOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("dims", &dims_));
  }

  void Compute(OpKernelContext* c) override {
    auto pos = c->input(0).vec<int64>();
    auto pos_split = c->input(1).vec<int64>();
    auto value = c->input(2).vec<float>();
    auto value_offset_map = c->input(3).vec<int64>();
    auto value_offset_map_split = c->input(4).vec<int64>();
    core::RefCountPtr<SharedTensor> st = nullptr;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 5), &st));
    auto value_buffer = st->val.vec<float>();
    int64 value_offset = 0;
    OP_REQUIRES(
        c, pos_split.size() == dims_.size() + 1,
        errors::InvalidArgument("Pos's first dim doesn't match dim size. ",
                                pos_split.size() - 1, " v.s. ", dims_.size()));
    int j = 0;
    for (int i = 0; i < pos.size(); ++i) {
      while (i == pos_split(j + 1)) {
        ++j;
      }
      OP_REQUIRES(
          c, pos(i) < value_offset_map.size(),
          errors::InvalidArgument("pos is bigger than offset size. ", pos(i),
                                  " v.s. ", value_offset_map.size()));
      const int64 value_offset_end = value_offset + dims_[j];
      OP_REQUIRES(c, value_offset_end <= value.size(),
                  errors::InvalidArgument(FormatValueError(pos_split, value)));
      for (int64 offset_pos = value_offset_map_split(pos(i));
           offset_pos < value_offset_map_split(pos(i) + 1); ++offset_pos) {
        std::memcpy(value_buffer.data() + value_offset_map(offset_pos),
                    value.data() + value_offset, dims_[j] * sizeof(float));
      }
      value_offset = value_offset_end;
    }
    c->set_output(0, c->input(5));
  }

 private:
  std::string FormatValueError(TTypes<const int64>::Vec pos_split,
                               TTypes<const float>::Vec value) {
    std::string s;
    std::vector<int64> pos_split_vec;
    for (int i = 0; i < pos_split.size(); ++i) {
      pos_split_vec.push_back(pos_split(i));
    }
    int64 expected_size = 0;
    for (int i = 0; i < dims_.size(); ++i) {
      expected_size += (pos_split_vec[i + 1] - pos_split_vec[i]) * dims_[i];
    }
    absl::StrAppend(&s,
                    absl::StrFormat("Value size doesn't match expected size. "
                                    "expected: %d, actual: %d. \n",
                                    expected_size, value.size()));
    absl::StrAppend(&s, absl::StrFormat("Pos split: %s",
                                        absl::StrJoin(pos_split_vec, ",")));
    return s;
  }

  std::vector<int> dims_;
};

REGISTER_OP("MonolithFillWithOffsetMap")
    .Input("pos: int64")
    .Input("pos_split: int64")
    .Input("value : float")
    .Input("value_offset_map: int64")
    .Input("value_offset_map_split: int64")
    .Input("value_buffer: resource")
    .Output("out_value_buffer: resource")
    .Attr("dims: list(int)")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(Name("MonolithFillWithOffsetMap").Device(DEVICE_CPU),
                        FillWithOffsetMapOp);

class FillWithOffsetMapGradientOp : public OpKernel {
 public:
  explicit FillWithOffsetMapGradientOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("dims", &dims_));
  }

  void Compute(OpKernelContext* c) override {
    auto pos = c->input(0).vec<int64>();
    auto pos_split = c->input(1).vec<int64>();
    auto grad = c->input(2).vec<float>();
    auto grad_offset_map = c->input(3).vec<int64>();
    auto grad_offset_map_split = c->input(4).vec<int64>();

    int64 bgrad_size = 0;
    for (int j = 0; j < dims_.size(); ++j) {
      bgrad_size += dims_[j] * (pos_split(j + 1) - pos_split(j));
    }
    Tensor* t;
    OP_REQUIRES_OK(c, c->allocate_output(0, {bgrad_size}, &t));
    auto bgrad = t->vec<float>();
    bgrad.setZero();
    int64 bgrad_offset = 0;

    int j = 0;
    for (int i = 0; i < pos.size(); ++i) {
      while (i == pos_split(j + 1)) {
        ++j;
      }
      OP_REQUIRES(
          c, pos(i) < grad_offset_map.size(),
          errors::InvalidArgument("pos is bigger than offset size. ", pos(i),
                                  " v.s. ", grad_offset_map.size()));
      for (int64 offset_pos = grad_offset_map_split(pos(i));
           offset_pos < grad_offset_map_split(pos(i) + 1); ++offset_pos) {
        const int64 grad_offset = grad_offset_map(offset_pos);
        for (int k = 0; k < dims_[j]; ++k) {
          bgrad(bgrad_offset + k) += grad(grad_offset + k);
        }
      }
      bgrad_offset += dims_[j];
    }
  }

 private:
  std::vector<int> dims_;
};

REGISTER_OP("MonolithFillWithOffsetMapGradient")
    .Input("pos: int64")
    .Input("pos_split: int64")
    .Input("grad: float")
    .Input("grad_offset_map: int64")
    .Input("grad_offset_map_split: int64")
    .Output("backprop_grad: float")
    .Attr("dims: list(int)")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Vector(c->UnknownDim()));
      return Status::OK();
    });

REGISTER_KERNEL_BUILDER(
    Name("MonolithFillWithOffsetMapGradient").Device(DEVICE_CPU),
    FillWithOffsetMapGradientOp);

class FusedValueRowidsOp : public OpKernel {
 public:
  explicit FusedValueRowidsOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    auto splits = c->input(0).vec<int64>();
    Tensor* t;
    const int len = splits.size() - 1;
    OP_REQUIRES_OK(c, c->allocate_output(0, {splits(len)}, &t));
    auto rowids = t->vec<int64>();
    for (int64 i = 0; i < len; ++i) {
      for (int64 j = splits(i); j < splits(i + 1); ++j) {
        rowids(j) = i;
      }
    }
  }
};

REGISTER_OP("MonolithFusedValueRowids")
    .Input("splits: int64")
    .Output("rowids: int64")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Vector(c->UnknownDim()));
      return Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name("MonolithFusedValueRowids").Device(DEVICE_CPU),
                        FusedValueRowidsOp);

}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow
