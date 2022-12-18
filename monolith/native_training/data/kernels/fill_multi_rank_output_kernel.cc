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

#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/hash/internal/city.h"
#include "absl/strings/str_format.h"
#include "monolith/native_training/data/kernels/internal/label_utils.h"
#include "monolith/native_training/data/training_instance/cc/instance_utils.h"
#include "monolith/native_training/data/training_instance/cc/pb_variant.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace monolith_tf {

using IFeature = ::idl::matrix::proto::Feature;
using Instance = ::parser::proto::Instance;
using Example = ::monolith::io::proto::Example;
using LineId = ::idl::matrix::proto::LineId;

class FillMultiRankOutputOp : public OpKernel {
 public:
  explicit FillMultiRankOutputOp(OpKernelConstruction *context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("enable_draw_as_rank",
                                             &enable_draw_as_rank_));
    OP_REQUIRES_OK(context, context->GetAttr("enable_chnid_as_rank",
                                             &enable_chnid_as_rank_));
    OP_REQUIRES_OK(context, context->GetAttr("enable_lineid_rank_as_rank",
                                             &enable_lineid_rank_as_rank_));
    if (!(enable_draw_as_rank_ || enable_chnid_as_rank_ ||
          enable_lineid_rank_as_rank_)) {
      LOG(FATAL)
          << "At least one of enable_draw_as_rank, enable_chnid_as_rank, "
             "enable_lineid_rank_as_rank must be set";
    }

    OP_REQUIRES_OK(context, context->GetAttr("rank_num", &rank_num_));
    OP_REQUIRES_OK(context, context->GetAttr("variant_type", &variant_type_));

    if (variant_type_ != "instance" && variant_type_ != "example") {
      LOG(FATAL) << "Invalid 'variant_type', please choose on from "
                    "['instance', 'example']!";
    }
  }

  void Compute(OpKernelContext *context) override {
    /* Parse data fields from input tensor. */
    const Tensor &input_tensor = context->input(0);
    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));

    bool is_instance = variant_type_ == "instance";
    if (is_instance) {
      Instance instance;
      instance.CopyFrom(*input_tensor.scalar<Variant>()().get<Instance>());
      output_tensor->scalar<Variant>()() = std::move(instance);
    } else {
      Example example;
      example.CopyFrom(*input_tensor.scalar<Variant>()().get<Example>());
      output_tensor->scalar<Variant>()() = std::move(example);
    }

    LineId *line_id = GetLineId(output_tensor, is_instance);
    auto label = GetLabel(output_tensor, is_instance);

    /* fill_multi_rank_output() from matrix processor:
     */

    if (enable_draw_as_rank_) {
      int rank = line_id->is_draw() ? 1 : 0;
      label->Add(rank);
      return;
    }
    if (enable_chnid_as_rank_) {
      int rank = 0, chnid = line_id->chnid();
      if (chnid == 0 || chnid == 1) {
        rank = chnid;
      } else {
        rank = 2;
      }
      label->Add(rank);
      return;
    }
    if (enable_lineid_rank_as_rank_) {
      int rank = line_id->rank();
      if (rank >= rank_num_) {
        rank = rank_num_ - 1;
      }
      label->Add(rank);
      return;
    }
  }

 private:
  static LineId *GetLineId(Tensor *output_tensor, bool is_instance) {
    if (is_instance) {
      return output_tensor->scalar<Variant>()()
          .get<Instance>()
          ->mutable_line_id();
    } else {
      return output_tensor->scalar<Variant>()()
          .get<Example>()
          ->mutable_line_id();
    }
  }

  static ::google::protobuf::RepeatedField<float> *GetLabel(
      Tensor *output_tensor, bool is_instance) {
    if (is_instance) {
      return output_tensor->scalar<Variant>()()
          .get<Instance>()
          ->mutable_label();
    } else {
      return output_tensor->scalar<Variant>()().get<Example>()->mutable_label();
    }
  }

  static std::vector<uint64_t> GetFids(Tensor *output_tensor,
                                       bool is_instance) {
    std::vector<uint64_t> fids;
    if (is_instance) {
      auto instance = output_tensor->scalar<Variant>()().get<Instance>();
      for (uint64_t fid : instance->fid()) {
        fids.push_back(fid);
      }
    } else {
      auto example = output_tensor->scalar<Variant>()().get<Example>();
      for (const auto &named_feature : example->named_feature()) {
        if (named_feature.feature().has_fid_v1_list()) {
          for (const auto &fid :
               named_feature.feature().fid_v1_list().value()) {
            fids.push_back(fid);
          }
        }
      }
    }
    return fids;
  }

  bool enable_draw_as_rank_;
  bool enable_chnid_as_rank_;
  bool enable_lineid_rank_as_rank_;
  int rank_num_;
  std::string variant_type_;
};

namespace {

REGISTER_KERNEL_BUILDER(Name("FillMultiRankOutput").Device(DEVICE_CPU),
                        FillMultiRankOutputOp)

}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow
