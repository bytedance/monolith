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

#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>

#include "absl/hash/internal/city.h"
#include "monolith/native_training/data/training_instance/cc/pb_variant.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace monolith_tf {
using NamedRawFeature = ::monolith::io::proto::NamedRawFeature;
using RawFeature = ::monolith::io::proto::RawFeature;
using NamedFeature = ::monolith::io::proto::NamedFeature;
using Feature = ::monolith::io::proto::Feature;
using Example = ::monolith::io::proto::Example;

class FeatureHashOp : public OpKernel {
 public:
  using OpKernel::OpKernel;
  using ConstFlatSplits = typename TTypes<int64>::ConstFlat;

  explicit FeatureHashOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    std::vector<std::string> names;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("names", &names));
    names_.insert(names.begin(), names.end());
  }

  void Compute(OpKernelContext *context) override {
    // Grab the input tensor
    const Tensor *pb_input;
    OP_REQUIRES_OK(context, context->input("input", &pb_input));
    TTypes<Variant>::ConstVec pb_variant_tensor = pb_input->vec<Variant>();
    const int batch_size = pb_variant_tensor.dimension(0);

    // Create an output tensor
    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, pb_input->shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<Variant>();

    for (int i = 0; i < batch_size; ++i) {
      const Example *in_pb = pb_variant_tensor(i).get<Example>();
      Example out_pb;
      out_pb.mutable_line_id()->CopyFrom(in_pb->line_id());
      out_pb.mutable_label()->CopyFrom(in_pb->label());
      for (size_t i = 0; i < in_pb->named_raw_feature_size(); ++i) {
        const NamedRawFeature &named_raw_feature = in_pb->named_raw_feature(i);
        std::string name = named_raw_feature.name();
        if (names_.find(name) == names_.end()) continue;

        NamedFeature *out_nf = out_pb.add_named_feature();
        out_nf->set_id(named_raw_feature.id());
        out_nf->set_name(name);
        raw_feature_to_feature(name, named_raw_feature.raw_feature(),
                               out_nf->mutable_feature());
      }

      output_flat(i) = std::move(out_pb);
    }
  }

 private:
  std::unordered_set<std::string> names_;

  void raw_feature_to_feature(const std::string &name,
                              const RawFeature &raw_feature, Feature *feature) {
    for (size_t i = 0; i < raw_feature.feature_size(); ++i) {
      const auto &rf = raw_feature.feature(i);
      if (rf.has_float_list()) {
        feature->mutable_float_list()->MergeFrom(rf.float_list());
      }

      if (rf.has_double_list()) {
        feature->mutable_double_list()->MergeFrom(rf.double_list());
      }

      if (rf.has_int64_list()) {
        feature->mutable_int64_list()->MergeFrom(rf.int64_list());
      }

      if (rf.has_bytes_list()) {
        const auto &bytes_list = rf.bytes_list();
        auto *out_list = feature->mutable_fid_v2_list();
        for (size_t j = 0; j < bytes_list.value_size(); ++j) {
          const std::string &value =
              absl::StrCat(bytes_list.value(j), "-", name);
          int64 hash_val = absl::hash_internal::CityHash64(value.c_str(), 8);
          out_list->add_value(hash_val);
        }
      }

      if (rf.has_float_lists()) {
        feature->mutable_float_lists()->MergeFrom(rf.float_lists());
      }

      if (rf.has_double_lists()) {
        feature->mutable_double_lists()->MergeFrom(rf.double_lists());
      }

      if (rf.has_int64_lists()) {
        feature->mutable_int64_lists()->MergeFrom(rf.int64_lists());
      }

      if (rf.has_bytes_lists()) {
        const auto &bytes_lists = rf.bytes_lists();
        for (size_t j = 0; j < bytes_lists.list_size(); ++j) {
          const auto &bytes_list = bytes_lists.list(j);
          auto *out_list = feature->mutable_fid_v2_lists()->add_list();
          for (size_t k = 0; k < bytes_list.value_size(); ++k) {
            const std::string &value =
                absl::StrCat(bytes_list.value(j), "-", name);
            int64 hash_val = absl::hash_internal::CityHash64(value.c_str(), 8);
            out_list->add_value(hash_val);
          }
        }
      }
    }
  }
};

namespace {
REGISTER_KERNEL_BUILDER(Name("FeatureHash").Device(DEVICE_CPU), FeatureHashOp);

}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow
