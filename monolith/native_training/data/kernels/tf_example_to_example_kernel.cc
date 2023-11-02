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
#include "absl/strings/str_format.h"
#include "monolith/native_training/data/data_op_config.pb.h"
#include "monolith/native_training/data/training_instance/cc/fid.h"
#include "monolith/native_training/data/training_instance/cc/pb_variant.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace monolith_tf {

using ::monolith::io::proto::Example;
using ::monolith::io::proto::Feature;
using ::monolith::io::proto::NamedFeature;
using ::monolith::native_training::data::config::TFRecordFeatureDescription;

class TFExampleToExampleOp : public OpKernel {
 public:
  explicit TFExampleToExampleOp(OpKernelConstruction* context)
      : OpKernel(context) {
    std::string serialized;
    OP_REQUIRES_OK(context,
                   context->GetAttr("feature_description", &serialized));
    OP_REQUIRES(context, feature_description_.ParseFromString(serialized),
                errors::InvalidArgument("Corrupted data!"));
    LOG(INFO) << feature_description_.DebugString();
    const auto& s = feature_description_.sparse_features();
    const auto& d = feature_description_.dense_features();
    absl::flat_hash_set<int32_t> slot_ids, duplicates;
    for (const auto& kv : s) {
      sparse_features_.insert(kv.first);
      auto ret = slot_ids.insert(kv.second);
      if (!ret.second) {
        duplicates.insert(kv.second);
      }
    }
    dense_features_.insert(d.begin(), d.end());
    std::set<string> intersection;
    std::set_intersection(sparse_features_.begin(), sparse_features_.end(),
                          dense_features_.begin(), dense_features_.end(),
                          std::inserter(intersection, intersection.begin()));
    OP_REQUIRES(context, intersection.empty(),
                errors::InvalidArgument(absl::StrFormat(
                    "%s occur in sparse_features and dense_features "
                    "simultaneously, please investigate and retry!",
                    absl::StrJoin(intersection, ","))));
    const auto& label = feature_description_.label();
    const auto& instance_weight = feature_description_.instance_weight();
    if (!label.empty()) {
      OP_REQUIRES(context, !sparse_features_.contains(label),
                  errors::InvalidArgument(absl::StrFormat(
                      "label: {%s} should NOT occur in sparse_features, "
                      "please investigate and retry!",
                      label)));
      OP_REQUIRES(context, !dense_features_.contains(label),
                  errors::InvalidArgument(absl::StrFormat(
                      "label: {%s} should NOT occur in dense_features, "
                      "please investigate and retry!",
                      label)));
    }
    if (!instance_weight.empty()) {
      OP_REQUIRES(
          context, !sparse_features_.contains(instance_weight),
          errors::InvalidArgument(absl::StrFormat(
              "instance_weight: {%s} should NOT occur in sparse_features, "
              "please investigate and retry!",
              instance_weight)));
      OP_REQUIRES(
          context, !dense_features_.contains(instance_weight),
          errors::InvalidArgument(absl::StrFormat(
              "instance_weight: {%s} should NOT occur in dense_features, "
              "please investigate and retry!",
              instance_weight)));
    }

    OP_REQUIRES(context, duplicates.empty(),
                errors::InvalidArgument(
                    absl::StrFormat("{%s} have multiple sparse feature name "
                                    "mapping, please investigate and retry!",
                                    absl::StrJoin(duplicates, ","))));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    const auto& serialized = input_tensor.scalar<tstring>()();
    google::protobuf::Arena arena;
    auto* tf_example =
        google::protobuf::Arena::CreateMessage<tensorflow::Example>(&arena);
    OP_REQUIRES(context, tf_example->ParseFromString(serialized),
                errors::DataLoss("Corrupted data!"));
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));

    auto* example = google::protobuf::Arena::CreateMessage<Example>(&arena);
    const auto& feature_map = tf_example->features().feature();
    const auto& label_name = feature_description_.label();
    if (!label_name.empty() && !feature_map.contains(label_name)) {
      LOG(ERROR) << "label_name: " << label_name
                 << " doest not exist in tf.example.features.feature()!";
    }

    example->set_instance_weight(1.f);
    const auto& m = feature_description_.sparse_features();
    for (const auto& kv : feature_map) {
      const std::string& name = kv.first;
      const tensorflow::Feature& f = kv.second;

      // label
      if (name == feature_description_.label()) {
        example->mutable_label()->CopyFrom(f.float_list().value());
        continue;
      }

      // instance_weight
      if (name == feature_description_.instance_weight()) {
        if (!f.has_float_list()) {
          LOG(ERROR) << absl::StrFormat(
              "instance_weight: %s does not have float list!", name);
        } else if (f.float_list().value_size() != 1) {
          LOG(ERROR) << absl::StrFormat(
              "instance_weight: %s value_size should be 1", name);
        } else {
          example->set_instance_weight(f.float_list().value(0));
        }
        continue;
      }

      // sparse & dense
      if (!sparse_features_.contains(name) && !dense_features_.contains(name)) {
        continue;
      }

      NamedFeature* named_feature = example->add_named_feature();
      named_feature->set_name(name);
      // TODO(zhangbiao.david): set_sorted_id()?
      // named_feature->set_sorted_id();

      Feature* feature = named_feature->mutable_feature();
      if (sparse_features_.contains(name)) {
        int32_t slot_id = m.at(name);
        named_feature->set_id(slot_id);

        std::vector<FIDV2> fids;
        if (f.has_int64_list()) {
          fids.reserve(f.int64_list().value_size());
          for (int64_t value : f.int64_list().value()) {
            fids.push_back(FIDV2(slot_id, value));
          }
        } else if (f.has_float_list()) {
          fids.reserve(f.float_list().value_size());
          for (float value : f.float_list().value()) {
            int64_t hash_value = CalcHashValue(value);
            fids.push_back(FIDV2(slot_id, hash_value));
          }
        } else {
          LOG(ERROR) << "Only supports int64/float32 sparse features!";
        }

        for (FIDV2 fid : fids) {
          feature->mutable_fid_v2_list()->mutable_value()->Add(fid);
        }
      } else if (dense_features_.contains(name)) {
        if (f.has_int64_list()) {
          feature->mutable_int64_list()->mutable_value()->CopyFrom(
              f.int64_list().value());
        } else if (f.has_float_list()) {
          feature->mutable_float_list()->mutable_value()->CopyFrom(
              f.float_list().value());
        } else if (f.has_bytes_list()) {
          feature->mutable_bytes_list()->mutable_value()->CopyFrom(
              f.bytes_list().value());
        }
      }
    }

    output_tensor->scalar<Variant>()() = std::move(*example);
  }

 private:
  int64_t CalcHashValue(float value) const {
    return static_cast<int64_t>(std::log2(std::abs(value) + 1));
  }

  TFRecordFeatureDescription feature_description_;
  absl::flat_hash_set<std::string> sparse_features_;
  absl::flat_hash_set<std::string> dense_features_;
};

namespace {

REGISTER_KERNEL_BUILDER(Name("MonolithTFExampleToExample").Device(DEVICE_CPU),
                        TFExampleToExampleOp)

}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow
