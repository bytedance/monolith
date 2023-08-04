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

#include "monolith/native_training/data/kernels/transform_dataset_kernel.h"

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/stats_aggregator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/str_util.h"

#include "monolith/native_training/data/kernels/internal/label_utils.h"
#include "monolith/native_training/data/training_instance/cc/pb_variant.h"
#include "monolith/native_training/data/transform/cc/transforms.h"
#include "monolith/native_training/runtime/common/linalg_utils.h"
#include "third_party/nlohmann/json.hpp"

namespace tensorflow {
namespace data {
namespace monolith_tf {
using IFeature = ::idl::matrix::proto::Feature;
using Instance = ::parser::proto::Instance;
using Example = ::monolith::io::proto::Example;
using EFeature = ::monolith::io::proto::Feature;
using LineId = ::idl::matrix::proto::LineId;
using Action = google::protobuf::RepeatedField<int>;
using ::monolith::common::IsAlmostEqual;
using monolith::native_training::data::TransformConfig;
using tensorflow::monolith_tf::NewTransformFromConfig;
using tensorflow::monolith_tf::TransformInterface;

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

/* static */ constexpr const char *const TransformDatasetOp::kDatasetType;
/* static */ constexpr const char *const TransformDatasetOp::kInputDataset;
/* static */ constexpr const char *const TransformDatasetOp::kConfig;
/* static */ constexpr const char *const TransformDatasetOp::kVariantType;

class TransformDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext *ctx, const DatasetBase *input,
          std::string config_serialized, std::string variant_type)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        config_serialized_(std::move(config_serialized)),
        variant_type_(std::move(variant_type)) {
    input_->Ref();
    OP_REQUIRES(ctx, config_.ParseFromString(config_serialized_),
                errors::InvalidArgument("Unable to parse config. Make sure it "
                                        "is serialized version of "
                                        "TransformConfig."));
    transform_ = NewTransformFromConfig(config_);
  }

  ~Dataset() override { input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string &prefix) const override {
    return absl::make_unique<Iterator>(
        Iterator::Params{this, strings::StrCat(prefix, "::", kDatasetType)});
  }

  const DataTypeVector &output_dtypes() const override {
    return input_->output_dtypes();
  }

  const std::vector<PartialTensorShape> &output_shapes() const override {
    return input_->output_shapes();
  }

  string DebugString() const override {
    return "This is the customized Dataset: Mixup";
  }

  Status InputDatasets(
      std::vector<const DatasetBase *> *inputs) const override {
    inputs->push_back(input_);
    return Status::OK();
  }

  Status CheckExternalState() const override {
    return input_->CheckExternalState();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext *ctx,
                            DatasetGraphDefBuilder *b,
                            Node **output) const override {
    Node *input_graph_node;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));

    AttrValue config_node;
    b->BuildAttrValue(config_serialized_, &config_node);
    AttrValue variant_type_node;
    b->BuildAttrValue(variant_type_, &variant_type_node);

    TF_RETURN_IF_ERROR(b->AddDataset(
        this, {input_graph_node},
        {{kConfig, config_node}, {kVariantType, variant_type_node}}, output));

    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params &params)
        : DatasetIterator<Dataset>(params) {}

    Status Initialize(IteratorContext *ctx) override {
      return dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_);
    }

    Status GetNextInternal(IteratorContext *ctx,
                           std::vector<Tensor> *out_tensors,
                           bool *end_of_sequence) override {
      out_tensors->clear();
      out_tensors->reserve(1);
      tensorflow::mutex_lock l(mu_);

      Status status;
      if (dataset()->variant_type_ == "instance") {
        status = NextInternalImpl<Instance>(ctx, out_tensors, end_of_sequence);
      } else {
        status = NextInternalImpl<Example>(ctx, out_tensors, end_of_sequence);
      }

      return status;
    }

    template <typename T>
    Status NextInternalImpl(IteratorContext *ctx,
                            std::vector<Tensor> *out_tensors,
                            bool *end_of_sequence) {
      while (!*end_of_sequence) {
        std::vector<Tensor> batch_variant;
        TF_RETURN_IF_ERROR(
            input_impl_->GetNext(ctx, &batch_variant, end_of_sequence));

        if (!*end_of_sequence) {
          T *instance_or_example = GetCurrent<T>(&batch_variant.back());
          std::shared_ptr<T> instance_or_example_ptr;
          instance_or_example_ptr.reset(instance_or_example, [](...) {});
          std::vector<std::shared_ptr<T>> instance_or_example_list;
          dataset()->transform_->Transform(instance_or_example_ptr,
                                           &instance_or_example_list);
          if (!instance_or_example_list.empty()) {
            CHECK_EQ(instance_or_example_list.size(), 1);
            out_tensors->push_back(batch_variant.back());
            return Status::OK();
          }
        }
      }

      return Status::OK();
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext *ctx, model::Node::Args args) const override {
      return model::MakeUnknownRatioNode(std::move(args));
    }

    Status SaveInternal(SerializationContext *ctx,
                        IteratorStateWriter *writer) override {
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext *ctx,
                           IteratorStateReader *reader) override {
      return Status::OK();
    }

   private:
    template <typename T>
    inline T *GetCurrent(Tensor *t) {
      Variant *variant = &t->scalar<Variant>()();
      return variant->get<T>();
    }

    tensorflow::mutex mu_;
    std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
  };

  const DatasetBase *const input_;
  std::string config_serialized_;
  TransformConfig config_;
  std::string variant_type_;
  std::unique_ptr<TransformInterface> transform_;
};

TransformDatasetOp::TransformDatasetOp(OpKernelConstruction *ctx)
    : UnaryDatasetOpKernel(ctx) {
  std::string config_serialized;
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kConfig, &config_serialized));
  OP_REQUIRES(ctx, config_.ParseFromString(config_serialized),
              errors::InvalidArgument("Unable to parse config. Make sure it "
                                      "is serialized version of "
                                      "TransformConfig."));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kVariantType, &variant_type_));
  LOG(INFO) << "variant_type: " << variant_type_ << ", config: \n"
            << config_.DebugString();
}

void TransformDatasetOp::MakeDataset(OpKernelContext *ctx, DatasetBase *input,
                                     DatasetBase **output) {
  *output = new Dataset(ctx, input, config_.SerializeAsString(), variant_type_);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("TransformDataset").Device(DEVICE_CPU),
                        TransformDatasetOp)
}  // namespace

}  // namespace monolith_tf
}  // namespace data
}  // namespace tensorflow
