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
#include <string>

#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "monolith/native_training/runtime/hash_table/embedding_hash_table.pb.h"
#include "monolith/native_training/runtime/ops/embedding_hash_table_tf_bridge.h"
#include "monolith/native_training/runtime/ops/multi_hash_table.h"
#include "monolith/native_training/runtime/ops/parameter_sync_tf_bridge.h"
#include "monolith/native_training/runtime/parameter_sync/parameter_sync_client.h"
#include "tensorflow/core/framework/lookup_interface.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/default/integral_types.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace monolith_tf {

using ::monolith::hash_table::MultiEmbeddingHashTableConfig;
using ::monolith::parameter_sync::ParameterSyncClient;

class CreateMultiHashTableOp : public ResourceOpKernel<MultiHashTable> {
 public:
  explicit CreateMultiHashTableOp(OpKernelConstruction* ctx)
      : ResourceOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    absl::MutexLock l(&mu_);
    if (resource_ == nullptr) {
      const tstring& serialized_config = ctx->input(0).scalar<tstring>()();
      OP_REQUIRES(ctx, config_.ParseFromArray(serialized_config.data(),
                                              serialized_config.size()),
                  errors::InvalidArgument("Unable to parse config."));
      n_ = config_.names_size();
      OP_REQUIRES(
          ctx, config_.configs_size() == n_,
          errors::InvalidArgument(
              "`table_configs` size must equal to `N`, got filter_handles (",
              config_.names_size(), ") vs N (", n_, ")"));

      const auto& filter_handle = ctx->input(1).scalar<ResourceHandle>()();
      OP_REQUIRES_OK(ctx, LookupResource(ctx, filter_handle, &hash_filter_));

      const auto& sync_client_handle = ctx->input(2).scalar<ResourceHandle>()();
      OP_REQUIRES_OK(ctx,
                     LookupResource(ctx, sync_client_handle, &sync_client_));
    }
    ResourceOpKernel::Compute(ctx);
  }

  Status CreateResource(MultiHashTable** resource)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    auto* mtable = new MultiHashTable(cinfo_.name());
    for (int i = 0; i < n_; i++) {
      EmbeddingHashTableTfBridge* hash_table;
      TF_RETURN_IF_ERROR(EmbeddingHashTableTfBridge::New(
          config_.configs(i), hash_filter_.get(), &hash_table,
          config_.names(i)));
      hash_table->SetHopscotchHashSet(sync_client_->GetTouchedKeySet());
      mtable->add_table(
          config_.names(i),
          core::RefCountPtr<EmbeddingHashTableTfBridge>(hash_table));
    }
    sync_client_->SetMultiHashTableResource(mtable);
    *resource = mtable;
    return Status::OK();
  }

 private:
  absl::Mutex mu_;
  int n_ ABSL_GUARDED_BY(mu_);
  MultiEmbeddingHashTableConfig config_ ABSL_GUARDED_BY(mu_);
  core::RefCountPtr<HashFilterTfBridge> hash_filter_ ABSL_GUARDED_BY(mu_);
  core::RefCountPtr<ParameterSyncClientTfBridge> sync_client_
      ABSL_GUARDED_BY(mu_);
};

REGISTER_OP("CreateMonolithMultiHashTable")
    .Input("config: string")
    .Input("filter_handle: resource")
    .Input("sync_client_handle: resource")
    .Output("multi_hash_table_handle: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(Name("CreateMonolithMultiHashTable").Device(DEVICE_CPU),
                        CreateMultiHashTableOp);

class ReadMultiHashTableOp : public OpKernel {
 public:
  explicit ReadMultiHashTableOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* ctx) override {
    absl::MutexLock l(&mu_);
    if (cinfo_.name().empty()) {
      OP_REQUIRES_OK(ctx, cinfo_.Init(ctx->resource_manager(), def()));
    }
    OP_REQUIRES_OK(ctx, MakeResourceHandleToOutput(
                            ctx, 0, cinfo_.container(), cinfo_.name(),
                            TypeIndex::Make<MultiHashTable>()));
  }

 private:
  absl::Mutex mu_;
  ContainerInfo cinfo_ TF_GUARDED_BY(mu_);
};

REGISTER_OP("ReadMonolithMultiHashTable")
    .Output("multi_hash_table_handle: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(Name("ReadMonolithMultiHashTable").Device(DEVICE_CPU),
                        ReadMultiHashTableOp);

}  // namespace monolith_tf
}  // namespace tensorflow
