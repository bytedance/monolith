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

#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "monolith/native_training/runtime/ops/embedding_hash_table_tf_bridge.h"
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

using ::monolith::hash_table::EmbeddingHashTableConfig;
// using ::monolith::hopscotch::HopscotchHashSet;
using ::monolith::parameter_sync::ParameterSyncClient;

class HashTableOp : public OpKernel {
 public:
  explicit HashTableOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("config", &config_serialized_));
    OP_REQUIRES(ctx, config_.ParseFromString(config_serialized_),
                errors::InvalidArgument("Unable to parse config. Make sure it "
                                        "is serialized version of "
                                        "EmbeddingHashTableConfig."));
  }

  ~HashTableOp() override {
    if (hash_table_ != nullptr) {
      if (cinfo_.resource_is_private_to_kernel()) {
        cinfo_.resource_manager()
            ->Delete<EmbeddingHashTableTfBridge>(cinfo_.container(),
                                                 cinfo_.name())
            .IgnoreError();
      }
      // here we use different way than ResourceKernelOp. Otherwise,
      // we got crash and I believe it is our compiler's problem.
      hash_table_->Unref();
    }

    if (hash_filter_ != nullptr) {
      hash_filter_->Unref();
    }
  }

  void Compute(OpKernelContext* ctx) override {
    absl::MutexLock l(&mu_);
    if (hash_filter_ == nullptr) {
      OP_REQUIRES_OK(
          ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &hash_filter_));
    }
    if (hash_table_ == nullptr) {
      ResourceMgr* rmgr = ctx->resource_manager();
      OP_REQUIRES_OK(ctx, cinfo_.Init(rmgr, def()));

      core::RefCountPtr<ParameterSyncClientTfBridge> sync_client = nullptr;
      OP_REQUIRES_OK(
          ctx, LookupResource(ctx, HandleFromInput(ctx, 1), &sync_client));
      auto sync_client_ptr = sync_client.get();

      auto creator =
          [this, sync_client_ptr,
           ctx](EmbeddingHashTableTfBridge** out_hash_table) -> Status {
#ifdef GOOGLE_CUDA
        TF_RETURN_IF_ERROR(EmbeddingHashTableTfBridge::New(
            config_, hash_filter_, out_hash_table, cinfo_.name(),
            ctx->eigen_gpu_device().stream()));
#else
        TF_RETURN_IF_ERROR(EmbeddingHashTableTfBridge::New(
            config_, hash_filter_, out_hash_table, cinfo_.name()));
#endif
        // #ifndef GOOGLE_CUDA
        if (sync_client_ptr->IsDummySyncClient()) {
          LOG(INFO) << absl::StrFormat(
              "Hash table %s will not be attached to the sync client",
              cinfo_.name());
        } else {
          // TODO(zhangbiao.david) Make hopscotch hash set configurable
          auto* touched_key_set = sync_client_ptr->GetTouchedKeySet();
          (*out_hash_table)->SetHopscotchHashSet(touched_key_set);
          sync_client_ptr->AddHashTableResource(cinfo_.name(), *out_hash_table);
          LOG(INFO) << absl::StrFormat(
              "Hash table %s will be attached to the sync client",
              cinfo_.name());
        }
        // #endif
        return Status::OK();
      };
      OP_REQUIRES_OK(
          ctx, rmgr->LookupOrCreate<EmbeddingHashTableTfBridge>(
                   cinfo_.container(), cinfo_.name(), &hash_table_, creator));
    }
    OP_REQUIRES_OK(ctx, MakeResourceHandleToOutput(
                            ctx, 0, cinfo_.container(), cinfo_.name(),
                            TypeIndex::Make<EmbeddingHashTableTfBridge>()));
  }

 private:
  std::string config_serialized_;
  EmbeddingHashTableConfig config_;
  absl::Mutex mu_;
  EmbeddingHashTableTfBridge* hash_table_ ABSL_GUARDED_BY(mu_) = nullptr;
  HashFilterTfBridge* hash_filter_ ABSL_GUARDED_BY(mu_) = nullptr;
  ContainerInfo cinfo_ ABSL_GUARDED_BY(mu_);
};

REGISTER_OP("MonolithHashTable")
    .Input("filter_handle: resource")
    .Input("sync_client_handle: resource")
    .Output("handle: resource")
    .Attr("config: string")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);
REGISTER_KERNEL_BUILDER(Name("MonolithHashTable").Device(DEVICE_CPU),
                        HashTableOp);

}  // namespace monolith_tf
}  // namespace tensorflow