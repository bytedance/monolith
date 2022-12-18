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

#include <memory>

#include "grpcpp/ext/proto_server_reflection_plugin.h"
#include "grpcpp/grpcpp.h"
#include "grpcpp/health_check_service_interface.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/platform/tstring.h"

#include "monolith/native_training/runtime/ops/parameter_sync_tf_bridge.h"
#include "monolith/native_training/runtime/parameter_sync/dummy_sync_client.h"
#include "monolith/native_training/runtime/parameter_sync/parameter_sync.pb.h"
#include "monolith/native_training/runtime/parameter_sync/parameter_sync_client.h"

namespace tensorflow {
namespace monolith_tf {

class DummySyncServerOp : public ResourceOpKernel<DummySyncServerTfBridge> {
 public:
  explicit DummySyncServerOp(OpKernelConstruction* ctx)
      : ResourceOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("address", &address_));
  }

  ~DummySyncServerOp() override = default;

 private:
  Status CreateResource(DummySyncServerTfBridge** server_bridge)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *server_bridge = new DummySyncServerTfBridge(address_);
    return Status::OK();
  };

  std::string address_;
};

class DummySyncServerShutdownOp : public OpKernel {
 public:
  explicit DummySyncServerShutdownOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    DummySyncServerTfBridge* server = nullptr;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &server));
    core::ScopedUnref unref(server);

    server->Shutdown();
    // TODO(zhangbiao.david): remove
    LOG(INFO) << server->DebugString() << " has been shutdown";

    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {1}, &output));
    auto output_vec = output->vec<int64>();
    output_vec(0) = 100;
  }
};

class DummySyncServerGetPortOp : public OpKernel {
 public:
  explicit DummySyncServerGetPortOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    DummySyncServerTfBridge* server = nullptr;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &server));
    core::ScopedUnref unref(server);

    int port = server->GetSelectedPort();
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {1}, &output));
    output->scalar<int32>()() = port;
  }
};

class DummySyncClientOp : public ResourceOpKernel<ParameterSyncClientTfBridge> {
 public:
  explicit DummySyncClientOp(OpKernelConstruction* ctx)
      : ResourceOpKernel(ctx) {}

  ~DummySyncClientOp() override = default;

 private:
  Status CreateResource(ParameterSyncClientTfBridge** client_bridge)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *client_bridge =
        new ParameterSyncClientTfBridge(true, [](const std::string& target) {
          return std::make_unique<monolith::parameter_sync::DummySyncClient>(
              target);
        });
    return Status::OK();
  };
};

class ParameterSyncClientOp
    : public ResourceOpKernel<ParameterSyncClientTfBridge> {
 public:
  explicit ParameterSyncClientOp(OpKernelConstruction* ctx)
      : ResourceOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("config", &config_serialized_));
    OP_REQUIRES(ctx, config_.ParseFromString(config_serialized_),
                errors::InvalidArgument("Unable to parse config. Make "
                                        "sure it is serialized version of "
                                        "ClientConfig"));
  }

  ~ParameterSyncClientOp() override = default;

 private:
  Status CreateResource(ParameterSyncClientTfBridge** client_bridge)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *client_bridge =
        new ParameterSyncClientTfBridge(false, [](const std::string& target) {
          return std::make_unique<
              monolith::parameter_sync::ParameterSyncClient>(target);
        });
    (*client_bridge)->TryReplace(config_.targets());

    return Status::OK();
  };

  std::string config_serialized_;

  monolith::parameter_sync::ClientConfig config_;
};

class ParameterSyncOp : public OpKernel {
 public:
  explicit ParameterSyncOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
  }

  void Compute(OpKernelContext* ctx) override {
    ParameterSyncClientTfBridge* client = nullptr;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &client));
    core::ScopedUnref unref(client);

    const Tensor* config_str;
    monolith::parameter_sync::ClientConfig config;
    OP_REQUIRES_OK(ctx, ctx->input("config_str", &config_str));
    OP_REQUIRES(ctx, config.ParseFromString(config_str->flat<tstring>()(0)),
                errors::InvalidArgument("Unable to parse config. Make "
                                        "sure it is serialized version of "
                                        "ClientConfig"));


    client->TryReplace(config.targets());
    LOG_EVERY_N_SEC(INFO, 600) << client->DebugString();
    LOG_EVERY_N_SEC(INFO, 600)
        << "ClientConfig: " << config.ShortDebugString() << std::endl;
    monolith::parameter_sync::PushResult result;
    OP_REQUIRES_OK(ctx,
                   client->Push(config.model_name(), config.signature_name(),
                                config.timeout_in_ms(), &result));

    std::string json;
    auto option = google::protobuf::util::JsonOptions();
    option.add_whitespace = true;
    option.preserve_proto_field_names = true;
    google::protobuf::util::MessageToJsonString(result, &json, option);
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {1}, &output));
    auto result_vec = output->vec<tstring>();
    result_vec(0) = json;
  }
};

REGISTER_OP("MonolithDummySyncServer")
    .Output("handle: resource")
    .Attr("address: string")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(Name("MonolithDummySyncServer").Device(DEVICE_CPU),
                        DummySyncServerOp);

REGISTER_OP("MonolithDummySyncServerShutdown")
    .Input("handle: resource")
    .Output("size: int64")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(
    Name("MonolithDummySyncServerShutdown").Device(DEVICE_CPU),
    DummySyncServerShutdownOp);

REGISTER_OP("MonolithDummySyncServerGetPort")
    .Input("handle: resource")
    .Output("size: int32")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(
    Name("MonolithDummySyncServerGetPort").Device(DEVICE_CPU),
    DummySyncServerGetPortOp);

REGISTER_OP("MonolithParameterSyncClient")
    .Output("handle: resource")
    .Attr("config: string")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(Name("MonolithParameterSyncClient").Device(DEVICE_CPU),
                        ParameterSyncClientOp);

REGISTER_OP("MonolithDummySyncClient")
    .Output("handle: resource")
    .Attr("config: string = ''")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);
REGISTER_KERNEL_BUILDER(Name("MonolithDummySyncClient").Device(DEVICE_CPU),
                        DummySyncClientOp);
REGISTER_OP("MonolithParameterSync")
    .Input("handle: resource")
    .Input("config_str: string")
    .Output("result: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(Name("MonolithParameterSync").Device(DEVICE_CPU),
                        ParameterSyncOp);

}  // namespace monolith_tf
}  // namespace tensorflow
