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

#include "monolith/native_training/runtime/parameter_sync/dummy_sync_server.h"

namespace monolith {
namespace parameter_sync {

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

DummySyncServer::DummySyncServer(std::string target)
    : target_(std::move(target)), selected_port_(0) {
  grpc::EnableDefaultHealthCheckService(true);
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();
  ServerBuilder builder;

  // Listen on the given address without any authentication mechanism.
  builder.AddListeningPort(target_, grpc::InsecureServerCredentials(),
                           &selected_port_);

  // Register "service" as the instance through which we'll communicate with
  // clients. In this case it corresponds to an *synchronous* service.
  builder.RegisterService(&service_);

  // Finally assemble the server.
  server_ = builder.BuildAndStart();
  LOG(INFO) << "Server listening on " << target_ << ", selecting port "
            << selected_port_ << std::endl;
}

void DummySyncServer::Shutdown() const { server_->Shutdown(); }

const std::string& DummySyncServer::GetTarget() const { return target_; }

int DummySyncServer::GetSelectedPort() const { return selected_port_; }

}  // namespace parameter_sync
}  // namespace monolith
