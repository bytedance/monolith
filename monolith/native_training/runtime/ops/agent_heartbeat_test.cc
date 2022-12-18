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



#ifdef TEST_USE_GRPC
#include "monolith/native_training/runtime/ops/prediction_service_grpc.h"
#else
#include "monolith/native_training/runtime/ops/prediction_service_archon.h"
#endif
#include "gmock/gmock.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "gtest/gtest.h"
#include "monolith/agent_service/agent_service_mock.grpc.pb.h"
#include "monolith/native_training/runtime/ops/agent_heartbeat.h"

namespace tensorflow {
namespace monolith_tf {
namespace {

namespace tf_serving = ::tensorflow::serving;
using DoneCallback = std::function<void()>;

#ifdef TEST_USE_GRPC
using PredictionServiceType = PredictionServiceGrpc;
#else
using PredictionServiceType = PredictionServiceArchon;
#endif

using ::monolith::serving::agent_service::AddressList;
using ::monolith::serving::agent_service::AgentService;
using ::monolith::serving::agent_service::HeartBeatResponse;
using ::monolith::serving::agent_service::MockAgentServiceStub;
using ::testing::DoAll;
using ::testing::ElementsAre;
using ::testing::InSequence;
using ::testing::Pair;
using ::testing::Return;
using ::testing::SetArgPointee;
using ::testing::UnorderedElementsAre;

const absl::Duration kNoHeartbeat = absl::Hours(1000);

TEST(AgentHeartbeatTest, Basic) {
  auto stub = std::make_unique<MockAgentServiceStub>();
  {
    InSequence s;
    HeartBeatResponse resp1;
    (*resp1.mutable_addresses())["model_name"].add_address("localhost:1");
    EXPECT_CALL(*stub, HeartBeat)
        .WillOnce(DoAll(SetArgPointee<2>(resp1), Return(grpc::Status::OK)));

    HeartBeatResponse resp2;
    (*resp2.mutable_addresses())["model_name"].add_address("localhost:2");
    EXPECT_CALL(*stub, HeartBeat)
        .WillRepeatedly(DoAll(SetArgPointee<2>(resp2), Return(grpc::Status::OK)));
  }

  AgentHeartbeat<PredictionServiceType> agent(std::move(stub), kNoHeartbeat);
  for (const auto &p : agent.TestOnly_GetModelAddrs()) {
    printf("model_name = %s\n", p.first.c_str());
    for (const auto addr : p.second) {
      printf("%s ", addr.c_str());
    }
    puts("");
  }
  EXPECT_THAT(
      agent.TestOnly_GetModelAddrs(),
      UnorderedElementsAre(Pair("model_name", ElementsAre("localhost:1"))));
  agent.TestOnly_UpdateAddrs();
  EXPECT_THAT(
      agent.TestOnly_GetModelAddrs(),
      UnorderedElementsAre(Pair("model_name", ElementsAre("localhost:2"))));
}

TEST(AgentHeartbeatTest, HeartBeat) {
  auto stub = std::make_unique<MockAgentServiceStub>();
  {
    InSequence s;
    HeartBeatResponse resp1;
    (*resp1.mutable_addresses())["model_name"].add_address("localhost:1");
    EXPECT_CALL(*stub, HeartBeat)
        .WillOnce(DoAll(SetArgPointee<2>(resp1), Return(grpc::Status::OK)));

    HeartBeatResponse resp2;
    (*resp2.mutable_addresses())["model_name"].add_address("localhost:2");
    EXPECT_CALL(*stub, HeartBeat)
        .WillRepeatedly(
            DoAll(SetArgPointee<2>(resp2), Return(grpc::Status::OK)));
  }

  AgentHeartbeat<PredictionServiceType> agent(std::move(stub),
                                              absl::ZeroDuration());
  // Waits for heartbeat update.
  absl::SleepFor(absl::Seconds(0.2));
  EXPECT_THAT(
      agent.TestOnly_GetModelAddrs(),
      UnorderedElementsAre(Pair("model_name", ElementsAre("localhost:2"))));
}

TEST(AgentHeartbeatTest, DefaultInstance) {
  setenv(kAgentPortEnvVar, "1234", 1);
  AgentHeartbeat<PredictionServiceType>::GetInstance();
}

class MockPredictionService : public tf_serving::PredictionService::Service {
 public:
  MOCK_METHOD(grpc::Status, Predict,
              (grpc::ServerContext *, const tf_serving::PredictRequest *,
               tf_serving::PredictResponse *));
};

std::unique_ptr<grpc::Server> StartServer(
    tf_serving::PredictionService::Service *service, int *port) {
  grpc::ServerBuilder builder;
  builder.AddListeningPort(absl::StrCat(GetMyHostIp(), ":0"),
                           grpc::InsecureServerCredentials(), port);
  builder.RegisterService(service);
  return builder.BuildAndStart();
}

TEST(AgentHeartbeatTest, StubTest) {
  MockPredictionService service;
  EXPECT_CALL(service, Predict);

  int port;
  auto server = StartServer(&service, &port);

  auto stub = std::make_unique<MockAgentServiceStub>();
  HeartBeatResponse resp;
  (*resp.mutable_addresses())["model_name"].add_address(
      absl::StrCat(GetMyHostIp(), ":", port));
  EXPECT_CALL(*stub, HeartBeat)
      .WillRepeatedly(DoAll(SetArgPointee<2>(resp), Return(grpc::Status::OK)));

  AgentHeartbeat<PredictionServiceType> agent(std::move(stub), kNoHeartbeat);
  std::shared_ptr<PredictionServiceType> predict =
      agent.GetPredictionService("model_name");
  tf_serving::PredictRequest predict_req;
  tf_serving::PredictResponse predict_resp;
  absl::Notification notify;
  predict->Predict(
      &predict_req, &predict_resp,
      [&notify](absl::Status s, DoneCallback &&op_done) { notify.Notify(); },
      1000, [] {});
  notify.WaitForNotification();
}

TEST(AgentHeartbeatTest, ApiVersion) {
  auto stub = std::make_unique<MockAgentServiceStub>();
  HeartBeatResponse resp;
  (*resp.mutable_addresses())["ps:0"].add_address("local_host:0");
  EXPECT_CALL(*stub, HeartBeat)
      .WillRepeatedly(DoAll(SetArgPointee<2>(resp), Return(grpc::Status::OK)));
  AgentHeartbeat<PredictionServiceType> agent(std::move(stub), kNoHeartbeat);
  EXPECT_THAT(agent.api_version(), 0);
}

TEST(AgentHeartbeatTest2, ApiVersion) {
  auto stub = std::make_unique<MockAgentServiceStub>();
  HeartBeatResponse resp;
  (*resp.mutable_addresses())["model_name:ps_0"].add_address("local_host:0");
  EXPECT_CALL(*stub, HeartBeat)
      .WillRepeatedly(DoAll(SetArgPointee<2>(resp), Return(grpc::Status::OK)));
  AgentHeartbeat<PredictionServiceType> agent(std::move(stub), kNoHeartbeat);
  EXPECT_THAT(agent.api_version(), 1);
}

}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow
