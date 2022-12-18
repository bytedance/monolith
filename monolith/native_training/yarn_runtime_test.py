# Copyright 2022 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from concurrent import futures
import os
import types
import unittest
from unittest import mock

import grpc

from monolith.native_training import yarn_runtime
from monolith.native_training.proto import primus_am_service_pb2
from monolith.native_training.proto import primus_am_service_pb2_grpc


class YarnRuntimeTest(unittest.TestCase):

  @mock.patch.dict(os.environ, {"YARN_INET_ADDR": "1.2.3.4"})
  def test_get_local_host_overwrite(self):
    self.assertEqual(yarn_runtime.get_local_host(), "1.2.3.4")

  @mock.patch.dict(os.environ, {"CLOUDNATIVE_INET_ADDR": "1.2.3.4,5.6.7.8"})
  def test_get_local_host_overwrite_by_cloudnative(self):
    self.assertEqual(yarn_runtime.get_local_host(), "1.2.3.4")

  def test_get_local_host_basic(self):
    yarn_runtime.get_local_host()

  @mock.patch.dict(os.environ, {
      "PRIMUS_AM_RPC_HOST": "unix",
      "PRIMUS_AM_RPC_PORT": "test_kill"
  })
  def test_kill(self):
    servicer = primus_am_service_pb2_grpc.AppMasterServiceServicer()

    called = False
    reason = "TestKill"

    def kill(servicer_self, request, context):
      nonlocal called
      called = True
      self.assertEqual(request.diagnose, reason)
      return primus_am_service_pb2.KillResponse()

    servicer.kill = types.MethodType(kill, servicer)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    primus_am_service_pb2_grpc.add_AppMasterServiceServicer_to_server(
        servicer, server)
    addr = "unix:test_kill"
    server.add_insecure_port(addr)
    server.start()
    yarn_runtime.maybe_kill_application(reason)
    self.assertTrue(called)
    server.stop(True)

  @mock.patch.dict(os.environ, {
      "PRIMUS_AM_RPC_HOST": "unix",
      "PRIMUS_AM_RPC_PORT": "test_succeed"
  })
  def test_finish(self):
    servicer = primus_am_service_pb2_grpc.AppMasterServiceServicer()
    called = False

    def succeed(self, request, context):
      nonlocal called
      called = True
      return primus_am_service_pb2.SucceedResponse()

    servicer.succeed = types.MethodType(succeed, servicer)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    primus_am_service_pb2_grpc.add_AppMasterServiceServicer_to_server(
        servicer, server)
    addr = "unix:test_succeed"
    server.add_insecure_port(addr)
    server.start()
    yarn_runtime.maybe_finish_application()
    self.assertTrue(called)
    server.stop(True)

  @mock.patch.dict(os.environ, {
      "PRIMUS_AM_RPC_HOST": "unix",
      "PRIMUS_AM_RPC_PORT": "test_save_primus"
  })
  def test_save_primus(self):
    servicer = primus_am_service_pb2_grpc.AppMasterServiceServicer()
    create_called = False
    status_called = False
    dst = "test"

    def createSavepoint(self, request, context):
      nonlocal create_called
      create_called = True
      resp = primus_am_service_pb2.CreateSavepointResponse()
      resp.savepoint_id = "123"
      return resp

    def createSavepointStatus(self, request, context):
      nonlocal status_called
      status_called = True
      resp = primus_am_service_pb2.CreateSavepointStatusResponse()
      resp.create_savepoint_state = primus_am_service_pb2.CreateSavepointStatusResponse.CreateSavepointState.SUCCEEDED
      return resp

    servicer.createSavepoint = types.MethodType(createSavepoint, servicer)
    servicer.createSavepointStatus = types.MethodType(createSavepointStatus,
                                                      servicer)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    primus_am_service_pb2_grpc.add_AppMasterServiceServicer_to_server(
        servicer, server)
    addr = "unix:test_save_primus"
    server.add_insecure_port(addr)
    server.start()
    resp = yarn_runtime.create_primus_save_point(dst)
    self.assertTrue(resp)
    self.assertTrue(create_called)
    self.assertTrue(status_called)
    server.stop(True)


if __name__ == "__main__":
  unittest.main()
