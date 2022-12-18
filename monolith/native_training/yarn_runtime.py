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

"""Functionalities will help to get/set some information from the yarn runtime."""

import collections
import os
import socket
import time

from absl import logging
import grpc

from monolith.native_training.proto import primus_am_service_pb2
from monolith.native_training.proto import primus_am_service_pb2_grpc


def get_local_host():
  if "CLOUDNATIVE_INET_ADDR" in os.environ:
    ips = os.environ["CLOUDNATIVE_INET_ADDR"]
    local_host = ips.split(",")[0]
  elif "YARN_INET_ADDR" in os.environ:
    local_host = os.environ["YARN_INET_ADDR"]
  else:
    try:
      local_host = socket.gethostbyname(socket.gethostname())
    except socket.gaierror as e:
      logging.info('Failed to get ipv4, try to get ipv6 instead')
      local_host = socket.getaddrinfo(socket.gethostname(), None,
                                      socket.AF_INET6)[0][4][0]
      local_host = '[{}]'.format(local_host)
  assert local_host
  return local_host


def _get_primus_am_host():
  if "PRIMUS_AM_RPC_HOST" in os.environ and "PRIMUS_AM_RPC_PORT" in os.environ:
    host = os.environ["PRIMUS_AM_RPC_HOST"]
    port = os.environ["PRIMUS_AM_RPC_PORT"]
    return host + ":" + port
  return ""


_CHANNEL_MAP = {}


def _get_channel(addr: str) -> grpc.Channel:
  if not addr in _CHANNEL_MAP:
    _CHANNEL_MAP[addr] = grpc.insecure_channel(addr)
  return _CHANNEL_MAP[addr]


def maybe_kill_application(reason: str) -> bool:
  """Send a request to AM to kill application."""
  if _get_primus_am_host():
    stub = primus_am_service_pb2_grpc.AppMasterServiceStub(
        _get_channel(_get_primus_am_host()))
    req = primus_am_service_pb2.KillRequest()
    req.exit_code = 1
    req.diagnose = reason
    req.graceful_shutdown_timeout_ms.value = 20000
    try:
      resp = stub.kill(req, timeout=10)
      logging.info("Successfully killed application.")
      return True
    except grpc.RpcError as e:
      logging.info("Failed to kill application: %s", e)
      return False
  logging.info("Current framework doesn't support kill. Ignore killing...")
  return False


def maybe_finish_application():
  if _get_primus_am_host():
    stub = primus_am_service_pb2_grpc.AppMasterServiceStub(
        _get_channel(_get_primus_am_host()))
    req = primus_am_service_pb2.SucceedRequest()
    req.graceful_shutdown_timeout_ms.value = 20000
    try:
      resp = stub.succeed(req, timeout=10)
      logging.info("Successfully mark the application success.")
      return True
    except grpc.RpcError as e:
      logging.info("Failed to finish application: %s", e)


def create_primus_save_point(dst):
  if _get_primus_am_host():
    stub = primus_am_service_pb2_grpc.AppMasterServiceStub(
        _get_channel(_get_primus_am_host()))
    create_req = primus_am_service_pb2.CreateSavepointRequest()
    create_req.savepoint_dir = dst
    try:
      create_resp = stub.createSavepoint(create_req, timeout=10)
      if create_resp.code != 0:
        logging.error("Failed to create primus save point: %s",
                      create_resp.message)
        return False
      savepoint_id = create_resp.savepoint_id

      status_req = primus_am_service_pb2.CreateSavepointStatusRequest()
      status_req.savepoint_restore_id = savepoint_id
      while True:
        statue_resp = stub.createSavepointStatus(status_req, timeout=10)
        if statue_resp.create_savepoint_state in [
            primus_am_service_pb2.CreateSavepointStatusResponse.
            CreateSavepointState.PENDING, primus_am_service_pb2.
            CreateSavepointStatusResponse.CreateSavepointState.RUNNING
        ]:
          time.sleep(5)
          continue
        elif statue_resp.create_savepoint_state == primus_am_service_pb2.CreateSavepointStatusResponse.CreateSavepointState.SUCCEEDED:
          logging.info("Create primus save point succeeded.")
          return True
        else:
          logging.error("Failed to create primus save point: %s",
                        statue_resp.message)
          return False
    except grpc.RpcError as e:
      logging.info("Failed to create primus save point: %s", e)
      return False
