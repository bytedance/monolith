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
import logging
import grpc
from typing import List, Dict, Callable
from dataclasses import dataclass
from functools import singledispatchmethod

from monolith.agent_service.utils import AgentConfig, get_local_ip
from monolith.agent_service.resource_utils import cal_available_memory_v2
from monolith.agent_service.agent_service_pb2 import AddressList, GetReplicasRequest, GetReplicasResponse, \
  GetResourceRequest, GetResourceResponse, HeartBeatRequest, HeartBeatResponse
from monolith.agent_service.agent_service_pb2_grpc import AgentServiceServicer
from monolith.agent_service.agent_service_pb2_grpc import add_AgentServiceServicer_to_server
from monolith.agent_service.replica_manager import ReplicaWatcher
from monolith.agent_service.zk_mirror import ZKMirror
from monolith.agent_service.data_def import ReplicaMeta


class AgentDataProvider:

  def __init__(self, addrs_fn: Callable[[], Dict[str, List[str]]]):
    self._addrs_fn = addrs_fn


class AgentServiceImpl(AgentServiceServicer):

  @singledispatchmethod
  def __init__(self, arg):
    raise NotImplementedError('__init__ is not implemented!')

  @__init__.register
  def _(self, watcher: ReplicaWatcher, conf: AgentConfig = None):
    self._watcher: ReplicaWatcher = watcher
    self.conf = conf

  @__init__.register
  def _(self, zk: ZKMirror, conf: AgentConfig):
    self._zk: ZKMirror = zk
    self.conf = conf

  @__init__.register
  def _(self, data_provider: AgentDataProvider, conf: AgentConfig):
    self._data_provider = data_provider
    self.conf = conf

  def GetReplicas(self, request: GetReplicasRequest,
                  context) -> GetReplicasResponse:
    response = GetReplicasResponse()
    if self.conf is None or self.conf.agent_version == 1:
      idc, cluster = self._watcher._conf.idc, self._watcher._conf.cluster
      address = self._watcher.get_replicas(request.server_type, request.task,
                                           idc, cluster)
      response.address_list.address.extend(address)
    elif self.conf.agent_version == 2:
      rms: List[ReplicaMeta] = self._zk.get_task_replicas(
          request.model_name, request.server_type, request.task)
      response.address_list.address.extend([rm.address for rm in rms])
    else:
      raise NotImplementedError("not implement for agent v3")
    return response

  def HeartBeat(self, request: HeartBeatRequest, context) -> HeartBeatResponse:
    response = HeartBeatResponse()
    addresses = response.addresses
    if self.conf is None or self.conf.agent_version == 1:
      dc_aware = self._watcher._conf.dc_aware
      idc, cluster = self._watcher._conf.idc, self._watcher._conf.cluster
      addrs = self._watcher.get_all_replicas(request.server_type, idc, cluster)
      for key, values in addrs.items():
        key = key.split('/')[-1] if dc_aware else key
        addr_list = AddressList()
        addr_list.address.extend(values)
        addresses[key].CopyFrom(addr_list)
    elif self.conf.agent_version == 2:
      rm_dict: Dict[str, List[ReplicaMeta]] = self._zk.get_all_replicas(
          request.server_type)
      for key, rms in rm_dict.items():
        addr_list = AddressList()
        addr_list.address.extend([rm.address for rm in rms])
        addresses[key].CopyFrom(addr_list)
    else:
      addrs_map = self._data_provider._addrs_fn()
      if addrs_map:
        for saved_model_name, addrs in addrs_map.items():
          addr_list = AddressList()
          addr_list.address.extend(addrs)
          addresses[saved_model_name].CopyFrom(addr_list)
    logging.info(f"heartbeat response: {response}")
    return response

  def GetResource(self, request: GetResourceRequest,
                  context) -> GetResourceResponse:
    if self.conf is None or self.conf.agent_version == 1:
      return GetResourceResponse()
    else:
      return GetResourceResponse(
          address=f'{get_local_ip()}:{self.conf.agent_port}',
          shard_id=int(self.conf.shard_id),
          replica_id=int(self.conf.replica_id),
          memory=cal_available_memory_v2(),
          cpu=-1.0,
          network=-1.0,
          work_load=-1.0)


class AgentService:

  @singledispatchmethod
  def __init__(self, arg):
    raise NotImplementedError('__init__ is not implemented!')

  @__init__.register
  def _(self, watcher: ReplicaWatcher, port: int = None, max_workers: int = 10):
    self._server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers))
    add_AgentServiceServicer_to_server(AgentServiceImpl(watcher), self._server)
    self._server.add_insecure_port(f'[::]:{port or 0}')

  @__init__.register
  def _(self, zk: ZKMirror, conf: AgentConfig, max_workers: int = 10):
    self._server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers))
    add_AgentServiceServicer_to_server(AgentServiceImpl(zk, conf), self._server)
    self._server.add_insecure_port(f'[::]:{conf.agent_port or 0}')

  @__init__.register
  def _(self, data_provider: AgentDataProvider, conf: AgentConfig, max_workers: int = 10):
    self._server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers))
    add_AgentServiceServicer_to_server(AgentServiceImpl(data_provider, conf),
                                       self._server)
    self._server.add_insecure_port(f'[::]:{conf.agent_port or 0}')

  def start(self):
    self._server.start()

  def wait_for_termination(self):
    self._server.wait_for_termination()

  def stop(self, grace=None):
    self._server.stop(grace=grace)
