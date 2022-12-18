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

from absl import logging
import grpc
from kazoo.exceptions import NoNodeError, NodeExistsError
import os
import socket
import unittest

from monolith.agent_service import utils
from monolith.agent_service.agent_service import AgentService
from monolith.agent_service.agent_service_pb2 import HeartBeatRequest, ServerType, \
  GetReplicasRequest
from monolith.agent_service.agent_service_pb2_grpc import AgentServiceStub
from monolith.agent_service.mocked_zkclient import FakeKazooClient
from monolith.agent_service.replica_manager import ReplicaWatcher, ReplicaMeta, ModelState
from monolith.agent_service.svr_client import SvrClient

MODEL_NAME = 'test_model_ctr'
BASE_PATH = f'/test_model/{MODEL_NAME}/saved_models'
NUM_PS_REPLICAS = 2
NUM_ENTRY_REPLICAS = 2


class AgentServiceTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls) -> None:
    os.environ['TCE_INTERNAL_IDC'] = 'lf'
    os.environ['TCE_LOGICAL_CLUSTER'] = 'default'
    cls.zk = FakeKazooClient()
    cls.zk.start()
    cls.agent_conf: utils.AgentConfig = utils.AgentConfig(bzid='test_model',
                                                          base_name=MODEL_NAME,
                                                          deploy_type='ps',
                                                          base_path=BASE_PATH,
                                                          num_ps=20,
                                                          dc_aware=True)
    cls.watcher = ReplicaWatcher(cls.zk, cls.agent_conf)
    cls.register(cls.zk)
    cls.watcher.watch_data()
    cls.agent = AgentService(cls.watcher, port=cls.agent_conf.agent_port)
    cls.agent.start()
    cls.client = SvrClient(cls.agent_conf)

  @classmethod
  def tearDownClass(cls) -> None:
    cls.agent.stop()
    cls.watcher.stop()

  @classmethod
  def register(cls, zk):
    path_prefix = cls.agent_conf.path_prefix
    path_to_meta, idx = {}, 2
    for task_id in range(cls.agent_conf.num_ps):
      for replica_id in range(NUM_PS_REPLICAS):
        meta = ReplicaMeta(address=f'192.168.1.{idx}:{utils.find_free_port()}',
                           stat=ModelState.AVAILABLE)
        replica_path = f'{path_prefix}/ps:{task_id}/{replica_id}'
        print(replica_path, flush=True)
        path_to_meta[replica_path] = meta
        idx += 1

    for replica_id in range(NUM_ENTRY_REPLICAS):
      replica_path = f'{path_prefix}/entry:0/{replica_id}'
      meta = ReplicaMeta(address=f'192.168.1.{idx}:{utils.find_free_port()}',
                         stat=ModelState.AVAILABLE)
      path_to_meta[replica_path] = meta
      idx += 1

    for replica_path, meta in path_to_meta.items():
      replica_meta_bytes = bytes(meta.to_json(), encoding='utf-8')

      try:
        zk.retry(zk.create,
                 path=replica_path,
                 value=replica_meta_bytes,
                 ephemeral=True,
                 makepath=True)
      except NodeExistsError:
        logging.info(f'{replica_path} has already exists')
        zk.retry(zk.set, path=replica_path, value=replica_meta_bytes)

  def test_heart_beat(self):
    resp = self.client.heart_beat(server_type=ServerType.PS)
    self.assertTrue(len(resp.addresses) == 20)

  def test_get_replicas(self):
    resp = self.client.get_replicas(server_type=ServerType.PS,
                                    task=NUM_PS_REPLICAS - 1)
    self.assertTrue(len(resp.address_list.address) == NUM_PS_REPLICAS)


if __name__ == "__main__":
  unittest.main()
