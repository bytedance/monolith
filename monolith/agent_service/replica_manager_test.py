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
from kazoo.exceptions import NodeExistsError
import socket
import time
import os
import threading

import unittest

from monolith.agent_service import constants
from monolith.agent_service import utils
from monolith.agent_service.agent_service_pb2 import ServerType
from monolith.agent_service.mocked_tfserving import FakeTFServing
from monolith.agent_service.mocked_zkclient import FakeKazooClient
from monolith.agent_service.replica_manager import ReplicaUpdater, ReplicaWatcher, \
  ReplicaManager, ReplicaMeta, ModelState

MODEL_NAME = 'test_model'
BASE_PATH = f'/model/{MODEL_NAME}/saved_models'
NUM_REPLICAS = 3


class ReplicaMgrTest(unittest.TestCase):
  tfs: FakeTFServing = None
  agent_conf: utils.AgentConfig = None

  @classmethod
  def setUpClass(cls) -> None:
    os.environ[constants.HOST_SHARD_ENV] = '5'
    os.environ['SHARD_ID'] = '1'
    os.environ['REPLICA_ID'] = '2'
    os.environ['TCE_INTERNAL_IDC'] = 'lf'
    os.environ['TCE_LOGICAL_CLUSTER'] = 'default'
    cls.agent_conf = utils.AgentConfig(bzid='bzid',
                                       base_name=MODEL_NAME,
                                       deploy_type='mixed',
                                       base_path=BASE_PATH,
                                       num_ps=20,
                                       num_shard=5,
                                       dc_aware=True)

    entry_cmd = cls.agent_conf.get_cmd('tensorflow_serving',
                                       server_type=utils.TFSServerType.ENTRY)
    start = entry_cmd.index('model_config_file') + len('model_config_file') + 1
    end = entry_cmd.find(' ', start)
    cls.tfs_entry = FakeTFServing(model_config_file=entry_cmd[start:end],
                                  num_versions=1,
                                  port=cls.agent_conf.tfs_entry_port)

    ps_cmd = cls.agent_conf.get_cmd('tensorflow_serving',
                                    server_type=utils.TFSServerType.PS)
    start = ps_cmd.index('model_config_file') + len('model_config_file') + 1
    end = ps_cmd.find(' ', start)
    cls.tfs_ps = FakeTFServing(model_config_file=ps_cmd[start:end],
                               num_versions=1,
                               port=cls.agent_conf.tfs_ps_port)

    cls.threads = [
        threading.Thread(target=lambda: cls.tfs_entry.start()),
        threading.Thread(target=lambda: cls.tfs_ps.start())
    ]
    for thread in cls.threads:
      thread.start()
    time.sleep(1)

  @classmethod
  def tearDownClass(cls) -> None:
    cls.tfs_entry.stop()
    cls.tfs_ps.stop()
    for thread in cls.threads:
      thread.join()

  def register(self, zk):
    path_prefix = self.agent_conf.path_prefix
    path_to_meta, idx = {}, 2
    for replica_id in range(NUM_REPLICAS):
      for shard_id in range(self.agent_conf.num_shard):
        if shard_id == self.agent_conf.shard_id and replica_id == self.agent_conf.replica_id:
          continue

        for task_id in range(self.agent_conf.num_ps):
          if task_id % self.agent_conf.num_shard == shard_id:
            meta = ReplicaMeta(
                address=f'192.168.1.{idx}:{utils.find_free_port()}',
                stat=ModelState.AVAILABLE)
            replica_path = f'{path_prefix}/ps:{task_id}/{replica_id}'
            path_to_meta[replica_path] = meta
            idx += 1

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

  def test_get_replicas(self):
    zk = FakeKazooClient()
    zk.start()

    self.register(zk)

    mgr = ReplicaManager(zk, self.agent_conf)
    mgr.start()

    time.sleep(3)  # waiting for model available
    try:
      task_01 = mgr.get_replicas(ServerType.PS, task=1)
      self.assertTrue(task_01[0] is None and task_01[1] is None and
                      task_01[2] is not None)
      self.assertTrue(
          len(mgr.get_replicas(ServerType.ENTRY, task=0)) == NUM_REPLICAS)
    finally:
      mgr.stop()
      zk.stop()


if __name__ == "__main__":
  unittest.main()
