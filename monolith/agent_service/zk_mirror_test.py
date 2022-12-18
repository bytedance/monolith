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
import os
from random import shuffle
from queue import Queue
from kazoo.exceptions import NodeExistsError
import socket
import time
import threading

import unittest

from monolith.agent_service import constants
from monolith.agent_service import utils
from monolith.agent_service.agent_service_pb2 import ServerType
from monolith.agent_service.mocked_tfserving import FakeTFServing
from monolith.agent_service.mocked_zkclient import FakeKazooClient
from monolith.agent_service.zk_mirror import ZKMirror
from monolith.agent_service.data_def import PublishMeta, PublishType, ReplicaMeta, ResourceSpec, \
  SubModelName, VersionPath, ModelMeta, EventType

MODEL_NAME = 'model'
BASE_PATH = f'/tmp/{MODEL_NAME}/saved_models'
NUM_REPLICAS = 3


class ZKMirrorTest(unittest.TestCase):
  tfs: FakeTFServing = None
  agent_conf: utils.AgentConfig = None

  @classmethod
  def setUpClass(cls) -> None:
    os.environ[constants.HOST_SHARD_ENV] = '10'
    os.environ['SHARD_ID'] = '2'
    os.environ['REPLICA_ID'] = '2'
    cls.bzid = 'bzid'
    cls.shard_id = 2
    cls.num_tce_shard = 10
    cls.replica_id = 2

    cls.zk = ZKMirror(zk=FakeKazooClient(),
                      bzid=cls.bzid,
                      queue=Queue(),
                      tce_shard_id=cls.shard_id,
                      num_tce_shard=cls.num_tce_shard)
    cls.zk.start()

    cls.resource = ResourceSpec(
        address=f'{utils.get_local_ip()}:1234',  # host:port
        shard_id=cls.shard_id,
        replica_id=cls.replica_id,
        memory=12345,
        cpu=5.6,
        network=3.2,
        work_load=0.7)

  @classmethod
  def tearDownClass(cls) -> None:
    cls.zk.stop()

  def test_crud(self):
    # ensure_path
    self.zk.ensure_path(path='/model/crud')
    # exists
    self.assertTrue(self.zk.exists(path='/model/crud'))
    # create
    self.zk.create(path='/model/crud/data', value=b'test', makepath=True)
    # get/set
    value, _ = self.zk._zk.get(path='/model/crud/data')
    self.assertEqual(value, b'test')
    self.zk.set(path='/model/crud/data', value=b'new_test')
    value, _ = self.zk._zk.get(path='/model/crud/data')
    self.assertEqual(value, b'new_test')
    # delete
    self.zk.delete(path='/model/crud', recursive=False)
    self.assertFalse(self.zk.exists(path='/model/crud'))

    # porperties
    self.assertEqual(self.zk.num_tce_shard, 10)
    self.assertEqual(self.zk.tce_replica_id, 2)
    self.assertEqual(self.zk.tce_shard_id, 2)

  def test_zk_mirror(self):
    # 0) test_step0_request_loading
    self.zk.watch_portal()
    self.zk.watch_resource()

    path = os.path.join(self.zk.portal_base_path, MODEL_NAME)
    mm = ModelMeta(model_name=MODEL_NAME, model_dir=BASE_PATH, num_shard=5)
    self.zk.create(path, mm.serialize())

    # 1) test_step1_scheduler
    path = os.path.join(self.zk.portal_base_path, MODEL_NAME)
    event = self.zk.queue.get()
    self.assertEqual(event.etype, EventType.PORTAL)
    self.assertEqual(event.path, path)
    mm = ModelMeta.deserialize(event.data)

    version, num_ps, num_tce_shard = 123456, 10, self.zk.num_tce_shard
    pms = []
    tce_shards = list(range(self.zk.num_tce_shard))
    shuffle(tce_shards)

    # scheduler
    for i in range(mm.num_shard):
      sub_models: Dict[SubModelName, VersionPath] = {
          f'ps_{k}': f'{mm.model_dir}/ps_{k}/{version}'
          for k in range(num_ps)
          if k % mm.num_shard == i
      }
      sub_models['entry'] = f'{mm.model_dir}/entry/{version}'

      # random schedule, and ensure current shard included
      if i == 0:
        shard_id = self.shard_id
      else:
        shard_id = tce_shards.pop()
        if shard_id == self.shard_id:
          shard_id = tce_shards.pop()

      for replica_id in range(NUM_REPLICAS):
        pm = PublishMeta(shard_id=shard_id,
                         replica_id=replica_id,
                         model_name=mm.model_name,
                         num_ps=10,
                         sub_models=sub_models)
        pms.append(pm)

    for pm in pms:
      pm.total_publish_num = len(pms)
    self.zk.publish_loadding(pms)

    # 2) test_step2_loading
    expected_loading = self.zk.expected_loading()
    for model_name, pm in expected_loading.items():
      self.assertEqual(model_name, MODEL_NAME)
      self.assertEqual(self.shard_id, pm.shard_id)
      self.assertTrue('entry' in pm.sub_models)

    # 3) test_step3_update_service
    expected_loading = self.zk.expected_loading()
    for model_name, pm in expected_loading.items():
      replicas = []
      for sub_model_name, vp in pm.sub_models.items():
        if sub_model_name == 'entry':
          server_type, task = 'entry', 0
        else:
          server_type, task = sub_model_name.split('_')
          task = int(task)

        rm = ReplicaMeta(
            address=f'{utils.get_local_ip()}:8080',  # host:port
            model_name=model_name,
            server_type=server_type,
            task=task,
            replica=self.replica_id,
            stat=utils.ModelState.AVAILABLE)
        replicas.append(rm)
      self.zk.update_service(replicas)

    # 4) test_step4_replicas_ops
    local_ip = utils.get_local_ip()
    entry_replica = ReplicaMeta(address=f'{local_ip}:8080',
                                model_name='model',
                                server_type='entry',
                                task=0,
                                replica=2,
                                stat=30)
    ps0_replica = ReplicaMeta(address=f'{local_ip}:8080',
                              model_name='model',
                              server_type='ps',
                              task=0,
                              replica=2,
                              stat=30)
    ps5_replica = ReplicaMeta(address=f'{local_ip}:8080',
                              model_name='model',
                              server_type='ps',
                              task=5,
                              replica=2,
                              stat=30)

    all_replicas = self.zk.get_all_replicas(server_type='ps')
    self.assertEqual(all_replicas['model:ps:0'][0], ps0_replica)
    self.assertEqual(all_replicas['model:ps:5'][0], ps5_replica)

    model_replicas = self.zk.get_model_replicas(model_name=MODEL_NAME,
                                                server_type='entry')
    self.assertEqual(model_replicas['model:entry:0'][0], entry_replica)

    task_replicas = self.zk.get_task_replicas(model_name=MODEL_NAME,
                                              server_type='ps',
                                              task=0)
    self.assertEqual(task_replicas[0], ps0_replica)

    self.assertEqual(
        ps5_replica,
        self.zk.get_replica(model_name=MODEL_NAME,
                            server_type='ps',
                            task=5,
                            replica=2))

    local_replica_paths = {
        '/bzid/service/model/ps:0/2', '/bzid/service/model/entry:0/2',
        '/bzid/service/model/ps:5/2'
    }
    self.assertSetEqual(local_replica_paths, self.zk.local_replica_paths)

    # 5) test_step5_report_resources
    self.zk.report_resource(self.resource)

    # 6) test_step6_get_resources
    self.assertEqual(self.zk.resources[0], self.resource)


if __name__ == "__main__":
  unittest.main()
