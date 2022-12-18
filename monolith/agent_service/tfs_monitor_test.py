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

import os
import socket
import time
import threading
import random
import unittest

from monolith.agent_service import constants
from monolith.agent_service import utils
from monolith.agent_service.tfs_monitor import TFSMonitor
from monolith.agent_service.mocked_tfserving import FakeTFServing

from tensorflow_serving.config.model_server_config_pb2 import ModelServerConfig
from monolith.agent_service.data_def import PublishMeta, SubModelName, TFSModelName, \
  VersionPath, PublishType as PType
from monolith.agent_service.utils import AgentConfig, gen_model_spec, gen_model_config, \
  TFSServerType, get_local_ip

MODEL_NAME = 'test_model'
BASE_PATH = f'/tmp/{MODEL_NAME}/monolith'
version = '1634631496'
path = '/tmp/monolith/agent_service/test_data/ckpt/exported_models/{}/{}'


class TFSMonitorTest(unittest.TestCase):
  tfs: FakeTFServing = None
  monitor: TFSMonitor = None

  @classmethod
  def setUpClass(cls) -> None:
    os.environ[constants.HOST_SHARD_ENV] = '10'
    os.environ['SHARD_ID'] = '1'
    os.environ['REPLICA_ID'] = '2'
    cls.agent_conf = utils.AgentConfig(bzid='bzid', deploy_type='mixed')

    cls.tfs_entry = FakeTFServing(num_versions=2,
                                  port=cls.agent_conf.tfs_entry_port,
                                  model_config_file=ModelServerConfig())
    cls.tfs_ps = FakeTFServing(num_versions=2,
                               port=cls.agent_conf.tfs_ps_port,
                               model_config_file=ModelServerConfig())

    entry = threading.Thread(target=lambda: cls.tfs_entry.start())
    entry.start()
    ps = threading.Thread(target=lambda: cls.tfs_ps.start())
    ps.start()
    time.sleep(2)

    cls.monitor = TFSMonitor(cls.agent_conf)
    cls.monitor.connect()

    cls.data = {}

  @classmethod
  def tearDownClass(cls) -> None:
    cls.monitor.stop()
    cls.tfs_entry.stop()
    cls.tfs_ps.stop()

  def setUp(self):
    sub_models: Dict[SubModelName, VersionPath] = {
        'entry': path.format('entry', version),
        'ps_0': path.format('ps_0', version),
        'ps_3': path.format('ps_3', version),
        'ps_5': path.format('ps_5', version)
    }
    pm = PublishMeta(shard_id=self.agent_conf.shard_id,
                     replica_id=self.agent_conf.replica_id,
                     model_name='test_1',
                     num_ps=5,
                     sub_models=sub_models)
    self.data['setUp'] = self.monitor.get_model_status(pm)

  def tearDown(self):
    sub_models: Dict[SubModelName, VersionPath] = {
        'entry': path.format('entry', version),
        'ps_0': path.format('ps_0', version),
        'ps_3': path.format('ps_3', version),
        'ps_5': path.format('ps_5', version)
    }
    pm = PublishMeta(shard_id=self.agent_conf.shard_id,
                     replica_id=self.agent_conf.replica_id,
                     model_name='test_1',
                     num_ps=5,
                     sub_models=sub_models)

    time.sleep(1)
    before_status = self.data['setUp']
    after_status = self.monitor.get_model_status(pm)
    self.assertEqual(len(before_status), len(after_status))

    if self.data['execute'] == 'reload_config':
      for tfs_model_name, (bvp, bstate) in before_status.items():
        (avp, astate) = after_status[tfs_model_name]
        self.assertEqual(bvp, avp)
        self.assertTrue(bstate.version == -1 and
                        bstate.status.error_code == 5)  # NOT_FOUND
        if astate.version == -1:
          pass
        elif astate.version == 1:
          self.assertTrue(tfs_model_name.endswith('entry'))
        else:
          self.assertEqual(astate.version, int(os.path.basename(bvp)))
    else:
      for tfs_model_name, (bvp, bstate) in before_status.items():
        (avp, astate) = after_status[tfs_model_name]
        self.assertEqual(astate.version, -1)
        self.assertEqual(bvp, avp)
        if bstate.version == -1:
          self.assertTrue(bstate.status.error_code == 5)  # NOT_FOUND
        else:
          self.assertTrue(bstate.version > 0)

  def test_reload_config(self):
    pms = []
    for i in range(10):
      num_ps = random.randint(5, 20)
      sub_models = {
          f'ps_{i}': path.format(f'ps_{i}', version)
          for i in range(num_ps)
          if i % 3 == 0
      }
      sub_models[TFSServerType.ENTRY] = path.format(f'entry', version)

      pm = PublishMeta(shard_id=self.agent_conf.shard_id,
                       replica_id=self.agent_conf.replica_id,
                       model_name=f'test_{i}',
                       num_ps=num_ps,
                       sub_models=sub_models)
      pms.append(pm)

    model_configs: Dict[str,
                        ModelServerConfig] = self.monitor.gen_model_config(pms)
    for service_type, model_config in model_configs.items():
      if len(model_config.model_config_list.config) > 0:
        status = self.monitor.handle_reload_config_request(
            service_type, model_config)
    self.data['execute'] = 'reload_config'

  def test_remove_config(self):
    pms = []
    for i in range(5, 10):
      num_ps = random.randint(5, 20)
      sub_models = {
          f'ps_{i}': path.format(f'ps_{i}', version)
          for i in range(num_ps)
          if i % 3 == 0
      }
      sub_models[TFSServerType.ENTRY] = path.format(f'entry', version)

      pm = PublishMeta(shard_id=self.agent_conf.shard_id,
                       replica_id=self.agent_conf.replica_id,
                       model_name=f'test_{i}',
                       num_ps=num_ps,
                       sub_models=sub_models)
      pms.append(pm)

    model_configs: Dict[str,
                        ModelServerConfig] = self.monitor.gen_model_config(pms)
    for service_type, model_config in model_configs.items():
      if len(model_config.model_config_list.config) > 0:
        status = self.monitor.handle_reload_config_request(
            service_type, model_config)
    self.data['execute'] = 'remove_config'


if __name__ == "__main__":
  unittest.main()
