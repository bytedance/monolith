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

import json
import time
import unittest
import os

from absl import logging

from monolith.agent_service import utils
from monolith.agent_service import backends
from monolith.agent_service.agent_v3 import AgentV3
from monolith.agent_service.tfs_wrapper import FakeTFSWrapper
from monolith.agent_service.mocked_zkclient import FakeKazooClient


class AgentV3Test(unittest.TestCase):

  @classmethod
  def setUpClass(cls) -> None:
    cls.bzid = 'gip'
    agent_conf = utils.AgentConfig(bzid='gip',
                                   deploy_type='unified',
                                   agent_version=3,
                                   layout_pattern="/gip/layout",
                                   zk_servers="127.0.0.1:8888")
    base_path = os.environ["TEST_TMPDIR"]

    cls.agent = AgentV3(config=agent_conf,
                        conf_path=os.path.join(base_path,
                                               '/monolith_serving/conf'),
                        tfs_log=os.path.join('monolith_serving/logs/log.log'))
    model_config_path = cls.agent._model_config_path
    # replace tfs wrapper and zk
    cls.tfs_wrapper = FakeTFSWrapper(model_config_path)
    cls.agent._tfs_wrapper = cls.tfs_wrapper
    cls.zk = FakeKazooClient()
    cls.backend = cls.agent._backend
    cls.backend._zk = cls.zk

    cls.agent.start()
    logging.info('setUpClass finished!')

    def base_path(sub_graph):
      return os.path.join(os.environ["TEST_TMPDIR"],
                          "test_ffm_model/exported_models", sub_graph)

    for sub_graph in ['entry', 'ps_0', 'ps_1', 'ps_2']:
      config = {
          'model_base_path': base_path(sub_graph),
          'version_policy': 'latest'
      }
      path = f'/gip/saved_models/test_ffm_model/{sub_graph}'
      value = json.dumps(config).encode('utf-8')
      cls.zk.create(path, value=value, makepath=True)

  @classmethod
  def tearDownClass(cls) -> None:
    cls.agent.stop()
    logging.info('tearDownClass finished!')

  def test_service_info(self):
    self.assertEqual(self.agent._service_info,
                     self.backend.get_service_info(self.agent._container))

  def test_publish_models(self):
    self.assertEqual(self.tfs_wrapper.list_saved_models(), [])
    # publish
    self.zk.ensure_path("/gip/layout/test_ffm_model:entry")
    self.zk.ensure_path("/gip/layout/test_ffm_model:ps_0")
    # check tfs serving
    self.assertEqual(self.tfs_wrapper.list_saved_models(),
                     ['test_ffm_model:entry', 'test_ffm_model:ps_0'])
    # force binding info to propagate
    self.agent.sync_available_saved_models()
    self.assertEqual(
        self.backend.get_service_map(), {
            'test_ffm_model': {
                'entry': [self.agent._service_info],
                'ps_0': [self.agent._service_info]
            }
        })

    # unload one model
    self.zk.delete("/gip/layout/test_ffm_model:ps_0")
    # check tfs serving
    self.assertEqual(self.tfs_wrapper.list_saved_models(),
                     ['test_ffm_model:entry'])
    # force binding info to propagate
    self.agent.sync_available_saved_models()
    self.assertEqual(self.backend.get_service_map(),
                     {'test_ffm_model': {
                         'entry': [self.agent._service_info]
                     }})


if __name__ == "__main__":
  logging.use_absl_handler()
  logging.get_absl_handler().setFormatter(fmt=logging.PythonFormatter())
  logging.set_verbosity(logging.INFO)
  unittest.main()