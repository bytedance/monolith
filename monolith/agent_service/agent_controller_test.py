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
import os
import unittest

from monolith.agent_service import agent_controller
from monolith.agent_service import backends
from monolith.agent_service.mocked_zkclient import FakeKazooClient


def saved_model(sub_graph):
  return backends.SavedModel('test_ffm_model', sub_graph)


class AgentControllerTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls) -> None:
    cls.bzid = 'gip'
    cls.bd = backends.ZKBackend(cls.bzid, zk_servers='127.0.0.1:9999')
    cls.zk = FakeKazooClient()
    cls.bd._zk = cls.zk
    cls.bd.start()

  @classmethod
  def tearDownClass(cls) -> None:
    cls.bd.stop()
    print('tearDownClass finished!')

  def test_decl_saved_models(self):
    agent_controller.declare_saved_model(
        self.bd,
        os.path.join(
            os.environ['TEST_SRCDIR'], os.environ["TEST_WORKSPACE"],
            "monolith/native_training/model_export/testdata/saved_model"),
        'test_ffm_model',
        overwrite=True)
    saved_models = self.bd.list_saved_models('test_ffm_model')
    self.assertEqual(
        set(saved_models), {
            saved_model(sub_graph)
            for sub_graph in ['ps_0', 'ps_1', 'ps_2', 'ps_3', 'ps_4', 'entry']
        })

  def test_pub(self):
    self.maxDiff = None
    agent_controller.declare_saved_model(
        self.bd,
        os.path.join(
            os.environ['TEST_SRCDIR'], os.environ["TEST_WORKSPACE"],
            "monolith/native_training/model_export/testdata/saved_model"),
        'test_ffm_model',
        overwrite=True)
    agent_controller.map_model_to_layout(self.bd,
                                         "test_ffm_model:entry",
                                         "/gip/layouts/test_layout1",
                                         action="pub")
    self.assertEqual(self.bd.bzid_info()['layout_info']['test_layout1'],
                     ['test_ffm_model:entry'])
    agent_controller.map_model_to_layout(self.bd,
                                         "test_ffm_model:ps_*",
                                         "/gip/layouts/test_layout1",
                                         action="pub")
    self.assertEqual(self.bd.bzid_info()['layout_info']['test_layout1'], [
        'test_ffm_model:entry', 'test_ffm_model:ps_0', 'test_ffm_model:ps_1',
        'test_ffm_model:ps_2', 'test_ffm_model:ps_3', 'test_ffm_model:ps_4'
    ])
    agent_controller.map_model_to_layout(self.bd,
                                         "test_ffm_model:ps_*",
                                         "/gip/layouts/test_layout1",
                                         action="unpub")
    self.assertEqual(self.bd.bzid_info()['layout_info']['test_layout1'],
                     ['test_ffm_model:entry'])
    agent_controller.map_model_to_layout(self.bd,
                                         "test_ffm_model:entry",
                                         "/gip/layouts/test_layout1",
                                         action="unpub")
    self.assertEqual(self.bd.bzid_info()['layout_info']['test_layout1'], [])


if __name__ == "__main__":
  unittest.main()
