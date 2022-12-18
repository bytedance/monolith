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

from monolith.agent_service import utils
from monolith.agent_service import backends
from monolith.agent_service.mocked_zkclient import FakeKazooClient


class ZKBackendTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls) -> None:
    cls.bzid = 'gip'
    cls.container = backends.Container("default", "asdf")
    cls.service_info = backends.ContainerServiceInfo(grpc="localhost:8888",
                                                     http="localhost:8889",
                                                     archon="localhost:8890",
                                                     agent="localhost:8891",
                                                     idc="IDC")
    cls.backend = backends.ZKBackend(cls.bzid, zk_servers='127.0.0.1:9999')
    cls.zk = FakeKazooClient()
    cls.backend._zk = cls.zk
    cls.layout_record = None

    def layout_callback(saved_models):
      cls.layout_record = saved_models

    cls.backend.start()
    cls.backend.report_service_info(cls.container, cls.service_info)
    cls.layout_path = "/gip/layouts/test_layout/mixed"
    cls.backend.register_layout_callback(cls.layout_path, layout_callback)
    print("setUpClass finished!")

  @classmethod
  def tearDownClass(cls) -> None:
    cls.backend.stop()
    print('tearDownClass finished!')

  def test_register_service(self):
    service_info = self.backend.get_service_info(self.container)
    self.assertEqual(service_info, self.service_info)

  def test_layout_callback(self):

    def base_path(sub_graph):
      return os.path.join(os.environ["TEST_TMPDIR"],
                          "test_ffm_model/exported_models", sub_graph)

    for sub_graph in ['entry', 'ps_0', 'ps_1', 'ps_2']:
      saved_model = backends.SavedModel('test_ffm_model', sub_graph)
      self.backend.decl_saved_model(
          saved_model,
          backends.SavedModelDeployConfig(base_path(sub_graph), 'latest'))
      self.backend.add_to_layout(self.layout_path, saved_model)
    expected_saved_models = [
        (backends.SavedModel("test_ffm_model", "entry"),
         backends.SavedModelDeployConfig(base_path('entry'), 'latest')),
        (backends.SavedModel("test_ffm_model", "ps_0"),
         backends.SavedModelDeployConfig(base_path('ps_0'), 'latest')),
        (backends.SavedModel("test_ffm_model", "ps_1"),
         backends.SavedModelDeployConfig(base_path('ps_1'), 'latest')),
        (backends.SavedModel("test_ffm_model", "ps_2"),
         backends.SavedModelDeployConfig(base_path('ps_2'), 'latest')),
    ]
    self.assertEqual(self.layout_record, expected_saved_models)
    self.backend.remove_from_layout(
        self.layout_path, backends.SavedModel('test_ffm_model', 'entry'))
    self.assertEqual(self.layout_record, [
        (backends.SavedModel("test_ffm_model", "ps_0"),
         backends.SavedModelDeployConfig(base_path('ps_0'), 'latest')),
        (backends.SavedModel("test_ffm_model", "ps_1"),
         backends.SavedModelDeployConfig(base_path('ps_1'), 'latest')),
        (backends.SavedModel("test_ffm_model", "ps_2"),
         backends.SavedModelDeployConfig(base_path('ps_2'), 'latest')),
    ])

  def test_sync_available_models(self):
    self.backend.sync_available_saved_models(
        self.container, {
            backends.SavedModel("test_ffm_model", "entry"),
            backends.SavedModel("test_ffm_model", "ps_0"),
            backends.SavedModel("test_ffm_model", "ps_1"),
        })
    self.assertTrue(
        self.zk.exists(f"/gip/binding/test_ffm_model/entry:{self.container}"))
    self.assertTrue(
        self.zk.exists(f"/gip/binding/test_ffm_model/ps_0:{self.container}"))
    self.assertTrue(
        self.zk.exists(f"/gip/binding/test_ffm_model/ps_1:{self.container}"))

  def test_service_map(self):
    self.backend.sync_available_saved_models(
        self.container, {
            backends.SavedModel("test_ffm_model", "entry"),
            backends.SavedModel("test_ffm_model", "ps_0")
        })
    expected = {
        'test_ffm_model': {
            'ps_0': [self.service_info],
            'entry': [self.service_info]
        }
    }
    self.assertTrue(self.backend.get_service_map(), expected)

  def test_sync_backend(self):
    self.backend.subscribe_model("test_ffm_model")
    self.backend.sync_available_saved_models(
        self.container, {
            backends.SavedModel("test_ffm_model", "ps_0"),
            backends.SavedModel("test_ffm_model", "ps_1"),
            backends.SavedModel("test_ffm_model", "ps_2"),
        })
    model_name, targets = self.backend.get_sync_targets("ps_1")
    self.assertEqual(model_name, "test_ffm_model:ps_1")
    self.assertEqual(targets, [self.service_info.grpc])


if __name__ == "__main__":
  unittest.main()
