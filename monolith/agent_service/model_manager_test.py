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

from absl import logging, app, flags
import json
import os
import shutil
import unittest
import time

from monolith.agent_service.model_manager import ModelManager

FLAGS = flags.FLAGS


class ModelManagerTest(unittest.TestCase):

  def create_file(self, model_name, timestamp, p2p_data_path):
    # model_data/test_model/ps_item_embedding_0/1234567

    # p2p/test_model@1234567/test_model/ps_item_embedding_0/1234567
    os.makedirs(
        os.path.join(p2p_data_path, model_name + '@' + timestamp, model_name,
                     'ps_item_embedding_0', timestamp))
    os.makedirs(
        os.path.join(p2p_data_path, model_name + '@' + timestamp, model_name,
                     'ps_item_embedding_1', timestamp))

    f = open(
        os.path.join(p2p_data_path,
                     model_name + '@' + timestamp + '.write.done'), 'w+')
    f.close()

  def test_start(self):
    base_path = os.path.join(os.environ["TEST_TMPDIR"], "test_model_manager")
    p2p_data_path = os.path.join(base_path, 'p2p')
    model_data_path = os.path.join(base_path, 'model_data')

    model_name = "test_model"
    timestamp = "1234567"

    self.create_file(model_name, timestamp, p2p_data_path)

    model_manager = ModelManager(model_name, p2p_data_path, model_data_path,
                                 False)
    model_manager._wait_timeout = 5
    model_manager._loop_interval = 5
    ret = model_manager.start()
    self.assertTrue(ret)

    ready_path1 = os.path.join(model_data_path, model_name,
                               'ps_item_embedding_0', timestamp)
    ready_path2 = os.path.join(model_data_path, model_name,
                               'ps_item_embedding_1', timestamp)

    self.assertTrue(os.path.exists(ready_path1))
    self.assertTrue(os.path.exists(ready_path2))

    model_manager.stop()
    shutil.rmtree(p2p_data_path)
    shutil.rmtree(model_data_path)

  def test_ignore_old(self):
    base_path = os.path.join(os.environ["TEST_TMPDIR"], "test_model_manager")
    p2p_data_path = os.path.join(base_path, 'p2p')
    model_data_path = os.path.join(base_path, 'model_data')

    model_name = "test_model"
    timestamp = "1234567"
    timestamp_old = "1234566"

    self.create_file(model_name, timestamp, p2p_data_path)

    model_manager = ModelManager(model_name, p2p_data_path, model_data_path,
                                 False)
    model_manager._wait_timeout = 5
    model_manager._loop_interval = 5
    ret = model_manager.start()
    self.assertTrue(ret)

    self.create_file(model_name, timestamp_old, p2p_data_path)
    time.sleep(11)

    ready_path1 = os.path.join(model_data_path, model_name,
                               'ps_item_embedding_0', timestamp_old)
    ready_path2 = os.path.join(model_data_path, model_name,
                               'ps_item_embedding_1', timestamp_old)

    self.assertFalse(os.path.exists(ready_path1))
    self.assertFalse(os.path.exists(ready_path2))

    model_manager.stop()
    shutil.rmtree(p2p_data_path)
    shutil.rmtree(model_data_path)


def main(_):
  unittest.main()


if __name__ == "__main__":
  app.run(main)
