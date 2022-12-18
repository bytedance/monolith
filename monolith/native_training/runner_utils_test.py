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

from absl import flags, logging
import json
import os
from google.protobuf import text_format
from kazoo.handlers.threading import KazooTimeoutError

import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow.python.training.checkpoint_state_pb2 import CheckpointState

from monolith.native_training.runner_utils import RunnerConfig, get_discovery
from monolith.native_training.service_discovery import ServiceDiscoveryType, \
  ConsulServiceDiscovery, TfConfigServiceDiscovery, ZKServiceDiscovery


class RunnerUtilsTest(tf.test.TestCase):

  def test_get_discovery_local(self):
    config = RunnerConfig(is_local=True)
    discovery = get_discovery(config)
    config.is_local = False
    self.assertEqual(discovery, None)

  def test_get_discovery_primus(self):
    tf_config = {
        'cluster': {
            'ps': ['localhost:1111', 'localhost:1112'],
            'worker': ['localhost:1113', 'localhost:1114'],
            'chief': ['localhost:1115']
        },
        'task': {
            'type': 'chief',
            'index': 0
        }
    }

    config = config = RunnerConfig(is_local=False,
                                   tf_config=json.dumps(tf_config),
                                   discovery_type=ServiceDiscoveryType.PRIMUS)
    discovery = get_discovery(config)
    self.assertEqual(isinstance(discovery, TfConfigServiceDiscovery), True)

  def test_get_discovery_consul(self):
    psm = 'data.monolith.123456'
    config = RunnerConfig(is_local=False,
                          discovery_type=ServiceDiscoveryType.CONSUL)
    discovery = get_discovery(config, psm)
    self.assertEqual(isinstance(discovery, ConsulServiceDiscovery), True)

  def test_get_discovery_zk(self):
    config = RunnerConfig(is_local=False,
                          discovery_type=ServiceDiscoveryType.ZK,
                          zk_server="127.0.0.1:0")
    try:
      discovery = get_discovery(config)
      self.assertEqual(isinstance(discovery, ZKServiceDiscovery), True)
    except KazooTimeoutError as e:
      logging.info('kazoo example: {}'.format(e))

  def test_copy_ckpt(self):
    restore_dir = os.path.join(os.environ["TEST_TMPDIR"], "runner_utils_test",
                               "restore_dir")
    if not tf.io.gfile.exists(restore_dir):
      tf.io.gfile.makedirs(restore_dir)
    ckpt = CheckpointState(model_checkpoint_path='model.ckpt-61')
    ckpt.all_model_checkpoint_paths.extend(
        ['model.ckpt-61', 'model.ckpt-30', 'model.ckpt-0'])
    file_io.atomic_write_string_to_file(os.path.join(restore_dir, 'checkpoint'),
                                        text_format.MessageToString(ckpt))

    model_dir = os.path.join(os.environ["TEST_TMPDIR"], "runner_utils_test",
                             "model_dir")
    if not tf.io.gfile.exists(model_dir):
      tf.io.gfile.makedirs(model_dir)
    config = RunnerConfig(is_local=True,
                          restore_dir=restore_dir,
                          model_dir=model_dir,
                          restore_ckpt='model.ckpt-30')
    ckpt2 = tf.train.get_checkpoint_state(model_dir)
    self.assertTrue(
        tf.io.gfile.exists(os.path.join(model_dir, 'monolith_checkpoint')))
    self.assertTrue(tf.io.gfile.exists(os.path.join(model_dir, 'restore_ckpt')))
    self.assertEqual(os.path.basename(ckpt2.model_checkpoint_path),
                     'model.ckpt-30')
    # Make sure other workers can go through once chief init the dir
    config = RunnerConfig(server_type="worker",
                          index=2,
                          restore_dir=restore_dir,
                          model_dir=model_dir)


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
