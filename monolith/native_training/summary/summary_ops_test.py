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
import unittest
import tensorflow as tf
from tensorboard import plugin_util
from tensorboard.backend.event_processing.plugin_event_multiplexer import EventMultiplexer
from tensorboard.backend.event_processing.data_provider import MultiplexerDataProvider
from tensorflow.python.training.checkpoint_state_pb2 import CheckpointState
from tensorboard.data.provider import DataProvider, RunTagFilter

from monolith.native_training.summary.utils import PLUGIN_NAME, prepare_head
from monolith.native_training.summary import summary_ops


tf.compat.v1.disable_eager_execution()


class SummaryTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    cls.log_dir = "demo_logs_v1"
    cls.sess = tf.compat.v1.Session()

    segment_names = ['f1', 'f2', 'f3']
    segment_sizes = [3, 5, 9]
    group_info = [['f1', 'f2'], ['f3', 'f4'], ['f5', 'f6']]
    cls.weight = [0.5, 0.8]
    weight=tf.constant(value=cls.weight, dtype=tf.float32, name='weight')
    summary_ops.nas_data(weight, segment_names, segment_sizes, group_info)
    
    input_tensor = tf.random.uniform(shape=(3, 17), dtype=tf.float32)
    label = tf.constant(value=[1, 0, 1], shape=(3,), dtype=tf.float32)
    weight_tensor = tf.random.uniform(shape=(17, 2), dtype=tf.float32)
    summary_ops.feature_insight_data(input_tensor, segment_names, segment_sizes,
                                     label=label, weight=weight_tensor)
    with cls.sess.as_default():
      with tf.compat.v1.summary.FileWriter(cls.log_dir) as writer:
        summaries = tf.compat.v1.summary.merge_all()
        for global_step in range(10):
          summaries_out = cls.sess.run(summaries)
          writer.add_summary(summaries_out, global_step)

    multiplexer = EventMultiplexer()
    multiplexer.AddRunsFromDirectory(path=cls.log_dir)
    multiplexer.Reload()
    cls.data_provider: DataProvider = MultiplexerDataProvider(
      multiplexer, logdir=cls.log_dir)

  @classmethod
  def tearDownClass(cls) -> None:
    if tf.io.gfile.exists(cls.log_dir):
      tf.io.gfile.rmtree(cls.log_dir)

  def test_nas_data(self):
    ctx = plugin_util.context({})
    run = '.'
    tag = 'gating/monolith_nas_weight'
    tag_info = self.data_provider.list_tensors(ctx,
                                               experiment_id='0',
                                               plugin_name=PLUGIN_NAME,
                                               run_tag_filter=RunTagFilter(runs=[run], tags=[tag]))
    tensors = self.data_provider.read_tensors(ctx,
                                              experiment_id='0',
                                              plugin_name=PLUGIN_NAME,
                                              downsample=100,
                                              run_tag_filter=RunTagFilter(runs=[run], tags=[tag]))
    tensor_tts = tag_info.get(run, {}).get(tag, None)
    tensor_datum = tensors.get(run, {}).get(tag, None)
    self.assertTrue(tensor_datum is not None)
    if isinstance(tensor_datum, (list, tuple)):
      tensor_datum = tensor_datum[-1]
    
    plugin_content = str(tensor_tts.plugin_content, encoding='utf-8')
    plugin_content_exp = '{"tag_type": "gating", "segment_names": ["f1", "f2", "f3"], "segment_sizes": [3, 5, 9], "group_index": [0, 0, 1]}'
    self.assertEqual(plugin_content, plugin_content_exp)
    for x, y in zip(tensor_datum.numpy, self.weight):
      self.assertAlmostEqual(x, y)

  def test_feature_insight_data(self):
    ctx = plugin_util.context({})
    run = '.'
    tag = 'fi_train/monolith_feature_insight'
    tag_info = self.data_provider.list_tensors(ctx,
                                               experiment_id='0',
                                               plugin_name=PLUGIN_NAME,
                                               run_tag_filter=RunTagFilter(runs=[run], tags=[tag]))
    tensors = self.data_provider.read_tensors(ctx,
                                              experiment_id='0',
                                              plugin_name=PLUGIN_NAME,
                                              downsample=100,
                                              run_tag_filter=RunTagFilter(runs=[run], tags=[tag]))
    tensor_tts = tag_info.get(run, {}).get(tag, None)
    tensor_datum = tensors.get(run, {}).get(tag, None)
    self.assertTrue(tensor_datum is not None)
    if isinstance(tensor_datum, (list, tuple)):
      tensor_datum = tensor_datum[-1]

    plugin_content = str(tensor_tts.plugin_content, encoding='utf-8')
    label_size = json.loads(plugin_content)['label_size']
    dim = 2 if label_size > 0 else 1
    plugin_content_exp = '{"tag_type": "fi_train", "segment_names": ["f1", "f2", "f3"], "segment_sizes": [2, 2, 2], "group_index": [0, 1, 2], "label_size": 1}'
    self.assertEqual(plugin_content, plugin_content_exp)

    shape_exp = (3, 7 if label_size > 0 else 3)
    self.assertTupleEqual(tensor_datum.numpy.shape, shape_exp)
    

if __name__ == "__main__":
  unittest.main()
