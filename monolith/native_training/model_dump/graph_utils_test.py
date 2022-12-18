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

import tensorflow as tf

from monolith.native_training.data.datasets import PBDataset
from monolith.native_training.model_dump.graph_utils import GraphDefHelper
from monolith.native_training.model_export.export_context import get_current_export_ctx
from monolith.native_training.model_dump.dump_utils import DumpUtils

file_name = "monolith/native_training/data/training_instance/examplebatch.data"


class GraphUtilsTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.dump_utils = DumpUtils()
    cls.dump_utils.load(
        "monolith/native_training/model_dump/test_data/model_dump")

  def test_load_input_fn(self):
    proto_model = self.dump_utils.get_proto_model(mode='train')
    graph_helper = self.dump_utils.get_graph_helper(mode='train')
    result = graph_helper.import_input_fn(input_conf=proto_model.input_fn,
                                          file_name=file_name)

    for fname, ts_repr in proto_model.input_fn.output_features.items():
      ts_dict = eval(ts_repr)
      if ts_dict['is_ragged']:
        self.assertTrue(isinstance(result[fname], tf.RaggedTensor))
      else:
        self.assertTrue(isinstance(result[fname], tf.Tensor))

  def test_load_receiver(self):
    proto_model = self.dump_utils.get_proto_model(mode='infer')
    graph_helper = self.dump_utils.get_graph_helper(mode='infer')
    features, receiver_tensors = graph_helper.import_receiver_fn(
        receiver_conf=proto_model.serving_input_receiver_fn)

    for fname, ts_repr in proto_model.serving_input_receiver_fn.features.items(
    ):
      ts_dict = eval(ts_repr)
      if ts_dict['is_ragged']:
        self.assertTrue(isinstance(features[fname], tf.RaggedTensor))
      else:
        self.assertTrue(isinstance(features[fname], tf.Tensor))

    self.assertTrue(len(receiver_tensors) == 1)

  def test_load_mode(self):
    mode = tf.estimator.ModeKeys.TRAIN
    proto_model = self.dump_utils.get_proto_model(mode=mode)
    graph_helper = self.dump_utils.get_graph_helper(mode=mode)
    self.assertTrue(isinstance(graph_helper, GraphDefHelper))

    graph = tf.compat.v1.get_default_graph()
    graph.dry_run = True
    proto_model = self.dump_utils.get_proto_model(mode=mode)
    graph_helper = self.dump_utils.get_graph_helper(mode=mode)
    self.assertTrue(isinstance(graph_helper, GraphDefHelper))

    mode = tf.estimator.ModeKeys.PREDICT
    proto_model = self.dump_utils.get_proto_model(mode=mode)
    graph_helper = self.dump_utils.get_graph_helper(mode=mode)
    self.assertTrue(isinstance(graph_helper, GraphDefHelper))


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

  tf.test.main()
