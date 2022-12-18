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

from unittest import mock

import tensorflow as tf

from monolith.native_training.metric import utils


class DeepInsightTest(tf.test.TestCase):

  @mock.patch(
      "monolith.native_training.metric.deep_insight_ops.write_deep_insight")
  def test_basic(self, deep_insight_op):

    def fake_call(uids, **kwargs):
      del kwargs
      with self.session() as sess:
        uids = sess.run(uids)
        self.assertAllEqual(uids, [1, 2, 3])

    deep_insight_op.side_effect = fake_call

    features = {
        "uid": tf.constant([1, 2, 3], dtype=tf.int64),
        "req_time": tf.constant([1, 2, 3], dtype=tf.int64),
        "sample_rate": tf.constant([0.5, 0.5, 0.5], dtype=tf.float32),
    }
    labels = tf.constant([1.0, 0.0, 1.0], dtype=tf.float32)
    preds = tf.constant([0.9, 0.2, 0.8], dtype=tf.float32)
    model_name = "test_model"
    target = "target"
    utils.write_deep_insight(features, 0.01, labels, preds, model_name, target)


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
