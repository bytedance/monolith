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

import numpy as np
import tensorflow as tf

from monolith.native_training.losses.batch_softmax_loss import batch_softmax_loss


class BatchSoftmaxLossTest(tf.test.TestCase):

  def test_batch_softmax_loss(self):
    batch_size, dim = 4, 3
    query = tf.constant(np.random.random([batch_size, dim]), dtype=tf.float32)
    item = tf.constant(np.random.random([batch_size, dim]), dtype=tf.float32)
    item_step_interval = tf.constant(
        [np.random.randint(1, 10) for _ in range(batch_size)], dtype=tf.float32)
    r = tf.ones((batch_size,), dtype=tf.float32)
    loss = batch_softmax_loss(query, item, item_step_interval, r)
    self.assertAllClose([loss], [6.5931373])


if __name__ == '__main__':
  tf.test.main()
