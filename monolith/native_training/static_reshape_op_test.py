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
from tensorflow.python.framework import ops

from monolith.native_training import static_reshape_op


class StaticReshapeOpTest(tf.test.TestCase):

  def test_static_reshape_n(self):
    inputs = [
        tf.ones(shape=(5,), dtype=tf.int32),
        tf.ones(shape=(4, 10), dtype=tf.float32),
        tf.ones(shape=(2, 2, 3), dtype=tf.int64),
    ]
    shapes = [
        (1, 5),
        (5, 8),
        (None, 2),
    ]
    with tf.compat.v1.Session() as sess:
      res = static_reshape_op.static_reshape(inputs, shapes)
      outputs, sizes = sess.run(res)

    self.assertAllEqual(sizes, [5, 40, 12])

    for out, shape in zip(outputs, shapes):
      self.assertEqual(len(out.shape), len(shape))
      for d in range(len(shape)):
        if shape[d] is not None:
          self.assertEqual(out.shape[d], shape[d])

  def test_nested_reshape_n(self):
    builder = static_reshape_op.StaticReshapeNBuilder()

    structure = [{
        "key_0": tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32),
        "key_1": tf.constant([[3, 3], [5, 5]], dtype=tf.float32)
    }, {
        "key_1": tf.constant([[0], [1], [2]], dtype=tf.float32)
    }]
    target = [{
        "key_0": np.array([1, 2, 3, 4, 5, 6], dtype=np.float32),
        "key_1": np.array([3, 3, 5, 5], dtype=np.float32)
    }, {
        "key_1": np.array([0, 1, 2], dtype=np.float32)
    }]

    def flatten(tensor):
      return builder.add(tensor, (None,))

    list_keyed_id = tf.nest.map_structure(flatten, structure)
    res = builder.build()

    with tf.compat.v1.Session() as sess:
      outputs, _ = sess.run(res)

    for di, dt in zip(list_keyed_id, target):
      for k in sorted(dt.keys()):
        self.assertAllEqual(outputs[di[k]], dt[k])


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
