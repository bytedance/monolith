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
from monolith.native_training.touched_key_set_ops import TouchedKeySet


class TouchedKeySetOpsTest(tf.test.TestCase):

  def test_touched_key_set_basic(self):
    touched_key_set = TouchedKeySet(1000, 1)
    ids = tf.constant([x for x in range(1000)], dtype=tf.int64)
    total_dropped_num = touched_key_set.insert(ids)
    with tf.control_dependencies([total_dropped_num]):
      output_ids = touched_key_set.steal()

    with self.session() as sess:
      ids, total_dropped_num, output_ids = sess.run(
          [ids, total_dropped_num, output_ids])
      self.assertEqual(0, total_dropped_num)
      self.assertAllEqual(ids, sorted(output_ids))

  def test_touched_key_set_overflow(self):
    touched_key_set = TouchedKeySet(1000, 1)
    ids = tf.constant([x for x in range(1005)], dtype=tf.int64)
    total_dropped_num = touched_key_set.insert(ids)

    with tf.control_dependencies([total_dropped_num]):
      output_ids = touched_key_set.steal()

    with self.session() as sess:
      ids, total_dropped_num, output_ids = sess.run(
          [ids, total_dropped_num, output_ids])
      self.assertEqual(1001, total_dropped_num)
      self.assertAllEqual([1001, 1002, 1003, 1004], sorted(output_ids))


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
