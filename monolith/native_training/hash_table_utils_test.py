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

from monolith.native_training import hash_table_utils
from monolith.native_training import hash_table_ops


class HashTableUtilsTest(tf.test.TestCase):

  def test_iterate_table_and_apply(self):
    with self.session() as sess:
      table = hash_table_ops.test_hash_table(1)
      sess.run(
          table.assign(tf.range(100, dtype=tf.int64), [[0.0]] * 100).as_op())
      count_var = tf.Variable(0)
      sess.run(count_var.initializer)

      def count_fn(dump: tf.Tensor):
        return count_var.assign_add(tf.size(dump), use_locking=True)

      sess.run(
          hash_table_utils.iterate_table_and_apply(table,
                                                   count_fn,
                                                   limit=2,
                                                   nshards=10))
      count = sess.run(count_var)
      self.assertEqual(count, 100)


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
