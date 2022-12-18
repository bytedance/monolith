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

from monolith.native_training import utils


class UtilsTest(tf.test.TestCase):

  def test_propagate_back_dict_gradients(self):
    x = tf.Variable(8.0)
    y = 2 * x
    # Use a grad related to x
    grad_y = 3 * y
    valid_vars = set([y])
    grouped = utils.propagate_back_dict_gradients(zip([grad_y], [y]),
                                                  {x: "group1"}, valid_vars)
    with self.session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      dx_and_x = sess.run(grouped["group1"])
      self.assertAllEqual(dx_and_x, [(96, 8)])

  def test_check_ops_dependence(self):
    v = tf.Variable(0)
    add = v.assign_add(1)
    with tf.control_dependencies([add]):
      t1 = tf.constant(0)
      t2 = tf.constant(0)
    with self.assertRaises(Exception):
      utils.check_ops_dependence(t1.op.name, add.name)
    # OK to check
    utils.check_ops_dependence(t1.op.name, t2.op.name)

  def test_collections(self):
    utils.add_to_collections('int', 1)
    utils.add_to_collections('int', 2)
    utils.add_to_collections('str', 'str')
    utils.add_to_collections('str', None)
    utils.add_to_collections('bool', True)
    utils.add_to_collections('int_list', [1, 2, 3])
    utils.add_to_collections('str_list', None)
    utils.add_to_collections('bool_list', [])
    utils.add_to_collections('int_list', [4, 5, 6])
    utils.add_to_collections('str_list', ['hello', 'world'])
    utils.add_to_collections('bool_list', [False])

    self.assertTrue(utils.get_collection('int')[-1] == 2)
    self.assertTrue(utils.get_collection('str')[-1] == 'str')
    self.assertTrue(utils.get_collection('bool')[-1])
    self.assertListEqual(utils.get_collection('int_list')[-1], [4, 5, 6])
    self.assertListEqual(
        utils.get_collection('str_list')[-1], ['hello', 'world'])
    self.assertListEqual(utils.get_collection('bool_list')[-1], [False])


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
