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

from absl import logging

import tensorflow as tf
from tensorflow.python.framework.ops import name_from_scope_name
from tensorflow.python.framework import test_util

from monolith.native_training.optimizers import rmsprop


def build_graph() -> tf.Operation:
  v = tf.Variable([0.1], name="var")
  loss = 0.12 * v
  opt = rmsprop.RmspropOptimizer(learning_rate=0.1,
                                 weight_decay=1,
                                 beta1=0.9,
                                 beta2=0.9,
                                 epsilon=0.1)
  return opt.minimize(loss)


class RmspropTest(tf.test.TestCase):

  def testBasic(self):
    with tf.Graph().as_default(), test_util.use_gpu():
      train_op = build_graph()
      with self.session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(train_op)
        all_vars = tf.compat.v1.all_variables()
        if tf.test.is_gpu_available():
          for var in all_vars:
            self.assertEqual(var.device, '/device:GPU:0')
        vars_map_maybe_on_gpu = sess.run({var.name: var for var in all_vars})
    eps = 1e-8
    found_count = 0
    for name, val in vars_map_maybe_on_gpu.items():
      if name.find("/m") >= 0:
        found_count += 1
        self.assertNear(val, 0.06794526153774846, eps)
      elif name.find("/v") >= 0:
        found_count += 1
        self.assertNear(val, 0.00484, eps)
      else:
        found_count += 1
        # Must be variable
        self.assertNear(val, 0.03205473846225154, eps)
    self.assertEqual(found_count, 3)

    with tf.Graph().as_default(), test_util.force_cpu():
      train_op = build_graph()
      with self.session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(train_op)
        all_vars = tf.compat.v1.all_variables()
        for var in all_vars:
          self.assertEqual(var.device, '/device:CPU:0')
        vars_map_on_cpu = sess.run({var.name: var for var in all_vars})
    self.assertEqual(vars_map_maybe_on_gpu, vars_map_on_cpu)


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
