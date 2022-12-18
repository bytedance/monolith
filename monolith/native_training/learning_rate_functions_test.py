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

from monolith.native_training import learning_rate_functions


class PolynomialDecayTest(tf.test.TestCase):

  def test_basic(self):
    with tf.compat.v1.Session() as sess:
      global_step = tf.compat.v1.train.get_or_create_global_step()
      self.evaluate(tf.compat.v1.global_variables_initializer())
      self.evaluate(tf.compat.v1.assign_add(global_step, 1))
      learning_rate_fn = learning_rate_functions.PolynomialDecay(
          initial_learning_rate=0.01, decay_steps=10, end_learning_rate=0.11)
      learning_rate = self.evaluate(learning_rate_fn())
      self.assertAllClose(learning_rate, 0.02, 1e-6)

      self.evaluate(tf.compat.v1.assign_add(global_step, 1))
      learning_rate = self.evaluate(learning_rate_fn())
      self.assertAllClose(learning_rate, 0.03, 1e-6)

      learning_rate_fn2 = learning_rate_functions.PolynomialDecay(
          initial_learning_rate=0.01, decay_steps=10, end_learning_rate=0.11)
      self.assertEqual(str(learning_rate_fn), str(learning_rate_fn2))

  def test_dense_optimizer(self):
    with tf.compat.v1.Session() as sess:
      global_step = tf.compat.v1.train.get_or_create_global_step()
      learning_rate_fn = learning_rate_functions.PolynomialDecay(
          initial_learning_rate=3.0, decay_steps=10, end_learning_rate=11.0)
      var0 = tf.Variable([1.0, 2.0], dtype=tf.float32)
      var1 = tf.Variable([3.0, 4.0], dtype=tf.float32)
      grads0 = tf.constant([0.1, 0.1], dtype=tf.float32)
      grads1 = tf.constant([0.01, 0.01], dtype=tf.float32)

      ada_opt = tf.compat.v1.train.AdagradOptimizer(
          learning_rate_fn, initial_accumulator_value=0.1)

      ada_update = ada_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      self.evaluate(tf.compat.v1.global_variables_initializer())

      # Fetch params to validate initial values
      v0_val, v1_val = self.evaluate([var0, var1])
      self.assertAllClose([1.0, 2.0], v0_val)
      self.assertAllClose([3.0, 4.0], v1_val)

      # Run 3 steps of adagrad
      for _ in range(3):
        self.evaluate(ada_update)

      # Validate updated params
      v0_val, v1_val = self.evaluate([var0, var1])
      self.assertAllCloseAccordingToType(
          np.array([-1.6026098728179932, -0.6026098728179932]), v0_val)
      self.assertAllCloseAccordingToType(
          np.array([2.715679168701172, 3.715679168701172]), v1_val)


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
