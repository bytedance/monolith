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

from monolith.native_training.layers.agru import AGRUCell, \
  dynamic_rnn_with_attention, static_rnn_with_attention


class AGRUTest(tf.test.TestCase):

  def test_agru_instantiate(self):
    dense_layer_template = AGRUCell.params()

    test_params0 = dense_layer_template.copy()
    test_params0.name = 'test_dense0'
    test_params0.units = 10
    test_params0.activation = tf.keras.activations.sigmoid
    test_params0.initializer = tf.keras.initializers.GlorotNormal()
    mlp1 = test_params0.instantiate()
    print(mlp1)

    mlp2 = AGRUCell(units=10,
                    activation=tf.keras.activations.sigmoid,
                    initializer=tf.keras.initializers.HeUniform())
    print(mlp2)

  def test_agru_serde(self):
    mlp1 = AGRUCell(units=10,
                    activation=tf.keras.activations.sigmoid,
                    initializer=tf.keras.initializers.HeUniform())

    cfg = mlp1.get_config()
    mlp2 = AGRUCell.from_config(cfg)

    print(mlp1, mlp2)

  def test_agru_call(self):
    dense_layer_template = AGRUCell.params()

    test_params0 = dense_layer_template.copy()
    test_params0.name = 'test_dense0'
    test_params0.units = 10
    test_params0.activation = tf.keras.activations.sigmoid
    test_params0.initializer = tf.keras.initializers.GlorotNormal()
    layer = test_params0.instantiate()
    print(layer)

    data = tf.keras.backend.variable(np.ones((100, 100)))
    state = tf.keras.backend.variable(np.ones((100, 10)))
    attr = tf.keras.backend.variable(np.ones((100, 1)))
    _, out = layer((data, state, attr))
    sum_out = tf.reduce_sum(out)
    with self.session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      print(sess.run(sum_out))

  def test_agru_static_rnn_call(self):
    dense_layer_template = AGRUCell.params()

    test_params0 = dense_layer_template.copy()
    test_params0.name = 'test_dense0'
    test_params0.units = 10
    test_params0.activation = tf.keras.activations.sigmoid
    test_params0.initializer = tf.keras.initializers.GlorotNormal()
    cell = test_params0.instantiate()
    print(cell)

    data = tf.keras.backend.variable(np.ones((100, 20, 10)))
    attr = tf.keras.backend.variable(np.ones((100, 20)))
    _, out = static_rnn_with_attention(cell, inputs=data, att_scores=attr)
    sum_out = tf.reduce_sum(out)
    with self.session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      print(sess.run(sum_out))

  def test_agru_dynamic_rnn_call(self):
    dense_layer_template = AGRUCell.params()

    test_params0 = dense_layer_template.copy()
    test_params0.name = 'test_dense0'
    test_params0.units = 10
    test_params0.activation = tf.keras.activations.sigmoid
    test_params0.initializer = tf.keras.initializers.GlorotNormal()
    cell = test_params0.instantiate()
    print(cell)

    data = tf.random.uniform(shape=(100, 20, 10))
    attr = tf.random.uniform(shape=(100, 20))
    _, out = dynamic_rnn_with_attention(cell, inputs=data, att_scores=attr)
    sum_out = tf.reduce_sum(out)
    with self.session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      print(sess.run(sum_out))


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.compat.v1.disable_v2_behavior()
  tf.test.main()
