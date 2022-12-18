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

from monolith.native_training.layers.mlp import MLP


class MLPTest(tf.test.TestCase):

  def test_mlp_instantiate(self):
    dense_layer_template = MLP.params()

    test_params0 = dense_layer_template.copy()
    test_params0.name = 'test_dense0'
    test_params0.output_dims = [1, 3, 4, 5]
    test_params0.activations = None
    test_params0.initializers = tf.keras.initializers.GlorotNormal()
    mlp1 = test_params0.instantiate()
    print(mlp1)

    mlp2 = MLP(output_dims=[1, 3, 4, 5],
               activations=None,
               initializers=tf.keras.initializers.HeUniform())
    print(mlp2)

  def test_mlp_serde(self):
    dense_layer_template = MLP.params()

    test_params0 = dense_layer_template.copy()
    test_params0.name = 'test_dense0'
    test_params0.output_dims = [1, 3, 4, 5]
    test_params0.activations = None
    test_params0.initializers = tf.keras.initializers.GlorotNormal()
    mlp1 = test_params0.instantiate()

    cfg = mlp1.get_config()
    mlp2 = MLP.from_config(cfg)

    print(mlp1, mlp2)

  def test_mlp_call(self):
    dense_layer_template = MLP.params()

    test_params0 = dense_layer_template.copy()
    test_params0.name = 'test_dense0'
    test_params0.output_dims = [100, 50, 10, 1]
    test_params0.enable_batch_normalization = True
    test_params0.activations = [
        'relu', tf.keras.activations.tanh, tf.keras.layers.PReLU, None
    ]
    test_params0.initializers = tf.keras.initializers.GlorotNormal()
    layer = test_params0.instantiate()

    data = tf.keras.backend.variable(np.ones((100, 100)))
    sum_out = tf.reduce_sum(layer(data))
    self.assertEqual(len(layer._stacked_layers), 11)
    with self.session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      print(sess.run(sum_out))


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
