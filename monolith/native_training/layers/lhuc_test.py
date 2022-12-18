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

from monolith.native_training.layers.lhuc import LHUCTower


class LHUCTowerTest(tf.test.TestCase):

  def test_lhuc_instantiate(self):
    lhuc_layer_template = LHUCTower.params()

    test_params0 = lhuc_layer_template.copy()
    test_params0.name = 'test_dense0'
    test_params0.output_dims = [1, 3, 4, 5]
    test_params0.activations = None
    test_params0.initializers = tf.keras.initializers.GlorotNormal()
    lhuc1 = test_params0.instantiate()
    print(lhuc1)

    lhuc2 = LHUCTower(output_dims=[1, 3, 4, 5],
                      activations=None,
                      initializers=tf.keras.initializers.HeUniform())
    print(lhuc2)

  def test_lhuc_serde(self):
    lhuc_layer_template = LHUCTower.params()

    test_params0 = lhuc_layer_template.copy()
    test_params0.name = 'test_dense0'
    test_params0.output_dims = [1, 3, 4, 5]
    test_params0.activations = None
    test_params0.initializers = tf.keras.initializers.GlorotNormal()
    lhuc1 = test_params0.instantiate()

    cfg = lhuc1.get_config()
    lhuc2 = LHUCTower.from_config(cfg)

    print(lhuc1, lhuc2)

  def test_lhuc_call(self):
    layer = LHUCTower(output_dims=[50, 20, 1],
                      activations=None,
                      lhuc_output_dims=[[50, 50], [50, 50, 20], [100, 1]],
                      use_bias=True,
                      lhuc_use_bias=False,
                      initializers=tf.keras.initializers.HeUniform())

    dense_data = tf.keras.backend.variable(np.ones((100, 100)))
    lhuc_data = tf.keras.backend.variable(np.ones((100, 50)))
    sum_out = tf.reduce_sum(layer([dense_data, lhuc_data]))
    with self.session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      print(sess.run(sum_out))


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
