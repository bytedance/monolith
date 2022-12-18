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

from monolith.native_training.layers.norms import LayerNorm, GradNorm


class NormTest(tf.test.TestCase):

  def test_ln_instantiate(self):
    layer_template = LayerNorm.params()

    test_params0 = layer_template.copy()
    test_params0.initializer = tf.keras.initializers.GlorotNormal()
    bn1 = test_params0.instantiate()
    print(bn1)

    bn2 = LayerNorm(initializer=tf.keras.initializers.HeUniform())
    print(bn2)

  def test_ln_serde(self):
    layer_template = LayerNorm.params()

    test_params0 = layer_template.copy()
    test_params0.initializer = tf.keras.initializers.GlorotNormal()
    bn1 = test_params0.instantiate()
    print(bn1)

    cfg = bn1.get_config()
    bn2 = LayerNorm.from_config(cfg)

    print(bn1, bn2)

  def test_ln_call(self):
    bn = LayerNorm(initializer=tf.keras.initializers.HeUniform())

    data = tf.keras.backend.variable(np.ones((100, 100)))
    sum_out = tf.reduce_sum(bn(data))
    with self.session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      print(sess.run(sum_out))

  def test_gn_instantiate(self):
    layer_template = GradNorm.params()

    test_params0 = layer_template.copy()
    test_params0.loss_names = ["abc", 'defg']
    bn1 = test_params0.instantiate()
    print(bn1)

    bn2 = GradNorm(loss_names=["abc", 'defg'], relative_diff=True)
    print(bn2)

  def test_gn_serde(self):
    layer_template = GradNorm.params()

    test_params0 = layer_template.copy()
    test_params0.loss_names = ["abc", 'defg']
    bn1 = test_params0.instantiate()
    print(bn1)

    cfg = bn1.get_config()
    bn2 = GradNorm.from_config(cfg)

    print(bn1, bn2)


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
