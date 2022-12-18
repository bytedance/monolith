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

from monolith.native_training.layers.add_bias import AddBias


class AddBiasTest(tf.test.TestCase):

  def test_ab_instantiate(self):
    layer_template = AddBias.params()

    test_params0 = layer_template.copy()
    test_params0.initializer = tf.initializers.Zeros()
    ins1 = test_params0.instantiate()
    print(ins1)

    ins2 = AddBias(initializer=tf.initializers.Zeros())
    print(ins2)

  def test_ab_serde(self):
    layer_template = AddBias.params()

    test_params0 = layer_template.copy()
    test_params0.initializer = tf.initializers.Zeros()
    ins1 = test_params0.instantiate()
    print(ins1)

    cfg = ins1.get_config()
    ins2 = AddBias.from_config(cfg)

    print(ins1, ins2)

  def test_ab_call(self):
    layer_template = AddBias.params()

    test_params0 = layer_template.copy()
    test_params0.name = 'test_dense0'
    test_params0.initializer = tf.initializers.Zeros()
    layer = test_params0.instantiate()

    data = tf.keras.backend.variable(np.random.uniform(size=(100, 10)))
    sum_out = tf.reduce_sum(layer(data))
    with self.session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      print(sess.run(sum_out))


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
