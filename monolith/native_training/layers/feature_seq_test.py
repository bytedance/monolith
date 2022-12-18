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

from monolith.native_training.layers.feature_seq import DIN, DIEN, DMR_U2I


class FeatureSeqTest(tf.test.TestCase):

  def test_din_instantiate(self):
    layer_template = DIN.params()

    test_params0 = layer_template.copy()
    test_params0.hidden_units = [10, 1]
    test_params0.initializer = tf.initializers.GlorotNormal()
    ins1 = test_params0.instantiate()
    print(ins1)

    ins2 = DIN(hidden_units=[10, 1], initializer=tf.initializers.HeUniform())
    print(ins2)

  def test_din_serde(self):
    ins1 = DIN(hidden_units=[10, 1], initializer=tf.initializers.HeUniform())

    cfg = ins1.get_config()
    ins2 = DIN.from_config(cfg)

    print(ins1, ins2)

  def test_din_call(self):
    layer = DIN(hidden_units=[10, 1], initializer=tf.initializers.HeUniform())

    query = tf.keras.backend.variable(np.ones((100, 10)))
    keys = tf.keras.backend.variable(np.ones((100, 15, 10)))
    out = layer((query, keys))
    sum_out = tf.reduce_sum(out)
    with self.session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      print(sess.run(sum_out))

  def test_dien_instantiate(self):
    layer_template = DIEN.params()

    test_params0 = layer_template.copy()
    test_params0.num_units = 10
    test_params0.initializer = tf.initializers.GlorotNormal()
    ins1 = test_params0.instantiate()
    print(ins1)

    ins2 = DIEN(num_units=10, initializer=tf.initializers.HeUniform())
    print(ins2)

  def test_dien_serde(self):
    ins1 = DIEN(num_units=10, initializer=tf.initializers.HeUniform())

    cfg = ins1.get_config()
    ins2 = DIEN.from_config(cfg)

    print(ins1, ins2)

  def test_dien_call(self):
    layer = DIEN(num_units=10, initializer=tf.initializers.HeUniform())

    query = tf.keras.backend.variable(np.ones((100, 10)))
    keys = tf.keras.backend.variable(np.ones((100, 15, 10)))
    out = layer((query, keys))
    sum_out = tf.reduce_sum(out)
    with self.session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      print(sess.run(sum_out))

  def test_dmr_instantiate(self):
    layer_template = DMR_U2I.params()

    test_params0 = layer_template.copy()
    test_params0.cmp_dim = 10
    test_params0.activation = 'relu'
    test_params0.initializer = tf.initializers.GlorotNormal()
    ins1 = test_params0.instantiate()
    print(ins1)

    ins2 = DMR_U2I(cmp_dim=10,
                   activation='relu',
                   initializer=tf.initializers.HeUniform())
    print(ins2)

  def test_dmr_serde(self):
    ins1 = DMR_U2I(cmp_dim=10,
                   activation='relu',
                   initializer=tf.initializers.HeUniform())

    cfg = ins1.get_config()
    ins2 = DMR_U2I.from_config(cfg)

    print(ins1, ins2)

  def test_dmr_call(self):
    layer = DMR_U2I(cmp_dim=5,
                    activation='relu',
                    initializer=tf.initializers.HeUniform())

    query = tf.keras.backend.variable(np.ones((100, 10)))
    keys = tf.keras.backend.variable(np.ones((100, 15, 10)))
    out = layer((query, keys))
    sum_out = tf.reduce_sum(out)
    with self.session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      print(sess.run(sum_out))


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
