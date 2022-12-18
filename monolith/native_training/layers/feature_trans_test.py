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

from monolith.native_training.layers.feature_trans import AutoInt, iRazor, SeNet


class FeatureTransTest(tf.test.TestCase):

  def test_autoint_instantiate(self):
    layer_template = AutoInt.params()

    test_params0 = layer_template.copy()
    test_params0.layer_num = 1
    ins1 = test_params0.instantiate()
    print(ins1)

    ins2 = AutoInt(layer_num=1)
    print(ins2)

  def test_autoint_serde(self):
    layer_template = AutoInt.params()

    test_params0 = layer_template.copy()
    test_params0.layer_num = 1
    ins1 = test_params0.instantiate()
    print(ins1)

    cfg = ins1.get_config()
    ins2 = AutoInt.from_config(cfg)

    print(ins1, ins2)

  def test_autoint_call(self):
    layer_template = AutoInt.params()

    test_params0 = layer_template.copy()
    test_params0.name = 'test_dense0'
    test_params0.layer_num = 2
    layer = test_params0.instantiate()

    data = tf.keras.backend.variable(np.ones((100, 10, 10)))
    sum_out = tf.reduce_sum(layer(data))
    with self.session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      print(sess.run(sum_out))

  def test_senet_instantiate(self):
    layer_template = SeNet.params()

    test_params0 = layer_template.copy()
    test_params0.num_feature = 10
    test_params0.cmp_dim = 4
    test_params0.initializer = tf.initializers.GlorotNormal()
    ins1 = test_params0.instantiate()
    print(ins1)

    ins2 = SeNet(num_feature=10,
                 cmp_dim=4,
                 initializer=tf.initializers.HeUniform())
    print(ins2)

  def test_senet_serde(self):
    ins1 = SeNet(num_feature=10,
                 cmp_dim=4,
                 initializer=tf.initializers.HeUniform())

    cfg = ins1.get_config()
    ins2 = SeNet.from_config(cfg)

    print(ins1, ins2)

  def test_senet_call(self):
    layer_template = SeNet.params()

    test_params0 = layer_template.copy()
    test_params0.num_feature = 10
    test_params0.cmp_dim = 4
    test_params0.initializer = tf.initializers.GlorotNormal()
    layer = test_params0.instantiate()

    data = tf.keras.backend.variable(np.ones((100, 10, 10)))
    sum_out = tf.reduce_sum(layer(data))
    with self.session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      print(sess.run(sum_out))

  def test_irazor_instantiate(self):
    layer_template = iRazor.params()

    test_params0 = layer_template.copy()
    test_params0.nas_space = [0, 2, 5, 7, 10]
    test_params0.initializer = tf.initializers.GlorotNormal()
    ins1 = test_params0.instantiate()
    print(ins1)

    ins2 = iRazor(nas_space=[0, 2, 5, 7, 10],
                  t=0.08,
                  initializer=tf.initializers.HeUniform())
    print(ins2)

  def test_irazor_serde(self):
    ins1 = iRazor(nas_space=[0, 2, 5, 7, 10],
                  t=0.08,
                  initializer=tf.initializers.HeUniform())

    cfg = ins1.get_config()
    ins2 = iRazor.from_config(cfg)

    print(ins1, ins2)

  def test_irazor_call(self):
    layer = iRazor(nas_space=[0, 2, 5, 7, 10],
                   t=0.08,
                   initializer=tf.initializers.HeUniform())

    data = tf.keras.backend.variable(np.ones((100, 10, 10)))
    out = layer(data)
    sum_out = tf.reduce_sum(out)
    with self.session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      print(sess.run(sum_out))


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
