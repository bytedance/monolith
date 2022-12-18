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

from monolith.native_training.layers.feature_cross import GroupInt, AllInt, CDot, CAN, CIN, DCN


class FeatureCrossTest(tf.test.TestCase):

  def test_groupint_instantiate(self):
    layer_template = GroupInt.params()

    test_params0 = layer_template.copy()
    test_params0.interaction_type = 'dot'
    test_params0.use_attention = False
    test_params0.attention_units = [128, 256, 1]
    test_params0.activation = 'relu'
    ins1 = test_params0.instantiate()
    print(ins1)

    ins2 = GroupInt(interaction_type='multiply',
                    use_attention=True,
                    attention_units=[128, 256, 1],
                    activation='relu')
    print(ins2)

  def test_groupint_serde(self):
    ins1 = GroupInt(interaction_type='multiply',
                    use_attention=True,
                    attention_units=[128, 256, 1],
                    activation='relu')

    cfg = ins1.get_config()
    ins2 = GroupInt.from_config(cfg)

    print(ins1, ins2)

  def test_groupint_call(self):
    layer_template = GroupInt.params()

    test_params0 = layer_template.copy()
    test_params0.name = 'test_dense0'
    test_params0.out_type = 'concat'
    layer = test_params0.instantiate()

    left = [tf.keras.backend.variable(np.ones((100, 10))) for _ in range(5)]
    right = [tf.keras.backend.variable(np.ones((100, 10))) for _ in range(3)]
    sum_out = tf.reduce_sum(layer((left, right)))
    with self.session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      print(sess.run(sum_out))

  def test_groupint_attention_call(self):
    layer = GroupInt(interaction_type='multiply',
                     use_attention=True,
                     attention_units=[15, 10, 1],
                     activation='relu')

    left = [tf.keras.backend.variable(np.ones((100, 10))) for _ in range(5)]
    right = [tf.keras.backend.variable(np.ones((100, 10))) for _ in range(3)]
    sum_out = tf.reduce_sum(layer((left, right)))
    with self.session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      print(sess.run(sum_out))

  def test_allint_instantiate(self):
    layer_template = AllInt.params()

    test_params0 = layer_template.copy()
    test_params0.cmp_dim = 4
    ins1 = test_params0.instantiate()
    print(ins1)

    ins2 = AllInt(cmp_dim=4)
    print(ins2)

  def test_allint_serde(self):
    layer_template = AllInt.params()

    test_params0 = layer_template.copy()
    test_params0.cmp_dim = 4
    ins1 = test_params0.instantiate()
    print(ins1)

    cfg = ins1.get_config()
    ins2 = AllInt.from_config(cfg)

    print(ins1, ins2)

  def test_allint_call(self):
    layer_template = AllInt.params()

    test_params0 = layer_template.copy()
    test_params0.name = 'test_dense0'
    test_params0.cmp_dim = 4
    layer = test_params0.instantiate()

    data = tf.keras.backend.variable(np.ones((100, 10, 10)))
    sum_out = tf.reduce_sum(layer(data))
    with self.session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      print(sess.run(sum_out))

  def test_cdot_instantiate(self):
    layer_template = CDot.params()

    test_params0 = layer_template.copy()
    test_params0.project_dim = 8
    test_params0.compress_units = [128, 256]
    test_params0.activation = 'tanh'
    ins1 = test_params0.instantiate()
    print(ins1)

    ins2 = CDot(project_dim=8, compress_units=[128, 256], activation='tanh')
    print(ins2)

  def test_cdot_serde(self):
    ins1 = CDot(project_dim=8, compress_units=[128, 256], activation='tanh')

    cfg = ins1.get_config()
    ins2 = CDot.from_config(cfg)

    print(ins1, ins2)

  def test_cdot_call(self):
    layer = CDot(project_dim=8, compress_units=[128, 256], activation='tanh')

    data = tf.keras.backend.variable(np.ones((100, 10, 10)))
    sum_out = tf.reduce_sum(layer(data))
    with self.session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      print(sess.run(sum_out))

  def test_can_instantiate(self):
    layer_template = CAN.params()

    test_params0 = layer_template.copy()
    test_params0.layer_num = 8
    test_params0.activation = 'sigmoid'
    test_params0.is_seq = False
    test_params0.is_stacked = True
    ins1 = test_params0.instantiate()
    print(ins1)

    ins2 = CAN(layer_num=8, activation='tanh', is_seq=False, is_stacked=True)
    print(ins2)

  def test_can_serde(self):
    ins1 = CAN(layer_num=8, activation='tanh', is_seq=False, is_stacked=True)

    cfg = ins1.get_config()
    ins2 = CAN.from_config(cfg)

    print(ins1, ins2)

  def test_can_seq_call(self):
    layer = CAN(layer_num=2, activation='relu', is_seq=True, is_stacked=True)

    user = tf.keras.backend.variable(np.ones((128, 10, 12, 10)))
    item = tf.keras.backend.variable(np.ones((128, 220)))
    sum_out = tf.reduce_sum(layer((user, item)))
    with self.session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      print(sess.run(sum_out))

  def test_can_call(self):
    layer = CAN(layer_num=2, activation='relu', is_seq=False, is_stacked=True)

    user = tf.keras.backend.variable(np.ones((128, 10, 10)))
    item = tf.keras.backend.variable(np.ones((128, 220)))
    sum_out = tf.reduce_sum(layer((user, item)))
    with self.session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      print(sess.run(sum_out))

  def test_dcn_instantiate(self):
    layer_template = DCN.params()

    test_params0 = layer_template.copy()
    test_params0.layer_num = 8
    test_params0.dcn_type = 'matrix'
    test_params0.use_dropout = True
    test_params0.keep_prob = 0.5
    ins1 = test_params0.instantiate()
    print(ins1)

    ins2 = DCN(layer_num=8, dcn_type='matrix', use_dropout=True, keep_prob=0.5)
    print(ins2)

  def test_dcn_serde(self):
    ins1 = DCN(layer_num=8, dcn_type='matrix', use_dropout=True, keep_prob=0.5)

    cfg = ins1.get_config()
    ins2 = DCN.from_config(cfg)

    print(ins1, ins2)

  def test_dcn_vector_call(self):
    layer = DCN(layer_num=2,
                dcn_type='vector',
                allow_kernel_norm=True,
                use_dropout=True,
                keep_prob=0.5)

    data = tf.keras.backend.variable(np.ones((128, 10, 10)))
    sum_out = tf.reduce_sum(layer(data))
    with self.session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      print(sess.run(sum_out))

  def test_dcn_matrix_call(self):
    layer = DCN(layer_num=2,
                dcn_type='matrix',
                allow_kernel_norm=True,
                use_dropout=True,
                keep_prob=0.5)

    data = tf.keras.backend.variable(np.ones((128, 10, 10)))
    sum_out = tf.reduce_sum(layer(data))
    with self.session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      print(sess.run(sum_out))

  def test_dcn_mixed_call(self):
    layer = DCN(layer_num=2,
                dcn_type='mixed',
                num_experts=2,
                low_rank=5,
                allow_kernel_norm=True,
                use_dropout=True,
                keep_prob=0.5)

    data = tf.keras.backend.variable(np.ones((128, 10, 10)))
    sum_out = tf.reduce_sum(layer(data))
    with self.session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      print(sess.run(sum_out))

  def test_cin_instantiate(self):
    layer_template = CIN.params()

    test_params0 = layer_template.copy()
    test_params0.hidden_uints = [10, 5]
    test_params0.activation = 'sigmoid'
    ins1 = test_params0.instantiate()
    print(ins1)

    ins2 = CIN(hidden_uints=[10, 5], activation='tanh')
    print(ins2)

  def test_cin_serde(self):
    ins1 = CIN(hidden_uints=[10, 5], activation='tanh')

    cfg = ins1.get_config()
    ins2 = CIN.from_config(cfg)

    print(ins1, ins2)

  def test_cin_call(self):
    layer = CIN(hidden_uints=[10, 5], activation='relu')

    data = tf.keras.backend.variable(np.ones((128, 10, 10)))
    sum_out = tf.reduce_sum(layer(data))
    with self.session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      print(sess.run(sum_out))


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.compat.v1.disable_v2_behavior()
  tf.test.main()
