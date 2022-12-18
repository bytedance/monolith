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

from monolith.native_training.layers.multi_task import MMoE, SNR


class MultiTaskTest(tf.test.TestCase):

  def test_mmoe_instantiate(self):
    mmoe_layer_template = MMoE.params()

    test_params0 = mmoe_layer_template.copy()
    test_params0.name = 'test_mmoe'
    test_params0.num_tasks = 2
    test_params0.num_experts = 3
    test_params0.expert_output_dims = [128, 64, 64]
    test_params0.expert_activations = 'relu'
    test_params0.expert_initializers = tf.keras.initializers.GlorotNormal()
    mmoe1 = test_params0.instantiate()
    print(mmoe1)

    mmoe2 = MMoE(num_tasks=2,
                 num_experts=3,
                 expert_output_dims=[128, 64, 64],
                 expert_activations='relu',
                 expert_initializers=tf.keras.initializers.GlorotNormal())
    print(mmoe2)

  def test_mmoe_serde(self):
    mmoe_layer_template = MMoE.params()

    test_params0 = mmoe_layer_template.copy()
    test_params0.name = 'test_mmoe'
    test_params0.num_tasks = 2
    test_params0.num_experts = 3
    test_params0.expert_output_dims = [128, 64, 64]
    test_params0.expert_activations = 'relu'
    test_params0.expert_initializers = tf.keras.initializers.GlorotNormal()
    mmoe1 = test_params0.instantiate()

    cfg = mmoe1.get_config()
    mmoe2 = MMoE.from_config(cfg)

    print(mmoe1, mmoe2)

  def test_mmoe_call(self):
    layer = MMoE(num_tasks=2,
                 num_experts=3,
                 gate_type='topk',
                 top_k=2,
                 expert_output_dims=[[128, 64, 64], [64, 64], [128, 64]],
                 expert_activations='relu',
                 expert_initializers=tf.keras.initializers.GlorotNormal())

    dense_data = tf.keras.backend.variable(np.ones((100, 128)))
    # mmoe_data = tf.keras.backend.variable(np.ones((100, 64)))
    # sum_out = tf.reduce_sum(layer([dense_data, mmoe_data]))
    sum_out = tf.reduce_sum(layer(dense_data))
    with self.session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      print(sess.run(sum_out))

  def test_snr_instantiate(self):
    snr_layer_template = SNR.params()

    test_params0 = snr_layer_template.copy()
    test_params0.name = 'test_snr'
    test_params0.num_out_subnet = 3
    test_params0.out_subnet_dim = 128
    test_params0.use_ste = False
    snr1 = test_params0.instantiate()
    print(snr1)

    snr2 = SNR(num_out_subnet=3, out_subnet_dim=128, use_ste=False)
    print(snr2)

  def test_snr_serde(self):
    snr_layer_template = SNR.params()

    test_params0 = snr_layer_template.copy()
    test_params0.name = 'test_snr'
    test_params0.num_out_subnet = 3
    test_params0.out_subnet_dim = 128
    test_params0.use_ste = False
    snr1 = test_params0.instantiate()
    print(snr1)

    cfg = snr1.get_config()
    snr2 = SNR.from_config(cfg)

    print(snr1, snr2)

  def test_snr_call(self):
    layer = SNR(num_out_subnet=3,
                out_subnet_dim=128,
                snr_type='aver',
                use_ste=False,
                mode=tf.estimator.ModeKeys.PREDICT)

    snr_data1 = tf.keras.backend.variable(np.ones((100, 128)))
    snr_data2 = tf.keras.backend.variable(np.ones((100, 128)))
    snr_data3 = tf.keras.backend.variable(np.ones((100, 128)))
    snr_data4 = tf.keras.backend.variable(np.ones((100, 128)))

    sum_out = tf.reduce_sum(layer([snr_data1, snr_data2, snr_data3, snr_data4]))
    with self.session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      print(sess.run(sum_out))


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
