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
import os

import tensorflow as tf

from monolith.native_training.layers.dense import Dense


class DenseTest(tf.test.TestCase):

  def test_dense_instantiate(self):
    dense_layer_template = Dense.params()

    test_params0 = dense_layer_template.copy()
    test_params0.name = 'test_dense0'
    test_params0.units = 100
    test_params0.activation = tf.keras.activations.sigmoid
    test_params0.kernel_initializer = tf.keras.initializers.GlorotNormal()
    ins1 = test_params0.instantiate()
    print(ins1)

    ins2 = Dense(units=100,
                 activation=tf.keras.activations.sigmoid,
                 kernel_initializer=tf.keras.initializers.GlorotNormal())
    print(ins2)

  def test_dense_serde(self):
    dense_layer_template = Dense.params()

    test_params0 = dense_layer_template.copy()
    test_params0.name = 'test_dense0'
    test_params0.units = 100
    test_params0.activation = tf.keras.activations.sigmoid
    test_params0.kernel_initializer = tf.keras.initializers.GlorotNormal()
    ins1 = test_params0.instantiate()
    print(ins1)

    cfg = ins1.get_config()
    ins2 = Dense.from_config(cfg)

    print(ins1, ins2)

  def test_dense_call(self):
    layer = Dense(units=100,
                  activation=tf.keras.activations.sigmoid,
                  kernel_initializer=tf.keras.initializers.GlorotNormal())

    data = tf.keras.backend.variable(np.ones((100, 100)))
    sum_out = tf.reduce_sum(layer(data))
    with self.session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      print(sess.run(sum_out))

  def test_dense_kernel_norm_call(self):
    layer = Dense(units=100,
                  allow_kernel_norm=True,
                  kernel_norm_trainable=True,
                  activation=tf.keras.activations.sigmoid,
                  kernel_initializer=tf.keras.initializers.GlorotNormal())

    data = tf.keras.backend.variable(np.ones((100, 100)))
    sum_out = tf.reduce_sum(layer(data))
    with self.session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      print(sess.run(sum_out))

  def test_inactive_relu_monitor(self):
    dense_layer_template = Dense.params()

    test_params = dense_layer_template.copy()
    test_params.units = 10
    test_params.activation = tf.keras.activations.relu
    test_params.inactive_relu_monitor = True
    layer = test_params.instantiate()

    with tf.Graph().as_default():
      x = tf.constant([[1., 1., 1., 1., 1.]])
      _ = layer(x)
      graph = tf.compat.v1.get_default_graph()
      self.assertIn('Dense/inactive_relu_count_moving_avg_1',
                    [node.name for node in graph.as_graph_def().node])

  def test_dense_with_explicit_partition(self):
    layer = Dense(units=1024,
                  allow_kernel_norm=True,
                  kernel_norm_trainable=True,
                  activation=tf.keras.activations.sigmoid,
                  kernel_initializer=tf.keras.initializers.GlorotNormal(),
                  partitioner=tf.compat.v1.variable_axis_size_partitioner(
                      max_shard_bytes=1 << 17, max_shards=5))

    data = tf.keras.backend.variable(np.ones((100, 294)))
    sum_out = layer(data)
    partition_dims = []
    expected_dims = [59, 59, 59, 59, 58]
    for var in layer.kernel_var:
      partition_dims.append(var.shape[0])

    with self.session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      sum_out = sess.run(sum_out)
      self.assertEqual(sum_out.shape, (100, 1024))

  def test_dense_with_implicit_partition(self):
    with tf.compat.v1.variable_scope(
        "",
        partitioner=tf.compat.v1.variable_axis_size_partitioner(
            max_shard_bytes=1 << 17, max_shards=5)):
      # The dense kernel's shape is [294, 1024] and will be
      # partitioned into five shards(unevenly)
      layer = Dense(units=1024,
                    allow_kernel_norm=True,
                    kernel_norm_trainable=True,
                    activation=tf.keras.activations.sigmoid,
                    kernel_initializer=tf.keras.initializers.GlorotNormal(),
                    partitioner=None)
      data = tf.keras.backend.variable(np.ones((100, 294)))
      sum_out = layer(data)
      partition_dims = []
      expected_dims = [59, 59, 59, 59, 58]
      for var in layer.kernel_var:
        partition_dims.append(var.shape[0])

      self.assertEqual(partition_dims, expected_dims)
    with self.session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      sum_out = sess.run(sum_out)
      self.assertEqual(sum_out.shape, (100, 1024))


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
