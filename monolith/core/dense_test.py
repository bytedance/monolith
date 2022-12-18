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

"""Tests for the Dense layer."""

from __future__ import absolute_import, division, print_function

import textwrap

import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes, ops
from tensorflow.python.keras import backend

from monolith.core import testing_utils
from monolith.core.dense import Dense


class DenseTest(tf.test.TestCase):

  def test_dense_instantiate(self):
    dense_layer_template = Dense.params()

    test_params0 = dense_layer_template.copy()
    test_params0.name = 'test_dense0'
    test_params0.units = 3
    testing_utils.layer_test(Dense,
                             kwargs={'params': test_params0},
                             input_shape=(3, 2))

    test_params1 = dense_layer_template.copy()
    test_params1.name = 'test_dense1'
    test_params1.units = 3
    testing_utils.layer_test(Dense,
                             kwargs={'params': test_params1},
                             input_shape=(3, 4, 2))

    test_params2 = dense_layer_template.copy()
    test_params2.name = 'test_dense2'
    test_params2.units = 3
    testing_utils.layer_test(Dense,
                             kwargs={'params': test_params2},
                             input_shape=(None, None, 2))

    test_params3 = dense_layer_template.copy()
    test_params3.name = 'test_dense3'
    test_params3.units = 3
    testing_utils.layer_test(Dense,
                             kwargs={'params': test_params3},
                             input_shape=(3, 4, 5, 2))

  def test_dense_dtype(self):
    dense_layer_template = Dense.params()
    test_params0 = dense_layer_template.copy()
    test_params0.name = 'test_dense0'
    test_params0.units = 3

    inputs = ops.convert_to_tensor_v2(
        np.random.randint(low=0, high=7, size=(2, 2)))
    layer = Dense(test_params0, dtype='float32')
    outputs = layer(inputs)
    self.assertEqual(outputs.dtype, 'float32')

  def test_dense(self):
    dense_layer_template = Dense.params()
    test_params0 = dense_layer_template.copy()
    test_params0.name = 'test_dense0'
    test_params0.units = 3

    layer = Dense(test_params0)

    output = layer(keras.backend.variable(np.ones((2, 4))))
    self.assertAllEqual((2, 3), output.shape)

    with self.session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      sess.run(output)

  def test_dense_with_partitioner(self):
    param = Dense.params()
    param.name = "test_dense_with_partitioner"
    param.units = 5
    param.partitioner = tf.compat.v1.variable_axis_size_partitioner(1024)
    layer = Dense(param)
    output = layer(keras.backend.variable(np.ones((2, 4096))))
    self.assertAllEqual((2, 5), output.shape)

    with self.session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      sess.run(output)


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
