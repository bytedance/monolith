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

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Utilities for unit-testing layers.

The implementation for these utilities is similar to that in
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/testing_utils.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import threading

import numpy as np

import tensorflow as tf
from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
# from tensorflow.python.keras import testing_utils


@test_util.disable_cudnn_autotune
def layer_test(layer_cls,
               kwargs=None,
               input_shape=None,
               input_dtype=None,
               input_data=None,
               expected_output=None,
               expected_output_dtype=None,
               expected_output_shape=None,
               validate_training=True,
               adapt_data=None,
               custom_objects=None,
               test_harness=None):
  """Test routine for a BaseLayer with a single input and single output.

    Args:
        layer_cls: BaseLayer class object.
        kwargs: Optional dictionary of keyword arguments for instantiating the
            layer.
        input_shape: Input shape tuple.
        input_dtype: Data type of the input data.
        input_data: Numpy array of input data.
        expected_output: Numpy array of the expected output.
        expected_output_dtype: Data type expected for the output.
        expected_output_shape: Shape tuple for the expected shape of the output.
        validate_training: Whether to attempt to validate training on this layer.
            This might be set to False for non-differentiable layers that output
            string or integer values.
        adapt_data: Optional data for an 'adapt' call. If None, adapt() will not
            be tested for this layer. This is only relevant for PreprocessingLayers.
            custom_objects: Optional dictionary mapping name strings to custom objects
            in the layer class. This is helpful for testing custom layers.
            test_harness: The Tensorflow test, if any, that this function is being
            called in.

    Returns:
        The output data (Numpy array) returned by the layer, for additional
        checks to be done by the calling code.
    Raises:
        ValueError: if `input_shape is None`.
    """
  if input_data is None:
    if input_shape is None:
      raise ValueError('input_shape is None')
    if not input_dtype:
      input_dtype = 'float32'
    input_data_shape = list(input_shape)
    for i, e in enumerate(input_data_shape):
      if e is None:
        input_data_shape[i] = np.random.randint(1, 4)
    input_data = 10 * np.random.random(input_data_shape)
    if input_dtype[:5] == 'float':
      input_data -= 0.5
    input_data = input_data.astype(input_dtype)
  elif input_shape is None:
    input_shape = input_data.shape
  if input_dtype is None:
    input_dtype = input_data.dtype
  if expected_output_dtype is None:
    expected_output_dtype = input_dtype

  if dtypes.as_dtype(expected_output_dtype) == dtypes.string:
    if test_harness:
      assert_equal = test_harness.assertAllEqual
    else:
      assert_equal = string_test
  else:
    if test_harness:
      assert_equal = test_harness.assertAllClose
    else:
      # assert_equal = tf.python.keras.testing_utils.numeric_test
      assert_equal = testing_utils.numeric_test

  # instantiation
  kwargs = kwargs or {}
  layer = layer_cls(**kwargs)

  # Test adapt, if data was passed.
  if adapt_data is not None:
    layer.adapt(adapt_data)

  # test get_weights , set_weights at layer level
  weights = layer.get_weights()
  layer.set_weights(weights)

  # test and instantiation from weights
  if 'weights' in tf_inspect.getargspec(layer_cls.__init__):
    kwargs['weights'] = weights
    layer = layer_cls(**kwargs)

  # test in functional API
  x = layers.Input(shape=input_shape[1:], dtype=input_dtype)
  y = layer(x)
  if backend.dtype(y) != expected_output_dtype:
    raise AssertionError('When testing layer %s, for input %s, found output '
                         'dtype=%s but expected to find %s.\nFull kwargs: %s' %
                         (layer_cls.__name__, x, backend.dtype(y),
                          expected_output_dtype, kwargs))

  def assert_shapes_equal(expected, actual):
    """Asserts that the output shape from the layer matches the actual shape."""
    if len(expected) != len(actual):
      raise AssertionError(
          'When testing layer %s, for input %s, found output_shape='
          '%s but expected to find %s.\nFull kwargs: %s' %
          (layer_cls.__name__, x, actual, expected, kwargs))

    for expected_dim, actual_dim in zip(expected, actual):
      if isinstance(expected_dim, tensor_shape.Dimension):
        expected_dim = expected_dim.value
      if isinstance(actual_dim, tensor_shape.Dimension):
        actual_dim = actual_dim.value
      if expected_dim is not None and expected_dim != actual_dim:
        raise AssertionError(
            'When testing layer %s, for input %s, found output_shape='
            '%s but expected to find %s.\nFull kwargs: %s' %
            (layer_cls.__name__, x, actual, expected, kwargs))

  if expected_output_shape is not None:
    assert_shapes_equal(tensor_shape.TensorShape(expected_output_shape),
                        y.shape)

  # check shape inference
  model = models.Model(x, y)
  computed_output_shape = tuple(
      layer.compute_output_shape(
          tensor_shape.TensorShape(input_shape)).as_list())
  computed_output_signature = layer.compute_output_signature(
      tensor_spec.TensorSpec(shape=input_shape, dtype=input_dtype))
  actual_output = model.predict(input_data)
  actual_output_shape = actual_output.shape
  assert_shapes_equal(computed_output_shape, actual_output_shape)
  assert_shapes_equal(computed_output_signature.shape, actual_output_shape)
  if computed_output_signature.dtype != actual_output.dtype:
    raise AssertionError(
        'When testing layer %s, for input %s, found output_dtype='
        '%s but expected to find %s.\nFull kwargs: %s' %
        (layer_cls.__name__, x, actual_output.dtype,
         computed_output_signature.dtype, kwargs))
  if expected_output is not None:
    assert_equal(actual_output, expected_output)
