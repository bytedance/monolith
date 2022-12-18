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

"Code to implement custom Sail-like layer using TensorFlow Keras API."

from __future__ import absolute_import, division, print_function

import functools
import sys

import numpy as np
import scipy.stats as stats
import tensorflow as tf
from absl import logging
from tensorflow.python.framework import dtypes, tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints, initializers, regularizers
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import gen_math_ops, math_ops, nn

from monolith.core.base_layer import BaseLayer
from monolith.core.variance_scaling import VarianceScaling


class Dense(tf.keras.layers.Dense, BaseLayer):

  @classmethod
  def params(cls):
    p = super(Dense, cls).params()
    p.define('units', 512, 'Positive integer, dimensionality of the ' \
        'output space.')
    p.define('activation', None, 'Activation function to use.')
    p.define('use_bias', True, 'Boolean, whether the layer uses a bias ' \
        'vector.')
    p.define('kernel_initializer',
        VarianceScaling(mode='fan_avg', distribution='uniform'),
        'Initializer for the `kernel` weights matrix. Currently only '\
        'supporting variance scaling initializer.')
    p.define('bias_initializer', 'zeros', 'Initializer for the bias vector.')
    p.define('allow_kernel_norm', True,
             'Boolean, kernel normalization is only applicable when TRAINING.')
    p.define('kernel_norm_trainable', True,
             'Boolean, whether a trainable weight norm variable is allocated')
    p.define('partitioner', None,
             'VariablePartitioner, if we will use partitioned variable')

    return p

  def __init__(self, params, **kwargs):
    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
      kwargs['input_shape'] = (kwargs.pop('input_dim'),)
    # Call the __init__() function for BaseLayer
    BaseLayer.__init__(self, params)
    # Call the _init__() function for tf.keras.layers.Dense
    super(Dense, self).__init__(
        units=params.units,
        activation=params.activation,
        use_bias=params.use_bias,
        kernel_initializer=params.kernel_initializer,
        bias_initializer=params.bias_initializer,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    )
    # Change/Add some class properties to the tf.keras.layers.Dense
    # properties. Note that this Dense layer does not support regularizers
    # and constraints.
    self.p = params
    self.units = int(
        params.units) if not isinstance(params.units, int) else params.units
    self.activation = activations.get(params.activation)
    self.use_bias = params.use_bias
    self.kernel_initializer = params.kernel_initializer
    self.bias_initializer = initializers.get(params.bias_initializer)

    self.supports_masking = True
    self.input_spec = InputSpec(min_ndim=2)

    self.allow_kernel_norm = params.allow_kernel_norm
    self.kernel_norm_trainable = params.kernel_norm_trainable
    self.var_name_prefix = params.name
    self.partitioner = params.partitioner

  def build(self, input_shape):
    dtype = tf.dtypes.as_dtype(self.dtype or K.floatx())
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError('Unable to build `Dense` layer with non-floating point '
                      'dtype %s' % (dtype,))
    input_shape = tensor_shape.TensorShape(input_shape)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    last_dim = tensor_shape.dimension_value(input_shape[-1])
    self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})

    kernel_shape = [last_dim, self.units]
    init_kernel = self.kernel_initializer(shape=kernel_shape, dtype=self.dtype)
    if self.partitioner is None:
      kernel_initializer = lambda shape, dtype: init_kernel
    else:
      kernel_initializer = init_kernel

    self.kernel = tf.compat.v1.get_variable(initializer=kernel_initializer,
                                            trainable=True,
                                            name="{}/kernel".format(
                                                self.var_name_prefix),
                                            shape=kernel_shape,
                                            dtype=dtype,
                                            partitioner=self.partitioner)

    # Add the option for allow_kernel_norm
    if self.allow_kernel_norm:
      self.kernel = tf.nn.l2_normalize(self.kernel,
                                       axis=0,
                                       epsilon=1e-6,
                                       name='normalized_kernel')
      if self.kernel_norm_trainable:
        # Use np to mitigate the error thrown by tensorflow due to the variable
        # initializer inside a conditional.
        init_trainable_kernel_norm = np.linalg.norm(
            init_kernel,
            axis=0,
        )
        if self.partitioner is None:
          norm_initializer = lambda shape, dtype: init_trainable_kernel_norm
        else:
          norm_initializer = init_trainable_kernel_norm
        self.trainable_kernel_norm = tf.compat.v1.get_variable(
            initializer=norm_initializer,
            shape=init_trainable_kernel_norm.shape,
            trainable=True,
            name='{}/trainable_kernel_norm'.format(self.var_name_prefix),
            dtype=dtype,
            partitioner=self.partitioner)
        self.kernel = tf.multiply(self.kernel,
                                  self.trainable_kernel_norm,
                                  name='mul_of_kernel_and_trainable_norm')

    if self.use_bias:
      self.bias = self.add_weight(name='{}/bias'.format(self.var_name_prefix),
                                  shape=[
                                      self.units,
                                  ],
                                  initializer=self.bias_initializer,
                                  dtype=dtype,
                                  trainable=True)
    else:
      self.bias = None
    self.built = True

  def get_config(self):
    config = {
        'units': self.units,
        'activation': activations.serialize(self.activation),
        'use_bias': self.use_bias,
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'allow_kernel_norm': self.allow_kernel_norm,
        'kernel_norm_trainable': self.kernel_norm_trainable,
        'partitioner': self.partitioner,
    }
    base_config = super(Dense, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def fprop(self, inputs, **kwargs):
    return self.call(inputs)
