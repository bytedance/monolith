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

import tensorflow as tf
from tensorflow.python.ops import variables as variable_ops
from tensorflow.python.keras.layers import Layer
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.layers.ops import core as core_ops

from monolith.native_training.utils import with_params, get_uname
from monolith.native_training.monolith_export import monolith_export


@monolith_export
@with_params
class Dense(Layer):
  """Dense Layer实现 :math:`y = active(wx + b)`.

  之所以要重新实现一个Dense Layer, 是因为增加的了些额外的操作, 如kernel_norm, 论文可参考 https://arxiv.org/pdf/1602.07868.pdf
  kernel_norm的计算方式为: 
  
  .. math::
  
    y = active( norm_{kernel} * l2_{normalize}(W) x + b)
  
  先对W求 :math:`l2_{normalize}`, 将其取值限制在[-1, 1]之间, 然后乘以 :math:`norm_{kernel}`, 这样 :math:`norm_{kernel} * l2_{normalize}(W)` 
  的取值在 [-kernel_norm, kernel_norm]之间, 可以有效地防止梯度爆炸. :math:`norm_{kernel}` 一般由W的初值决定, 有 :math:`norm_{kernel} = morm(W_{init})`. 
  也可让 :math:`norm_{kernel}` 成为trainable, 让算法自已调节.
  
  Args:
    units (:obj:`tf.Tensor`): 输入, 也就是x
    activation (:obj:`tf.activation`, `str`): 激活函数, 可以用str表示, 也可以用TF中的activation
    use_bias (:obj:`bool`): 是否使用bias
    kernel_initializer (:obj:`tf.initializer`): kernel, 也就是W的初始化器
    bias_initializer (:obj:`tf.initializer`): bias, 也就是b的初始化器
    bias_regularizer (:obj:`tf.regularizer`): bias正侧化
    allow_kernel_norm (:obj:`bool`): 是否开启kernel_norm
    kernel_norm_trainable (:obj:`bool`):  是否让kernel_norm可训练
    partitioner (:obj:`tf.partitioner`, optional): 分区器, 可以将一个大变量分到不同的PS机器上
    inactive_relu_monitor (:obj:`bool`): 是否开启relu_monitor
    inactive_relu_monitor_decay (:obj:`float`): 因为relu的非0率是用指数平均来计算的, decay就是衰减因子
    optimizer (:obj:`tf.optimizer`): 优化器, 请参考TF

  >>> dense = Dense(units=100,
  >>>               activation=tf.keras.activations.sigmoid,
  >>>               kernel_initializer=tf.keras.initializers.GlorotNormal())
  >>> y = dense(x)
  
  """

  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               allow_kernel_norm=False,
               kernel_norm_trainable=False,
               partitioner=None,
               inactive_relu_monitor=False,
               inactive_relu_monitor_decay=0.1,
               optimizer=None,
               **kwargs):
    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
      kwargs['input_shape'] = (kwargs.pop('input_dim'),)
    # Call the _init__() function for tf.keras.layers.Dense
    super(Dense, self).__init__(**kwargs)

    # Change/Add some class properties to the tf.keras.layers.Dense
    # properties. Note that this Dense layer does not support regularizers
    # and constraints.
    self.units = units
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer or
                                               'glorot_uniform')
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_var = None

    self.supports_masking = True
    self.input_spec = InputSpec(min_ndim=2)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.allow_kernel_norm = allow_kernel_norm
    self.kernel_norm_trainable = kernel_norm_trainable
    self.partitioner = partitioner
    self.inactive_relu_monitor = inactive_relu_monitor
    self.inactive_relu_monitor_decay = inactive_relu_monitor_decay
    self.optimizer = optimizer

  def add_weight(self,
                 name=None,
                 shape=None,
                 dtype=None,
                 initializer=None,
                 regularizer=None,
                 trainable=None,
                 constraint=None,
                 use_resource=None,
                 synchronization=tf.VariableSynchronization.AUTO,
                 aggregation=tf.VariableAggregation.NONE,
                 **kwargs):
    var = super().add_weight(name=name,
                             shape=shape,
                             dtype=dtype,
                             initializer=initializer,
                             regularizer=regularizer,
                             trainable=trainable,
                             constraint=constraint,
                             use_resource=use_resource,
                             synchronization=synchronization,
                             aggregation=aggregation,
                             **kwargs)
    if isinstance(var, tf.Variable):
      var.optimizer = self.optimizer
    elif isinstance(var, variable_ops.PartitionedVariable):
      for var_p in var:
        var_p.optimizer = self.optimizer
    return var

  def get_variable(self,
                   name,
                   shape=None,
                   dtype=None,
                   initializer=None,
                   regularizer=None,
                   trainable=None,
                   collections=None,
                   caching_device=None,
                   partitioner=None,
                   validate_shape=True,
                   use_resource=None,
                   custom_getter=None,
                   constraint=None,
                   synchronization=tf.VariableSynchronization.AUTO,
                   aggregation=tf.VariableAggregation.NONE):
    cur_name_scope = tf.compat.v1.get_default_graph().get_name_scope()
    with tf.compat.v1.variable_scope(cur_name_scope,
                                     reuse=tf.compat.v1.AUTO_REUSE):
      var = tf.compat.v1.get_variable(name=name,
                                      shape=shape,
                                      dtype=dtype,
                                      initializer=initializer,
                                      regularizer=regularizer,
                                      trainable=trainable,
                                      collections=collections,
                                      caching_device=caching_device,
                                      partitioner=partitioner,
                                      validate_shape=validate_shape,
                                      use_resource=use_resource,
                                      custom_getter=custom_getter,
                                      constraint=constraint,
                                      synchronization=synchronization,
                                      aggregation=aggregation)
    if isinstance(var, tf.Variable):
      var.optimizer = self.optimizer
    elif isinstance(var, variable_ops.PartitionedVariable):
      for var_p in var:
        var_p.optimizer = self.optimizer

    if base_layer_utils.is_split_variable(var) or isinstance(
        var, variable_ops.PartitionedVariable):
      for v in var:
        K.track_variable(v)
        if trainable:
          self._trainable_weights.append(v)
        else:
          self._non_trainable_weights.append(v)
    else:
      K.track_variable(var)
      if trainable:
        self._trainable_weights.append(var)
      else:
        self._non_trainable_weights.append(var)
    return var

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

    self.kernel_var = self.get_variable(initializer=init_kernel,
                                        trainable=True,
                                        name='kernel',
                                        shape=None,
                                        dtype=dtype,
                                        regularizer=self.kernel_regularizer,
                                        partitioner=self.partitioner)
    self.kernel = self.kernel_var

    # Add the option for allow_kernel_norm
    if self.allow_kernel_norm:
      self.kernel = tf.nn.l2_normalize(self.kernel,
                                       axis=0,
                                       epsilon=1e-6,
                                       name='normalized_kernel')
      if self.kernel_norm_trainable:
        init_trainable_kernel_norm = tf.linalg.norm(init_kernel, axis=0)
        self.trainable_kernel_norm = self.get_variable(
            initializer=init_trainable_kernel_norm,
            shape=None,
            trainable=True,
            name='trainable_kernel_norm',
            dtype=dtype,
            partitioner=self.partitioner)
        self.kernel = tf.multiply(self.kernel,
                                  self.trainable_kernel_norm,
                                  name='mul_of_kernel_and_trainable_norm')

    if self.use_bias:
      self.bias = self.add_weight(name='bias',
                                  shape=[self.units],
                                  initializer=self.bias_initializer,
                                  regularizer=self.bias_regularizer,
                                  dtype=dtype,
                                  trainable=True)
    else:
      self.bias = None

    if self.inactive_relu_monitor and self.activation.__name__ == 'relu':
      self.inactive_relu_count_moving_avg = self.get_variable(
          initializer=tf.keras.initializers.zeros,
          trainable=False,
          name='inactive_relu_count_moving_avg',
          shape=[self.units],
          dtype=tf.float32,
          collections=[
              tf.compat.v1.GraphKeys.METRIC_VARIABLES,
              tf.compat.v1.GraphKeys.GLOBAL_VARIABLES
          ])

    super(Dense, self).build(input_shape)

  def call(self, inputs, **kwargs):
    output = core_ops.dense(inputs,
                            self.kernel,
                            self.bias,
                            self.activation,
                            dtype=self._compute_dtype_object)
    if self.inactive_relu_monitor:
      inactive_relu_count = self.units - tf.math.count_nonzero(output, axis=0)
      tf.compat.v1.summary.histogram('inactive_relu_count_moving_avg',
                                     self.inactive_relu_count_moving_avg)
      update_op = tf.compat.v1.assign(
          self.inactive_relu_count_moving_avg,
          (1. - self.inactive_relu_monitor_decay) *
          self.inactive_relu_count_moving_avg +
          self.inactive_relu_monitor_decay *
          tf.cast(inactive_relu_count, dtype=tf.float32),
      )
      with tf.control_dependencies([update_op]):
        output = tf.identity(output)
    return output

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError(
          'The innermost dimension of input_shape must be defined, but saw: %s'
          % input_shape)
    return input_shape[:-1].concatenate(self.units)

  def get_config(self):
    config = {
        'units': self.units,
        'activation': activations.serialize(self.activation),
        'use_bias': self.use_bias,
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        'allow_kernel_norm': self.allow_kernel_norm,
        'kernel_norm_trainable': self.kernel_norm_trainable,
        'partitioner': self.partitioner,
    }
    base_config = super(Dense, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
