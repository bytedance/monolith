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
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers
from tensorflow.python.keras import regularizers

from monolith.native_training.utils import get_ndim, int_shape, with_params
from monolith.native_training.monolith_export import monolith_export
from monolith.native_training.layers.utils import check_dim, dim_size


@monolith_export
@with_params
class AddBias(Layer):
  r"""AddBias 执行 :math:`y = x + b`, 与直接用`+`相比, AddBias处理了更多的shape问题
  
  例如image有两种表示方式NWHC, NCWH, 对于时间序列也有类似的问题. AddBias可以让用户透明增加Bias
  
  >>> add_bias = AddBias(initializer=tf.initializers.Zeros())
  >>> y = add_bias(x, data_format='channels_first')

  Args:
    initializer (:obj:`tf.initializer`): bias的初始化器
    regularizer (:obj:`tf.regularizer`): bias的正则化器
  
  """

  def __init__(self, initializer=None, regularizer=None, **kwargs):
    super(AddBias, self).__init__(**kwargs)
    self.initializer = initializers.get(initializer) or tf.initializers.Zeros()
    self.regularizer = regularizers.get(regularizer)

    # allowed input specification
    self.input_spec = InputSpec(min_ndim=2)
    self.bias = None

  def build(self, input_shape):
    shape = list(map(check_dim, input_shape[1:]))
    self.bias = self.add_weight(name='bias',
                                shape=shape,
                                dtype=tf.float32,
                                initializer=self.initializer,
                                regularizer=self.regularizer)

  def call(self, inputs, **kwargs):
    data_format = kwargs.get('data_format', 'channels_last')
    if data_format not in {'channels_first', 'channels_last'}:
      raise ValueError('Unknown data_format: ' + str(data_format))
    bias_shape = int_shape(self.bias)
    if len(bias_shape) != 1 and len(bias_shape) != get_ndim(inputs) - 1:
      raise ValueError(
          'Unexpected bias dimensions %d, expect to be 1 or %d dimensions' %
          (len(bias_shape), get_ndim(inputs)))
    if get_ndim(inputs) == 5:
      if data_format == 'channels_first':
        if len(bias_shape) == 1:
          inputs += tf.reshape(self.bias, (1, bias_shape[0], 1, 1, 1))
        else:
          inputs += tf.reshape(self.bias, (1, bias_shape[3]) + bias_shape[:3])
      elif data_format == 'channels_last':
        if len(bias_shape) == 1:
          inputs += tf.reshape(self.bias, (1, 1, 1, bias_shape[0]))
        else:
          inputs += tf.reshape(self.bias, (1,) + bias_shape)
    elif get_ndim(inputs) == 4:
      if data_format == 'channels_first':
        if len(bias_shape) == 1:
          inputs += tf.reshape(self.bias, (1, bias_shape[0], 1, 1))
        else:
          inputs += tf.reshape(self.bias, (1, bias_shape[2]) + bias_shape[:2])
      elif data_format == 'channels_last':
        if len(bias_shape) == 1:
          inputs = tf.nn.bias_add(inputs, self.bias, data_format='NHWC')
        else:
          inputs += tf.reshape(self.bias, (1,) + bias_shape)
    elif get_ndim(inputs) == 3:
      if data_format == 'channels_first':
        if len(bias_shape) == 1:
          inputs += tf.reshape(self.bias, (1, bias_shape[0], 1))
        else:
          inputs += tf.reshape(self.bias, (1, bias_shape[1], bias_shape[0]))
      elif data_format == 'channels_last':
        if len(bias_shape) == 1:
          inputs += tf.reshape(self.bias, (1, 1, bias_shape[0]))
        else:
          inputs += tf.reshape(self.bias, (1,) + bias_shape)
    else:
      inputs = tf.nn.bias_add(inputs, self.bias)
    return inputs

  def get_config(self):
    config = {
        'initializer': tf.keras.initializers.serialize(self.initializer),
        'regularizer': regularizers.serialize(self.regularizer),
    }
    base_config = super(AddBias, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
