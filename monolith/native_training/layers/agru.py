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
from tensorflow.python.eager import context
from tensorflow.python.keras import backend
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
import tensorflow.python.keras.layers.legacy_rnn.rnn_cell_impl as rnn_impl

from monolith.native_training.utils import with_params
from monolith.native_training.monolith_export import monolith_export
from monolith.native_training.layers.utils import check_dim, dim_size

_hasattr = rnn_impl._hasattr
_concat = rnn_cell_impl._concat
_zero_state_tensors = rnn_cell_impl._zero_state_tensors

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

# This can be used with self.assertRaisesRegexp for assert_like_rnncell.
ASSERT_LIKE_RNNCELL_ERROR_REGEXP = "is not an RNNCell"

__all__ = ['AGRUCell', 'dynamic_rnn_with_attention']


@monolith_export
@with_params
class AGRUCell(Layer):
  """带attention的GRU单元, 用于DIEN中.
  
  Args:
    units (:obj:`int`): GRU隐含层大小
    att_type (:obj:`str`): attention方式, 支持两种AGRU/AUGRU
    activation (:obj:`tf.activation`): 激活函数
    initializer (:obj:`tf.initializer`): kernel初始化器
    regularizer (:obj:`tf.regularizer`): kernel正则化
  
  """

  def __init__(self,
               units,
               att_type='AGRU',
               activation=None,
               initializer=None,
               regularizer=None,
               **kwargs):
    super(AGRUCell, self).__init__(**kwargs)

    # Inputs must be 2-dimensional.
    assert att_type.upper() in {'AGRU', 'AUGRU'}
    self.input_spec = [
        InputSpec(ndim=2),
        InputSpec(ndim=2),
        InputSpec(max_ndim=2)
    ]
    self.units = units
    self.att_type = att_type
    self.activation = activations.get(activation or math_ops.tanh)
    self.initializer = tf.initializers.get(
        initializer) or tf.initializers.HeNormal()
    self.regularizer = regularizers.get(regularizer)

  @property
  def state_size(self):
    return self.units

  @property
  def output_size(self):
    return self.units

  def build(self, inputs_shape):
    input_shape, state_shape, att_shape = inputs_shape
    assert check_dim(state_shape[-1]) == self.units
    input_depth = check_dim(input_shape[-1])
    if input_shape[-1] == -1:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" %
                       str(inputs_shape))
    self._gate_kernel = self.add_weight(
        name="gates/{}".format(_WEIGHTS_VARIABLE_NAME),
        dtype=self.dtype,
        shape=[input_depth + self.units, 2 * self.units],
        initializer=self.initializer,
        regularizer=self.regularizer)
    self._gate_bias = self.add_weight(
        name="gates/{}".format(_BIAS_VARIABLE_NAME),
        dtype=self.dtype,
        shape=[2 * self.units],
        initializer=initializers.Ones(),
        regularizer=self.regularizer)

    self._candidate_kernel = self.add_weight(
        name="candidate/{}".format(_WEIGHTS_VARIABLE_NAME),
        dtype=self.dtype,
        shape=[input_depth + self.units, self.units],
        initializer=self.initializer,
        regularizer=self.regularizer)
    self._candidate_bias = self.add_weight(
        name="candidate/{}".format(_BIAS_VARIABLE_NAME),
        dtype=self.dtype,
        shape=[self.units],
        initializer=initializers.Ones(),
        regularizer=self.regularizer)

    super(AGRUCell, self).build(inputs_shape)

  def call(self, inputs, **kwargs):
    x, state, att_score = inputs
    gate_inputs = math_ops.matmul(array_ops.concat([x, state], 1),
                                  self._gate_kernel)
    gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

    value = math_ops.sigmoid(gate_inputs)
    r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

    r_state = r * state
    candidate = math_ops.matmul(array_ops.concat([x, r_state], 1),
                                self._candidate_kernel)
    candidate = nn_ops.bias_add(candidate, self._candidate_bias)

    c = self.activation(candidate)
    if att_score is None:
      new_h = (1.0 - u) * state + u * c
    else:
      if self.att_type.upper() == 'AUGRU':
        # GRU with attentional update gate（AUGRU）
        u = (1.0 - att_score) * u
        new_h = u * state + (1 - u) * c
      else:  # self.att_type.upper() == 'AGRU':
        # Attention based GRU（AGRU）
        new_h = (1. - att_score) * state + att_score * c
    return new_h, new_h

  def zero_state(self, batch_size, dtype):
    # Try to use the last cached zero_state. This is done to avoid recreating
    # zeros, especially when eager execution is enabled.
    state_size = self.state_size
    is_eager = context.executing_eagerly()
    if is_eager and _hasattr(self, "_last_zero_state"):
      (last_state_size, last_batch_size, last_dtype,
       last_output) = getattr(self, "_last_zero_state")
      if (last_batch_size == batch_size and last_dtype == dtype and
          last_state_size == state_size):
        return last_output
    with backend.name_scope(type(self).__name__ + "ZeroState"):
      output = _zero_state_tensors(state_size, batch_size, dtype)
    if is_eager:
      self._last_zero_state = (state_size, batch_size, dtype, output)
    return output

  def get_config(self):
    config = {
        "units": self.units,
        "att_type": self.att_type,
        "initializer": initializers.serialize(self.initializer),
        "activation": activations.serialize(self.activation),
        'regularizer': regularizers.serialize(self.regularizer)
    }
    base_config = super(AGRUCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@monolith_export
def create_ta(name, size, dtype):
  """创建Tensor Array, 一般用于while循环中存放中间结果.

  Args:
    name (:obj:`str`): Array名称
    size (:obj:`int`): Array大小
    dtype (:obj:`tf.DType`): 数据类型
    
  """

  return tensor_array_ops.TensorArray(dtype=dtype,
                                      size=size,
                                      tensor_array_name=name)


@monolith_export
def static_rnn_with_attention(cell, inputs, att_scores, init_state=None):
  """带Attention的静态RNN, 利用python for循环直接将时间维度静态展开, 模型大小会增大

  Args:
    cell (:obj:`RNNCell`): RNN单元
    inputs (:obj:`tf.Tensor`): 输入数据, shape为(batch_size, seq_len, emb_size)
    att_scores (:obj:`tf.Tensor`): attention权重, shape为(batch_size, seq_len)
    init_state (:obj:`tf.Tensor`): 初始化状态
  
  """

  assert isinstance(cell, AGRUCell)
  if init_state is None:
    batch_size = dim_size(inputs, 0)
    if getattr(cell, "get_initial_state", None) is not None:
      state = cell.get_initial_state(inputs=None,
                                     batch_size=batch_size,
                                     dtype=dtype)
    else:
      state = cell.zero_state(batch_size, inputs.dtype)
  else:
    state = init_state

  inputs, outputs = tf.unstack(tf.transpose(inputs, [1, 0, 2])), []
  for time, inp in enumerate(inputs):
    attr = tf.reshape(att_scores[:, time], shape=(-1, 1))
    cell_out, new_state = cell((inp, state, attr))
    state = new_state
    outputs.append(state)

  outputs = tf.transpose(tf.stack(outputs), [1, 0, 2])

  return outputs, state


@monolith_export
def dynamic_rnn_with_attention(cell,
                               inputs,
                               att_scores,
                               parallel_iterations=1,
                               swap_memory=True,
                               init_state=None):
  """带Attention的动态RNN, 得用tf.while实现, 模型大小不会增大

  Args:
    cell (:obj:`RNNCell`): RNN单元
    inputs (:obj:`tf.Tensor`): 输入数据, shape为(batch_size, seq_len, emb_size)
    att_scores (:obj:`tf.Tensor`): attention权重, shape为(batch_size, seq_len)
    parallel_iterations (:obj:`int`): 并行迭代次数, 具体请参考`control_flow_ops.while_loop`
    swap_memory (:obj:`bool`): 是否swap内存, 具体请参考`control_flow_ops.while_loop`
    init_state (:obj:`tf.Tensor`): 初始化状态
  
  """

  assert isinstance(cell, AGRUCell)
  batch_size, time_steps = dim_size(inputs, 0), dim_size(inputs, 1)
  time = array_ops.constant(0, dtype=tf.dtypes.int32, name="time")

  if init_state is None:
    if getattr(cell, "get_initial_state", None) is not None:
      state = cell.get_initial_state(inputs=None,
                                     batch_size=batch_size,
                                     dtype=dtype)
    else:
      state = cell.zero_state(batch_size, inputs.dtype)
  else:
    state = init_state

  with ops.name_scope("dynamic_rnn"):
    output_ta = create_ta("output_ta", time_steps, inputs.dtype)
    input_ta = create_ta("input_ta", time_steps, inputs.dtype)

  # [batch_size, time, emb_dim] -> [time, batch_size, emb_dim]
  input_ta = input_ta.unstack(tf.transpose(inputs, [1, 0, 2]))

  def _body(time, output_ta, state, att_scores):
    att_score = tf.reshape(att_scores[:, time], shape=(-1, 1))  # [bz, 1]
    cell_out, new_state = cell((input_ta.read(time), state, att_score))
    output_ta = output_ta.write(time, cell_out)

    return (time + 1, output_ta, new_state, att_scores)

  _, output_final, final_state, _ = control_flow_ops.while_loop(
      cond=lambda time, *_: time < time_steps,
      body=_body,
      loop_vars=(time, output_ta, state, att_scores),
      parallel_iterations=parallel_iterations,
      swap_memory=swap_memory)

  outputs = output_final.stack()
  outputs = tf.transpose(outputs, [1, 0, 2])
  outputs.set_shape([None, time_steps, dim_size(outputs, -1)])

  return outputs, final_state
