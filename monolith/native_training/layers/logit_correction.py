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
from tensorflow.python.ops import math_ops
from tensorflow.keras import activations
from tensorflow.keras import initializers
from tensorflow.python.keras import regularizers

from monolith.native_training.utils import with_params
from monolith.native_training.layers.mlp import MLP
from monolith.native_training.monolith_export import monolith_export


@monolith_export
@with_params
class LogitCorrection(Layer):
  """Logit校正, 由于采样等原因, 会使得CTR/CVR的预测与后验均值有偏差, 需要对这种偏差进行校正

  Logit校正可以在训练时进行, 也可以在推理时进行, 为了减轻推理时负担, 一般选择训练时进行, LogitCorrection就是用于训练时校正的
  
  Args:
    activation (:obj:`tf.activation`): 激活函数, 默认为None
    sample_bias (:obj:`bool`): 是否校正样本采样偏差

  """

  def __init__(self, activation=None, sample_bias: bool = False, **kwargs):
    super(LogitCorrection, self).__init__(**kwargs)
    # compatible with older version forced sumpooling
    # self.input_spec = InputSpec(shape=[None, None, 1])
    self.input_spec = [InputSpec(max_ndim=2), InputSpec(max_ndim=2)]
    self.activation = activations.get(activation)
    self.sample_bias = sample_bias

  def call(self, inputs, **kwargs):
    # tensor with tf.shape([None,])
    logits, sample_rate = inputs
    corrected = self.get_sample_logits(logits, sample_rate, self.sample_bias)
    if self.activation is not None:
      corrected = self.activation(corrected)
    return corrected

  @staticmethod
  def safe_log_sigmoid(logits):
    zeros = tf.zeros_like(logits, dtype=logits.dtype)
    cond = (logits >= zeros)
    relu_logits = tf.where(cond, logits, zeros)
    neg_abs_logits = tf.where(cond, -logits, logits)
    return tf.negative(relu_logits - logits +
                       tf.compat.v1.log1p(tf.exp(neg_abs_logits)))

  @staticmethod
  def get_sample_logits(logits, sample_rate, sample_bias):
    if sample_rate is None and sample_bias:
      return LogitCorrection.safe_log_sigmoid(logits)
    elif sample_rate is not None and not sample_bias:
      return tf.add(logits, tf.negative(tf.compat.v1.log(sample_rate)))
    elif sample_rate is not None and sample_bias:
      return tf.add(LogitCorrection.safe_log_sigmoid(logits),
                    tf.negative(tf.compat.v1.log(sample_rate)))
    else:
      return logits

  def compute_output_shape(self, input_shape):
    return tuple(tf.shape([
        None,
    ]))

  def get_config(self):
    config = {
        'activation': activations.serialize(self.activation),
        'sample_bias': self.sample_bias
    }
    base_config = super(LogitCorrection, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
