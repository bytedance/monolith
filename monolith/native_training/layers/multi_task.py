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

import math
import numpy as np
from typing import Union, List, Optional, Any

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.keras.layers import Layer, InputSpec

from monolith.core.base_layer import add_layer_loss
from monolith.native_training.utils import with_params
import monolith.native_training.layers.advanced_activations as ad_acts
from monolith.native_training.monolith_export import monolith_export
from monolith.native_training.layers.mlp import MLP
from monolith.native_training.layers.dense import Dense


@monolith_export
@with_params
class MMoE(Layer):
  """MMoE (Multi-gate Mixture of Experts) 是 MTL (Multi-task Training) 多任务学习的一种结构。通过引入
  Multi-gate 来描述任务之间相关性以及每个任务对底层共享参数的依赖程度。
  论文可参考: https://www.kdd.org/kdd2018/accepted-papers/view/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-

  Args: 
    num_tasks (:obj:`int`): 任务训练的数量
    expert_output_dims (:obj:`List[int]`, `List[List[int]]`): 每个Expert MLP的output_dims, 可以通过两种方法来定义
                                                          1) 用`List[int]`指定, 此时, 每个Expert的结构是相同的;
                                                          2) 用`List[List[int]]`, 此时, 每个Expert MLP结构都可以不同, 内部不会处理最上层Dense
                                                          层, 所以用户必须确保每个Expert最上层的shape是相同的
    expert_activations (:obj:`List[Any]`, `str`): 每个Expert激活函数, 可以用str表示, 也可以用TF中的activation
    expert_initializers (:obj:`List[Any]`, `str`): W的初始化器, 可以是 str 也可以用户定义使用列表，默认使用 Glorot_uniform 初始化
    gate_type (:obj:`str`): 每个gate所使用的计算方式, 可以在 (softmax, topk, noise_topk) 。默认使用的是 softmax
    topk (:obj:`int`): 定义gate使用(topk, noise_topk)计算后保留最大的k个Expert, 默认是1
    num_experts (:obj:`int`): 定义 Expert 的个数, 默认会根据 Expert 的其他参数生成个数
    kernel_regularizer (:obj:`tf.regularizer`): kernel正侧化器
    use_weight_norm (:obj:`bool`): 是否开启kernel_norm, 默认为True
    use_learnable_weight_norm (:obj:`bool`): 是否让kernel_norm可训练, 默认为True
    use_bias (:obj:`bool`): 是否使用bias, 默认为True
    bias_regularizer (:obj:`tf.regularizer`): bias正侧化
    enable_batch_normalization (:obj:`bool`): 是否开启batch normalization, 如果开启, 会对输入数据, 及每个Dense Layer的输出匀做
                                              BatchNorm (最后一个Dense Layer除外).
    batch_normalization_momentum (:obj:`float`): BatchNorm中的动量因子
    batch_normalization_renorm (:obj:`bool`): 是否使用renorm, (论文可参考 https://arxiv.org/abs/1702.03275)
    batch_normalization_renorm_clipping (:obj:`bool`): renorm中的clipping, 具体请参考TF中的 `BatchNormalization`_
    batch_normalization_renorm_momentum (:obj:`float`): renorm中的momentum, 具体请参考TF中的 `BatchNormalization`_

    .. _BatchNormalization: https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization
  """

  def __init__(self,
               num_tasks: int,
               expert_output_dims: Union[List[int], List[List[int]]],
               expert_activations: Union[str, List[Any]],
               expert_initializers: Union[str, List[Any]] = 'glorot_uniform',
               gate_type: str = 'softmax',
               top_k: int = 1,
               num_experts: Optional[int] = None,
               kernel_regularizer=None,
               use_weight_norm=True,
               use_learnable_weight_norm=True,
               use_bias=True,
               bias_regularizer=None,
               enable_batch_normalization=False,
               batch_normalization_momentum=0.99,
               batch_normalization_renorm=False,
               batch_normalization_renorm_clipping=None,
               batch_normalization_renorm_momentum=0.99,
               **kwargs):
    super(MMoE, self).__init__(**kwargs)

    assert gate_type in {'softmax', 'topk', 'noise_topk'}
    self._gate_type = gate_type

    if num_experts is None:
      if all(isinstance(dims, (list, tuple)) for dims in expert_output_dims):
        self._num_experts = len(expert_output_dims)
      elif isinstance(expert_activations, (list, tuple)):
        self._num_experts = len(expert_activations)
      elif isinstance(expert_initializers, (list, tuple)):
        self._num_experts = len(expert_initializers)
      else:
        raise Exception('num_experts not set')
    else:
      self._num_experts = num_experts

    if all(isinstance(dims, (list, tuple)) for dims in expert_output_dims):
      last_dim = expert_output_dims[0][-1]
      for dims in expert_output_dims:
        assert last_dim == dims[-1]
      self._expert_output_dims = expert_output_dims
    else:
      self._expert_output_dims = [expert_output_dims] * self._num_experts

    if isinstance(expert_activations, (tuple, list)):
      assert len(expert_activations) == self._num_experts
      self._expert_activations = [
          activations.get(act) for act in expert_activations
      ]
    else:
      self._expert_activations = [
          activations.get(expert_activations) for _ in range(self._num_experts)
      ]

    if isinstance(expert_initializers, (tuple, list)):
      assert len(expert_initializers) == self._num_experts
      self._expert_initializers = [
          initializers.get(init) for init in expert_initializers
      ]
    else:
      self._expert_initializers = [
          initializers.get(expert_initializers)
          for _ in range(self._num_experts)
      ]

    self._top_k = top_k
    self._num_tasks = num_tasks
    self.use_weight_norm = use_weight_norm
    self.use_learnable_weight_norm = use_learnable_weight_norm
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.use_bias = use_bias
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.enable_batch_normalization = enable_batch_normalization
    self.batch_normalization_momentum = batch_normalization_momentum
    self.batch_normalization_renorm = batch_normalization_renorm
    self.batch_normalization_renorm_clipping = batch_normalization_renorm_clipping
    self.batch_normalization_renorm_momentum = batch_normalization_renorm_momentum

  def build(self, input_shape):
    self.experts = []
    for i in range(self._num_experts):
      mlp = MLP(name=f'expert_{i}',
                output_dims=self._expert_output_dims[i],
                activations=self._expert_activations[i],
                initializers=self._expert_initializers[i],
                kernel_regularizer=self.kernel_regularizer,
                use_weight_norm=self.use_weight_norm,
                use_learnable_weight_norm=self.use_learnable_weight_norm,
                use_bias=self.use_bias,
                bias_regularizer=self.bias_regularizer,
                enable_batch_normalization=self.enable_batch_normalization,
                batch_normalization_momentum=self.batch_normalization_momentum,
                batch_normalization_renorm=self.batch_normalization_renorm,
                batch_normalization_renorm_clipping=self.
                batch_normalization_renorm_clipping,
                batch_normalization_renorm_momentum=self.
                batch_normalization_renorm_momentum)
      self._trainable_weights.extend(mlp.trainable_weights)
      self._non_trainable_weights.extend(mlp.non_trainable_weights)
      self.experts.append(mlp)

    # input_shape: [TensorShape([bz, dim1]), TensorShape([bz, dim2])]
    if all(isinstance(shape, tf.TensorShape) for shape in input_shape):
      gate_input_dim = input_shape[-1].as_list()[-1]
    elif all(
        isinstance(shape, tf.compat.v1.Dimension) for shape in input_shape):
      gate_input_dim = input_shape[-1].value
    else:
      assert isinstance(input_shape[-1], int)
      gate_input_dim = input_shape[-1]

    gate_shape = (gate_input_dim, self._num_experts * self._num_tasks)
    self._gate_weight = self.add_weight(name="gate_weight",
                                        shape=gate_shape,
                                        dtype=tf.float32,
                                        initializer=initializers.Zeros(),
                                        trainable=True)
    if self._gate_type == 'noise_topk':
      self._gate_noise = self.add_weight(
          name="gate_noise",
          shape=gate_shape,
          dtype=tf.float32,
          initializer=initializers.GlorotNormal(),
          trainable=True)
    else:
      self._gate_noise = None

    super(MMoE, self).build(input_shape)

  def calc_gate(self, gate_input: tf.Tensor):
    # (batch, num_tasks * num_experts)
    gete_logit = tf.matmul(gate_input, self._gate_weight)

    if self._gate_type == 'noise_topk':
      noise = tf.random.normal(shape=tf.shape(gete_logit))
      noise = noise * tf.nn.softplus(tf.matmul(gate_input, self._gate_noise))
      gete_logit = gete_logit + noise

    # (batch, num_tasks, num_experts)
    gete_logit = tf.reshape(gete_logit,
                            shape=(-1, self._num_tasks, self._num_experts))
    gates = tf.nn.softmax(gete_logit, axis=2)

    if self._gate_type in {'topk', 'noise_topk'}:
      # (batch, num_tasks, top_k)
      top_gates, _ = tf.nn.top_k(gates, self._top_k)
      # (batch, num_tasks, 1)
      threshold = tf.reduce_min(top_gates, axis=2, keepdims=True)
      # (batch, num_tasks, num_experts)
      gates = tf.where(gates >= threshold, gates,
                       tf.zeros_like(gates, dtype=gates.dtype))
      gates /= tf.reduce_sum(gates, axis=2, keepdims=True)  # normalize

    # (batch, num_experts, num_tasks)
    return tf.transpose(gates, perm=[0, 2, 1])

  def call(self, inputs, **kwargs):
    if isinstance(inputs, (list, tuple)):
      assert len(inputs) == 2
      expert_input, gate_input = inputs
    else:
      inputs = tf.convert_to_tensor(inputs)
      expert_input = inputs
      gate_input = inputs

    # (batch, output_dim, num_experts)
    expert_outputs = tf.stack([expert(expert_input) for expert in self.experts],
                              axis=2)

    # (batch, num_experts, num_tasks)
    gates = self.calc_gate(gate_input)
    if self._gate_type != 'softmax':  # add layer loss
      # (num_experts, num_tasks)
      importance = tf.reduce_sum(gates, axis=0)
      # (num_tasks, )
      mean, variance = tf.nn.moments(importance, [0])
      cv_square = variance / tf.square(mean)
      self.add_loss(cv_square)

    # (batch, output_dim, num_tasks)
    mmoe_output = tf.matmul(expert_outputs, gates)

    # (batch, output_dim) * num_tasks
    final_outputs = tf.unstack(mmoe_output, axis=2)

    return final_outputs

  def get_config(self):
    config = {
        'num_tasks':
            self._num_tasks,
        'num_experts':
            self._num_experts,
        'expert_output_dims':
            self._expert_output_dims,
        "expert_activations": [
            ad_acts.serialize(act) for act in self._expert_activations
        ],
        "expert_initializers": [
            tf.initializers.serialize(init)
            for init in self._expert_initializers
        ],
        'gate_type':
            self._gate_type,
        'top_k':
            self._top_k,
        "use_weight_norm":
            self.use_weight_norm,
        "use_learnable_weight_norm":
            self.use_learnable_weight_norm,
        "enable_batch_normalization":
            self.enable_batch_normalization,
        "batch_normalization_momentum":
            self.batch_normalization_momentum,
        "use_bias":
            self.use_bias,
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'batch_normalization_renorm':
            self.batch_normalization_renorm,
        'batch_normalization_renorm_clipping':
            self.batch_normalization_renorm_clipping,
        'batch_normalization_renorm_momentum':
            self.batch_normalization_renorm_momentum
    }
    base_config = super(MMoE, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.custom_gradient
def hard_concrete_ste(x):
  out = tf.minimum(1.0, tf.maximum(x, 0.0))

  def grad(dy):
    return dy

  return out, grad


@monolith_export
@with_params
class SNR(Layer):
  """SNR (Sub-Network Routing) 是为了解决多任务学习(MTL)中任务之间相关性不大导致出现训练效果不好(Negative Transfer)而提出的
      一种灵活共享参数的方法, 论文可参考: https://ojs.aaai.org/index.php/AAAI/article/view/3788
  
  Args:
    num_out_subnet (:obj:`int`): 表示Sub_Network (Expert) 输出的个数
    out_subnet_dim (:obj:`int`): 表示Sub_Network (Expert) 输出的维度
    snr_type (:obj:`str`): 表示Sub_Networks之前的连接的结构, 可以在 ('trans', 'aver'), 默认使用 'trans'
    zeta (:obj:`float`): 表示改变Conrete分布范围的上界
    gamma (:obj:`float`): 表示改变Conrete分布范围的下界
    beta (:obj:`float`): 表示Concrete分布的温度因子, 用于决定分布的平滑程度
    use_ste: (:obj:`bool`): 表示是否使用STE (Straight-Through Estimator), 默认为False
    mode (:obj:`str`): 表示tf.esitimator.Estimator的模式, 默认是训练模式
    initializer (:obj:`str`): 表示参数W的初始化器, 配合 'trans' 结构默认使用glorot_uniform
    regularizer (:obj:`tf.regularizer`): 表示参数W的正则化
  """

  def __init__(self,
               num_out_subnet: int,
               out_subnet_dim: int,
               snr_type: str = 'trans',
               zeta: float = 1.1,
               gamma: float = -0.1,
               beta: float = 0.5,
               use_ste: bool = False,
               mode: str = tf.estimator.ModeKeys.TRAIN,
               initializer='glorot_uniform',
               regularizer=None,
               **kwargs):
    assert snr_type in {'trans', 'aver'}
    self._mode = mode
    self._num_out_subnet = num_out_subnet
    self._out_subnet_dim = out_subnet_dim
    self._num_in_subnet = None
    self._in_subnet_dim = None

    self._snr_type = snr_type
    self._weight = None
    self._log_alpha = None
    self._beta = beta
    self._zeta = zeta
    self._gamma = gamma
    self._use_ste = use_ste
    self._mode = mode

    self._initializer = initializers.get(initializer)
    self._regularizer = regularizers.get(regularizer)
    super(SNR, self).__init__(**kwargs)

  def build(self, input_shape):
    assert isinstance(input_shape, (list, tuple))
    self._num_in_subnet = len(input_shape)
    in_subnet_dim = 0
    for i, shape in enumerate(input_shape):
      last_dim = shape[-1]
      if not isinstance(last_dim, int):
        last_dim = last_dim.value

      if i == 0:
        in_subnet_dim = last_dim
      else:
        assert in_subnet_dim == last_dim
    self._in_subnet_dim = in_subnet_dim

    num_route = self._num_in_subnet * self._num_out_subnet
    block_size = self._in_subnet_dim * self._out_subnet_dim

    self._log_alpha = self.add_weight(name='log_alpha',
                                      shape=(num_route, 1),
                                      initializer=initializers.Zeros(),
                                      trainable=True)

    factor = self._beta * math.log(-self._gamma / self._zeta)
    l0_loss = tf.reduce_sum(tf.sigmoid(self._log_alpha - factor))
    self.add_loss(l0_loss)

    if self._snr_type == 'trans':
      self._weight = self.add_weight(name='weight',
                                     dtype=tf.float32,
                                     shape=(num_route, block_size),
                                     initializer=self._initializer,
                                     regularizer=self._regularizer,
                                     trainable=True)
    else:
      assert self._snr_type == 'aver' and self._in_subnet_dim == self._out_subnet_dim
      self._weight = tf.tile(tf.reshape(tf.eye(self._in_subnet_dim),
                                        shape=(1, block_size)),
                             multiples=(num_route, 1))

    super(SNR, self).build(input_shape)

  def sample(self):
    if self._mode != tf.estimator.ModeKeys.PREDICT:
      num_route = self._num_in_subnet * self._num_out_subnet

      u = tf.random.uniform(shape=(num_route, 1), minval=0, maxval=1)
      s = tf.sigmoid((tf.math.log(u) - tf.math.log(1.0 - u) + self._log_alpha) /
                     self._beta)
    else:
      s = tf.sigmoid(self._log_alpha)

    s_ = s * (self._zeta - self._gamma) + self._gamma

    if self._use_ste:
      z = hard_concrete_ste(s_)
    else:
      z = tf.minimum(1.0, tf.maximum(s_, 0.0))

    return z

  def call(self, inputs, **kwargs):
    z = self.sample()
    weight = tf.multiply(self._weight, z)

    shape1 = (self._num_in_subnet, self._num_out_subnet, self._in_subnet_dim,
              self._out_subnet_dim)
    shape2 = (self._num_in_subnet * self._in_subnet_dim,
              self._num_out_subnet * self._out_subnet_dim)
    weight = tf.reshape(
        tf.transpose(tf.reshape(weight, shape1), perm=[0, 2, 1, 3]), shape2)

    return tf.split(tf.matmul(tf.concat(inputs, axis=1), weight),
                    num_or_size_splits=self._num_out_subnet,
                    axis=1)

  def get_config(self):
    config = {
        'num_out_subnet': self._num_out_subnet,
        'out_subnet_dim': self._out_subnet_dim,
        'snr_type': self._snr_type,
        'zeta': self._zeta,
        'gamma': self._gamma,
        'beta': self._beta,
        'use_ste': self._use_ste,
        'mode': self._mode,
        'initializer': initializers.serialize(self._initializer),
        'regularizer': regularizers.serialize(self._regularizer)
    }

    base_config = super(SNR, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
