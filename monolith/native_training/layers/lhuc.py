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

# -*- encoding=utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Layer
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization as BatchNorm

from monolith.native_training.layers.mlp import MLP
from monolith.native_training.layers.dense import Dense
from monolith.native_training.utils import extend_as_list, with_params
import monolith.native_training.layers.advanced_activations as ad_acts
from monolith.native_training.monolith_export import monolith_export


@monolith_export
@with_params
class LHUCTower(Layer):
  """LHUCTower, 对MLP的改进, 在MLP的基础上增加了一系列的 LHUC MLP 当作Gate. 论文可参考 https://arxiv.org/abs/1601.02828

  Args:
    output_dims (:obj:`List[int]`): 主Tower的每一层的输出神经元个数
    lhuc_output_dims (:obj:`List[int]`, `List[List[int]]`): 每个LHUC MLP的output_dims, 其长度与output_dims相同, 可以有两种方式指定, 
                                                            1) 用`List[int]`指定, 此时, 除最上层外, 所有LHUC MLP结构相同, 最上层的Dense会在内部自动加上
                                                            并处理shape; 2) 用`List[List[int]]`, 此时, 每个LHUC MLP结构都可以不同, 内部不会处理最上层Dense
                                                            层, 所以用户必须确保shape是正确的. lhuc_output_dims默认为None, 等价于[]. 
    activations (:obj:`List[tf.activation]`, `List[str]`, `tf.activation`, `str`): 激活函数, 可以用str表示, 也可以用TF中的activation
    initializers (:obj:`List[tf.initializer]`): kernel, 也就是W的初始化器, 是一个列表
    kernel_regularizer (:obj:`tf.regularizer`): kernel正侧化器
    use_weight_norm (:obj:`bool`): 是否开启kernel_norm 
    use_learnable_weight_norm (:obj:`bool`): 是否让kernel_norm可训练
    use_bias (:obj:`bool`): 是否使用bias, 默认为True
    bias_regularizer (:obj:`tf.regularizer`): bias正侧化
    enable_batch_normalization (:obj:`bool`): 是否开启batch normalization, 如果开启, 会对输入数据, 及每个Dense Layer的输出匀做
                                              BatchNorm (最后一个Dense Layer除外).
    batch_normalization_momentum (:obj:`float`): BatchNorm中的动量因子
    batch_normalization_renorm (:obj:`bool`): 是否使用renorm, (论文可参考 https://arxiv.org/abs/1702.03275)
    batch_normalization_renorm_clipping (:obj:`bool`): renorm中的clipping, 具体请参考TF中的 `BatchNormalization`_
    batch_normalization_renorm_momentum (:obj:`float`): renorm中的momentum, 具体请参考TF中的 `BatchNormalization`_
  
  此外, 对于 weight_norm, batch_normalization 相关参数, 主MLP与LHUC MLP共用, 如果要为LHUC MLP指定不同的参数, 可用 "lhuc_{params_name}" 来指定

  .. _BatchNormalization: https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization
  
  """

  def __init__(self,
               output_dims,
               lhuc_output_dims=None,
               activations='relu',
               initializers=None,
               use_bias=True,
               use_weight_norm=True,
               use_learnable_weight_norm=True,
               kernel_regularizer=None,
               bias_regularizer=None,
               enable_batch_normalization=False,
               batch_normalization_momentum=0.99,
               batch_normalization_renorm=False,
               batch_normalization_renorm_clipping=None,
               batch_normalization_renorm_momentum=0.99,
               **kwargs):
    self._lhuc_kwargs = {
        k: v for k, v in kwargs.items() if k.startswith('lhuc_')
    }
    for lhuc_key in self._lhuc_kwargs:
      del kwargs[lhuc_key]

    super(LHUCTower, self).__init__(**kwargs)
    self.output_dims = output_dims
    self.n_layers = len(output_dims)

    if activations is None:
      self.activations = [ad_acts.get('relu')] * (self.n_layers - 1) + [None]
    elif isinstance(activations, (list, tuple)):
      assert len(activations) == self.n_layers
      self.activations = [ad_acts.get(act) for act in activations]
    else:
      self.activations = [
          ad_acts.get(activations) if i != self.n_layers - 1 else None
          for i in range(self.n_layers)
      ]

    self.initializers = extend_as_list(initializers, self.n_layers)

    self.use_bias = use_bias
    self.use_weight_norm = use_weight_norm
    self.use_learnable_weight_norm = use_learnable_weight_norm
    self.kernel_regularizer = kernel_regularizer
    self.bias_regularizer = bias_regularizer

    self.enable_batch_normalization = enable_batch_normalization
    self.batch_normalization_momentum = batch_normalization_momentum
    self.batch_normalization_renorm = batch_normalization_renorm
    self.batch_normalization_renorm_clipping = batch_normalization_renorm_clipping
    self.batch_normalization_renorm_momentum = batch_normalization_renorm_momentum

    if lhuc_output_dims:
      assert isinstance(lhuc_output_dims, (list, tuple))
      if all(isinstance(dims, (list, tuple)) for dims in lhuc_output_dims):
        for i, dims in enumerate(lhuc_output_dims):
          assert dims[-1] == output_dims[
              i], "the last dim of lhuc must be identity with dense output"
        self.lhuc_output_dims = lhuc_output_dims
      elif all(isinstance(dims, int) for dims in lhuc_output_dims):
        self.lhuc_output_dims = []
        for dim in self.output_dims:
          self.lhuc_output_dims.append(lhuc_output_dims + [dim])
      else:
        raise Exception("lhuc_output_dims is error")
    else:
      self.lhuc_output_dims = [[i] for i in self.output_dims]
    self.lhuc_activations = [[
        ad_acts.get('relu') if i != len(dims) - 1 else ad_acts.get('sigmoid2')
        for i in range(len(dims))
    ]
                             for dims in self.lhuc_output_dims]

    self.layers = []
    self.lhuc_layers = []
    self.extra_layers = []

  def lhuc_params(self, name):
    params = self._lhuc_kwargs.get(f"lhuc_{name}")
    if params is None and hasattr(self, name):
      params = getattr(self, name)
    return params

  def build(self, input_shape):
    if self.enable_batch_normalization:
      bn_layer = BatchNorm(
          name='batch_norm',
          momentum=self.batch_normalization_momentum,
          renorm=self.batch_normalization_renorm,
          renorm_clipping=self.batch_normalization_renorm_clipping,
          renorm_momentum=self.batch_normalization_renorm_momentum)
      self._trainable_weights.extend(bn_layer.trainable_weights)
      self._non_trainable_weights.extend(bn_layer.non_trainable_weights)
      self.extra_layers.append(bn_layer)

    for i, dim in enumerate(self.output_dims):
      layer_name = f'layer_{i + 1}'
      sequential = Sequential(name=layer_name)  # one block in dense tower
      dense = Dense(name=f'{layer_name}/dense',
                    units=dim,
                    activation=None,
                    use_bias=self.use_bias,
                    kernel_initializer=self.initializers[i],
                    bias_initializer=tf.initializers.zeros(),
                    allow_kernel_norm=self.use_weight_norm,
                    kernel_norm_trainable=self.use_learnable_weight_norm,
                    kernel_regularizer=regularizers.get(
                        self.kernel_regularizer),
                    bias_regularizer=regularizers.get(self.bias_regularizer))
      self._trainable_weights.extend(dense.trainable_weights)
      self._non_trainable_weights.extend(dense.non_trainable_weights)
      sequential.add(dense)

      if i != (self.n_layers - 1) and self.enable_batch_normalization:
        bn_layer = BatchNorm(
            name=f'{layer_name}/batch_norm',
            momentum=self.batch_normalization_momentum,
            renorm=self.batch_normalization_renorm,
            renorm_clipping=self.batch_normalization_renorm_clipping,
            renorm_momentum=self.batch_normalization_renorm_momentum)
        self._trainable_weights.extend(bn_layer.trainable_weights)
        self._non_trainable_weights.extend(bn_layer.non_trainable_weights)
        sequential.add(bn_layer)

      if self.activations[i] is not None:
        sequential.add(self.activations[i])

      self.layers.append(sequential)

      # for lhuc tower
      mlp = MLP(name=f'{layer_name}/lhuc',
                output_dims=self.lhuc_output_dims[i],
                activations=self.lhuc_activations[i],
                initializers=self.initializers[i],
                kernel_regularizer=self.lhuc_params('kernel_regularizer'),
                use_weight_norm=self.lhuc_params('use_weight_norm'),
                use_learnable_weight_norm=self.lhuc_params(
                    'use_learnable_weight_norm'),
                use_bias=self.lhuc_params('use_bias'),
                bias_regularizer=self.lhuc_params('bias_regularizer'),
                enable_batch_normalization=self.lhuc_params(
                    'enable_batch_normalization'),
                batch_normalization_momentum=self.lhuc_params(
                    'batch_normalization_momentum'),
                batch_normalization_renorm=self.lhuc_params(
                    'batch_normalization_renorm'),
                batch_normalization_renorm_clipping=self.lhuc_params(
                    'batch_normalization_renorm_clipping'),
                batch_normalization_renorm_momentum=self.lhuc_params(
                    'batch_normalization_renorm_momentum'))
      self._trainable_weights.extend(mlp.trainable_weights)
      self._non_trainable_weights.extend(mlp.non_trainable_weights)
      self.lhuc_layers.append(mlp)

      super(LHUCTower, self).build(input_shape)

  def call(self, inputs, **kwargs):
    if isinstance(inputs, (list, tuple)):
      assert len(inputs) == 2
      dense_input, lhuc_input = inputs
    else:
      inputs = tf.convert_to_tensor(inputs)
      dense_input = inputs
      lhuc_input = inputs

    input_t = dense_input
    for layer in self.extra_layers:
      input_t = layer(input_t)

    for layer, lhuc_layer in zip(self.layers, self.lhuc_layers):
      output_t = layer(input_t) * lhuc_layer(lhuc_input)
      input_t = output_t

    return output_t

  def get_config(self):
    config = {
        "output_dims":
            self.output_dims,
        "lhuc_output_dims":
            self.lhuc_output_dims,
        "activations": [ad_acts.serialize(act) for act in self.activations],
        "initializers": [
            tf.initializers.serialize(init) for init in self.initializers
        ],
        "use_bias":
            self.use_bias,
        "use_weight_norm":
            self.use_weight_norm,
        "use_learnable_weight_norm":
            self.use_learnable_weight_norm,
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        "enable_batch_normalization":
            self.enable_batch_normalization,
        "batch_normalization_momentum":
            self.batch_normalization_momentum,
        'batch_normalization_renorm':
            self.batch_normalization_renorm,
        'batch_normalization_renorm_clipping':
            self.batch_normalization_renorm_clipping,
        'batch_normalization_renorm_momentum':
            self.batch_normalization_renorm_momentum
    }

    config.update(self._lhuc_kwargs)
    base_config = super(LHUCTower, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    p = cls.params().copy()
    need_pop = []
    for key, value in config.items():
      if key in p:
        if key == 'initializers':
          p[key] = [tf.initializers.deserialize(init) for init in config[key]]
        elif key == 'activations':
          p[key] = [ad_acts.deserialize(act) for act in config[key]]
        elif key == 'kernel_regularizer':
          regularizers.deserialize(value),
        elif key == 'bias_regularizer':
          regularizers.deserialize(value),
        else:
          p[key] = value
        need_pop.append(key)

    for key in need_pop:
      config.pop(key)
    return p.instantiate()
