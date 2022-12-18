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

from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import BatchNormalization as BatchNorm
from tensorflow.python.keras import regularizers

from monolith.native_training.layers.dense import Dense
from monolith.native_training.utils import extend_as_list, with_params
import monolith.native_training.layers.advanced_activations as ad_acts
from monolith.native_training.monolith_export import monolith_export


@monolith_export
@with_params
class MLP(Layer):
  """多层感知器(Multilayer Perceptron), 最经典的人工神经网络, 由一系列层叠起来的Dense层组成

  Args:
    output_dims (:obj:`List[int]`): 每一层的输出神经元个数
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
  
  .. _BatchNormalization: https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization
  
  """

  def __init__(self,
               output_dims,
               activations=None,
               initializers=None,
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
    super(MLP, self).__init__(**kwargs)
    self.output_dims = output_dims
    self.use_weight_norm = use_weight_norm
    self.use_learnable_weight_norm = use_learnable_weight_norm
    self.use_bias = use_bias
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.enable_batch_normalization = enable_batch_normalization
    self.batch_normalization_momentum = batch_normalization_momentum
    self.batch_normalization_renorm = batch_normalization_renorm
    self.batch_normalization_renorm_clipping = batch_normalization_renorm_clipping
    self.batch_normalization_renorm_momentum = batch_normalization_renorm_momentum
    self._stacked_layers = []
    self._n_layers = len(self.output_dims)
    self._activations = None
    self._initializers = [
        tf.initializers.get(init)
        for init in extend_as_list(initializers, self._n_layers)
    ]

    if activations is None:
      self._activations = [ad_acts.get('relu')] * (self._n_layers - 1) + [None]
    elif isinstance(activations, (list, tuple)):
      assert len(activations) == self._n_layers
      self._activations = [ad_acts.get(act) for act in activations]
    else:
      self._activations = [
          ad_acts.get(activations) if i != self._n_layers - 1 else None
          for i in range(self._n_layers)
      ]

  def build(self, input_shape):
    if self.enable_batch_normalization:
      bn = BatchNorm(momentum=self.batch_normalization_momentum,
                     renorm=self.batch_normalization_renorm,
                     renorm_clipping=self.batch_normalization_renorm_clipping,
                     renorm_momentum=self.batch_normalization_renorm_momentum,
                     name=f"BatchNorm/in")
      self._trainable_weights.extend(bn.trainable_weights)
      self._non_trainable_weights.extend(bn.non_trainable_weights)
      self.add_loss(bn.losses)
      self._stacked_layers.append(bn)

    for i, dim in enumerate(self.output_dims):
      is_final_layer = (i == (self._n_layers - 1))
      dense = Dense(name=f"dense_{i}",
                    units=dim,
                    activation=None,
                    use_bias=self.use_bias,
                    kernel_initializer=self._initializers[i],
                    bias_initializer=tf.initializers.zeros(),
                    allow_kernel_norm=self.use_weight_norm,
                    kernel_norm_trainable=self.use_learnable_weight_norm,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer)
      self._trainable_weights.extend(dense.trainable_weights)
      self._non_trainable_weights.extend(dense.non_trainable_weights)
      self.add_loss(dense.losses)
      self._stacked_layers.append(dense)

      if not is_final_layer and self.enable_batch_normalization:
        bn = BatchNorm(momentum=self.batch_normalization_momentum,
                       renorm=self.batch_normalization_renorm,
                       renorm_clipping=self.batch_normalization_renorm_clipping,
                       renorm_momentum=self.batch_normalization_renorm_momentum,
                       name=f"BatchNorm/out")
        self._trainable_weights.extend(bn.trainable_weights)
        self._non_trainable_weights.extend(bn.non_trainable_weights)
        self.add_loss(bn.losses)
        self._stacked_layers.append(bn)

      if self._activations[i] is not None:
        self._stacked_layers.append(self._activations[i])

    super(MLP, self).build(input_shape)

  def call(self, input, **kwargs):
    input_t, output_t = input, None
    for layer in self._stacked_layers:
      output_t = layer(input_t)
      input_t = output_t
    return output_t

  def get_config(self):
    config = {
        'output_dims':
            self.output_dims,
        "activations": [ad_acts.serialize(act) for act in self._activations],
        "initializers": [
            tf.initializers.serialize(init) for init in self._initializers
        ],
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
    base_config = super(MLP, self).get_config()
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
        else:
          p[key] = value
        need_pop.append(key)

    for key in need_pop:
      config.pop(key)
    return p.instantiate()

  def get_layer(self, index: int):
    assert index < len(self._stacked_layers)
    return self._stacked_layers[index]
