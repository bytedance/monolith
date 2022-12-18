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

from monolith.native_training.layers.advanced_activations import serialize
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer, InputSpec
from tensorflow.python.keras import activations
import tensorflow.keras.initializers as initializers
from tensorflow.python.keras import regularizers
from monolith.core.base_layer import add_layer_loss
from monolith.native_training.layers.mlp import MLP
from monolith.native_training.layers.agru import AGRUCell, dynamic_rnn_with_attention
from monolith.native_training.utils import with_params
from monolith.native_training.monolith_export import monolith_export
from monolith.native_training.layers.utils import check_dim, dim_size


@monolith_export
@with_params
class DIN(Layer):
  """Deep Interest Network, 是阿里的原创, 基于兴趣序列特征聚合, 论文可参考 https://arxiv.org/pdf/1706.06978.pdf

  为了更好地描述用户, 仅用静态特征是不够的, 需要加入行为特征. 行为特征往往是一个序列, 如点击过的app, 购买过的商品等等. 
  一方面, 用户的行为是由内在兴趣(Interest)与外部条件(Target)一起促成的. 用户行为是用户兴趣的体现, 简单起见, 用户行为表示兴趣
  
  DIN的三个假设:
    - Behavior/Interest: 将用户行为序列表示为embedding序列, 这个序列同时也表示用户兴趣
    - Target Representation: 将用户物品(Target)表示为embedding, 它与行为/兴趣处于同一空间, 因为它能满足用户的兴趣, 促进行为的发生
    - Interest Match: 用户对物品发生行为, 是因为物品满足了用户的`某些`兴趣, 用Attention来表达
  
  为了简单, 以单个特征为例:
    - queries: 表示召回的物品(Target), emb_size为k, shape为(k, )
    - keys   : 表示用户序列特征(Interest), emb_size为k, 序列长长度为t, shape为(t, k)
  
  先将queries tile成shape为(t, k), 即将数据copy t次, 使queries, key同shape. 然后作如下操作
    din_all = concat([queries, keys, queries - keys, queries * keys])
  
  也就是将queries, keys及其差值, 乘值等concat起来, 然后输入MLP, 得到attention weight(即物品对兴趣的满足程度)
    attention_weight = mlp(din_all)
  
  最后, 线性组合, 实现attention (兴趣汇总), 如下:
    output = matmul(attention_weight * keys)
  
  结果的shape为(k, ), 与原始queries同shape. 

  Args:
    hidden_units (:obj:`list`): DIN中MLP layers 的hidden_units, 最后一维为1
    activation (:obj:`tf.activation`): 激活函数
    initializer (:obj:`tf.initializer`): kernel/bias初始化器
    regularizer (:obj:`tf.regularizer`): kernel正则化
    mode (:obj:`str`): 输出模式, 如果为 `sum`, 则会进行线性组合, 反回的shape与queries一样, 否则只相乘不组合, 返架的shape与keys一样
    decay (:obj:`bool`): 是否在attention weight上做decay, 默认为False
    
  """

  def __init__(self,
               hidden_units,
               activation=None,
               initializer=None,
               regularizer=None,
               mode: str = 'sum',
               decay: bool = False,
               **kwargs):
    super(DIN, self).__init__(**kwargs)
    assert hidden_units[-1] == 1
    self.input_spec = [InputSpec(ndim=2), InputSpec(ndim=3)]
    self.hidden_units = hidden_units
    self.activation = activations.get(activation)
    self.initializer = initializers.get(
        initializer) or initializers.GlorotNormal()
    self.regularizer = regularizers.get(regularizer)
    self.dense_tower = None
    self.mode = mode
    self.decay = decay

  def build(self, input_shape):
    self.dense_tower = MLP(name='compress_tower',
                           activations=self.activation,
                           output_dims=self.hidden_units,
                           initializers=self.initializer,
                           kernel_regularizer=self.regularizer)
    self._trainable_weights.extend(self.dense_tower.trainable_weights)
    self._non_trainable_weights.extend(self.dense_tower.non_trainable_weights)
    self.add_loss(self.dense_tower.losses)
    super(DIN, self).build(input_shape)

  def call(self, inputs, **kwargs):
    queries, keys = inputs
    mask = kwargs.get('mask', None)

    T, H = dim_size(keys, 1), dim_size(keys, 2)
    if self.hidden_units is None:
      self.hidden_units = [T, 1]

    # tf.tile(input, multiples, name=None), creates a new tensor by replicating `input` `multiples` times
    # The output tensor's i'th dimension has input.dims(i) * multiples[i] elements,
    # and the values of input are replicated multiples[i] times along the 'i'th dimension
    queries = tf.reshape(tf.tile(queries, [1, T]),
                         [-1, T, H])  # [B, H] -> [B, T * H] --> [B, T, H]

    # DIN
    din_all = tf.concat([queries, keys, queries - keys, queries * keys],
                        axis=-1)  # [B, T, 4 * H]
    # dense_tower on the last dim, [B, T, 4 * H] -> [B, T, 1]
    attention_weight = self.dense_tower(din_all)
    if self.decay:
      attention_weight /= (H**0.5)

    # Mask
    if mask is not None:
      mask = tf.greater_equal(mask, tf.ones_like(mask))
      key_masks = tf.expand_dims(mask, 2)  # [B, T, 1]
      attention_weight = tf.where(key_masks, attention_weight,
                                  tf.zeros_like(attention_weight))  # [B, 1, T]
      tf.compat.v1.summary.histogram(
          '{name}_attention_outputs'.format(name=self.name), attention_weight)

    if self.mode == 'sum':
      # Weighted sum
      # [B, T, 1]^T * [B, T, H] -> [B, 1, H]
      attention_out = tf.matmul(attention_weight, keys, transpose_a=True)
      outputs = tf.squeeze(attention_out, [1])  # [B, 1, H] -> [B, H]
    else:
      # [B, T, H] * [B, T, 1] -> [B, T, H]
      outputs = keys * attention_weight
    return outputs

  def get_config(self):
    config = {
        'hidden_units': self.hidden_units,
        'activation': activations.serialize(self.activation),
        'initializer': initializers.serialize(self.initializer),
        'regularizer': regularizers.serialize(self.regularizer),
    }
    base_config = super(DIN, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@monolith_export
@with_params
class DIEN(Layer):
  """DIN的升级版, Deep Interest Evolution Network, 阿里原创, 基于兴趣演进的序列特征聚合, 论文可参考 https://arxiv.org/pdf/1809.03672.pdf
  
  在推荐场景，用户无需输入搜索关键词来表达意图，这种情况下捕捉用户兴趣并考虑兴趣的动态变化将是提升模型效果的关键.
  大多该类模型将用户的行为直接看做兴趣，而用户的潜在兴趣往往很难通过行为来完全表示, 需要挖掘行为背后的用户真实兴趣，并考虑用户兴趣的动态变化
  
  DIEN的假设:
    - Behavior Layer: 也就是将用户行为序列表示为embedding序列, embedding表达的意义是行为本身, 不再直接代表兴趣, 这与DIN不同
    - Interest Extractor Layer: 用GRU从用户行为中提取兴趣(Interest), 兴趣是随时间演变的, DIN没有考虑这一点
    - Interest Evolving Layer: 随着外部环境(Target attention)和内部认知(Interest)的变化，用户兴趣也不断变化, 最终兴起促使行为发生
        - 物品表示与DIN一样, 它与兴趣处于同一空间, 因为它能满足用户的兴趣, 促进行为的发生
        - 物品与兴趣的关系建模与DIN不一样, DIN是静态地看物品能否满足用户兴趣; DIEN中, 用户兴趣是演进的(Evolving), 物品会诱导/挖掘用户兴趣
          在网络结构上表示为AGRU, 即attention + GRU

  Args:
    num_units (:obj:`int`): GRU隐含层的大小
    att_type (:obj:`str`): attention的类型, 目前支持AGRU/AUGRU两种
    activation (:obj:`tf.activation`): 激活函数
    initializer (:obj:`tf.initializer`): kernel/bias初始化器
    regularizer (:obj:`tf.regularizer`): kernel正则化

  """

  def __init__(self,
               num_units,
               att_type='AGRU',
               activation=tf.keras.activations.relu,
               initializer=tf.initializers.HeUniform,
               regularizer=None,
               **kwargs):
    super(DIEN, self).__init__(**kwargs)
    self.num_units, self.att_type = num_units, att_type
    self.activation = tf.keras.activations.get(activation)
    self.initializer = initializers.get(
        initializer) or initializers.GlorotNormal()
    self.regularizer = regularizers.get(regularizer)

  def build(self, input_shape):
    self.gru_cell = tf.keras.layers.GRUCell(
        name='gru_cell',
        units=self.num_units,
        activation=self.activation,
        kernel_initializer=self.initializer,
        bias_initializer=tf.initializers.Zeros(),
        kernel_regularizer=self.regularizer)
    self._trainable_weights.extend(self.gru_cell.trainable_weights)
    self._non_trainable_weights.extend(self.gru_cell.non_trainable_weights)
    self.add_loss(self.gru_cell.losses)

    self.augru_cell = AGRUCell(name='augru_cell',
                               units=self.num_units,
                               activation=self.activation,
                               att_type='AGRU',
                               initializer=self.initializer,
                               regularizer=self.regularizer)
    self._trainable_weights.extend(self.augru_cell.trainable_weights)
    self._non_trainable_weights.extend(self.augru_cell.non_trainable_weights)
    self.add_loss(self.augru_cell.losses)

    self.weight = self.add_weight(name='attention_weight',
                                  dtype=tf.float32,
                                  shape=(self.num_units, self.num_units),
                                  initializer=self.initializer,
                                  regularizer=self.regularizer)
    super(DIEN, self).build(input_shape)

  def _attention(self, queries, keys):
    emb_size = dim_size(keys, 2)
    query_weight = tf.reshape(tf.matmul(queries, self.weight, transpose_b=True),
                              [-1, emb_size, 1])
    logit = tf.squeeze(tf.matmul(keys, query_weight), [2])

    return tf.nn.softmax(logit)

  def call(self, inputs, **kwargs):
    if isinstance(inputs, (list, tuple)):
      if len(inputs) == 3:
        queries, keys, mask = inputs[:]
      elif len(inputs) == 2:
        queries, keys = inputs[:]
      else:
        queries = inputs[0]
        keys = kwargs['keys']
    else:
      queries = inputs
      keys = kwargs['keys']

    # interest extractor layer to capture temporal interests
    outputs, _ = tf.compat.v1.nn.dynamic_rnn(self.gru_cell,
                                             keys,
                                             dtype=tf.float32)  # [B, T, H]

    # interest evolving layer to capture interest evolving process that is relative to the target item
    attn_scores = self._attention(queries, outputs)  # [B, T]
    _, final_state = dynamic_rnn_with_attention(self.augru_cell, outputs,
                                                attn_scores)  # [B, T, H]

    return final_state

  def get_config(self):
    config = {
        'num_units': self.num_units,
        'att_type': self.att_type,
        'activation': activations.serialize(self.activation),
        'initializer': initializers.serialize(self.initializer),
        'regularizer': regularizers.serialize(self.regularizer),
    }
    base_config = super(DIEN, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@monolith_export
@with_params
class DMR_U2I(Layer):
  """Deep Match to Rank, DMR, 深度配匹排序, 与RNN不同, 主要考虑序列顺序 
  
  与DIN一样, 还DMR还是用attention的方式来聚合序列特征. 不同的是MR考虑了序列顺序, 即增加了位置embedding来处理用户序列的选后顺序
  由于原始论文中最后的输出是点积, 梯度回传时只有一个值, 会导致训练不充分, 所以引入辅助loss, 但是辅助loss要用到负采样, 系统实现上比较
  麻烦, 这里用element wise乘积代替点积, 去除辅助loss. 论文可参考 https://ojs.aaai.org/index.php/AAAI/article/view/5346/5202
  
  Args:
    cmp_dim (:obj:`int`): 压缩维度
    activation (:obj:`tf.activation`): 激活函数
    initializer (:obj:`tf.initializer`): kernel/bias初始化器
    regularizer (:obj:`tf.regularizer`): kernel正则化
  
  """

  def __init__(self,
               cmp_dim: int,
               activation="PReLU",
               initializer="glorot_uniform",
               regularizer=None,
               **kwargs):
    super(DMR_U2I, self).__init__(**kwargs)
    self.cmp_dim = cmp_dim
    self.activation = activations.get(activation)
    self.initializer = initializers.get(
        initializer) or initializers.GlorotNormal()
    self.regularizer = regularizers.get(regularizer)

  def build(self, input_shape):
    item_sh, user_seq_sh = input_shape
    (bs1, seq_length, ue_size) = tuple(map(check_dim, user_seq_sh))
    (bs2, ie_size) = tuple(map(check_dim, item_sh))
    assert bs1 == bs2

    # position embedding
    self.pos_emb = self.add_weight(name="pos_emb",
                                   shape=(seq_length, self.cmp_dim),
                                   initializer=self.initializer,
                                   regularizer=self.regularizer)

    self.emb_weight = self.add_weight(name="emb_weight",
                                      shape=(ue_size, self.cmp_dim),
                                      initializer=self.initializer,
                                      regularizer=self.regularizer)

    self.z_weight = self.add_weight(name="z_weight",
                                    shape=(self.cmp_dim, 1),
                                    initializer=initializers.Ones())

    self.bias = self.add_weight(name="bias",
                                shape=(self.cmp_dim,),
                                initializer=initializers.Zeros())

    self.linear = Dense(name="dense",
                        units=ie_size,
                        activation=self.activation,
                        kernel_initializer=self.initializer,
                        kernel_regularizer=self.regularizer,
                        use_bias=True)
    self._trainable_weights.extend(self.linear.trainable_weights)
    self._non_trainable_weights.extend(self.linear.non_trainable_weights)

  def call(self, inputs, **kwargs):
    items, user_seq = inputs

    # 1) calculate compressed represention
    emb_cmp = tf.matmul(user_seq, self.emb_weight)  # (bs, seq_length, cmp_dim)
    comped = self.pos_emb + emb_cmp + self.bias  # (bs, seq_length, cmp_dim)

    # 2) prepare attention weight
    # (bs, seq_length, cmp_dim) * (cmp_dim, 1)  -> (bs, seq_length, 1)
    alpha = tf.matmul(comped, self.z_weight)  # (bs, seq_length, 1)
    alpha = tf.nn.softmax(alpha, axis=1)  # (bs, seq_length, 1)

    # 3) execute attention
    user_seq_trans = tf.transpose(user_seq,
                                  perm=(0, 2, 1))  # (bs, ue_size, seq_length)
    # (bs, ue_size, seq_length) * (bs, seq_length, 1) -> (bs, ue_size, 1) -> (bs, ue_size)
    user_seq_merged = tf.squeeze(tf.matmul(user_seq_trans, alpha),
                                 axis=-1)  # (bs, ue_size)

    # 4) linear transform
    user_seq_merged = self.linear(user_seq_merged)  # (bs, ie_size)

    return user_seq_merged * items

  def get_config(self):
    config = {
        'cmp_dim': self.cmp_dim,
        'activation': activations.serialize(self.activation),
        'initializer': initializers.serialize(self.initializer),
        'regularizer': regularizers.serialize(self.regularizer),
    }
    base_config = super(DMR_U2I, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
