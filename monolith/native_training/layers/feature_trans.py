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

import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec
import tensorflow.keras.initializers as initializers
from tensorflow.python.keras import regularizers
from monolith.core.base_layer import add_layer_loss
from monolith.native_training.layers.mlp import MLP
from monolith.native_training.layers.utils import merge_tensor_list
from monolith.native_training.utils import with_params
from monolith.native_training.monolith_export import monolith_export
from monolith.native_training.layers.utils import check_dim, dim_size


@monolith_export
@with_params
class AutoInt(Layer):
  r"""Auto-Interaction的缩写, 基于Self-attention的特征变换. 论文可参考 https://arxiv.org/pdf/1810.11921.pdf

  一个样本有n个特征, 每个特征用一个k维的embedding表示, 则样本可以表示为(n, k)的矩阵. 所谓attention, 本质上是一种线性组合, 关键是确定组合系数
  
  AutoInt中确定组合系数的方式为: 
  
  .. math::
  
    coeff_{n, n} = softmax( X_{n, k} * X_{n, k}^T )
  
  即先计算自相关, 确定特征与其它特征的`相似性`, 然后用softmax的方式归一化, 得到组合系数. 最后是组性组合, 计算attention: 
  
  .. math::
  
    O_{n, k} = coeff_{n, n} * X_{n, k}
  
  在AutoInt中, 上述过程可以迭代进行多次, 一次为一个layer

  Args:
    layer_num (:obj:`int`): auto int layer的层数, 一层为一个完整的auto int
    out_type (:obj:`str`): 输出类型, 可以为stack, concat, None
    keep_list (:obj:`bool`): 输出是否保持list
    
  """

  def __init__(self,
               layer_num=1,
               out_type='concat',
               keep_list: bool = False,
               **kwargs):
    super(AutoInt, self).__init__(**kwargs)
    self.layer_num = layer_num
    self.out_type = out_type
    self.keep_list = keep_list

  def call(self, embeds, **kwargs):
    assert len(embeds.shape) == 3

    autoint_input = embeds
    for i in range(self.layer_num):
      layer_name = '{name}_{idx}'.format(name=self.name, idx=i)
      with tf.name_scope(layer_name):
        # [batch_size, num_feat, emb_dim] -> [batch_size, num_feat, num_feat]
        attn = tf.nn.softmax(tf.matmul(autoint_input,
                                       autoint_input,
                                       transpose_b=True),
                             axis=-1)
        autoint_input = tf.matmul(attn,
                                  autoint_input)  # [batch, num_feats, emb_dim]

    return merge_tensor_list(autoint_input,
                             merge_type=self.out_type,
                             keep_list=self.keep_list)

  def get_config(self):
    config = {
        'layer_num': self.layer_num,
        'out_type': self.out_type,
        'keep_list': self.keep_list
    }
    base_config = super(AutoInt, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@with_params
class iRazor(Layer):
  """特征选择和Embedding维度搜索

  一个样本有n个特征, 每个特征用一个k维的embedding表示. 可以为每一个Embedding分配一个先择概率(一个0~1之间的数), 训练出完成后, 如果概率较大, 则保留,
  否则去除, 从而实现Embedding维度搜索. 也可以给特征分配一个移除概率, 这个概率与Embedding分配的概率可以归一化, 如果概率越大, 则将特征移除. 当训练完成
  后, 可以用后剪枝算法CPT(cumulative probability threshold, 累积概率阈值)来对网络裁剪, 从而达到特征选择和Embedding维度搜索的目的
  
  .. note::
  
    从另一个角度看, 不同的特征, 重要程度不一样, 同一个特征Embedding, 不同的维度重要程度也不一样. 常用内积, 在Euclid空间中计算内积就是点乘, 
    因此每个维度的重要性一样. 所以可以引入一个`度量空间`, 在这个空间算内积. 为了简单, 度量矩阵用对角阵(半正定), 此时, 直观理解就是每个embedding维度权重不
    一样, 而且权重匀为正数, 是可以学习的. 前面的分析是假设`度量空间`存在, 可以为`度量空间`情况也分配权重, 而且这个权重与Embedding权重是归一化的, 
    从而实现不同特征重要性不一样. 此时, iRazor的目的是做特征变换
  
  给定一个 nas_space, 假设emb_size=8, 则nas_space=[0, 1, 3, 5, 8], 是对embedding的一个划分:{}, {0}, {1, 2}, {3, 4}, {5, 6, 7}, 共5段, 每段出现的概率为p_i
  
  .. code-block:: text
  
    rigid_masks = [
      [0, 0, 0, 0, 0, 0, 0, 0],   -> p_0, 表示`度量空间`不存在的概率
      [1, 0, 0, 0, 0, 0, 0, 0],   -> p_1, 表示0号位置的概率/重要性
      [0, 1, 1, 0, 0, 0, 0, 0],   -> p_2, 表示1-2号位置的概率/重要性
      [0, 0, 0, 1, 1, 0, 0, 0],   -> p_3, 表示3-4号位置的概率/重要性
      [0, 0, 0, 0, 0, 1, 1, 1]    -> p_4, 表示5-8号位置的概率/重要性
    ]
    P = (p_0, p_1, p_2, p_3, p_4), 且有 p_0 + ... + p_4 = 1
    soft_masks = P * rigid_masks = (p_1, p_2, p_2, p_3, p_3, p_4, p_4, p_4)
  
  从上面可以看出, nas_space是对测度空间的限制, 强制某几个维度(分组)有相同的权重. 如果 nas_space = [0,1,2,3,4,5,6,7,8], 可以去除这种强制. 
  rigid_masks中第一行全为0, 表示表示`度量空间`不存在. 可以加一个辅助loss, 强制`度量空间`不存在, 因为可以减少参数/省内存/评估特征重要性
  
  .. code-block:: text
  
    loss = feature_weight * sum(soft_masks)
  
  Args:
    nas_space (:obj:`list`): 用于定义embedding特征分组, 第一个元素是0, 最后一个元素是emb_size, 元素是有序的. 
                            0表示`度量空间`不存在, nas_space[i-1]:nas_space[i] 表示一个分组, 位于同一组内的元素有相同的权重
    t (:obj:`float`): softmax平滑因子
    initializer (:obj:`tf.initializer`): kernel/bias初始化器
    regularizer (:obj:`tf.regularizer`): kernel正则化
    feature_weight (:obj:`tf.Tensor`): 特征权重, 用于计算辅助loss
    out_type (:obj:`str`): 输出类型, 可以为stack, concat, None
    keep_list (:obj:`bool`): 输出是否保持list

  """

  def __init__(self,
               nas_space,
               t=0.05,
               initializer=None,
               regularizer=None,
               feature_weight=None,
               out_type='concat',
               keep_list=False,
               **kwargs):
    super(iRazor, self).__init__(**kwargs)
    self.out_type = out_type
    self.keep_list = keep_list
    self.nas_space = nas_space
    self.t = t

    self.nas_logits = None
    self.emb_size = max(self.nas_space)
    self.nas_len = len(self.nas_space)
    self.initializer = initializers.get(initializer)
    self.regularizer = regularizers.get(regularizer)

    if feature_weight is not None:
      if isinstance(feature_weight, (tf.Tensor, tf.Variable)):
        self.feature_weight = tf.reshape(feature_weight, shape=(1, -1))
      else:
        self.feature_weight = tf.constant(feature_weight,
                                          shape=(1, -1),
                                          dtype=tf.float32)
    else:
      self.feature_weight = feature_weight

  @property
  def rigid_masks(self):
    masks = np.zeros(shape=(self.nas_len, self.emb_size), dtype=np.float32)
    for i, j in enumerate(self.nas_space):
      if i > 0:
        masks[i, self.nas_space[i - 1]:j] = 1.0
    return tf.constant(masks, name="masks", dtype=tf.float32)

  def build(self, input_shape):
    # input_shape: [bath_size, num_feat, emb_dim]
    shape = (check_dim(input_shape[1]), self.nas_len)
    self.nas_logits = self.add_weight(name="nas_weight",
                                      shape=shape,
                                      dtype=tf.float32,
                                      initializer=self.initializer,
                                      regularizer=self.regularizer)
    super(iRazor, self).build(input_shape)

  def call(self, embeds, **kwargs):
    assert check_dim(embeds.shape[-1]) == max(self.nas_space)

    nas_weight = tf.nn.softmax(self.nas_logits / self.t,
                               axis=1,
                               name="nas_concat_choice_probs")
    tf.compat.v1.summary.histogram(name='nas_weight', values=nas_weight)

    # create soft mask for each embedding dim with nas
    soft_masks = tf.matmul(nas_weight, self.rigid_masks, name="choice_matrix")

    if self.feature_weight is not None:
      nas_loss = tf.matmul(self.feature_weight,
                           tf.reduce_sum(soft_masks, axis=1, keepdims=True))
      add_layer_loss(self.name, tf.reduce_sum(nas_loss))

    # re-weight embeds
    out_embeds = embeds * soft_masks

    return merge_tensor_list(out_embeds,
                             merge_type=self.out_type,
                             keep_list=self.keep_list)

  def get_config(self):
    config = {
        'nas_space': self.nas_space,
        't': self.t,
        'initializer': initializers.serialize(self.initializer),
        'regularizer': regularizers.serialize(self.regularizer),
        'feature_weight': self.feature_weight,
        'out_type': self.out_type,
        'keep_list': self.keep_list,
    }
    base_config = super(iRazor, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@monolith_export
@with_params
class SeNet(Layer):
  """SeNet最早用于图像中, 这里是借用其概念, 不同特征具有不同重要性. 论文可参考 https://arxiv.org/pdf/1709.01507.pdf

  一个样本有n个特征, 每个特征用一个k维的embedding表示. 但是并不是每个特征都一样重要, 所以想给每个特征一个权重, 以调整其重要性.
  权重计算是用一个MLP完成的, 一般有三层input - cmp_layer - output. 其中input/output是同shape的, input是通过 reduce_mean输入矩阵(n, k)的最后一维得到.
  最后用 weight(n) * (n, k) 为特征加权

  Args:
    num_feature (:obj:`int`): 输入特征数
    cmp_dim (:obj:`int`): 压缩维的维度
    initializer (:obj:`tf.initializer`): kernel/bias初始化器
    kernel_regularizer (:obj:`tf.regularizer`): kernel正则化
    bias_regularizer (:obj:`tf.regularizer`): bias正则化
    on_gpu: 计算是否发生在GPU上, 如果是, 则用GPU优化版本
    out_type (:obj:`str`): 输出类型, 可以为stack, concat, None
    keep_list (:obj:`bool`): 输出是否保持list
    
  """

  def __init__(self,
               num_feature,
               cmp_dim,
               initializer=None,
               regularizer=None,
               on_gpu=False,
               out_type='concat',
               keep_list=False,
               **kwargs):
    super(SeNet, self).__init__(**kwargs)
    self.num_feat = num_feature
    self.cmp_dim = cmp_dim
    self.initializer = initializers.get(initializer)
    self.regularizer = regularizers.get(regularizer)
    self.on_gpu = on_gpu
    self.out_type = out_type
    self.keep_list = keep_list

  def build(self, input_shape):
    if self.cmp_dim is None:
      self.cmp_tower = lambda x: x
    else:
      self.cmp_tower = MLP(name='cmp_tower',
                           output_dims=[self.cmp_dim, self.num_feat],
                           activations=['relu', 'sigmoid'],
                           initializers=self.initializer,
                           kernel_regularizer=self.regularizer)
      self._trainable_weights.extend(self.cmp_tower.trainable_weights)
      self._non_trainable_weights.extend(self.cmp_tower.non_trainable_weights)
      self.add_loss(self.cmp_tower.losses)
    super(SeNet, self).build(input_shape)

  def call(self, inputs, **kwargs):
    senet_input_concat, emb_dim = None, None
    if isinstance(inputs, (tf.Tensor, tf.Variable)):
      # [batch_size, slots_num, emb_dim]
      num_feat, emb_dim = dim_size(inputs, 1), dim_size(inputs, 2)
      senet_input_concat = tf.reshape(inputs, [-1, num_feat, emb_dim])
      sequeeze_embedding = tf.reduce_mean(senet_input_concat,
                                          axis=2)  # [batch, slots_num]
    else:  # isinstance(inputs, (list, tuple))
      num_feat = len(inputs)
      if self.on_gpu:
        slots_lens = [dim_size(embed, 1) for embed in inputs]
        ids = tf.constant(
            np.concatenate([[i] * length for i, length in enumerate(slots_lens)
                           ]))
        lens = tf.constant([1.0 / slot_len for slot_len in slots_lens])

        concat_trans = tf.transpose(tf.concat(inputs, axis=1))
        sequeeze_embedding = tf.compat.v1.segment_sum(concat_trans, ids)
        sequeeze_embedding = tf.transpose(sequeeze_embedding)
        sequeeze_embedding = tf.reshape(sequeeze_embedding,
                                        shape=(-1, num_feat))
        sequeeze_embedding = tf.multiply(sequeeze_embedding, lens)
      else:
        sequeeze_embedding = tf.concat(
            [tf.reduce_mean(embed, axis=1, keepdims=True) for embed in inputs],
            axis=1,
            name='concat')

    weight_out = self.cmp_tower(sequeeze_embedding)
    if isinstance(inputs, (tf.Tensor, tf.Variable)):
      # [batch, num_feat] -> # [batch, num_feat, 1]
      weight_out = tf.reshape(weight_out, [-1, num_feat, 1])
      senet_weighted = tf.multiply(weight_out, senet_input_concat)
    else:
      weight_out = tf.split(weight_out, num_feat, axis=1)
      senet_weighted = [
          tf.multiply(embed, weight)
          for embed, weight in zip(inputs, weight_out)
      ]

    return merge_tensor_list(senet_weighted,
                             merge_type=self.out_type,
                             keep_list=self.keep_list,
                             num_feature=num_feat)

  def get_config(self):
    config = {
        'num_feature': self.num_feat,
        'cmp_dim': self.cmp_dim,
        'initializer': initializers.serialize(self.initializer),
        'regularizer': regularizers.serialize(self.regularizer),
        'on_gpu': self.on_gpu,
        'out_type': self.out_type,
        'keep_list': self.keep_list
    }
    base_config = super(SeNet, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
