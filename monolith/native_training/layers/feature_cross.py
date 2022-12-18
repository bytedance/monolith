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

from typing import List
from absl import logging

import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv1D
from tensorflow.python.keras import activations
import tensorflow.keras.initializers as initializers
from tensorflow.python.keras import regularizers
from monolith.native_training.layers.mlp import MLP
from monolith.native_training.utils import with_params, get_uname
from monolith.native_training.layers.utils import merge_tensor_list, DCNType
from monolith.native_training.monolith_export import monolith_export
from monolith.native_training.layers.layer_ops import ffm
from tensorflow.python.ops import variables as variable_ops
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras import backend as K

from monolith.native_training.layers.utils import check_dim, dim_size


@monolith_export
@with_params
class GroupInt(Layer):
  """Group Interaction的缩写, 一种简单的特征交叉方式, 同时支持attention. 论文可参考 https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf

  特征交叉可以在多个层面做, 一种方法是在特征工程中做, 即在特征工程阶段直接生成一个新特征, 这个特征是由多个原始征特拼接起来的, 然后再做Embedding. 
  这样做的好处是记忆性较好, 但由于稀疏性, 有时训练不够充分, 也存在过拟合的风险. 另一种是在模型层面做, 代表算法为FM, DeepFM等
  
  在模型中做二阶特征交叉存在如下问题:
    - 输出维度高: FM用点积表示特征交叉, 如果输入有n个特征, 输出有 n(n-1)/2 维, 当特征较多时, 给训练/推理带来很大的负担
    - 重复交叉: 特征交叉可以在两个地方做, 现实中往往同时做. FM等算法并不区分参与交叉的是原始特征还是交叉特征. 所以存在重复交叉. 不过, 也有人认为
      重复交叉会生成更高阶的特征, 不是重复
  
  为了克服FM等算法的不足, 可以使用GroupInt. 它先将特征分组(Group), 哪些特征属于一个组由算法开发人员确定. 然后用sumpooling来将特征聚合
  得到group embedding. 最后用group embedding做两两交叉输出
  
  GroupInt输出有如下几种形式:
    - 交叉用dot, 直接输出. 此时输出的大小远小于原始FM, 而且, 人工确定group, 减少了重复交叉
    - 交叉用multiply, 输出有两种选择:
        - 直接concat输出
        - 用attention, 将所以结果线性组合后输出(与AFM一样, 论文可参考 https://www.ijcai.org/proceedings/2017/0435.pdf)

  Args:
    interaction_type (:obj:`str`): Interaction的方式有两种, dot和multiply
    use_attention (:obj:`bool`): 是否使用attention, 当interaction_type为'multiply'时才可用
    attention_units (:obj:`List[int]`): 使用一个MLP生成attention, attention_units表示MLP每一层的dim, 最后一维必须是1
    activation (:obj:`tf.activation`): MLP的激活函数
    initializer (:obj:`tf.initializer`): MLP的初始化器
    regularizer (:obj:`tf.regularizer`): MLP的正则化器
    out_type (:obj:`str`): 输出类型, 可以为stack, concat, None
    keep_list (:obj:`bool`): 输出是否保持list
  
  """

  def __init__(self,
               interaction_type='multiply',
               use_attention: bool = False,
               attention_units: List[int] = None,
               activation='relu',
               initializer=None,
               regularizer=None,
               out_type='concat',
               keep_list: bool = False,
               **kwargs):
    super(GroupInt, self).__init__(**kwargs)
    assert interaction_type in ['multiply', 'dot']
    self.interaction_type = interaction_type

    self.use_attention = use_attention
    if use_attention:
      assert interaction_type == 'multiply'

    self.attention_units = attention_units
    self.activation = activations.get(activation)
    self.initializer = initializers.get(
        initializer) or initializers.GlorotNormal()
    self.regularizer = regularizers.get(regularizer)

    self.out_type = out_type
    self.keep_list = keep_list

  def build(self, input_shape):
    if self.use_attention:
      assert self.attention_units[-1] == 1
      self.mlp = MLP(name='groupint_attention_mlp',
                     output_dims=self.attention_units,
                     activations=self.activation,
                     initializers=self.initializer,
                     kernel_regularizer=self.regularizer)
    else:
      self.mlp = None

    return super().build(input_shape)

  def call(self, inputs, **kwargs):
    left_fields, right_fields = inputs
    left, right = tf.concat(left_fields, axis=1), tf.concat(right_fields,
                                                            axis=1)
    last_dim_size = dim_size(left_fields[0], -1)
    ffm_embeddings = ffm(left=left,
                         right=right,
                         dim_size=last_dim_size,
                         int_type=self.interaction_type)

    if self.interaction_type == 'multiply':
      if self.use_attention:
        num_feature = len(left_fields) * len(
            right_fields
        )  #int(dim_size(left, 1) * dim_size(right, 1) / last_dim_size)
        stacked = tf.reshape(ffm_embeddings,
                             shape=(-1, num_feature, last_dim_size))
        attention = self.mlp(stacked)  # (bs, num_feature, 1)
        ffm_embeddings = tf.reshape(stacked * attention,
                                    shape=(-1, num_feature * last_dim_size))
    return [ffm_embeddings] if self.keep_list else ffm_embeddings

  def get_config(self):
    config = {
        'interaction_type': self.interaction_type,
        'use_attention': self.use_attention,
        'attention_units': self.attention_units,
        'activation': activations.serialize(self.activation),
        'initializer': initializers.serialize(self.initializer),
        'regularizer': regularizers.serialize(self.regularizer),
        'out_type': self.out_type,
        'keep_list': self.keep_list
    }
    base_config = super(GroupInt, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


FFM = GroupInt


@monolith_export
@with_params
class AllInt(Layer):
  r"""AllInt是All Interaction的缩写, 是一种简单的特征交叉方式, 通过引入压缩矩阵, 减少输出大小. 论文可参考 https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf

  GroupInt虽然能克服FM带来的输出膨胀的问题, 但也有其它问题, 如Group要人工决定, 给算法开发人员带来较大的负担. AllInt将所有特征都做交叉, 不用人工选择, 
  同时引入压缩矩阵来减少输出大小
  
  All Interaction中引入压缩矩阵. 如下:
  
  .. math::
  
    O_{n, c} = X_{n, k} * X_{n, k}^T * C_{n, c}
  
  为了避免生成(n, n)的大中间矩阵, 在计算上进行了一些优化, 即先算 :math:`X_{n, k}^T * C_{n, c}`, 这样得到的(k, c)矩阵小很多, 计算效率高

  Args:
    cmp_dim (:obj:`int`): 压缩维的维度
    initializer (:obj:`tf.initializer`): 初始化器
    regularizer (:obj:`tf.regularizer`): kernel正则化器
    use_bias (:obj:`bool`) 是否启用bias
    out_type (:obj:`str`): 输出类型, 可以为stack, concat, None
    keep_list (:obj:`bool`): 输出是否保持list
    
  """

  def __init__(self,
               cmp_dim,
               initializer=None,
               regularizer=None,
               use_bias=True,
               out_type='concat',
               keep_list=False,
               **kwargs):
    super(AllInt, self).__init__(**kwargs)
    self.cmp_dim = cmp_dim
    self.initializer = initializers.get(
        initializer) or initializers.GlorotNormal()
    self.regularizer = regularizers.get(regularizer)
    self.use_bias = use_bias

    self.out_type = out_type
    self.keep_list = keep_list

  def build(self, input_shape):
    num_feat = check_dim(input_shape[1])
    self.kernel = self.add_weight(name='allint_kernel',
                                  shape=(num_feat, self.cmp_dim),
                                  dtype=tf.float32,
                                  initializer=self.initializer,
                                  regularizer=self.regularizer,
                                  trainable=True)
    if self.use_bias:
      self.bias = self.add_weight(name='allint_bias',
                                  shape=(self.cmp_dim,),
                                  dtype=tf.float32,
                                  initializer=initializers.Zeros(),
                                  trainable=True)
    return super(AllInt, self).build(input_shape)

  def call(self, embeddings, **kwargs):
    # embeddings: [batch_size, num_feat, emb_size]
    transposed = tf.transpose(embeddings,
                              perm=[0, 2,
                                    1])  # [batch_size, emb_size, num_feat]
    feature_comp = tf.matmul(transposed,
                             self.kernel)  # [batch_size, emb_size, cmp_dim]
    if self.use_bias:
      feature_comp += self.bias
    # [batch_size, num_feat, emb_size] * [batch_size, emb_size, cmp_dim] -> [batch_size, num_feat, cmp_dim]
    interaction = tf.matmul(embeddings,
                            feature_comp)  # [batch_size, num_feat, cmp_dim]

    return merge_tensor_list(interaction,
                             merge_type=self.out_type,
                             keep_list=self.keep_list)

  def get_config(self):
    config = {
        'cmp_dim': self.cmp_dim,
        'initializer': initializers.serialize(self.initializer),
        'regularizer': regularizers.serialize(self.regularizer),
        'use_bias': self.use_bias,
        'out_type': self.out_type,
        'keep_list': self.keep_list
    }

    base_config = super(AllInt, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@monolith_export
@with_params
class CDot(Layer):
  """Compression and Dot Interaction, CDot. 可以看成是Allint的升级版, 也是一种自动做特征交叉的方法. 论文可参考 https://arxiv.org/pdf/1803.05170.pdf
  
  Allint通过引入压缩矩阵, 减少相对FM的输出大小, 同时移除了GroupInt中人工定义Group的不足, CDot与Allint十分相似
  
  CDot相对Allint的改进在于:
    - AllInt引入的压缩矩阵与输入无关, 在CDot中, 压缩矩阵是与输入数据相关, 可以根据输入, 自适应地调节压缩矩阵. 
    - CDot输出时, 会将压缩后的中间特征也输出, 作为上层MLP的输入, Allint不会做这一步
  
  一般提取高阶特征交叉时使用MLP, MLP的输入是直接接拼起来的Embedding. 一些实验表明, 可以先用CDot提取二阶特征, 再在二阶特征基础上提取高阶
  特征效果更好. 所以CDot也可以与MLP联用, 用于高阶特征提取
  
  Args:
    project_dim (:obj:`int`): 投影dim
    compress_units (:obj:`List[int]`): 用一个MLP来压缩, 压缩MLP的各层dims
    activation (:obj:`tf.activation`): MLP的激活函数
    initializer (:obj:`tf.initializer`): 初始化器
    regularizer (:obj:`tf.regularizer`): kernel正则化器
    
  """

  def __init__(self,
               project_dim,
               compress_units,
               activation='relu',
               initializer=None,
               regularizer=None,
               **kwargs):
    super(CDot, self).__init__(**kwargs)
    self.activation = activations.get(activation)
    self.initializer = initializers.get(
        initializer) or initializers.GlorotNormal()
    self.regularizer = regularizers.get(regularizer)

    self.project_dim = project_dim
    self.compress_units = compress_units

  def build(self, input_shape):
    (_, num_feature, emd_size) = input_shape
    self._num_feature = check_dim(num_feature)
    self._emd_size = check_dim(emd_size)

    self.project_weight = self.add_weight(name="project_weight",
                                          shape=(num_feature, self.project_dim),
                                          dtype=tf.float32,
                                          initializer=self.initializer,
                                          regularizer=self.regularizer)

    self.compress_tower = MLP(output_dims=self.compress_units +
                              [emd_size * self.project_dim],
                              activations=self.activation,
                              initializers=self.initializer,
                              kernel_regularizer=self.regularizer,
                              name="compress_tower")
    self._trainable_weights.extend(self.compress_tower.trainable_weights)
    self._non_trainable_weights.extend(
        self.compress_tower.non_trainable_weights)
    return super(CDot, self).build(input_shape)

  def call(self, inputs, **kwargs):
    # 1) project the origin feature into raw compressed space
    transed_input = tf.transpose(inputs,
                                 perm=[0, 2, 1
                                      ])  # (batch_size, emd_size, num_feature)
    # (batch_size, emd_size, num_feature) * (num_feature, project_dim) -> (batch_size, emd_size, project_dim)
    projected = tf.matmul(transed_input, self.project_weight)

    # 2) concat the raw compressed features, and go through mlp to cast to compressed space
    concated = tf.reshape(
        projected,
        shape=(-1, self._emd_size *
               self.project_dim))  # (batch_size, emd_size * project_dim)
    compressed = self.compress_tower(
        concated)  # (batch_size, emd_size * project_dim)

    # 3) feature cross
    # (batch_size, num_feature, emd_size) * (batch_size, emd_size, project_dim)  -> (batch_size, num_feature, project_dim)
    crossed = tf.matmul(
        inputs,
        tf.reshape(compressed, shape=(-1, self._emd_size, self.project_dim)))
    crossed = tf.reshape(
        crossed,
        shape=(-1, self._num_feature *
               self.project_dim))  # (batch_size, num_feature * project_dim)

    # 4) concat the compressed features and crossed features
    return tf.concat([crossed, compressed], axis=1)

  def get_config(self):
    config = {
        'project_dim': self.project_dim,
        'compress_units': self.compress_units,
        'activation': activations.serialize(self.activation),
        'initializer': initializers.serialize(self.initializer),
        'regularizer': regularizers.serialize(self.regularizer),
    }

    base_config = super(CDot, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@monolith_export
@with_params
class CAN(Layer):
  """Co-action Network, CAN, 协同作用网络 论文可参考 https://arxiv.org/pdf/2011.05625.pdf
  
  在模型中做特征交叉, 同一份Embedding, 同时要拟合原始特征/交叉特征, 容易两个都拟合不好. CAN是为了改善这种情况提出的, 通过拓展参数, 使得交叉特征与原始特征的学习相对独立
  
  CAN Unit将要建模的”特征对”分为weight side(item)和input side(user):
    - weight side可以reshape成MLP的参数
    - input side作为MLP的输入，通过多层MLP来建模co-action
  
  Args:
    layer_num (:obj:`int`): Layer的层数
    activation (:obj:`tf.activation`): 激活函数
    is_seq (:obj:`bool`): 是否为序列特征
    is_stacked (:obj:`bool`): User侧是否是多个特征stack起来的
    
  """

  def __init__(self,
               layer_num: int = 2,
               activation='relu',
               is_seq: bool = False,
               is_stacked: bool = True,
               **kwargs):
    super(CAN, self).__init__(**kwargs)
    self.layer_num = layer_num
    self.activation = activations.get(activation)
    self.is_seq = is_seq
    self.is_stacked = is_stacked

  def build(self, input_shape):
    user_emb_sh, item_emb_sh = input_shape
    self._batch_size = check_dim(user_emb_sh[0])
    assert user_emb_sh[0] == item_emb_sh[0]
    u_emb_size = check_dim(user_emb_sh[-1])
    iemb_size = check_dim(item_emb_sh[-1])

    assert iemb_size == (u_emb_size * (u_emb_size + 1)) * self.layer_num
    self._splits = [u_emb_size * u_emb_size, u_emb_size] * self.layer_num

    return super(CAN, self).build(input_shape)

  def call(self, inputs, **kwargs):
    user_emb, item_emb = inputs
    if self._batch_size == -1:
      self._batch_size = dim_size(user_emb, 0)

    dims = self._splits[1]
    if self.is_seq and self.is_stacked:
      # user_emb shape: (bs, num_feat, seq_len, u_emb_size)
      weight_shape = (self._batch_size, 1, dims, dims)
      bias_shape = (self._batch_size, 1, 1, dims)
    elif not self.is_seq and self.is_stacked:
      # user_emb shape: (bs, num_feat, u_emb_size)
      weight_shape = (self._batch_size, dims, dims)
      bias_shape = (self._batch_size, 1, dims)
    elif self.is_seq and not self.is_stacked:
      # user_emb shape: (bs, seq_len, u_emb_size)
      weight_shape = (self._batch_size, dims, dims)
      bias_shape = (self._batch_size, 1, dims)
    else:
      # user_emb shape: (bs, u_emb_size)
      user_emb = tf.expand_dims(user_emb, axis=1)  # (bs, 1, u_emb_size)
      weight_shape = (self._batch_size, dims, dims)
      bias_shape = (self._batch_size, 1, dims)

    params = tf.split(item_emb, num_or_size_splits=self._splits, axis=1)
    for i in range(self.layer_num):
      weight = tf.reshape(params[2 * i], shape=weight_shape)
      bias = tf.reshape(params[2 * i + 1], shape=bias_shape)
      if self.activation is not None:
        user_emb = self.activation(tf.matmul(user_emb, weight) + bias)
      else:
        user_emb = tf.matmul(user_emb, weight) + bias

    if self.is_seq and self.is_stacked:
      # user_emb shape: (bs, num_feat, seq_len, u_emb_size)
      return tf.reduce_sum(user_emb, axis=2)  # (bs, num_feat, u_emb_size)
    elif not self.is_seq and self.is_stacked:
      # user_emb shape: (bs, num_feat, u_emb_size)
      return user_emb  # (bs, num_feat, u_emb_size)
    elif self.is_seq and not self.is_stacked:
      # user_emb shape: (bs, seq_len, u_emb_size)
      return tf.reduce_sum(user_emb, axis=1)  # (bs, u_emb_size)
    else:
      # user_emb shape: (bs, 1, u_emb_size)
      return tf.squeeze(user_emb)  # (bs, u_emb_size)

  def get_config(self):
    config = {
        'layer_num': self.layer_num,
        'activation': activations.serialize(self.activation),
        "is_seq": self.is_seq,
        "is_stacked": self.is_stacked
    }
    base_config = super(CAN, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@monolith_export
@with_params
class DCN(Layer):
  r"""二阶特征交叉可用FM等方法显式提取, 更高阶的交叉用MLP隐式提取. Deep & Cross Network (DCN)可替代MLP做高阶特征交叉, 
  通过加入残差联接, 达到比MLP更好的效果
  
  DCN现在有三个版本(论文可参考 https://arxiv.org/pdf/1708.05123.pdf):
    - vector, :math:`x_{l+1} = x_0 * x_l w + b + x_l`, 其中w的shape为(dim, 1)
    - matrix, :math:`x_{l+1} = x_0 * (x_l w + b) + x_l`, 其中w的shape为(dim, dim)
    - mixed, :math:`x_{l+1} = \sum_i x_0 * (x_l V C U^T + b) * softmax(x_l g) + x_l`
  
  Args:
    layer_num (:obj:`int`): DCN的层数
    dcn_type (:obj:`str`): DCN类型, 目前支持三种vector/matrix/mixed
    initializer (:obj:`tf.initializer`): 初始化器
    regularizer (:obj:`tf.regularizer`): 正则化器
    num_experts (:obj:`int`): 只在mixed模式下有用, 用于指定expert个数
    low_rank (:obj:`int`): 只在mixed模式下有用, 用于指定低秩
    use_dropout (:obj:`bool`): 只否使用dropout
    keep_prob (:obj:`float`): dropout的保留概率
    mode (:obj:`str`): 运行模式, 可以是train/eval/predict
  
  """

  def __init__(self,
               layer_num: int = 1,
               dcn_type: str = DCNType.Matrix,
               initializer=None,
               regularizer=None,
               num_experts: int = 1,
               low_rank: int = 0,
               allow_kernel_norm: bool = False,
               use_dropout=False,
               keep_prob=0.95,
               mode: str = tf.estimator.ModeKeys.TRAIN,
               **kwargs):
    super(DCN, self).__init__(**kwargs)
    self.layer_num = layer_num
    self.dcn_type = dcn_type
    self.num_experts = num_experts
    self.low_rank = low_rank
    self.initializer = initializers.get(
        initializer) or initializers.GlorotNormal()
    self.regularizer = regularizers.get(regularizer)
    self.allow_kernel_norm = allow_kernel_norm
    self.use_dropout = use_dropout
    self.keep_prob = keep_prob
    self.mode = mode

  def build(self, input_shape):
    dims = check_dim(input_shape[-1])
    if self.dcn_type == DCNType.Vector:
      self.kernel = [
          self.get_variable(name='kernel_{}'.format(i),
                            shape=[dims, 1],
                            dtype=tf.float32,
                            initializer=self.initializer,
                            regularizer=self.regularizer,
                            trainable=True) for i in range(self.layer_num)
      ]
    elif self.dcn_type == DCNType.Matrix:
      self.kernel = [
          self.get_variable(name='kernel_{}'.format(i),
                            shape=[dims, dims],
                            dtype=tf.float32,
                            initializer=self.initializer,
                            regularizer=self.regularizer,
                            trainable=True) for i in range(self.layer_num)
      ]
    else:
      self.U = [[
          self.get_variable(name='U_{}_{}'.format(i, j),
                            shape=[dims, self.low_rank],
                            dtype=tf.float32,
                            initializer=self.initializer,
                            regularizer=self.regularizer,
                            trainable=True) for j in range(self.num_experts)
      ] for i in range(self.layer_num)]

      self.V = [[
          self.get_variable(name='V_{}_{}'.format(i, j),
                            shape=[dims, self.low_rank],
                            dtype=tf.float32,
                            initializer=self.initializer,
                            regularizer=self.regularizer,
                            trainable=True) for j in range(self.num_experts)
      ] for i in range(self.layer_num)]

      self.C = [[
          self.get_variable(name='C_{}_{}'.format(i, j),
                            shape=[self.low_rank, self.low_rank],
                            dtype=tf.float32,
                            initializer=self.initializer,
                            regularizer=self.regularizer,
                            trainable=True) for j in range(self.num_experts)
      ] for i in range(self.layer_num)]

      self.G = [[
          self.get_variable(name='G_{}_{}'.format(i, j),
                            shape=[dims, 1],
                            dtype=tf.float32,
                            initializer=self.initializer,
                            regularizer=self.regularizer,
                            trainable=True) for j in range(self.num_experts)
      ] for i in range(self.layer_num)]

    self.bias = [
        self.get_variable(name='bias_{}'.format(i),
                          shape=[1, dims],
                          dtype=tf.float32,
                          initializer=initializers.Zeros(),
                          regularizer=None,
                          trainable=True) for i in range(self.layer_num)
    ]

    return super(DCN, self).build(input_shape)

  def call(self, inputs, **kwargs):
    x0 = inputs
    xl = x0

    for i in range(self.layer_num):
      if self.dcn_type == DCNType.Vector:
        xl = x0 * tf.matmul(xl, self.kernel[i]) + self.bias[i] + xl
      elif self.dcn_type == DCNType.Matrix:
        xl = x0 * (tf.matmul(xl, self.kernel[i]) + self.bias[i]) + xl
      else:
        output_of_experts = []
        gating_score_of_experts = []
        for expert_id in range(self.num_experts):
          # (1) G(x_l)
          # compute the gating score by x_l: (batch_size, 1)
          gating_score_of_experts.append(tf.matmul(xl, self.G[i][expert_id]))

          # (2) E(x_l)
          # project the input x_l to $\mathbb{R}^{r}$
          v_x = tf.matmul(xl, self.V[i][expert_id])  # (batch_size, low_rank)
          v_x = tf.tanh(v_x)

          # nonlinear activation in low rank space
          cv_x = tf.matmul(v_x, self.C[i][expert_id])  # (batch_size, low_rank)
          cv_x = tf.tanh(cv_x)

          # project back to $\mathbb{R}^{d}$
          ucv_x = tf.matmul(cv_x, self.U[i][expert_id],
                            transpose_b=True)  # (batch_size, num_feat)

          out = x0 * (ucv_x + self.bias[i])
          output_of_experts.append(out)

        # (3) mixture of low-rank experts
        output_of_experts = tf.stack(output_of_experts,
                                     -1)  # (batch_size, num_feat, num_experts)
        gating_score_of_experts = tf.stack(gating_score_of_experts,
                                           -2)  # (bs, num_experts, 1)
        gating_score_of_experts = tf.nn.softmax(gating_score_of_experts,
                                                axis=-1)
        moe_out = tf.matmul(output_of_experts, gating_score_of_experts)
        xl = tf.squeeze(moe_out, -1) + xl

      if self.use_dropout and self.mode == tf.estimator.ModeKeys.TRAIN:
        xl = tf.nn.dropout(xl, rate=1 - self.keep_prob)

    return xl

  def get_variable(self, name, shape, dtype, initializer, regularizer,
                   trainable):
    # ref https://arxiv.org/pdf/1602.07868.pdf
    if self.allow_kernel_norm:
      upper_ns = tf.compat.v1.get_default_graph().get_name_scope()
      var_init = initializer(shape, dtype)
      with tf.compat.v1.name_scope(f'{upper_ns}/{name}/') as name_scope:
        var_name = name_scope.strip('/')
        with tf.compat.v1.variable_scope('', reuse=tf.compat.v1.AUTO_REUSE):
          var = tf.compat.v1.get_variable(initializer=var_init,
                                          name=var_name,
                                          dtype=dtype,
                                          regularizer=regularizer,
                                          trainable=trainable)

        normalized = tf.nn.l2_normalize(var,
                                        axis=0,
                                        epsilon=1e-6,
                                        name='normalized_var')
        var_norm_init = tf.norm(var_init, axis=0, name='init_trainable_norm')
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

        with tf.compat.v1.variable_scope('', reuse=tf.compat.v1.AUTO_REUSE):
          trainable_var_norm = tf.compat.v1.get_variable(
              initializer=var_norm_init,
              name=f'{var_name}/trainable_norm',
              dtype=dtype)

        if base_layer_utils.is_split_variable(trainable_var_norm) or isinstance(
            trainable_var_norm, variable_ops.PartitionedVariable):
          for v in trainable_var_norm:
            K.track_variable(v)
            if trainable:
              self._trainable_weights.append(v)
            else:
              self._non_trainable_weights.append(v)
        else:
          K.track_variable(trainable_var_norm)
          if trainable:
            self._trainable_weights.append(trainable_var_norm)
          else:
            self._non_trainable_weights.append(trainable_var_norm)
        var = tf.multiply(normalized, trainable_var_norm, name='mul_var_norm')
    else:
      var = self.add_weight(initializer=initializer,
                            shape=shape,
                            name=name,
                            dtype=dtype,
                            regularizer=regularizer,
                            trainable=trainable)

    return var

  def get_config(self):
    config = {
        'layer_num': self.layer_num,
        'dcn_type': self.dcn_type,
        'initializer': initializers.serialize(self.initializer),
        'regularizer': regularizers.serialize(self.regularizer),
        'num_experts': self.num_experts,
        'low_rank': self.low_rank,
        'allow_kernel_norm': self.allow_kernel_norm,
        'use_dropout': self.use_dropout,
        'keep_prob': self.keep_prob,
        'mode': self.mode
    }

    base_config = super(DCN, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@monolith_export
@with_params
class CIN(Layer):
  r"""Compressed Interaction Network, CIN, 压缩相互作用网络. 它是高阶(二阶以上)特征提取方法, 形式上是DCN与FM的结合体, 也是xDeepFM的核心. 论文可参考 https://arxiv.org/pdf/1703.04247.pdf
  
  DCN的计算: 
    - :math:`x_{l+1} = f_{\theta}(x_0, x_l) + x_l`, 即它是一个残差网络, 并且每一层的计算都与 :math:`x_0` 有关
  
  FM的计算: 
    - 相对于LR, 增加了二阶交叉项, 并且用embedding的形式压缩表达, 计算特征交叉的方式是点积
    
  CIN的计算: 
    - 与DCN一样, 并且每一层的计算都与 :math:`x_0` 有关, 但是并不使用残差, :math:`f_{\theta}(x,y)` 不是线性的, 而是与FM类似, 用embedding计算得到, 
      但使用的不是点积(bit-wise), 而是对应元素相乘, 然后线性组合(vector-wise). :math:`f_{\theta}(x,y)` 是类似于FM的方法显式交叉, 所以它是一种显式高阶特征交叉方法
    - 计算上, CIN还有一个特点是它可以转化成CNN高效计算

  .. math::

    X_{h,*}^k = \sum_{i=1}^{H_{k-1}} \sum_{j=1}^m W_{ij}^{k,k} (x_{i,*}^{k-1} \circ x_{j,*}^0)
  
  CIN的主要特点是:
    - 相互作用在vector-wise level, 而不是在bit-wise level 
    - 高阶特征交叉是显性的, 而非隐性的
    - 模型大小并不会随因交叉度增加而指数增加
  
  Args:
    hidden_uints (:obj:`List[int]`): CIN隐含层uints个数
    activation (:obj:`tf.activation`): 激活函数
    initializer (:obj:`tf.initializer`): 初始化器
    regularizer (:obj:`tf.regularizer`): 正则化器
  
  """

  def __init__(self,
               hidden_uints,
               activation=None,
               initializer='glorot_uniform',
               regularizer=None,
               **kwargs):
    super(CIN, self).__init__(**kwargs)
    self.hidden_uints = hidden_uints
    self.activation = activations.get(activation)
    self.initializer = initializers.get(initializer)
    self.regularizer = regularizers.get(regularizer)

    self._layer_num = len(self.hidden_uints)
    self._batch_size = None
    self._num_feat = None
    self._emb_size = None

  def build(self, input_shape):
    assert len(input_shape) == 3
    (batch_size, num_feat, emb_size) = input_shape
    self._batch_size = check_dim(batch_size)
    self._num_feat = check_dim(num_feat)
    self._emb_size = check_dim(emb_size)

    self._conv1d = []
    for i, uints in enumerate(self.hidden_uints):
      if i == 0:
        last_hidden_dim = num_feat
      else:
        last_hidden_dim = self.hidden_uints[i - 1]

      if i != self._layer_num - 1:
        self._conv1d.append(
            Conv1D(filters=uints,
                   kernel_size=1,
                   strides=1,
                   activation=self.activation,
                   kernel_initializer=self.initializer,
                   kernel_regularizer=self.regularizer,
                   input_shape=(emb_size, last_hidden_dim * num_feat)))
      else:
        self._conv1d.append(
            Conv1D(filters=uints,
                   kernel_size=1,
                   strides=1,
                   kernel_initializer=self.initializer,
                   kernel_regularizer=self.regularizer,
                   input_shape=(emb_size, last_hidden_dim * num_feat)))

      self._trainable_weights.extend(self._conv1d[-1].trainable_weights)
      self._non_trainable_weights.extend(self._conv1d[-1].non_trainable_weights)
    return super(CIN, self).build(input_shape)

  def call(self, inputs, **kwargs):
    x0 = tf.transpose(inputs, perm=[0, 2,
                                    1])  # (batch_size, emb_size, num_feat)
    xl = x0

    final_result = []
    for i in range(self._layer_num):
      # (batch_size, emb_size, -1)
      xl_last_dim = dim_size(xl, -1)
      zl = tf.reshape(tf.einsum('bdh,bdm->bdhm', xl, x0),
                      shape=(self._batch_size, self._emb_size,
                             xl_last_dim * self._num_feat))
      xl = self._conv1d[i](zl)  # (batch_size, emb_size, num_hidden)

      final_result.append(xl)

    return tf.concat([tf.reduce_sum(hi, axis=1) for hi in final_result], axis=1)

  def get_config(self):
    config = {
        'hidden_uints': self.hidden_uints,
        'activation': activations.serialize(self.activation),
        "initializer": initializers.serialize(self.initializer),
        "regularizer": regularizers.serialize(self.regularizer)
    }

    base_config = super(CIN, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
