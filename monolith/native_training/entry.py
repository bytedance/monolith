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

import abc
import copy
from typing import Any, List, Union

import tensorflow as tf

from monolith.native_training.monolith_export import monolith_export

from monolith.native_training.runtime.hash_table import \
  embedding_hash_table_pb2


class Optimizer(abc.ABC):
  """The abstract base class for optimizer."""

  @abc.abstractmethod
  def as_proto(self) -> embedding_hash_table_pb2.OptimizerConfig:
    pass


def _convert_to_proto(obj: object, proto: object):
  proto.SetInParent()
  for k, v in obj.__dict__.items():
    if v is not None:
      setattr(proto, k, v)


class StochasticRoundingFloat16OptimizerWrapper(Optimizer):

  def __init__(self, optimizer):
    self._optimizer = optimizer

  def as_proto(self):
    proto = self._optimizer.as_proto()
    proto.stochastic_rounding_float16 = True
    return proto


@monolith_export
class SgdOptimizer(Optimizer):
  r"""随机梯度下降优化器. 
  定义参数为x, 梯度为grad, 第i次更新梯度有
  
  .. math::
  
    x_{i+1} = x_{i} - \eta * grad

  Args:
    learning_rate (:obj:`float`): 学习率
    
  """

  def __init__(self, learning_rate=None):
    self.learning_rate = learning_rate

  def as_proto(self):
    opt = embedding_hash_table_pb2.OptimizerConfig()
    _convert_to_proto(self, opt.sgd)
    return opt


@monolith_export
class AdagradOptimizer(Optimizer):
  r"""Adagrad优化器, 论文可参考 http://jmlr.org/papers/v12/duchi11a.html
  定义参数为x, 梯度为grad, 第i次更新梯度时有
  
  .. math::
  
    g_{i+1} = g_{i} + grad^2
    
    x_{i+1} = x_{i} - \frac{\eta}{\sqrt{g_i + \epsilon}} grad

  Args:
    learning_rate (:obj:`float`): 学习率
    initial_accumulator_value (:obj:`float`): accmulator的起始值
    hessian_compression_times (:obj:`float`): 在训练的时候，对accumulator使用hessian sketching算法进行压缩. 1代表没有压缩，值越大，压缩效果越好
    warmup_steps (:obj:`int`): 已弃用
  
  """

  def __init__(
      self,
      learning_rate=None,  # alpha
      initial_accumulator_value=None,  # beta
      hessian_compression_times=1,
      warmup_steps=0,
      weight_decay_factor=0.0):
    self.learning_rate = learning_rate
    self.initial_accumulator_value = initial_accumulator_value
    self.hessian_compression_times = hessian_compression_times
    self.weight_decay_factor = weight_decay_factor
    self.warmup_steps = warmup_steps

  def as_proto(self):
    opt = embedding_hash_table_pb2.OptimizerConfig()
    _convert_to_proto(self, opt.adagrad)
    return opt


@monolith_export
class AdadeltaOptimizer(Optimizer):

  def __init__(self,
               learning_rate=None,
               weight_decay_factor=0.0,
               averaging_ratio=0.9,
               epsilon=0.01,
               warmup_steps=0):
    self.learning_rate = learning_rate
    self.weight_decay_factor = weight_decay_factor
    self.averaging_ratio = averaging_ratio
    self.epsilon = epsilon
    self.warmup_steps = warmup_steps

  def as_proto(self):
    opt = embedding_hash_table_pb2.OptimizerConfig()
    _convert_to_proto(self, opt.adadelta)
    return opt


@monolith_export
class AdamOptimizer(Optimizer):
  r"""Adam优化器, 论文可参考 https://arxiv.org/abs/1412.6980
  
  定义参数为x, 梯度为grad, 第i次更新梯度时有
  
  .. math::
  
    m_{i+1} = \beta_1 * m_i + (1 - \beta_1) * grad
    
    v_{i+1} = \beta_2 * v_i + (1 - \beta_2) * grad^2
    
    w_{i+1} = w_i - \eta * \frac{m_i}{\sqrt{v_i + \epsilon}}
  
  Args:
    learning_rate (:obj:`float`): 学习率
    beta1 (:obj:`float`): 一阶矩估计的指数衰减率
    beta2 (:obj:`float`): 二阶矩估计的指数衰减率
    epsilon (:obj:`float`): 用来保证除数不为0的偏移量
    warmup_steps (:obj:`int`): 已弃用
  
  """

  def __init__(self,
               learning_rate=None,
               beta1=0.9,
               beta2=0.99,
               use_beta1_warmup=False,
               weight_decay_factor=0.0,
               use_nesterov=False,
               epsilon=0.01,
               warmup_steps=0):
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.beta2 = beta2
    self.use_beta1_warmup = use_beta1_warmup
    self.weight_decay_factor = weight_decay_factor
    self.use_nesterov = use_nesterov
    self.epsilon = epsilon
    self.warmup_steps = warmup_steps

  def as_proto(self):
    opt = embedding_hash_table_pb2.OptimizerConfig()
    _convert_to_proto(self, opt.adam)
    return opt


class AmsgradOptimizer(Optimizer):

  def __init__(self,
               learning_rate=None,
               beta1=0.9,
               beta2=0.99,
               weight_decay_factor=0.0,
               use_nesterov=False,
               epsilon=0.01,
               warmup_steps=0):
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.beta2 = beta2
    self.weight_decay_factor = weight_decay_factor
    self.use_nesterov = use_nesterov
    self.epsilon = epsilon
    self.warmup_steps = warmup_steps

  def as_proto(self):
    opt = embedding_hash_table_pb2.OptimizerConfig()
    _convert_to_proto(self, opt.amsgrad)
    return opt


@monolith_export
class BatchSoftmaxOptimizer(Optimizer):
  r"""Batch softmax优化器, 论文可参考 https://research.google/pubs/pub48840/

  Args:
    learning_rate (:obj:`float`): 学习率
  """

  def __init__(
      self,
      learning_rate=None,  # alpha
  ):
    self.learning_rate = learning_rate

  def as_proto(self):
    opt = embedding_hash_table_pb2.OptimizerConfig()
    _convert_to_proto(self, opt.batch_softmax)
    return opt


@monolith_export
class MomentumOptimizer(Optimizer):

  def __init__(self,
               learning_rate=None,
               weight_decay_factor=0.0,
               use_nesterov=False,
               momentum=0.9,
               warmup_steps=0):
    self.learning_rate = learning_rate
    self.weight_decay_factor = weight_decay_factor
    self.use_nesterov = use_nesterov
    self.momentum = momentum
    self.warmup_steps = warmup_steps

  def as_proto(self):
    opt = embedding_hash_table_pb2.OptimizerConfig()
    _convert_to_proto(self, opt.momentum)
    return opt


class MovingAverageOptimizer(Optimizer):

  def __init__(self, momentum=0.9):
    self.momentum = momentum

  def as_proto(self):
    opt = embedding_hash_table_pb2.OptimizerConfig()
    _convert_to_proto(self, opt.moving_average)
    return opt


@monolith_export
class RmspropOptimizer(Optimizer):

  def __init__(self, learning_rate=None, weight_decay_factor=0.0, momentum=0.9):
    self.learning_rate = learning_rate
    self.weight_decay_factor = weight_decay_factor
    self.momentum = momentum

  def as_proto(self):
    opt = embedding_hash_table_pb2.OptimizerConfig()
    _convert_to_proto(self, opt.rmsprop)
    return opt


@monolith_export
class RmspropV2Optimizer(Optimizer):

  def __init__(self, learning_rate=None, weight_decay_factor=0.0, momentum=0.9):
    self.learning_rate = learning_rate
    self.weight_decay_factor = weight_decay_factor
    self.momentum = momentum

  def as_proto(self):
    opt = embedding_hash_table_pb2.OptimizerConfig()
    _convert_to_proto(self, opt.rmspropv2)
    return opt


class FTRLWithGroupSparsityOptimizer(Optimizer):

  def __init__(
      self,
      learning_rate=None,  # alpha
      initial_accumulator_value=None,
      beta=None,
      warmup_steps=0,
      l1_regularization=None,  # lambda1
      l2_regularization=None):  # lambda2
    self.learning_rate = learning_rate
    self.initial_accumulator_value = initial_accumulator_value
    self.beta = beta
    self.l1_regularization_strength = l1_regularization
    self.l2_regularization_strength = l2_regularization
    self.warmup_steps = warmup_steps

  def as_proto(self):
    opt = embedding_hash_table_pb2.OptimizerConfig()
    _convert_to_proto(self, opt.group_ftrl)
    return opt


# TODO: put DcOptimizer into entry.py


class DynamicWdAdagradOptimizer(Optimizer):

  def __init__(
      self,
      learning_rate=None,  # alpha
      initial_accumulator_value=None,  # beta
      hessian_compression_times=1,
      warmup_steps=0,
      weight_decay_factor=0.0,
      decouple_weight_decay=True,
      enable_dynamic_wd=True,
      flip_direction=True,
      dynamic_wd_temperature=1.0):
    self.learning_rate = learning_rate
    self.initial_accumulator_value = initial_accumulator_value
    self.hessian_compression_times = hessian_compression_times
    self.weight_decay_factor = weight_decay_factor
    self.warmup_steps = warmup_steps
    self.decouple_weight_decay = decouple_weight_decay
    self.enable_dynamic_wd = enable_dynamic_wd
    self.flip_direction = flip_direction
    self.dynamic_wd_temperature = dynamic_wd_temperature

  def as_proto(self):
    opt = embedding_hash_table_pb2.OptimizerConfig()
    _convert_to_proto(self, opt.dynamic_wd_adagrad)
    return opt


@monolith_export
class FtrlOptimizer(Optimizer):
  """FTRL优化器, 论文可参考 https://dl.acm.org/citation.cfm?id=2488200
  
  Args:
    initial_accumulator_value (:obj:`float`): accumulator的起始值
    beta (:obj:`float`): 论文中的beta值
  
  """

  def __init__(
      self,
      learning_rate=None,  # alpha
      initial_accumulator_value=None,
      beta=None,
      warmup_steps=0,
      l1_regularization=None,  # lambda1
      l2_regularization=None):  # lambda2
    self.learning_rate = learning_rate
    self.initial_accumulator_value = initial_accumulator_value
    self.beta = beta
    self.l1_regularization_strength = l1_regularization
    self.l2_regularization_strength = l2_regularization
    self.warmup_steps = warmup_steps

  def as_proto(self):
    opt = embedding_hash_table_pb2.OptimizerConfig()
    _convert_to_proto(self, opt.ftrl)
    return opt


class Initializer(abc.ABC):
  """The abstract base class for initializer"""

  @abc.abstractmethod
  def as_proto(self) -> embedding_hash_table_pb2.InitializerConfig:
    pass


@monolith_export
class ZerosInitializer(Initializer):
  """全0初始化器，将会把embedidng的初始值设为全0"""

  def as_proto(self):
    init = embedding_hash_table_pb2.InitializerConfig()
    _convert_to_proto(self, init.zeros)
    return init


@monolith_export
class ConstantsInitializer(Initializer):
  """常数初始化器，将会把embedidng的初始值设为常数"""

  def __init__(self, constant: float):
    self.constant = constant

  def as_proto(self):
    init = embedding_hash_table_pb2.InitializerConfig()
    _convert_to_proto(self, init.constants)
    return init


class RandomUniformInitializer(Initializer):
  """随机均匀的初始化器，将会把初始化区间默认为[minval, maxval]
  
  Args:
    minval, maxval (:obj:`float`): 初始化的区间
  
  """

  def __init__(self, minval=None, maxval=None):
    self.minval = minval
    self.maxval = maxval

  def as_proto(self):
    init = embedding_hash_table_pb2.InitializerConfig()
    _convert_to_proto(self, init.random_uniform)
    return init


class BatchSoftmaxInitializer(Initializer):

  def __init__(self, init_step_interval: float):
    if init_step_interval < 1:
      raise ValueError("init_step_interval should be >= 1, while got {}".format(
          init_step_interval))
    self.constant = init_step_interval

  def as_proto(self):
    init = embedding_hash_table_pb2.InitializerConfig()
    _convert_to_proto(self, init.constants)
    return init


class Compressor(abc.ABC):
  """The abstract base class for compressor"""

  @abc.abstractmethod
  def as_proto(self) -> embedding_hash_table_pb2.FloatCompressorConfig:
    pass


@monolith_export
class OneBitCompressor(Compressor):

  def __init__(self, step_size: int = 200, amplitude: float = 0.05):
    super().__init__()
    self.step_size = step_size
    self.amplitude = amplitude

  def as_proto(self):
    comp = embedding_hash_table_pb2.FloatCompressorConfig()
    comp.one_bit.step_size = self.step_size
    _convert_to_proto(self, comp.one_bit)
    return comp


@monolith_export
class FixedR8Compressor(Compressor):

  def __init__(self, fixed_range=1.0):
    super().__init__()
    self.r = fixed_range

  def as_proto(self):
    comp = embedding_hash_table_pb2.FloatCompressorConfig()
    _convert_to_proto(self, comp.fixed_r8)
    return comp


@monolith_export
class Fp16Compressor(Compressor):
  """当模型服务时，将会对embedding进行Fp16编码，从而达到在服务时节省内存的目的"""

  def as_proto(self):
    comp = embedding_hash_table_pb2.FloatCompressorConfig()
    _convert_to_proto(self, comp.fp16)
    return comp


@monolith_export
class Fp32Compressor(Compressor):
  """当模型服务时，将会对embedding进行Fp32编码"""

  def as_proto(self):
    comp = embedding_hash_table_pb2.FloatCompressorConfig()
    _convert_to_proto(self, comp.fp32)
    return comp


def CombineAsSegment(
    dim_size: int,
    initializer: Union[Initializer, embedding_hash_table_pb2.InitializerConfig],
    optimizer: Union[Optimizer, embedding_hash_table_pb2.OptimizerConfig],
    compressor: Union[Compressor,
                      embedding_hash_table_pb2.FloatCompressorConfig]
) -> embedding_hash_table_pb2.EntryConfig.Segment:
  segment = embedding_hash_table_pb2.EntryConfig.Segment()
  segment.dim_size = dim_size
  if hasattr(initializer, 'as_proto'):
    segment.init_config.CopyFrom(initializer.as_proto())
  else:
    segment.init_config.CopyFrom(initializer)

  if hasattr(optimizer, 'as_proto'):
    segment.opt_config.CopyFrom(optimizer.as_proto())
  else:
    segment.opt_config.CopyFrom(optimizer)

  if hasattr(compressor, 'as_proto'):
    segment.comp_config.CopyFrom(compressor.as_proto())
  else:
    segment.comp_config.CopyFrom(compressor)
  return segment


class HashTableConfig(abc.ABC):
  """For hash table since we are not sure which field to update, we use an update function"""

  @abc.abstractmethod
  def mutate_table(
      self, table_config: embedding_hash_table_pb2.EmbeddingHashTableConfig):
    pass


class CuckooHashTableConfig(HashTableConfig):

  def __init__(self, initial_capacity=1, feature_evict_every_n_hours=0):
    self._initial_capacity = initial_capacity
    self._feature_evict_every_n_hours = feature_evict_every_n_hours

  def mutate_table(
      self, table_config: embedding_hash_table_pb2.EmbeddingHashTableConfig):
    table_config.initial_capacity = self._initial_capacity
    table_config.cuckoo.SetInParent()
    if self._feature_evict_every_n_hours > 0:
      table_config.enable_feature_eviction = True
      table_config.feature_evict_every_n_hours = self._feature_evict_every_n_hours




class HashTableConfigInstance():
  """The config instance for generating HashTable"""

  def __init__(self,
               table_config: embedding_hash_table_pb2.EmbeddingHashTableConfig,
               learning_rate_fns: List[Any],
               extra_restore_names=None):
    self._table_config = table_config
    self.extra_restore_names = copy.copy(extra_restore_names) or []
    self._learning_rate_fns = learning_rate_fns
    self._learning_rate_tensor = None

  # Used to check whether two slots share the same config.
  def __str__(self):
    return "TableConfigPB: %s, LearningRateFns: [%s]" % (
        self._table_config.SerializeToString(), ", ".join(
            [str(fn) for fn in self._learning_rate_fns]))

  @property
  def table_config(self):
    return self._table_config

  @property
  def learning_rate_fns(self):
    return self._learning_rate_fns

  @property
  def learning_rate_tensor(self):
    return self._learning_rate_tensor

  def set_learning_rate_tensor(self, learning_rate_tensor: tf.Tensor):
    self._learning_rate_tensor = learning_rate_tensor

  def call_learning_rate_fns(self) -> tf.Tensor:
    """Call learning rate function if callable and return a tf.Tensor"""
    with tf.name_scope("learning_rate"):
      learning_rates = list()
      for learning_rate_fn in self._learning_rate_fns:
        if not callable(learning_rate_fn):
          learning_rate = tf.cast(learning_rate_fn, dtype=tf.float32)
        else:
          learning_rate = learning_rate_fn()
        learning_rates.append(learning_rate)

      if len(learning_rates) > 0:
        learning_rate_tensor = tf.stack(learning_rates)
      else:
        raise Exception("Learning_rate_fns must be not empty.")

      return learning_rate_tensor
