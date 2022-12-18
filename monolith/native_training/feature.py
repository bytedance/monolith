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
from collections import namedtuple
import copy
import enum
from dataclasses import dataclass, asdict, field
from typing import Callable, Dict, Iterable, List, Tuple, Set, NamedTuple, Union
import sys
import os

from absl import logging

import tensorflow as tf

from monolith.native_training import device_utils
from monolith.native_training import distribution_ops
from monolith.native_training import embedding_combiners
from monolith.native_training import entry
from monolith.native_training import learning_rate_functions
from monolith.native_training import ragged_utils
from monolith.native_training.runtime.hash_table import \
    embedding_hash_table_pb2
from monolith.native_training.model_export.export_context import is_exporting
from monolith.native_training.monolith_export import monolith_export

_FEATURE_STRAT_END_KEY = "{}:{}_{}"

# Default expire time is 100 years.
DEFAULT_EXPIRE_TIME = 36500


class FeatureEmbTable(abc.ABC):
  """Used by framework. Do not use in the user code directly. Instead, using FeatureSlot."""

  def add_feature_slice(self,
                        segment: embedding_hash_table_pb2.EntryConfig.Segment,
                        learning_rate_fn=None):
    """
    Add one feature slice for this embedding table.
    """
    pass

  @abc.abstractmethod
  def embedding_lookup(self, feature_name: str, start: int,
                       end: int) -> tf.Tensor:
    """
    Returns combined embedding tensors for the given feature name.
    """
    pass

  def set_feature_metadata(self, feature_name: str,
                           combiner: embedding_combiners.Combiner):
    pass


class FeatureSlice(NamedTuple):
  """Represents a slice of a feature slot."""
  feature_slot: "FeatureSlot"
  start: int
  end: int


@dataclass
class FeatureSlotConfig:
  name: str = None
  has_bias: bool = False
  bias_initializer: entry.Initializer = entry.ZerosInitializer()
  bias_optimizer: entry.Optimizer = entry.FtrlOptimizer(
      initial_accumulator_value=1e-6, beta=1.0)
  bias_compressor: entry.Compressor = entry.Fp32Compressor()
  bias_learning_rate_fn: Callable = None
  default_vec_initializer: entry.Initializer = entry.RandomUniformInitializer()
  default_vec_optimizer: entry.Optimizer = entry.AdagradOptimizer(
      initial_accumulator_value=1.0)
  default_vec_compressor: entry.Compressor = entry.Fp16Compressor()
  default_vec_learning_rate_fn: Callable = None
  hashtable_config: entry.HashTableConfig = entry.CuckooHashTableConfig()
  slot_id: int = None
  occurrence_threshold: int = 0
  expire_time: int = DEFAULT_EXPIRE_TIME

  def __post_init__(self):
    if not self.name:
      self.name = str(self.slot_id)


@monolith_export
class FeatureSlot:
  """维护特征与HashTable的关系, 隐藏HashTable的细节. FeatureSlot可以看成是用户视角的HashTable
  
  Args:
    table (:obj:`FeatureEmbTable`): 内部HashTable
    config (:obj:`FeatureSlotConfig`): 特征配置
  
  """

  def __init__(self, table: FeatureEmbTable, config: FeatureSlotConfig):
    self._table = table
    self._config = config
    self._current_dim_size = 0
    self._feature_columns = set()

    if self._config.has_bias:
      self._bias_slice = self.add_feature_slice(
          1, self._config.bias_initializer, self._config.bias_optimizer,
          self._config.bias_compressor, self._config.bias_learning_rate_fn)

  def add_feature_slice(self,
                        dim_size: int,
                        initializer: entry.Initializer = None,
                        optimizer: entry.Optimizer = None,
                        compressor: entry.Compressor = None,
                        learning_rate_fn=None) -> FeatureSlice:
    """
    在哈希表中增加一段长度为|dim_size|，并采用|initializer|作为初始化器，|optimizer|作为
    优化器，同时在serving中使用|compressor|作为压缩器的embedding. 
    返回一个feature slice被FeatureColumn使用

    Args:
      dim_size (:obj:`float`): 这段embedding slice的长度
      optimizer （:obj:`entry.Optimizer`): 这段embedding slice的初始化器
      compressor （:obj:`entry.Compressor`): 这段embedding slice的初始化器
      learning_rate_fn (:obj:`Callable`): 如果不为None，覆盖在optimizer中定义的学习率
    """
    initializer = initializer or self._config.default_vec_initializer
    optimizer = optimizer or self._config.default_vec_optimizer
    compressor = compressor or self._config.default_vec_compressor
    learning_rate_fn = learning_rate_fn or self._config.default_vec_learning_rate_fn
    segment = entry.CombineAsSegment(dim_size, initializer, optimizer,
                                     compressor)
    self._table.add_feature_slice(segment, learning_rate_fn=learning_rate_fn)

    s = FeatureSlice(self, self._current_dim_size,
                     self._current_dim_size + dim_size)
    self._current_dim_size = self._current_dim_size + dim_size
    return s

  def get_bias_slice(self):
    assert self._config.has_bias
    return self._bias_slice

  def _add_feature_column(self, fc):
    self._feature_columns.add(fc)
    self._table.set_feature_metadata(fc.feature_name, fc.combiner)

  def _fc_embedding_lookup(self, feature_name: str, s: FeatureSlice):
    return self._table.embedding_lookup(feature_name, s.start, s.end)

  def get_feature_columns(self):
    return self._feature_columns

  @property
  def slot(self):
    return int(self._config.name)

  @property
  def name(self):
    return self._config.name


@monolith_export
class FeatureColumn:
  """将FeatureColumn与输入的Feature进行链接
  
  Args:
    feature_slot (:obj:`FeatureSlot`): 这个类对应的FeatureSlot
    feature_name (:obj:`str`): 这个类对应的链接的feature_name（在input_fn返回的结果）
  
  """

  @classmethod
  def reduce_sum(cls):
    return embedding_combiners.ReduceSum()

  @classmethod
  def reduce_mean(cls):
    return embedding_combiners.ReduceMean()

  @classmethod
  def first_n(cls, seq_length: int):
    return embedding_combiners.FirstN(seq_length)

  def __init__(self,
               feature_slot: FeatureSlot,
               feature_name: str,
               combiner=None):
    self._feature_name = feature_name
    self._feature_slot = feature_slot
    self._combiner = combiner or self.reduce_sum()
    self._size_tensor = None
    feature_slot._add_feature_column(self)

  def embedding_lookup(self, s: FeatureSlice) -> tf.Tensor:
    """返回feature_name在feature_slot中进行查询之后的结果. """
    assert s.feature_slot == self._feature_slot, "Slice must come from the dedicated feature slot."
    return self._feature_slot._fc_embedding_lookup(self._feature_name, s)

  def get_all_embeddings_concat(self) -> tf.Tensor:
    """
    Returns concatenated all embeddings owned by this column. Used in calculate gradients
    """
    return self._feature_slot._table.embedding_lookup(self._feature_name, None,
                                                      None)

  def get_all_embedding_slices(self) -> List[tf.Tensor]:
    """
    Returns concatenated all embedding slices owned by this column. Used in computing gradients
    """
    output_list = []
    for k, v in self._embedding_slices.items():
      if self._feature_name in k:
        output_list.append(v)
    return output_list

  @property
  def feature_name(self):
    return self._feature_name

  @property
  def feature_slot(self) -> FeatureSlot:
    return self._feature_slot

  @property
  def combiner(self) -> embedding_combiners.Combiner:
    return self._combiner

  def get_bias(self) -> tf.Tensor:
    """字节内部使用. 请勿直接使用"""
    bias_slice = self._feature_slot.get_bias_slice()
    return self._feature_slot._fc_embedding_lookup(self._feature_name,
                                                   bias_slice)

  def set_size_tensor(self, row_lengths: tf.Tensor):
    assert isinstance(self._combiner, embedding_combiners.FirstN
                     ), "This function is only supported in a sequence feature."
    seq_length = self._combiner.max_seq_length
    # Convert row_lengths to [B, max_seq_length] Tensor, in which
    # the first row_length elements of each row are 1, and the rest are
    # 0. This is used as the size_tensor
    batch_size = tf.size(row_lengths)  # 0-D Tensor
    boolean_mask = tf.less(
        tf.reshape(
            tf.tile(tf.range(0, seq_length), [batch_size]),
            [batch_size, -1],
        ), tf.expand_dims(row_lengths, 1))  # [B, max_seq_length] Tensor
    self._size_tensor = tf.cast(boolean_mask, tf.int32, name='size_tensor')

  def get_size_tensor(self):
    return self._size_tensor


FeatureColumnV1 = FeatureColumn

SliceConfig = namedtuple("SliceConfig", ["segment", "learning_rate_fn"])


class TableConfig(NamedTuple):
  slice_configs: List[SliceConfig]
  feature_names: Set[str]
  unmerged_slice_dims: List[int]
  hashtable_config: entry.HashTableConfig
  feature_to_combiners: Dict[str, embedding_combiners.Combiner]


class FeatureFactory(abc.ABC):
  """Used to get features in the model_fn."""

  def __init__(self):
    self.slot_to_occurrence_threshold = {}
    self.slot_to_expire_time = {}

  @abc.abstractmethod
  def create_feature_slot(self, config: FeatureSlotConfig) -> FeatureSlot:
    """Creates a feature slot by config."""

  def apply_gradients(self,
                      grads_and_vars: Iterable[Tuple[tf.Tensor, tf.Tensor]],
                      req_time: tf.Tensor = None) -> tf.Operation:
    """
    Applies the gradients to Features owned by this factory.
    The reason why we do not make per table based apply_gradients is because of
    performance reason. In the runtime, we may do a batch lookup. 
    Args: 
      grads_and_vars - vars must be the all_embedding_concat from each FeatureColumn.
    """
    raise NotImplementedError(
        "apply_gradients is not supported in this factory.")


class DummyFeatureEmbTable(FeatureEmbTable):
  """It is used to collect config of table from model_fn."""

  def __init__(self, batch_size, hashtable_config):
    self._batch_size = batch_size
    self._hashtable_config = hashtable_config
    self._slices = []
    self._merged_slices = []
    self._feature_names = set()
    self._feature_to_combiner = {}
    self._dim_size = 0

  def add_feature_slice(self,
                        segment: embedding_hash_table_pb2.EntryConfig.Segment,
                        learning_rate_fn=None):
    # The learning_rate_fn can be an instance of LearningRateFunction or a float
    # value. By default, set the learning_rate_fn according to the optimizer config.
    if learning_rate_fn is None:
      opt_config = getattr(segment.opt_config,
                           segment.opt_config.WhichOneof("type"))
      if hasattr(opt_config, 'warmup_steps') and opt_config.warmup_steps > 0:
        learning_rate_fn = learning_rate_functions.PolynomialDecay(
            initial_learning_rate=0.0,
            decay_steps=opt_config.warmup_steps,
            end_learning_rate=opt_config.learning_rate)
      else:
        learning_rate_fn = opt_config.learning_rate
    self._dim_size += segment.dim_size
    self._slices.append(SliceConfig(segment, learning_rate_fn))

  def embedding_lookup(self, feature_name: str, start: int,
                       end: int) -> tf.Tensor:
    if start is None and end is None:
      # This is the special case for gradients.
      start = 0
      end = self._dim_size

    # TODO(leqi.zou): Maybe we should add a dict here to make sure for the
    # same look up we should return same result.
    emb_ph = tf.compat.v1.placeholder(tf.float32,
                                      shape=[self._batch_size, end - start])
    key = tf.compat.v1.ragged.placeholder(tf.int64, 1, [])
    combiner = self._feature_to_combiner[feature_name]
    combined = combiner.combine(
        key,
        emb_ph,
        name=f'{combiner.__class__.__name__}_{feature_name}_{start}_{end}')
    if self._batch_size:
      shape = combined.shape.as_list()
      shape[0] = self._batch_size
      combined = tf.reshape(combined, shape)
    return combined

  def set_feature_metadata(self, feature_name: str,
                           combiner: embedding_combiners.Combiner):
    self._feature_names.add(feature_name)
    self._feature_to_combiner[feature_name] = combiner

  def get_table_config(self) -> TableConfig:
    """Returns merged slices of FeatureEmbTable"""
    self._merged_slices = self._merge_slices()

    # Note(youlong.cheng): This is mainly for tf.split after pooling embedding.
    # The alternative way uses strided_slice causes duplicated backward
    # calcualtion and unncessary memory write.
    unmerged_slice_dims = [config.segment.dim_size for config in self._slices]

    return TableConfig(self._merged_slices,
                       [feature_name for feature_name in self._feature_names],
                       unmerged_slice_dims, self._hashtable_config,
                       self._feature_to_combiner)

  def get_feature_names(self):
    return self._feature_names

  def _merge_slices(self):
    """Combines the slices which only differ in dim_size."""
    merged = []
    # Using deepcopy to prevent modifing the proto in self._slices.
    slices = copy.deepcopy(self._slices)
    for s in slices:
      if not merged:
        merged.append(s)
        continue
      last_s = merged[-1]

      last_dim_size = last_s.segment.dim_size
      last_s.segment.ClearField("dim_size")
      dim_size = s.segment.dim_size
      s.segment.ClearField("dim_size")

      if last_s.segment.SerializeToString() == s.segment.SerializeToString(
      ) and str(last_s.learning_rate_fn) == str(s.learning_rate_fn):
        # We can merge these two slices
        last_s.segment.dim_size = last_dim_size + dim_size
      else:
        last_s.segment.dim_size = last_dim_size
        s.segment.dim_size = dim_size
        merged.append(s)
    return merged


class DummyFeatureFactory(FeatureFactory):
  """Factory to collect the config."""

  def __init__(self, batch_size):
    super().__init__()
    self._batch_size = batch_size
    self._tables = {}

  def create_feature_slot(self, config: FeatureSlotConfig):
    """Creates a feature slot by config."""
    if config.name in self._tables:
      raise NameError("Duplicate names for the table. Name: {}".format(
          config.name))
    table = DummyFeatureEmbTable(self._batch_size, config.hashtable_config)
    self._tables.update({config.name: table})

    if config.slot_id is not None:
      self.slot_to_occurrence_threshold.update(
          {config.slot_id: config.occurrence_threshold})
      self.slot_to_expire_time.update({config.slot_id: config.expire_time})
    else:
      logging.warning(
          "feature[{}] slot is None. pls check feature_list.conf".format(
              config.name))

    return FeatureSlot(table, config)

  def apply_gradients(self, *args, **kwargs) -> tf.Operation:
    return tf.no_op()

  def get_table_name_to_table_config(self) -> Dict[str, TableConfig]:
    table_configs = {}
    for k, v in self._tables.items():
      table_config = v.get_table_config()
      if len(table_config.slice_configs) > 0:
        table_configs[k] = table_config
      else:
        raise RuntimeError(f'{k} has no slice, pls. check!')
    return table_configs


class EmbeddingFeatureEmbTable(FeatureEmbTable):
  """Actual emb table that provides the embedding tensor from embeddings."""

  def __init__(self, embeddings: Dict[str, tf.Tensor],
               embedding_slices: Dict[str, tf.Tensor]):
    self._embeddings = embeddings
    self._embedding_slices = embedding_slices

  def embedding_lookup(self, feature_name: str, start: int,
                       end: int) -> tf.Tensor:
    if start is None and end is None:
      # It is important to return the origin tensor since we may
      # use this tensor as map key.
      return self._embeddings[feature_name]
    k = _FEATURE_STRAT_END_KEY.format(feature_name, start, end)
    logging.vlog(1, "_embedding_slices: {}".format(self._embedding_slices))
    return self._embedding_slices[k]


class _FeatureFactoryFusionHelper:
  """Only for feature to be reduced. Not for features to keep the original dim."""

  def __init__(self):
    self._d = {}

  def append(self, name, ragged_ids, embeddings, slice_dims):
    self._d[name] = (ragged_ids.row_splits,
                     ragged_utils.fused_value_rowids(ragged_ids), embeddings,
                     ragged_ids.nrows(), slice_dims)

  def reduce_and_split(self):
    """(reduce -> split) * N: BASIC for both CPU and GPU."""
    feature_name_to_slices = {}
    for name, (_, value_rowids, embeddings, batch_size_tensor,
               slice_dims) in self._d.items():
      with tf.device("/device:CPU:0"):
        shape = tf.stack([batch_size_tensor,
                          embeddings.shape.as_list()[1]])  # (batch_size, dim)
      with device_utils.maybe_device_if_allowed('/device:GPU:0'):
        # scatter_nd (a.k.a embedding_combiners.ReduceSum) + split
        reduced_emb = tf.scatter_nd(
            tf.expand_dims(value_rowids, -1),
            embeddings,
            shape,
            name=name,
        )
        tf.compat.v1.add_to_collection("monolith_reduced_embs", reduced_emb)
        feature_name_to_slices[name] = tf.split(reduced_emb,
                                                slice_dims,
                                                axis=1,
                                                name=name + "_split")
    return feature_name_to_slices

  def fused_reduce_and_split(self):
    """(reduce + split) * N: For CPU Performance."""
    feature_name_to_slices = {}
    for name, (_, value_rowids, embeddings, batch_size_tensor,
               slice_dims) in self._d.items():
      # We do a simple fused operation that returns a list of tensors, split
      # across the column dimension, so it returns a list of tensors of shapes
      # [batch_size, split_dim[i]].
      with tf.device("/device:CPU:0"):
        slices = distribution_ops.fused_reduce_sum_and_split(
            value_rowids,
            embeddings,
            batch_size_tensor,
            slice_dims,
            name=f'ReduceSumAndSplit_{name}')
        feature_name_to_slices[name] = slices
    return feature_name_to_slices

  def fused_reduce_then_split(self):
    """reduce * N -> split: For GPU Performance.
    
    Note that we don't fuse the split here, so that split + downstream model op can be fused 
    when pattern matched at graph optimization level.
    """
    feature_name_to_slices = {}
    if not self._d:
      return feature_name_to_slices
    es, ss, ds = [], [], []
    for name, (row_splits, _, embeddings, _, slice_dims) in self._d.items():
      ss.append(row_splits)
      es.append(embeddings)
      ds.append(slice_dims)

    with device_utils.maybe_device_if_allowed('/device:GPU:0'):
      out = distribution_ops.fused_reduce_and_split_gpu(ss, es, ds)
      slice_idx = 0
      for name, (_, _, _, _, slice_dims) in self._d.items():
        feature_name_to_slices[name] = out[slice_idx:slice_idx +
                                           len(slice_dims)]
        slice_idx += len(slice_dims)
    return feature_name_to_slices


def create_embedding_slices(
    name_to_embeddings: Dict[str, tf.Tensor],
    name_to_embedding_ids: Dict[str, tf.RaggedTensor],
    feature_to_combiner: Dict[str, embedding_combiners.Combiner],
    feature_to_unmerged_slice_dims: Dict[str,
                                         List[int]]) -> Dict[str, tf.Tensor]:
  embedding_slices = {}
  feature_to_slices = {}
  helper = _FeatureFactoryFusionHelper()

  # Here we perform a fused reduce_sum+splitv operations.
  for name, embeddings in name_to_embeddings.items():
    ragged_ids = name_to_embedding_ids[name]
    combiner = feature_to_combiner[name]
    if isinstance(combiner, embedding_combiners.ReduceSum):
      # This is for a general case, where splits and reduce_sums both happen.
      # We do a simple fused operation that returns a list of tensors, split
      # across the column dimension, so it returns a list of tensors of shapes
      # [None, split_dim[i]], where None refers to the batch_size.
      helper.append(
          name,
          ragged_ids,
          embeddings,  # to combiner
          feature_to_unmerged_slice_dims[name])  # to split
    else:
      combined_emb = combiner.combine(
          ragged_ids,
          embeddings,
          name=f'{combiner.__class__.__name__}_{name}_vv')
      with device_utils.maybe_device_if_allowed('/device:GPU:0'):
        slices = tf.split(combined_emb,
                          feature_to_unmerged_slice_dims[name],
                          axis=-1)
      feature_to_slices[name] = slices

  with device_utils.maybe_device_if_allowed('/device:GPU:0'):
    # In a long term, this optimization should be on graph-transform level at runtime.
    if device_utils.within_placement_context_of("GPU"):
      if int(os.getenv("MONOLITH_GPU_FEATURE_FACTORY_FUSION_LEVEL", '1')) == 1:
        feature_to_slices.update(helper.fused_reduce_then_split())
      else:
        feature_to_slices.update(helper.reduce_and_split())
    else:
      if is_exporting():
        feature_to_slices.update(helper.reduce_and_split())
      else:
        feature_to_slices.update(helper.fused_reduce_and_split())

  # assign slice tensors to embedding table for lookup
  for name, slices in feature_to_slices.items():
    start = 0
    for i, dim in enumerate(feature_to_unmerged_slice_dims[name]):
      end = start + dim
      embedding_slices[_FEATURE_STRAT_END_KEY.format(name, start,
                                                     end)] = slices[i]
      start = end

  return embedding_slices


class FeatureFactoryFromEmbeddings(FeatureFactory):

  def __init__(self, name_to_embeddings: Dict[str, tf.Tensor],
               name_to_embedding_slices: Dict[str, tf.Tensor]):
    super().__init__()
    self._name_to_embeddings = name_to_embeddings
    self._name_to_embedding_slices = name_to_embedding_slices

  def create_feature_slot(self, config: FeatureSlotConfig) -> FeatureSlot:
    # TODO(zouxuan): self._embeddings is actually never updated or used.
    table = EmbeddingFeatureEmbTable(self._name_to_embeddings,
                                     self._name_to_embedding_slices)
    return FeatureSlot(table, config)


class EmbeddingLayoutFakeTable(FeatureEmbTable):

  def embedding_lookup(self, feature_name: str, start: int,
                       end: int) -> tf.Tensor:
    return None


class EmbeddingLayoutFactory(object):

  def __init__(self,
               hash_table: 'PartitionedHashTable',
               layout_embeddings: Dict[str, Union[tf.Tensor, List[tf.Tensor]]],
               auxiliary_bundle: Dict[str, tf.Tensor] = None):
    self.hash_table = hash_table
    self.layout_embeddings = layout_embeddings
    self.auxiliary_bundle = auxiliary_bundle

  def create_feature_slot(self, config: FeatureSlotConfig) -> FeatureSlot:
    table = EmbeddingLayoutFakeTable()
    return FeatureSlot(table, config)

  def apply_gradients(self,
                      grads_and_vars: Iterable[Tuple[tf.Tensor, tf.Tensor]],
                      req_time: tf.Tensor = None):
    return self.hash_table.apply_gradients(
        layout_grads_and_vars=grads_and_vars,
        global_step=tf.compat.v1.train.get_or_create_global_step(),
        req_time=req_time or self.auxiliary_bundle.get("req_time"),
        auxiliary_bundle=self.auxiliary_bundle)

  def get_layout(self, layout: str) -> Union[tf.Tensor, List[tf.Tensor]]:
    assert layout in self.layout_embeddings
    return self.layout_embeddings[layout]

  def flattened_layout(self) -> List[tf.Tensor]:
    return self.hash_table.flatten_layout(self.layout_embeddings)
