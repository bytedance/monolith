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

from absl import logging, flags
from enum import Enum
import os, sys, six
import types
from datetime import datetime, timedelta
from typing import Dict, List, Iterable, Callable, Optional, Union

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.data.experimental.ops import matching_files
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.data.util import convert
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.platform import resource_loader
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.framework import load_library
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
import tensorflow.python.data.experimental.service as dsvc

from monolith.native_training.hooks import ckpt_hooks
from monolith.utils import get_libops_path
from monolith.native_training.monolith_export import monolith_export
from monolith.native_training.data.feature_utils import create_item_pool, string_to_variant, \
  has_variant, kafka_resource_init, kafka_read_next
from monolith.native_training.data.feature_list import FeatureList
from monolith.native_training import native_task_context
from monolith.native_training.distribute import distributed_dataset
from monolith.native_training.runtime.ops import gen_monolith_ops
from kafka import KafkaConsumer
from threading import Thread, RLock
from queue import Queue

pb_datasource_ops = gen_monolith_ops

FLAGS = flags.FLAGS
POOL_KEY = "TF_ITEMPOOL"


class FeaturePruningType(object):
  AS_IS = 0
  PRUNING_FEATURE = 1
  PRUNING_RAW_FEATURE = 2


@monolith_export
class PbType(Enum):
  INSTANCE = 1
  EXAMPLEBATCH = 2
  EXAMPLE = 3
  PLAINTEXT = 4

  def to_name(self):
    return self.name.lower()


def _get_params(name, default=None):
  try:
    if name == 'data_type':
      attr_val = getattr(FLAGS, name)
      if attr_val:
        attr_val = attr_val.upper()
        if attr_val == 'EXAMPL_EBATCH':
          return PbType.EXAMPLEBATCH
        else:
          return PbType[attr_val]
      else:
        return default
    else:
      return getattr(FLAGS, name)
  except:
    return default


class DatasetMetaclass(type):

  def __call__(cls, *args, **kwargs):
    if kwargs.get('topics_or_files', None):
      value = kwargs['topics_or_files']
      if isinstance(value, str):
        kwargs['file_name'] = kwargs.get('file_name') or value
      else:
        kwargs['patterns'] = kwargs.get('patterns') or value
        kwargs['topics'] = kwargs.get('topics') or value
    if kwargs.get('buffer_size_or_group_id', None):
      value = kwargs['buffer_size_or_group_id']
      if isinstance(value, int):
        kwargs['buffer_size'] = kwargs.get('buffer_size') or value
      else:
        kwargs['group_id'] = kwargs.get('group_id') or value
    if kwargs.get('input_pb_type_or_servers', None):
      value = kwargs['input_pb_type_or_servers']
      if isinstance(value, (str, list)):
        kwargs['servers'] = kwargs.get('servers') or value
      else:
        kwargs['input_pb_type'] = kwargs.get('input_pb_type') or value

    try:
      # the first param is str, batch to streaming, use kafka params for cmd
      args = [
          kwargs.pop('topics', FLAGS.kafka_topics.split(',')),
          kwargs.pop('group_id', FLAGS.kafka_group_id),
          kwargs.pop('servers', FLAGS.kafka_servers)
      ]
      assert all(x is not None for x in args)
      logging.info('use KafkaDataset!')
      return KafkaDataset(*args, **kwargs)
    except:
      logging.info("it's not streaming training")

    if args is None or len(args) == 0:
      if 'patterns' in kwargs and 'group_id' not in kwargs and 'servers' not in kwargs:
        logging.info('use DistributedFilePBDataset!')
        return DistributedFilePBDataset(**kwargs)
      elif 'topics' in kwargs and 'group_id' in kwargs and 'servers' in kwargs:
        logging.info('use KafkaDataset!')
        return KafkaDataset(**kwargs)
      elif 'file_name' in kwargs or len(kwargs) == 0:
        return FilePBDataset(*args, **kwargs)
      else:
        return super(DatasetMetaclass, cls).__call__(*args, **kwargs)
    elif isinstance(args[0], str):
      logging.info('use FilePBDataset!')
      return FilePBDataset(*args, **kwargs)
    elif isinstance(args[0], (list, tuple)):
      if len(args) > 1:
        if isinstance(args[1], str):
          logging.info('use KafkaDataset!')
          return KafkaDataset(*args, **kwargs)
        else:
          logging.info('use DistributedFilePBDataset!')
          return DistributedFilePBDataset(*args, **kwargs)
      else:
        if 'group_id' in kwargs or 'servers' in kwargs:
          logging.info('use KafkaDataset!')
          return KafkaDataset(*args, **kwargs)
        else:
          logging.info('use DistributedFilePBDataset!')
          return DistributedFilePBDataset(*args, **kwargs)
    else:
      return super(DatasetMetaclass, cls).__call__(*args, **kwargs)


class PBDataset(metaclass=DatasetMetaclass):

  def __init__(
      self,
      topics_or_files: Union[str, List[str]] = '',
      buffer_size_or_group_id: Union[int, str] = None,
      input_pb_type_or_servers: Union[PbType, str] = None,
      output_pb_type: PbType = None,
      feature_pruning_type: int = FeaturePruningType.PRUNING_RAW_FEATURE,
      disable_iterator_save_restore: bool = True,
      *,
      has_header=True,
      variant_type: str = None,
      stream_timeout=-1,
      message_poll_timeout=10000,
      poll_batch_size: int = 1024,
      filter_empty: bool = False,
      configuration=None,
      container: str = '',
      shared_name: str = '',
      use_data_service: bool = False,
      cycle_length=None,
      block_length=None,
      num_parallel_calls=None,
      deterministic=None,
      **kwargs):
    pass

  @classmethod
  def gen_patterns(cls,
                   input_path: str = None,
                   start_date: int = None,
                   start_hour: int = None,
                   end_date: int = None,
                   end_hour: int = None,
                   is_hourly: bool = False,
                   wildcard: str = '*') -> List[str]:
    input_path = input_path or _get_params('input_path', None)
    if not input_path:
      return []

    start_date = start_date or _get_params('start_date', None)
    if not start_date:
      return []
    end_date = end_date or _get_params('end_date', None)
    if not end_date:
      end_date = datetime.today().strftime('%Y%m%d')

    is_hourly = is_hourly if is_hourly is not None else _get_params(
        'is_hourly', False)
    start_hour = start_hour or _get_params('start_hour', 0) or 0
    end_hour = end_hour or _get_params('end_hour', 0) or 0
    wildcard = wildcard or _get_params('wildcard', '*')
    start = datetime.strptime(f'{start_date}:{start_hour:02d}', '%Y%m%d:%H')
    if is_hourly:
      end = datetime.strptime(f'{end_date}:{end_hour:02d}', '%Y%m%d:%H')
    else:
      end = datetime.strptime(f'{end_date}:00', '%Y%m%d:%H')

    delta = timedelta(hours=1) if is_hourly else timedelta(days=1)

    cur = start
    patterns = []
    while cur < end:
      if is_hourly:
        pat = f"{cur.strftime('%Y%m%d/%H')}{wildcard}"
      else:
        pat = os.path.join(cur.strftime('%Y%m%d'), wildcard)
      patterns.append(os.path.join(input_path, pat))
      cur = cur + delta

    return patterns


class DynamicMatchingFilesDataset(dataset_ops.DatasetSource):
  """A `Dataset` that list the files according to the input patterns."""

  def __init__(self, patterns: List[str]):
    assert patterns is not None and len(patterns) > 0
    self._patterns = ops.convert_to_tensor(patterns,
                                           dtype=dtypes.string,
                                           name="patterns")
    variant_tensor = pb_datasource_ops.dynamic_matching_files_dataset(
        self._patterns)
    super(DynamicMatchingFilesDataset, self).__init__(variant_tensor)

  @property
  def element_spec(self):
    return tensor_spec.TensorSpec([], dtypes.string)


@monolith_export
class FilePBDataset(dataset_ops.DatasetSource):
  """从标准输入/pb文件中读取序列化数据, 并将其反序列化存于TF的Variant类型中. 这样做的好处是可以直接对PB对象进行过滤与修改, 
  不用等到parse以后. Monolith提供了一系列工具操作Variant变量, 如filter_by_fids, filter_by_value, negative_sample等
  
  另外, InstanceReweightDataset/NegativeGenDataset 这些DataSet也可以直接作用于Variant

  Args:
    file_name (:obj:`str`): 文件名, 如果为空, 则从stdin读取数据
    buffer_size (:obj:`int`): 读取文件时缓存大小, 默认100MB
    input_pb_type (:obj:`str`): 输入pb类型, 可以是example/example_batch/instance
    output_pb_type (:obj:`str`): 输入pb类型, 可以是example/instance/plaintext
    
  Raises:
    TypeError: 如果有任何参数与类型不匹配, 则抛TypeError
    ValueError: 如果有任何值与期望不匹配, 则抛ValueError
  
  """

  def __init__(
      self,
      file_name: str = "",
      buffer_size: int = None,
      input_pb_type: PbType = None,
      output_pb_type: PbType = None,
      feature_pruning_type: int = FeaturePruningType.PRUNING_RAW_FEATURE,
      disable_iterator_save_restore: bool = True,
      use_snappy: bool = None,
      **kwargs):

    input_pb_type = input_pb_type or _get_params('data_type', PbType.INSTANCE)
    output_pb_type = output_pb_type or (PbType.INSTANCE if input_pb_type
                                        == PbType.INSTANCE else PbType.EXAMPLE)

    feature_name_list = []
    feature_id_list = []
    if input_pb_type in [PbType.EXAMPLEBATCH, PbType.EXAMPLE]:
      try:
        feature_list = FeatureList.parse()
        for feature in feature_list:
          name, slot = feature.feature_name, feature.slot
          assert None not in [name, slot]
          feature_name_list.append(name)
          feature_id_list.append(slot)
      except Exception as e:
        logging.warning('Failed to parse feature_list.conf, %s', e)

    self._file_name = file_name
    self._buffer_size = buffer_size
    self._input_pb_type = input_pb_type
    self._output_pb_type = output_pb_type
    self._out_type = tf.string if output_pb_type == PbType.PLAINTEXT else tf.variant

    self._has_sort_id = kwargs.get('has_sort_id', _get_params('sort_id', True))
    self._kafka_dump = kwargs.get('kafka_dump',
                                  _get_params('kafka_dump', False))
    logging.info('input_pb_type: %s, kafka_dump: %s, output_pb_type: %s',
                 self._input_pb_type, self._kafka_dump, self._output_pb_type)
    self._kafka_dump_prefix = kwargs.get(
        'kafka_dump_prefix', _get_params('kafka_dump_prefix', False))
    self._lagrangex_header = kwargs.get('lagrangex_header',
                                        _get_params('lagrangex_header', False))

    if disable_iterator_save_restore and isinstance(file_name, str):
      # This is the special case that dataset uses stdin as the input.
      # In this case, we should diable the ckpt save/restore.
      if context.default_execution_mode == context.GRAPH_MODE:
        ckpt_hooks.disable_iterator_save_restore()

    default_buffer_size = 128 * 1024 * 1024 if input_pb_type == PbType.EXAMPLEBATCH else 64 * 1024 * 1024
    if use_snappy is None:
      if isinstance(file_name, str):
        use_snappy = file_name.endswith('.snappy')
    assert use_snappy is not None
    variant_tensor = pb_datasource_ops.pb_dataset(
        file_name=file_name,
        use_snappy=use_snappy,
        buffer_size=buffer_size or default_buffer_size,
        input_pb_type=input_pb_type.to_name(),
        output_pb_type=output_pb_type.to_name(),
        has_sort_id=self._has_sort_id,
        kafka_dump=self._kafka_dump,
        kafka_dump_prefix=self._kafka_dump_prefix,
        lagrangex_header=self._lagrangex_header,
        feature_pruning_type=feature_pruning_type,
        feature_name_list=feature_name_list,
        feature_id_list=feature_id_list,
        out_type=self._out_type,
    )
    logging.info("Start init of the pb instance dataset base.")
    super().__init__(variant_tensor)

  @property
  def element_spec(self):
    return tensor_spec.TensorSpec([], self._out_type)


class DistributedFilePBDataset(dataset_ops.DatasetSource):

  def __init__(
      self,
      patterns: Union[str, List[str]],
      use_snappy=False,
      buffer_size: int = None,
      input_pb_type: PbType = None,
      output_pb_type: PbType = None,
      feature_pruning_type: int = FeaturePruningType.PRUNING_RAW_FEATURE,
      exclude_fn: Callable[[tf.Tensor], bool] = None,
      use_data_service: bool = False,
      cycle_length=None,
      block_length=None,
      num_parallel_calls=None,
      deterministic=None,
      **kwargs):
    if not patterns:
      patterns = [""]
    elif isinstance(patterns, str):
      patterns = [patterns]
    else:
      logging.info(f'patterns: {patterns}')
    patterns.sort()
    enable_dynamic_sharding = kwargs.get(
        'enable_dynamic_sharding', _get_params('enable_dynamic_sharding',
                                               False))
    logging.info(f"enable_dynamic_sharding: {enable_dynamic_sharding}")

    map_func = lambda file_name: FilePBDataset(
        file_name=file_name,
        use_snappy=use_snappy,
        buffer_size=buffer_size,
        input_pb_type=input_pb_type,
        output_pb_type=output_pb_type,
        feature_pruning_type=feature_pruning_type,
        disable_iterator_save_restore=not enable_dynamic_sharding,
        **kwargs)
    if use_data_service:
      files_list = DynamicMatchingFilesDataset(patterns)
      if exclude_fn is not None:
        files_list = files_list.filter(predicate=exclude_fn)
      dataset = files_list.interleave(map_func,
                                      cycle_length=cycle_length,
                                      block_length=block_length,
                                      num_parallel_calls=num_parallel_calls,
                                      deterministic=deterministic)
    elif enable_dynamic_sharding:
      files_list = distributed_dataset.create_dynamic_sharding_dataset(patterns)
      if exclude_fn is not None:
        files_list = files_list.filter(predicate=exclude_fn)
      dataset = files_list.flat_map(map_func)
    else:
      files_list = matching_files.MatchingFilesDataset(patterns)
      if exclude_fn is not None:
        files_list = files_list.filter(predicate=exclude_fn)
      ctx = native_task_context.get()
      if ctx is not None:
        if ctx.num_workers > 1:
          files_list = files_list.shard(ctx.num_workers, ctx.worker_index)
      else:
        shard_num = kwargs.get('shard_num', 1)
        shard_index = kwargs.get('shard_index', 0)
        if shard_num > 1:
          files_list = files_list.shard(shard_num, shard_index)

      cycle_length = kwargs.get('cycle_length',
                                _get_params('max_task_num_per_worker', 4))
      num_parallel_calls = kwargs.get('num_parallel_calls',
                                      _get_params('max_task_num_per_worker', 4))
      block_length = kwargs.get('block_length', _get_params('block_length', 1))
      dataset = files_list.interleave(map_func=map_func,
                                      cycle_length=cycle_length,
                                      block_length=block_length,
                                      num_parallel_calls=num_parallel_calls,
                                      deterministic=False)
    self._dataset = dataset
    super(DistributedFilePBDataset,
          self).__init__(variant_tensor=self._dataset._variant_tensor)

  @property
  def element_spec(self):
    return self._dataset.element_spec


@monolith_export
class InstanceReweightDataset(dataset_ops.UnaryUnchangedStructureDataset):
  """样本重加权, 并根据action给样本打标签, 使用方式为 dataset.instance_reweight
  
  一个样本可能有多个action, 按`action_priority`, 找到最高优的action. 再用action找到对应的 `action:weight:label`, 
  让样本重复weight次(也有可能是0次, 即删除样本), 然后给样本打上label指定的标签 

  Args:
    input_dataset (:obj:`dataset`): 输入数据集
    action_priority (:obj:`str`): action用int表示, 以逗号分隔的int数组, 排在前面的优先级高
    reweight (:obj:`str`): 基本单元是`action:weight:label`, 可以用逗号分隔多个基本单元
      1) action: 动作, 用int表示, 与业务相关, 如download, install, click, exposure等
      2) weight: 权重, 用int表示, 表示样本重复的次数
      3) label: 标签, 一般用1/-1表示. 
    variant_type (:obj:`str`): 输入数据是variant类型的, 支持两种格式, instance/example
    
  Raises:
    TypeError: 如果有任何参数与类型不匹配, 则抛TypeError
    ValueError: 如果有任何值与期望不匹配, 则抛ValueError
  
  """

  def __init__(self,
               input_dataset,
               action_priority: str = None,
               reweight: str = None,
               variant_type: str = 'example'):
    self._label_priority = action_priority
    self._reweight = reweight
    self._variant_type = variant_type

    actions, weights, labels = [], [], []
    for item in reweight.strip().split(','):
      (action, weight, label) = item.strip().split(':')
      actions.append(int(action))
      weights.append(int(weight))
      labels.append(int(label))

    priorities = [int(p) for p in action_priority.strip().split(',')]
    variant_tensor = pb_datasource_ops.instance_reweight_dataset(
        input=input_dataset._variant_tensor,
        method=0,
        actions=actions,
        weights=weights,
        labels=labels,
        priorities=priorities,
        variant_type=variant_type)
    logging.info("Start init of the pb instance dataset base.")
    super(InstanceReweightDataset, self).__init__(input_dataset, variant_tensor)

  @property
  def element_spec(self):
    return tensor_spec.TensorSpec([], dtypes.variant)


@monolith_export
class NegativeGenDataset(dataset_ops.UnaryUnchangedStructureDataset):
  """负例生成. 有时, 样本中只有正例, 没有负例, 需要随机生成负例
  
  推荐系统中的样本通常是由user侧, item侧两部分组成. 这里的做法是: 
    - 先收集每个样本的item侧信息, 生成一个item池子
    - item池子并不是平铺的, 而是按某个特征(channel_slot)分类组织的. 如果在同一个channel随机取item得到的是hard负例, 在其它channel中抽样得到的是easy负例
    - 并不是一开始就生成负例, 而是要等item池子积累到一定大小才开始生成负例

  Args:
    input_dataset (:obj:`dataset`): 输入数据集
    neg_num (:obj:`int`): 为一个正例生成`neg_num`个负例
    channel_feature (:obj:`string`): 用于当item分类的字段
    per_channel (:obj:`bool`): 是否分类 
    start_num (:obj:`int`): 在item池子中积累多少个后才开始采样
    max_iten_num (:obj:`int`): 每一个channel最多收集多注个item
    item_features: (:obj:`List[str]`): item侧的特征名列表
    positive_label: 正例的label, 仅为正例生成负例
    negative_label: 生成的负例的被打上的label
    hard_easy_ratio: (:obj:`float`): 当使用 per_channel 的时候, hard和easy负例之间的比例。取值在 0 ~ 1 之间。举例：0.8就是大致80% easy负例

  Raises:
    TypeError: 如果有任何参数与类型不匹配, 则抛TypeError
    ValueError: 如果有任何值与期望不匹配, 则抛ValueError
  
  """

  def __init__(self,
               input_dataset,
               neg_num: int,
               per_channel: bool = False,
               channel_feature: Union[int, str] = '',
               item_features: Union[List[int], List[str]] = [],
               start_num: int = 500,
               max_item_num: int = 100000,
               positive_label: int = 1,
               negative_label: int = -1,
               negative_action: int = -99999,
               positive_actions: List[int] = [],
               label_index: int = 0,
               action_priority: str = '',
               index_feature: Union[int, str] = '',
               throw_origin: bool = False,
               throw_origin_neg: bool = False,
               cache_only_pos: bool = True,
               real_neg_instance_weight: float = 1.0,
               sampled_neg_instance_weight: float = -1.0,
               unbias_sampled_neg: bool = True,
               origin_neg_in_pool_proba: float = 1.0,
               neg_sample_declay_factor: float = 1.0,
               hard_easy_ratio: float = 0.0,
               variant_type: str = 'example'):
    pool = create_item_pool(start_num=start_num,
                            max_item_num_per_channel=max_item_num)
    tf.compat.v1.add_to_collection(POOL_KEY, pool)
    channel_feature = str(channel_feature)
    item_features = [str(item) for item in item_features]
    action_priority_items = action_priority.strip().split(',')
    assert len(action_priority_items) == len(set(action_priority_items))
    index_feature = str(index_feature)
    assert variant_type in {'instance', 'example'}
    assert label_index >= 0

    variant_tensor = pb_datasource_ops.instance_negative_gen_dataset(
        input=input_dataset._variant_tensor,
        pool=pool,
        neg_num=neg_num,
        per_channel=per_channel,
        channel_feature=channel_feature,
        item_features=item_features,
        label_index=label_index,
        positive_label=positive_label,
        negative_label=negative_label,
        negative_action=negative_action,
        action_priority=action_priority,
        positive_actions=positive_actions,
        index_feature=index_feature,
        throw_origin=throw_origin,
        throw_origin_neg=throw_origin_neg,
        cache_only_pos=cache_only_pos,
        real_neg_instance_weight=real_neg_instance_weight,
        sampled_neg_instance_weight=sampled_neg_instance_weight,
        unbias_sampled_neg=unbias_sampled_neg,
        origin_neg_in_pool_proba=origin_neg_in_pool_proba,
        neg_sample_declay_factor=neg_sample_declay_factor,
        hard_easy_ratio=hard_easy_ratio,
        variant_type=variant_type)
    super(NegativeGenDataset, self).__init__(input_dataset, variant_tensor)

  @property
  def element_spec(self):
    return tensor_spec.TensorSpec([], dtypes.variant)


def instance_reweight(self,
                      action_priority: str,
                      reweight: str,
                      variant_type: str = 'example'):
  return InstanceReweightDataset(self,
                                 action_priority,
                                 reweight,
                                 variant_type=variant_type)


@monolith_export
class SplitFlowDataset(dataset_ops.UnaryUnchangedStructureDataset):

  def __init__(self,
               input_dataset,
               data_flow: List[str],
               index: int,
               max_queue_size: int = 1024,
               variant_type: str = 'example'):
    variant_tensor = pb_datasource_ops.split_flow_dataset(
        input_dataset._variant_tensor,
        data_flow=data_flow,
        index=index,
        max_queue_size=max_queue_size,
        variant_type=variant_type)
    super(SplitFlowDataset, self).__init__(input_dataset, variant_tensor)

  @property
  def element_spec(self):
    return tensor_spec.TensorSpec([], dtypes.variant)


@monolith_export
class MergeFlowDataset(dataset_ops.DatasetV2):

  def __init__(self,
               input_dataset,
               dataset_to_merge,
               max_queue_size: int = 1024,
               variant_type: str = 'example'):
    self._input_dataset = input_dataset
    self._dataset_to_merge = dataset_to_merge

    output_types = dataset_ops.get_legacy_output_types(input_dataset)
    for ds in dataset_to_merge:
      ds_types = dataset_ops.get_legacy_output_types(ds)
      if output_types != ds_types:
        raise TypeError("Datasets to merge have different types %s and %s" %
                        (output_types, ds_types))

    input_shapes = dataset_ops.get_legacy_output_shapes(input_dataset)

    flat_sequence = None
    input_shapes_flatten = nest.flatten(input_shapes)
    for ds in dataset_to_merge:
      ds_shapes_flatten = nest.flatten(dataset_ops.get_legacy_output_shapes(ds))
      if flat_sequence is None:
        flat_sequence = [
            ts1.most_specific_compatible_shape(ts2)
            for (ts1, ts2) in zip(input_shapes_flatten, ds_shapes_flatten)
        ]
      else:
        tmp = [
            ts1.most_specific_compatible_shape(ts2)
            for (ts1, ts2) in zip(input_shapes_flatten, ds_shapes_flatten)
        ]
        assert all(ts1 == ts2 for (ts1, ts2) in zip(flat_sequence, tmp))
    output_shapes = nest.pack_sequence_as(input_shapes, flat_sequence)

    output_classes = dataset_ops.get_legacy_output_classes(input_dataset)
    for ds in dataset_to_merge:
      ds_classes = dataset_ops.get_legacy_output_classes(ds)
      if output_classes != ds_classes:
        raise TypeError("Datasets to merge have different classes %s and %s" %
                        (output_classes, ds_classes))

    self._structure = structure.convert_legacy_structure(
        output_types, output_shapes, output_classes)

    self._input_datasets = [input_dataset] + dataset_to_merge
    input_dataset_variant = [ds._variant_tensor for ds in self._input_datasets]
    data_flow = ['input_ds'] + [
        'ds_to_merge_{}'.format(i + 1)
        for i in range(len(self._dataset_to_merge))
    ]
    variant_tensor = pb_datasource_ops.merge_flow_dataset(
        input_dataset_variant,
        data_flow=data_flow,
        max_queue_size=max_queue_size,
        variant_type=variant_type)
    super(MergeFlowDataset, self).__init__(variant_tensor)

  def _inputs(self):
    return self._input_datasets

  @property
  def element_spec(self):
    return self._structure


def negative_gen(self,
                 neg_num: int,
                 per_channel: bool = False,
                 channel_feature: Union[int, str] = '',
                 item_features: Union[List[int], List[str]] = [],
                 start_num: int = 500,
                 max_item_num: int = 100000,
                 positive_label: int = 1,
                 negative_label: int = -1,
                 negative_action: int = -99999,
                 positive_actions: List[int] = [],
                 label_index: int = 0,
                 action_priority: str = '',
                 index_feature: Union[int, str] = '',
                 throw_origin: bool = False,
                 throw_origin_neg: bool = False,
                 cache_only_pos: bool = False,
                 real_neg_instance_weight: float = 1.0,
                 sampled_neg_instance_weight: float = -1.0,
                 unbias_sampled_neg: bool = True,
                 origin_neg_in_pool_proba: float = 1.0,
                 neg_sample_declay_factor: float = 1.0,
                 hard_easy_ratio: float = 0.0,
                 variant_type: str = 'example'):
  return NegativeGenDataset(
      self,
      neg_num=neg_num,
      per_channel=per_channel,
      channel_feature=channel_feature,
      item_features=item_features,
      start_num=start_num,
      max_item_num=max_item_num,
      label_index=label_index,
      positive_label=positive_label,
      negative_label=negative_label,
      negative_action=negative_action,
      action_priority=action_priority,
      positive_actions=positive_actions,
      index_feature=index_feature,
      throw_origin=throw_origin,
      throw_origin_neg=throw_origin_neg,
      cache_only_pos=cache_only_pos,
      real_neg_instance_weight=real_neg_instance_weight,
      sampled_neg_instance_weight=sampled_neg_instance_weight,
      unbias_sampled_neg=unbias_sampled_neg,
      origin_neg_in_pool_proba=origin_neg_in_pool_proba,
      neg_sample_declay_factor=neg_sample_declay_factor,
      hard_easy_ratio=hard_easy_ratio,
      variant_type=variant_type)


def split_flow(self,
               data_flow: List[str],
               index: int,
               max_queue_size: int = 1024,
               variant_type: str = 'example'):
  return SplitFlowDataset(self,
                          data_flow=data_flow,
                          index=index,
                          max_queue_size=max_queue_size,
                          variant_type=variant_type)


def merge_flow(self,
               dataset_to_merge,
               max_queue_size: int = 1024,
               variant_type: str = 'example'):
  return MergeFlowDataset(self,
                          dataset_to_merge,
                          max_queue_size=max_queue_size,
                          variant_type=variant_type)


class KafkaGen(object):

  def __init__(self,
               topics: List[str],
               group_id: str,
               servers: Union[str, List[str]],
               stream_timeout: int = -1,
               message_poll_timeout: int = 10000,
               poll_batch_size: int = 1024):
    if stream_timeout == -1:
      stream_timeout = sys.maxsize
    elif stream_timeout >= 0:
      stream_timeout = max(stream_timeout, message_poll_timeout)
    else:
      raise ValueError('stream_timeout must bigger then -1')

    if isinstance(topics, str):
      topics = [topics]

    self.topics, self.group_id, self.servers = topics, group_id, servers
    self._lock = RLock()
    self._stop_iteration = False  # lock
    self._consumer: KafkaConsumer = None  # lock
    self._queue = Queue(maxsize=1024)
    self.message_poll_timeout = message_poll_timeout
    self.poll_batch_size = poll_batch_size
    self._max_stream_timeout_polls = int(stream_timeout / message_poll_timeout)
    self._stream_timeout_polls = -1

  @property
  def consumer(self):
    with self._lock:
      if self._consumer is None:
        self._consumer = KafkaConsumer(*self.topics,
                                       group_id=self.group_id,
                                       bootstrap_servers=self.servers)
        thread = Thread(target=self._poll)
        thread.start()
    return self._consumer

  def __iter__(self):
    return self

  def __next__(self):
    assert self.consumer is not None
    while True:
      data = self._queue.get(timeout=self.message_poll_timeout)
      if data:
        return data
      with self._lock:
        if self._stop_iteration:
          raise StopIteration

  def __call__(self):
    return self

  def _poll(self):
    while self._stream_timeout_polls < self._max_stream_timeout_polls:
      try:
        msg = self._consumer.poll(timeout_ms=self.message_poll_timeout,
                                  max_records=self.poll_batch_size,
                                  update_offsets=True)
        if msg:
          poll_values = []
          for part, values in msg.items():
            part_vals = [value.value for value in values if value.value]
            if part_vals:
              poll_values.extend(part_vals)
          if poll_values:
            self._stream_timeout_polls = 0
            self._queue.put(poll_values)
          else:
            self._stream_timeout_polls += 1
            continue
        else:
          self._stream_timeout_polls += 1
      except Exception as e:
        logging.error(f'poll error: {e}')
        break

    with self._lock:
      self._consumer.close()
      self._stop_iteration = True


class PyKafkaDataset(dataset_ops.DatasetSource):

  def __init__(self,
               topics,
               group_id,
               servers,
               *,
               has_header=True,
               variant_type: str = None,
               stream_timeout=-1,
               message_poll_timeout=10000,
               poll_batch_size: int = 1024,
               filter_empty: bool = False,
               **kwargs):
    variant_type = variant_type or _get_params('data_type',
                                               PbType.INSTANCE).to_name()
    self._has_sort_id = kwargs.get('has_sort_id', _get_params('sort_id', False))
    self._kafka_dump = kwargs.get('kafka_dump',
                                  _get_params('kafka_dump', False))
    logging.info(f'pb_type: {variant_type}, kafka_dump: {self._kafka_dump}')
    self._kafka_dump_prefix = kwargs.get(
        'kafka_dump_prefix', _get_params('kafka_dump_prefix', False))
    self._lagrangex_header = kwargs.get('lagrangex_header',
                                        _get_params('lagrangex_header', False))

    if context.default_execution_mode == context.GRAPH_MODE:
      ckpt_hooks.disable_iterator_save_restore()

    kafka_gen = KafkaGen(topics, group_id, servers, stream_timeout,
                         message_poll_timeout, poll_batch_size)
    dataset = tf.data.Dataset.from_generator(generator=kafka_gen,
                                             output_types=tf.string,
                                             output_shapes=None)
    dataset = dataset.map(
        lambda v: string_to_variant(v,
                                    variant_type=variant_type.lower(),
                                    has_header=has_header,
                                    lagrangex_header=self._lagrangex_header,
                                    has_sort_id=self._has_sort_id,
                                    kafka_dump=self._kafka_dump,
                                    kafka_dump_prefix=self._kafka_dump_prefix),
        num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE).unbatch()
    if filter_empty:
      dataset = dataset.filter(predicate=lambda x: has_variant(
          input=x, variant_type=variant_type.lower()))

    self._dataset = dataset
    super().__init__(self._dataset._variant_tensor)

  @property
  def element_spec(self):
    return self._dataset.element_spec


class KafkaDataset(dataset_ops.DatasetSource):

  def __init__(self,
               topics: List[str],
               group_id: str,
               servers: str,
               *,
               has_header=True,
               variant_type: str = None,
               stream_timeout=-1,
               message_poll_timeout=10000,
               poll_batch_size: int = 1024,
               filter_empty: bool = False,
               configuration=None,
               container: str = '',
               shared_name: str = '',
               **kwargs):
    variant_type = variant_type or _get_params('data_type',
                                               PbType.INSTANCE).to_name()
    self._has_sort_id = kwargs.get('has_sort_id', _get_params('sort_id', False))
    self._kafka_dump = kwargs.get('kafka_dump',
                                  _get_params('kafka_dump', False))
    logging.info(f'pb_type: {variant_type}, kafka_dump: {self._kafka_dump}')
    self._kafka_dump_prefix = kwargs.get(
        'kafka_dump_prefix', _get_params('kafka_dump_prefix', False))
    self._lagrangex_header = kwargs.get('lagrangex_header',
                                        _get_params('lagrangex_header', False))

    if context.default_execution_mode == context.GRAPH_MODE:
      ckpt_hooks.disable_iterator_save_restore()
    self._chnids = kwargs.get('chnids', _get_params('chnids', None))
    self._datasources = kwargs.get('datasources',
                                   _get_params('datasources', None))
    self._default_datasource = kwargs.get('default_datasource',
                                          _get_params('default_datasource', ''))

    with tf.name_scope("MonolithKafkaDataset"):
      if stream_timeout == -1:
        stream_timeout = sys.maxsize
      elif stream_timeout >= 0:
        stream_timeout = max(stream_timeout, message_poll_timeout)
      else:
        raise ValueError(
            f"Invalid stream_timeout value: {stream_timeout} ,set it to -1 to block indefinitely."
        )
      metadata = list(configuration or [])
      if group_id is not None:
        metadata.append(f"group.id={group_id}")
      if servers is not None:
        metadata.append(f"bootstrap.servers={servers}")
      if poll_batch_size is not None:
        assert isinstance(poll_batch_size, int) and poll_batch_size > 0
        metadata.append(f"batch.num.messages={poll_batch_size}")

      resource = kafka_resource_init(topics=topics,
                                     metadata=metadata,
                                     container=container,
                                     shared_name=shared_name)
      self._resource = resource

      dataset = tf.data.experimental.Counter()
      dataset = dataset.map(lambda i: kafka_read_next(
          input=self._resource,
          index=i,
          message_poll_timeout=message_poll_timeout,
          stream_timeout=stream_timeout,
      ))
      dataset = dataset.apply(
          tf.data.experimental.take_while(
              lambda v: tf.greater(v.continue_fetch, 0)))

      dataset = dataset.map(lambda v: string_to_variant(
          v.message,
          variant_type=variant_type.lower(),
          has_header=has_header,
          lagrangex_header=self._lagrangex_header,
          has_sort_id=self._has_sort_id,
          kafka_dump=self._kafka_dump,
          kafka_dump_prefix=self._kafka_dump_prefix,
          chnids=self._chnids,
          datasources=self._datasources,
          default_datasource=self._default_datasource),
                            num_parallel_calls=tf.data.AUTOTUNE)
      dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE).unbatch()
      if filter_empty:
        dataset = dataset.filter(predicate=lambda x: has_variant(
            input=x, variant_type=variant_type.lower()))

      self._dataset = dataset
      super().__init__(self._dataset._variant_tensor)

  @property
  def element_spec(self):
    return self._dataset.element_spec


def distribute(self,
               target,
               *,
               job_name: str,
               num_worker: int,
               worker_idx: int,
               queue_device: str = "/device:CPU:0",
               max_outstanding_requests: int = None):
  try:
    if FLAGS.kafka_topics is not None and FLAGS.kafka_group_id is not None:
      return self
  except Exception as e:
    pass

  element_spec = self.element_spec
  with tf.compat.v1.device(queue_device):
    queue = tf.compat.v1.FIFOQueue(capacity=num_worker - 1,
                                   dtypes=[tf.int64],
                                   shared_name=f'{job_name}_queue')

  if worker_idx == 0:
    # data service try to register dataset, if the dataset has been registed, return dataset_id drectily
    # that means get or register dataset. for data parallel, the data pipeline assure to be identity
    # here we ues queue to ensure the same data pipeline for a job
    dataset_id = dsvc.register_dataset(target, self)
    enqueue_op = queue.enqueue_many(vals=[dataset_id] * (num_worker - 1))
    with tf.compat.v1.control_dependencies(control_inputs[enqueue_op]):
      # to share pipeline, job_name must be specified
      return dsvc.from_dataset_id(
          processing_mode="distributed_epoch",
          service=target,
          dataset_id=dataset_id,
          job_name=job_name,
          element_spec=element_spec,
          max_outstanding_requests=max_outstanding_requests)
  else:
    dataset_id = queue.dequeue()
    return dsvc.from_dataset_id(
        processing_mode="distributed_epoch",
        service=target,
        dataset_id=dataset_id,
        job_name=job_name,
        element_spec=element_spec,
        max_outstanding_requests=max_outstanding_requests)


Dataset.instance_reweight = instance_reweight
Dataset.negative_gen = negative_gen
Dataset.split_flow = split_flow
Dataset.merge_flow = merge_flow
Dataset.distribute = distribute
