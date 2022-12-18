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

import string
import numpy as np
from typing import Any, List, Union, Dict

import tensorflow as tf

from monolith.utils import get_libops_path
from monolith.native_training.monolith_export import monolith_export
from monolith.native_training.runtime.ops import gen_monolith_ops
from idl.matrix.proto.line_id_pb2 import LineId
from monolith.native_training.data.data_op_config_pb2 import LabelConf, \
  TaskLabelConf

ragged_data_ops = gen_monolith_ops


@monolith_export
def filter_by_fids(variant: tf.Tensor,
                   filter_fids: List[int] = None,
                   has_fids: List[int] = None,
                   select_fids: List[int] = None,
                   has_actions: List[int] = None,
                   req_time_min: int = 0,
                   select_slots: List[int] = None,
                   variant_type: str = 'instance'):
  """通过特征ID (FID) 过滤, 离散特征过滤
  
  Args:
    variant (:obj:`Tensor`): 输入数据, 必须是variant类型
    filter_fids (:obj:`List[int]`): 任意一个FID出现`filter_fids`中, 样本被过滤
    has_fids (:obj:`List[int]`): 任意一个FID出现在`has_fids`中, 则样本被选择
    select_fids (:obj:`List[int]`): 所有`select_fids`均出现在样本中, 则样本被选择
    has_actions (:obj:`List[int]`): 任意一个action出现在`has_actions`中, 则样本被选择
    req_time_min (:obj:`int`): 请求时间最小值
    select_slots (:obj:`List[int]`): 所有`select_slots`均出现在样本中, 样本才被选择
    variant_type (:obj:`str`): variant类型, 可以为instance/example
  
  Returns:
    variant tensor, 过滤后的数据, variant类型
  
  """

  filter_fids = [] if filter_fids is None else [
      np.uint64(fid).astype(np.int64) for fid in filter_fids
  ]
  has_fids = [] if has_fids is None else [
      np.uint64(fid).astype(np.int64) for fid in has_fids
  ]
  select_fids = [] if select_fids is None else [
      np.uint64(fid).astype(np.int64) for fid in select_fids
  ]
  select_slots = [] if select_slots is None else select_slots
  assert all([slot > 0 for slot in select_slots])

  return ragged_data_ops.set_filter(variant, filter_fids, has_fids, select_fids,
                                    has_actions or [], req_time_min,
                                    select_slots, variant_type)


@monolith_export
def filter_by_value(variant: tf.Tensor,
                    field_name: str,
                    op: str,
                    operand: Union[float, int, str, List[float], List[int],
                                   List[str]],
                    variant_type: str = 'instance',
                    keep_empty: bool = False,
                    operand_filepath: str = None):
  """通过值过滤, 连续特征过滤, 
  
  Args:
    variant (:obj:`Tensor`): 输入数据, 必须是variant类型
    field_name (:obj:`List[int]`): 任意一个FID出现`filter_fids`中, 样本被过滤
    op (:obj:`str`): 比较运算符, 可以是 gt/ge/eq/lt/le/neq/between/in/not-in 等
      布尔运算，也可以是 all/any/diff 等集合布尔运算
    operand (:obj:`float`): 操作数, 用于比较
    variant_type (:obj:`str`): variant类型, 可以为instance/example
    keep_empty (:obj:`bool`): False

  Returns:
    variant tensor, 过滤后的数据, variant类型
  """

  assert op in {
      'gt', 'ge', 'eq', 'lt', 'le', 'neq', 'between', 'in', 'not-in', 'all',
      'any', 'diff', 'startswith', 'endswith'
  }
  fields = LineId.DESCRIPTOR.fields_by_name
  assert field_name in fields
  assert (operand is None and operand_filepath) or (operand is not None and
                                                    not operand_filepath)
  field = fields[field_name]
  string_operand = []
  operand_filepath = '' if operand_filepath is None else operand_filepath
  if operand_filepath:
    assert op in {'in', 'not-in'}
    assert (isinstance(operand_filepath, str) and
            tf.io.gfile.exists(operand_filepath))
    int_operand, float_operand = [], []
  elif field.has_options:
    assert op in {'all', 'any', 'diff'}
    assert field.cpp_type in {
        field.CPPTYPE_INT32, field.CPPTYPE_INT64, field.CPPTYPE_UINT32,
        field.CPPTYPE_UINT64
    }
    if not isinstance(operand, (list, tuple)):
      assert isinstance(operand, int)
      int_operand, float_operand = [operand], []
    else:
      assert all(isinstance(o, int) for o in operand)
      int_operand, float_operand = list(operand), []
  elif field.cpp_type in {field.CPPTYPE_DOUBLE, field.CPPTYPE_FLOAT}:
    if op == 'between':
      assert all(isinstance(o, (int, float)) for o in operand)
      int_operand, float_operand = [], [float(o) for o in operand]
    else:
      int_operand, float_operand = [], [float(operand)]
  elif field.cpp_type in {
      field.CPPTYPE_INT32, field.CPPTYPE_INT64, field.CPPTYPE_UINT32,
      field.CPPTYPE_UINT64
  }:
    if op in {'in', 'not-in', 'between'}:
      assert all(isinstance(o, int) for o in operand)
      int_operand, float_operand = list(operand), []
    else:
      int_operand, float_operand = [int(operand)], []
  elif field.cpp_type == field.CPPTYPE_STRING:
    int_operand, float_operand = [], []
    if isinstance(operand, str):
      string_operand.append(operand)
    elif isinstance(operand, (list, tuple)):
      assert all(isinstance(o, str) for o in operand)
      string_operand.extend(operand)
    else:
      raise RuntimeError("params error!")
  else:
    raise RuntimeError("params error!")

  return ragged_data_ops.value_filter(variant,
                                      field_name=field_name,
                                      op=op,
                                      float_operand=float_operand,
                                      int_operand=int_operand,
                                      string_operand=string_operand,
                                      operand_filepath=operand_filepath,
                                      keep_empty=keep_empty,
                                      variant_type=variant_type)


@monolith_export
def add_action(
    variant: tf.Tensor,
    field_name: str,
    op: str,
    operand: Union[float, int, str, List[float], List[int], List[str]],
    action: int,
    variant_type: str = 'example',
):
  """根据指定 LineId 字段经过简单的关系运算，决定是否为 actions 字段增加值

  Args:
    variant (:obj:`Tensor`): 输入数据，必须是 variant 类型
    field_name (:obj:`List[int]`): 根据 field_name 对应值进行条件判断
    op (:obj:`str`): 比较运算符，可以是 gt/ge/eq/lt/le/neq/between/in
    operand (:obj:`float`): 操作数，用于比较
    action (:obj:`int`): 当条件满足时，需要往 LineId.actions 添加的值
    variant_type (:obj:`str`): 'instance' 或 'example'

  Returns:
    variant tensor, 改写后的数据，variant 类型
  """

  assert op in {'gt', 'ge', 'eq', 'lt', 'le', 'neq', 'between', 'in'}
  assert variant_type in {'instance', 'example'}
  fields = LineId.DESCRIPTOR.fields_by_name
  assert field_name in fields
  field = fields[field_name]
  string_operand = []

  if field.cpp_type in {field.CPPTYPE_DOUBLE, field.CPPTYPE_FLOAT}:
    if op == 'between':
      assert all(isinstance(o, (int, float)) for o in operand)
      int_operand, float_operand = [], [float(o) for o in operand]
    else:
      int_operand, float_operand = [], [float(operand)]
  elif field.cpp_type in {
      field.CPPTYPE_INT32, field.CPPTYPE_INT64, field.CPPTYPE_UINT32,
      field.CPPTYPE_UINT64
  }:
    if op in {'in', 'between'}:
      assert all(isinstance(o, int) for o in operand)
      int_operand, float_operand = list(operand), []
    else:
      int_operand, float_operand = [int(operand)], []
  elif field.cpp_type == field.CPPTYPE_STRING:
    int_operand, float_operand = [], []
    if isinstance(operand, str):
      string_operand.append(operand)
    elif isinstance(operand, (list, tuple)):
      assert all(isinstance(o, str) for o in operand)
      string_operand.extend(operand)
    else:
      raise RuntimeError("params error!")
  else:
    raise RuntimeError("params error!")

  return ragged_data_ops.add_action(variant,
                                    field_name=field_name,
                                    op=op,
                                    float_operand=float_operand,
                                    int_operand=int_operand,
                                    string_operand=string_operand,
                                    actions=[action],
                                    variant_type=variant_type)


@monolith_export
def add_label(
    variant: tf.Tensor,
    config: str,
    negative_value: float,
    new_sample_rate: float,
    variant_type: str = 'example',
):
  """根据给定配置决定是否添加 label，支持 multi-task label 生成，请务必配合
     filter_by_label 过滤算子同时使用，否则可能会有无效样本被喂入训练器。
  举例 config='1,2:3:1.0;4::0.5'，表示一共有两个 task（;分隔），
  task1 pos_actions = {1,2}, neg_actions = {3}, sample_rate = 1.0，而
  task2 pos_actions = {4}, neg_actions 为空，sample_rate = 0.5
  add_label 的执行逻辑如下
    - 对于 task1，如果当前样本的 actions 包含 {1, 2} 任一个则判定为正例，否则根据给定
      采样率决定是否采样（sample_rate < 1.0 方可触发采样），若触发采样且在采样范围内
      标为负例，不在采样范围内置为无效 label，若未触发采样直接标记为负例。这个例子里由于
      task1 的 sample_rate=1.0，因此不会触发负采样
    - 对于 task2，如果当前样本的 actions 包含 {4} 则判定为正例，由于未指定 neg_actions
      对于不包含 {4} 的样本直接进行负采样，在采样范围内标为负例，不在采样范围内置为
      无效 label。这个例子里由于 task2 的 sample_rate=0.5，因此会对于不包含 {4} 的样本
      触发负采样

  Args:
    variant (:obj:`Tensor`): 输入数据，必须是 variant 类型
    config (:obj:`str`): 形如 '1,2:3:1.0;4::0.5'
    negative_value (:obj:`float`): 如 -1.0 或 0.0
    new_sample_rate (:obj:`float`): 为 LineId.sample_rate 赋值
    variant_type (:obj:`str`): 'instance' 或 'example'

  Returns:
    variant tensor, 改写后的数据，variant 类型
  """

  assert variant_type in {'instance', 'example'}
  assert config, 'Please specify config and retry!'
  assert 0 < new_sample_rate <= 1.0, 'new_sample_rate should be in (0, 1.0]'

  label_conf = LabelConf()
  for task in config.split(';'):
    # skip empty parts, e.g. config = '1,2:3:1.0;'
    if len(task) == 0:
      continue

    task_conf = label_conf.conf.add()
    pos_actions, neg_actions, sample_rate = task.split(':')
    pos_actions_list = [
        int(pos) for pos in pos_actions.split(',') if len(pos) > 0
    ]
    neg_actions_list = [
        int(neg) for neg in neg_actions.split(',') if len(neg) > 0
    ]
    task_conf.pos_actions.extend(pos_actions_list)
    task_conf.neg_actions.extend(neg_actions_list)
    task_conf.sample_rate = float(sample_rate)

  return ragged_data_ops.add_label(variant,
                                   config=label_conf.SerializeToString(),
                                   negative_value=negative_value,
                                   sample_rate=new_sample_rate,
                                   variant_type=variant_type)


@monolith_export
def scatter_label(
    variant: tf.Tensor,
    config: str,
    variant_type: str = 'example',
):
  """根据给定配置 scatter label 以支持 multi-task label 生成，配置形如
  'chnid0:index0,chnid1:index1'，请务必配合 filter_by_label 过滤算子使用，
  否则可能会有无效样本被喂入训练器。举例 config='100:3,200:1,300:4'，
  表示一共有 5 个 task（最大的 index=4），scatter_label 的执行逻辑如下
    1. 获取 label_value = label[0]，亦即默认待处理样本的 label.size() > 0
    2. 重置待处理样本的 label 长度为 5，并全部初始化为 INVALID_LABEL
    3. if 样本的 chnid = 100，label[3] = label_value
    4. else if 样本的 chnid = 200，label[1] = label_value
    5. else if 样本的 chnid = 300，label[4] = label_value
    6. else 样本的 chnid not in {100, 200, 300}，则 label 中全部值为 INVALID_LABEL

  Args:
    variant (:obj:`Tensor`): 输入数据，必须是 variant 类型
    config (:obj:`str`): 形如 '100:3,200:1,300:4'
    variant_type (:obj:`str`): 'instance' 或 'example'

  Returns:
    variant tensor, 改写后的数据，variant 类型
  """

  assert variant_type in {'instance', 'example'}
  assert config, 'Please specify config and retry!'

  return ragged_data_ops.scatter_label(variant,
                                       config=config,
                                       variant_type=variant_type)


@monolith_export
def filter_by_label(
    variant: tf.Tensor,
    label_threshold: List[float],
    filter_equal: bool = False,
    variant_type: str = 'example',
) -> bool:
  """根据给定配置决定是否保留当前样本，支持 multi-task

  Args:
    variant (:obj:`Tensor`): 输入数据，必须是 variant 类型
    label_threshold (:obj:`List[float]`): 样本任一 label 值 >= 相应 label_threshold
    值则样本被保留，否则被丢弃。举例 label_threshold = [-100.0, 0.0]，假设样本
      - label = [-1000, -1]，则该样本被丢弃，即不存在任何合法 label 值
      - label = [-1000, 0]，则该样本被保留，即第 2 个 label 值合法
      - label = [-1, -1]，则该样本被保留，即第 1 个 label 值合法
      - label = [-1, 1]，则该样本被保留，即第 1, 2 个 label 值均合法
    filter_equal (:obj:`bool`): Whether to filter when label equals to threshold.
    variant_type (:obj:`str`): 'instance' 或 'example'

  Returns:
    valid tensor, 是否保留当前样本
  """

  assert variant_type in {'instance', 'example'}
  assert len(label_threshold) > 0, 'Please specify label_threshold and retry!'

  return ragged_data_ops.filter_by_label(variant,
                                         label_threshold=label_threshold,
                                         filter_equal=filter_equal,
                                         variant_type=variant_type)


@monolith_export
def special_strategy(variant: tf.Tensor,
                     strategy_list: List[int],
                     strategy_conf: str = None,
                     variant_type: str = 'instance',
                     keep_empty_strategy=True):
  """用LineID中的special_strategy进行过滤, 
  
  Args:
    variant (:obj:`Tensor`): 输入数据, 必须是variant类型
    strategy_list (:obj:`List[int]`): strategy列表
    strategy_conf (:obj:`str`): 配置方式为 `strategy:sample_rate:label`, 如果有多个可以用逗号分割.
                                用于实现采样, 包括对正例/负例/所有样本采样, 并修改样本标签 
    variant_type (:obj:`str`): variant类型, 可以为instance/example
    keep_empty_strategy (:obj:`bool`): 是否保留strategy为空的样本, 默认为False
  
  Returns:
    variant tensor, 过滤后的数据, variant类型
  """

  items = [] if strategy_conf is None else strategy_conf.strip().split(',')
  special_strategies, sample_rates, labels = [], [], []
  if len(items) > 0:
    for item in items:
      tl = item.strip().split(':')
      if len(tl) == 2:
        special_strategies.append(int(tl[0]))
        sample_rates.append(float(tl[1]))
      elif len(tl) == 3:
        special_strategies.append(int(tl[0]))
        sample_rates.append(float(tl[1]))
        labels.append(float(tl[2]))

  assert len(special_strategies) == len(sample_rates)
  assert len(special_strategies) == len(labels) or len(labels) == 0
  assert all(0 <= sr <= 1 for sr in sample_rates)
  return ragged_data_ops.special_strategy(
      variant,
      special_strategies=special_strategies,
      sample_rates=sample_rates,
      labels=labels,
      strategy_list=strategy_list,
      keep_empty_strategy=keep_empty_strategy,
      variant_type=variant_type)


@monolith_export
def negative_sample(variant: tf.Tensor,
                    drop_rate: float,
                    label_index: int = 0,
                    threshold: float = 0.0,
                    variant_type: str = 'instance'):
  """负例采样
  
  Args:
    variant (:obj:`Tensor`): 输入数据, 必须是variant类型
    drop_rate (:obj:`float`): 负例丢弃比例, 取值区间为[0, 1), sample_rate = 1 - drop_rate. 
    label_index (:obj:`int`): 样本中labels是一个列表, label_index表示本次启用哪一个index对应的label
    threshold (:obj:`float`): label是一个实数, 大于`threshold`的是正样本
    variant_type (:obj:`str`): variant类型, 可以为instance/example
  
  Returns:
    variant tensor, 过滤后的数据, variant类型
  """

  return ragged_data_ops.negative_sample(variant,
                                         drop_rate=drop_rate,
                                         label_index=label_index,
                                         threshold=threshold,
                                         variant_type=variant_type)


@monolith_export
def feature_combine(src1: tf.RaggedTensor, src2: tf.RaggedTensor,
                    slot: int) -> tf.RaggedTensor:
  """特征交叉, 用于对已抽取Sparse特征的交叉
  
  Args:
    src1 (:obj:`RaggedTensor`): 参与交叉的sparse特征, 可以是简单特征, 也可以是序列特征
    src1 (:obj:`RaggedTensor`): 参与交叉的sparse特征, 可以是简单特征, 也可以是序列特征
    slot (:obj:`int`): 输出特征的slot
  
  Returns:
    RaggedTensor, 交叉后的特征
  
  """

  assert isinstance(src1, tf.RaggedTensor)
  assert isinstance(src2, tf.RaggedTensor)

  splits, values = ragged_data_ops.feature_combine(
      rt_nested_splits_src1=src1.nested_row_splits,
      rt_dense_values_src1=src1.flat_values,
      rt_nested_splits_src2=src2.nested_row_splits,
      rt_dense_values_src2=src2.flat_values,
      slot=slot,
      fid_version=2)

  if splits[0].dtype == tf.float32:
    return tf.RaggedTensor.from_row_splits(values=values,
                                           row_splits=splits[1],
                                           validate=False)
  else:
    return tf.RaggedTensor.from_nested_row_splits(flat_values=values,
                                                  nested_row_splits=splits,
                                                  validate=False)


@monolith_export
def switch_slot(ragged: tf.RaggedTensor, slot: int) -> tf.RaggedTensor:
  """对Sparse特征切换slot
  
  Args:
    ragged (:obj:`RaggedTensor`): 输入sparse特征, 可以是简单特征, 也可以是序列特征
    slot (:obj:`int`): 输出特征的slot
  
  Returns:
    RaggedTensor, 切换后的特征
  
  """

  assert isinstance(ragged, tf.RaggedTensor)
  nested_row_splits = ragged.nested_row_splits

  splits, values = ragged_data_ops.switch_slot(
      rt_nested_splits=nested_row_splits,
      rt_dense_values=ragged.flat_values,
      slot=slot,
      fid_version=2)

  if splits[0].dtype == tf.float32:
    return tf.RaggedTensor.from_row_splits(values=values,
                                           row_splits=splits[1],
                                           validate=False)
  else:
    return ragged.with_flat_values(values)


@monolith_export
def label_upper_bound(variant: tf.Tensor,
                      label_upper_bounds: List[float],
                      variant_type: str = 'instance'):
  """给label设置upper_bound, instance的label超过upper_bound的会被设置成upper_bound.
  Args:
    variant (:obj:`Tensor`): 输入数据，必须是 variant 类型
    label_upper_bounds (:obj:`List[float]`): 样本任一 label 值 >= 相应 label_upper_bounds
    时，该label会被设置为upper_bound
    variant_type (:obj:`str`): 'instance' 或 'example'

  Returns:
    variant tensor, label根据upper_bound调整后的数据, variant类型
  """
  assert variant_type in {'instance', 'example'}
  assert len(
      label_upper_bounds) > 0, 'Please specify label_threshold and retry!'

  return ragged_data_ops.label_upper_bound(
      variant, label_upper_bounds=label_upper_bounds, variant_type=variant_type)


@monolith_export
def label_normalization(variant: tf.Tensor,
                        norm_methods: List[str],
                        norm_values: List[float],
                        variant_type: str = 'instance'):
  """对Label进行normalization, instance的label会被修改为norm之后的数值.
  Args:
    variant (:obj:`Tensor`): 输入数据，必须是 variant 类型
    norm_methods (:obj:`List[str]`): normlization的方法，例如log,scale,repow,scalelog
    norm_values (:obj:`List[float]`): 对应normalization方法使用的norm_value, 长度需要与norm_methods保持一致
    variant_type (:obj:`str`): 'instance' 或 'example'

  Returns:
    variant tensor, label根据upper_bound调整后的数据, variant类型
  """
  assert variant_type in {'instance', 'example'}
  assert len(norm_methods) == len(
      norm_values), 'norm_methods and norm_values should have the same length'

  return ragged_data_ops.label_normalization(variant,
                                             norm_methods=norm_methods,
                                             norm_values=norm_values,
                                             variant_type=variant_type)


@monolith_export
def use_field_as_label(variant: tf.Tensor,
                       field_name: str,
                       overwrite_invalid_value: bool = False,
                       label_threshold: float = 7200,
                       variant_type: str = 'instance'):
  """用line_id里的field作为新的label。
  Args:
    variant (:obj:`Tensor`): 输入数据，必须是 variant 类型
    overwrite_invalid_value (:obj:`bool`): 是否对新field进行overwrite，如果overwrite会在value >= label_threshold时overwrite成0.
    label_threshold (:obj:`List[float]`): 对新field进行overwrite的threshold值，如果value >= label_threshold则改写为0.
    variant_type (:obj:`str`): 'instance' 或 'example'

  Returns:
    variant tensor, label根据upper_bound调整后的数据, variant类型
  """
  assert variant_type in {'instance', 'example'}

  return ragged_data_ops.use_field_as_label(
      variant,
      field_name=field_name,
      overwrite_invalid_value=overwrite_invalid_value,
      label_threshold=label_threshold,
      variant_type=variant_type)


def create_item_pool(start_num: int,
                     max_item_num_per_channel: int,
                     container: str = '',
                     shared_name: str = '') -> tf.Tensor:
  assert start_num >= 0 and max_item_num_per_channel > 0
  handle = ragged_data_ops.ItemPoolCreate(
      start_num=start_num,
      max_item_num_per_channel=max_item_num_per_channel,
      container=container,
      shared_name=shared_name)
  return handle


def item_pool_random_fill(pool: tf.Tensor) -> tf.Tensor:
  handle = ragged_data_ops.ItemPoolRandomFill(ipool=pool)
  return handle


def item_pool_check(pool: tf.Tensor,
                    model_path: str,
                    global_step: int,
                    nshards: int = 1,
                    buffer_size: int = 10 * 1024 * 1024) -> tf.Tensor:
  handle = ragged_data_ops.ItemPoolCheck(ipool=pool,
                                         model_path=model_path,
                                         nshards=nshards,
                                         buffer_size=buffer_size,
                                         global_step=global_step)
  return handle


def save_item_pool(pool: tf.Tensor,
                   global_step: tf.Tensor,
                   model_path: str,
                   nshards: int = 1) -> tf.Tensor:
  handle = ragged_data_ops.ItemPoolSave(ipool=pool,
                                        global_step=global_step,
                                        model_path=model_path,
                                        nshards=nshards)
  return handle


def restore_item_pool(pool: tf.Tensor,
                      global_step: tf.Tensor,
                      model_path: str,
                      nshards: int = 1,
                      buffer_size: int = 10 * 1024 * 1024) -> tf.Tensor:
  handle = ragged_data_ops.ItemPoolRestore(ipool=pool,
                                           global_step=global_step,
                                           model_path=model_path,
                                           nshards=nshards,
                                           buffer_size=buffer_size)
  return handle


def fill_multi_rank_output(
    variant: tf.Tensor,
    enable_draw_as_rank: bool = False,
    enable_chnid_as_rank: bool = False,
    enable_lineid_rank_as_rank: bool = False,
    rank_num: int = 18,
    variant_type: str = 'instance',
):
  """When use_rank_multi_output flag is set.
  """
  assert variant_type in {'instance', 'example'}

  return ragged_data_ops.fill_multi_rank_output(
      input=variant,
      variant_type=variant_type,
      enable_draw_as_rank=enable_draw_as_rank,
      enable_chnid_as_rank=enable_chnid_as_rank,
      enable_lineid_rank_as_rank=enable_lineid_rank_as_rank,
      rank_num=rank_num)


def use_f100_multi_head(
    variant: tf.Tensor,
    variant_type: str = 'instance',
):
  """When use_f100_multihead flag is set.
  """
  assert variant_type in {'instance', 'example'}

  return ragged_data_ops.use_f100_multi_head(input=variant,
                                             variant_type=variant_type)


def map_id(tensor: tf.Tensor, map_dict: Dict[int, int], default: int = -1):
  assert map_dict is not None and len(map_dict) > 0
  from_value, to_value = zip(*map_dict.items())

  return ragged_data_ops.MapId(input=tensor,
                               from_value=list(from_value),
                               to_value=list(to_value),
                               default_value=default)


def multi_label_gen(variant: tf.Tensor,
                    head_to_index: Dict[Any, int],
                    head_field: str = 'chnid',
                    pos_actions: List[int] = None,
                    neg_actions: List[int] = None,
                    use_origin_label: bool = False,
                    pos_label: float = 1.0,
                    neg_label: float = 0.0,
                    action_priority: str = None,
                    task_num: int = None,
                    variant_type: str = 'example'):
  task_num = 0 if task_num is None else task_num
  head_to_index_list, max_idx = [], 0
  for head, idx in head_to_index.items():
    head_to_index_list.append(f'{head}:{idx}')
    max_idx = max(idx, max_idx)
  if task_num != 0:
    assert max_idx < task_num
  else:
    task_num = max_idx + 1

  action_priority = action_priority or ""
  pos_actions, neg_actions = pos_actions or [], neg_actions or []
  if use_origin_label:
    assert len(pos_actions) == 0 and len(neg_actions) == 0
  else:
    assert len(pos_actions) > 0

  fields = LineId.DESCRIPTOR.fields_by_name
  assert head_field in fields
  field = fields[head_field]
  assert field.cpp_type in {
      field.CPPTYPE_INT32, field.CPPTYPE_INT64, field.CPPTYPE_UINT32,
      field.CPPTYPE_UINT64, field.CPPTYPE_STRING
  }

  assert variant_type in {'instance', 'example'}
  return ragged_data_ops.multi_label_gen(
      variant,
      task_num=task_num,
      head_to_index=','.join(head_to_index_list),
      head_field=head_field,
      action_priority=action_priority,
      pos_actions=pos_actions,
      neg_actions=neg_actions,
      use_origin_label=use_origin_label,
      pos_label=pos_label,
      neg_label=neg_label,
      variant_type=variant_type)


def string_to_variant(tensor: tf.Tensor,
                      variant_type: str = 'example',
                      has_header: bool = False,
                      has_sort_id: bool = False,
                      lagrangex_header: bool = False,
                      kafka_dump_prefix: bool = False,
                      kafka_dump: bool = False,
                      chnids: List[int] = None,
                      datasources: List[str] = None,
                      default_datasource: str = ''):
  assert variant_type in {
      'instance', 'example', 'examplebatch', 'example_batch'
  }
  return ragged_data_ops.string_to_variant(
      input=tensor,
      has_header=has_header,
      has_sort_id=has_sort_id,
      lagrangex_header=lagrangex_header,
      kafka_dump_prefix=kafka_dump_prefix,
      kafka_dump=kafka_dump,
      input_type=variant_type,
      chnids=chnids or [],
      datasources=datasources or [],
      default_datasource=default_datasource)


def variant_to_zeros(tensor: tf.Tensor):
  return ragged_data_ops.variant_to_zeros(tensor)


def kafka_resource_init(topics: List[str],
                        metadata: List[str],
                        container: str = '',
                        shared_name: str = ''):
  return ragged_data_ops.KafkaGroupReadableInit(topics=topics,
                                                metadata=metadata,
                                                container=container,
                                                shared_name=shared_name)


def kafka_read_next(input, index: int, message_poll_timeout: int,
                    stream_timeout: int):
  return ragged_data_ops.KafkaGroupReadableNext(
      input=input,
      index=index,
      message_poll_timeout=message_poll_timeout,
      stream_timeout=stream_timeout)


def has_variant(input, variant_type: str = 'example'):
  return ragged_data_ops.HasVariant(input=input, variant_type=variant_type)
