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

from absl import logging
import os
import struct
from copy import deepcopy
from typing import Dict, List, Iterable, Callable
from collections import deque
from dataclasses import dataclass
import traceback

import tensorflow as tf

from idl.matrix.proto.line_id_pb2 import LineId
from idl.matrix.proto.example_pb2 import FeatureConfigs

from monolith.utils import get_libops_path
from monolith.native_training.monolith_export import monolith_export
from monolith.native_training.data.feature_list import get_feature_name_and_slot, FeatureList
from monolith.native_training.data.data_op_config_pb2 import LabelConf, TaskLabelConf
from monolith.native_training.runtime.ops import gen_monolith_ops
from monolith.native_training.utils import add_to_collections
from tensorflow.python.framework import ops
from tensorflow.python.framework import common_shapes

parse_instance_ops = gen_monolith_ops

_line_id_descriptor = LineId.DESCRIPTOR

_default_parser_ctx = None


@dataclass
class ShardingSparseFidsOpParams:
  num_ps: int
  use_native_multi_hash_table: bool
  unique: Callable
  transfer_float16: bool
  sub_table_name_to_config: Dict
  feature_configs: FeatureConfigs
  enable_gpu_emb: bool
  use_gpu: bool


class ParserCtx(object):
  sharding_sparse_fids_features_prefix = "__sharding_sparse_fids__"
  sharding_sparse_fids_sparse_features_key = "__sharding_sparse_fids__sparse_features"

  def __init__(self, enable_fused_layout: bool = False):
    self._old_parser_ctx = None
    self.parser_type = None
    self.enable_fused_layout = enable_fused_layout
    self.sharding_sparse_fids_op_params: ShardingSparseFidsOpParams = None
    self._ctx_kv = {}

  def __enter__(self):
    global _default_parser_ctx
    self._old_parser_ctx = _default_parser_ctx
    _default_parser_ctx = self
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    global _default_parser_ctx
    _default_parser_ctx = self._old_parser_ctx
    self._old_parser_ctx = None

  @classmethod
  def sharding_sparse_fids_features_insert_to_features(cls, inputs, features):
    if isinstance(inputs, Dict):
      for k, v in inputs.items():
        if isinstance(v, Dict):
          for sub_k, sub_v in v.items():
            features[cls.sharding_sparse_fids_features_prefix + k + "/" +
                     sub_k] = sub_v
        elif isinstance(v, List):
          raise ValueError("not support")
        else:
          features[cls.sharding_sparse_fids_features_prefix + k] = v
    else:
      raise ValueError("not support")

  @classmethod
  def sharding_sparse_fids_features_parse_from_features(cls, features):
    outputs = {}
    pop_key_list = []
    for k, v in features.items():
      if k.startswith(cls.sharding_sparse_fids_features_prefix):
        name = k[len(cls.sharding_sparse_fids_features_prefix):]
        level_ouput = outputs
        level_names = name.split("/")
        for level_name in level_names[:-1]:
          if level_name not in level_ouput:
            level_ouput[level_name] = {}
          level_ouput = level_ouput[level_name]
        level_ouput[level_names[-1]] = v
        pop_key_list.append(k)
    for k in pop_key_list:
      features.pop(k)
    return outputs

  def set(self, key, value):
    self._ctx_kv[key] = value

  def get(self, key, default_value=None):
    return self._ctx_kv.get(key, default_value)



def get_default_parser_ctx() -> ParserCtx:
  global _default_parser_ctx
  if _default_parser_ctx is None:
    _default_parser_ctx = ParserCtx(False)
  return _default_parser_ctx


class ProtoType:
  TYPE_BOOL: int = 8
  TYPE_BYTES: int = 12
  TYPE_DOUBLE: int = 1
  TYPE_ENUM: int = 14
  TYPE_FIXED32: int = 7
  TYPE_FIXED64: int = 6
  TYPE_FLOAT: int = 2
  TYPE_GROUP: int = 10
  TYPE_INT32: int = 5
  TYPE_INT64: int = 3
  TYPE_MESSAGE: int = 11
  TYPE_SFIXED32: int = 15
  TYPE_SFIXED64: int = 16
  TYPE_SINT32: int = 17
  TYPE_SINT64: int = 18
  TYPE_STRING: int = 9
  TYPE_UINT32: int = 13
  TYPE_UINT64: int = 4

  UNKNOWN = {TYPE_BOOL, TYPE_ENUM, TYPE_GROUP, TYPE_MESSAGE}
  STRING = {TYPE_BYTES, TYPE_STRING}
  FLOAT = {TYPE_FLOAT, TYPE_DOUBLE}
  INT = {
      TYPE_INT32, TYPE_INT64, TYPE_SINT32, TYPE_SINT64, TYPE_UINT32,
      TYPE_UINT64, TYPE_FIXED32, TYPE_FIXED64, TYPE_SFIXED32, TYPE_SFIXED64
  }

  @classmethod
  def get_tf_type(cls, proto_type: int):
    if proto_type in cls.INT:
      return tf.int64
    elif proto_type in cls.FLOAT:
      return tf.float32
    elif proto_type in cls.STRING:
      return tf.string
    else:
      raise Exception('proto_type {} is not support'.format(proto_type))


def _add_dense_features(names: List[str], shapes: List[int],
                        types: List[tf.compat.v1.dtypes.DType],
                        dense_features: List[str],
                        dense_feature_shapes: List[int],
                        dense_feature_types: List[tf.compat.v1.dtypes.DType]):
  assert dense_features is not None
  assert dense_feature_shapes is not None
  assert len(dense_features) == len(dense_feature_shapes)
  assert all([s > 0 for s in dense_feature_shapes])

  if dense_feature_types is None:
    dense_feature_types = [tf.float32] * len(dense_features)
  else:
    assert len(dense_features) == len(dense_feature_types)

  names.extend(dense_features)
  shapes.extend(dense_feature_shapes)
  types.extend(dense_feature_types)


def _add_extra_features(names: List[str], shapes: List[int],
                        types: List[tf.compat.v1.dtypes.DType],
                        extra_features: List[str],
                        extra_feature_shapes: List[int]):
  assert extra_features is not None
  assert extra_feature_shapes is not None
  assert len(extra_features) == len(extra_feature_shapes)
  assert all([s > 0 for s in extra_feature_shapes])

  extra_dtypes = []
  for name in extra_features:
    try:
      extra_dtypes.append(
          ProtoType.get_tf_type(_line_id_descriptor.fields_by_name[name].type))
    except:
      raise Exception(f"{name} is not in line id, pls check!")

  names.extend(extra_features)
  shapes.extend(extra_feature_shapes)
  types.extend(extra_dtypes)


def _assemble(sparse_features,
              names,
              shapes,
              types,
              out_list,
              batch_size: int = None):
  assert len(out_list) == len(types)
  features = {}
  for i, name in enumerate(names):
    if name in sparse_features:
      value = out_list[i + len(names)]
      if batch_size:
        batch_size = batch_size[0] if isinstance(batch_size,
                                                 (list, tuple)) else batch_size
        split = tf.reshape(out_list[i], shape=(batch_size + 1,))
      else:
        split = out_list[i]
      features[name] = tf.RaggedTensor.from_row_splits(value,
                                                       split,
                                                       validate=False)
    else:
      features[name] = out_list[i]

  return features


def parse_instances(tensor: tf.Tensor,
                    fidv1_features: List[int] = None,
                    fidv2_features: List[str] = None,
                    dense_features: List[str] = None,
                    dense_feature_shapes: List[int] = None,
                    dense_feature_types: List[tf.compat.v1.dtypes.DType] = None,
                    extra_features: List[str] = None,
                    extra_feature_shapes: List[int] = None):
  """从Tensor中解析instance
  
  Example格式中, 所有特征均存于feature中, 没有平铺的特征. Sparse特征由于长度不定, 输出RaggedTensor, 其它特征输出Tensor
  
  Args:
    tensor (:obj:`tf.Tensor`): 输入样本
    fidv1_features (:obj:`List[int]`): 在Instance中, fidv1_features是平铺的, 所以用slot指定, 可以是部分slot
    fidv2_features (:obj:`List[str]`): 在Instance中, fidv2_features存放于feature中, 可以用名字指定, 可以是部分特征名
    dense_features (:obj:`List[str]`): 稠密特征(或Label)名称, 可以有多个, 也可以有不同类型
    dense_feature_shapes (:obj:`List[int]`): 稠密特征名称的shape
    dense_feature_types (:obj:`List[dtype]`): 稠密特征名称的数据类型, 默认为`tf.float32`
    extra_features (:obj:`List[str]`): 主要指LineId中的字段, 可以有多个, Monolith会自动从LineId中提取数据类型
    extra_feature_shapes (:obj:`List[int]`): extra特征名称的shape
  
  Returns:
    Dict[str, Tensor] 解析出特征名到特征的字典
  
  """

  if dense_features:
    assert dense_feature_shapes is not None
    assert len(dense_feature_shapes) == len(dense_features)
    if dense_feature_types:
      assert len(dense_feature_types) == len(dense_features)
    else:
      dense_feature_types = [tf.float32] * len(dense_features)

  get_default_parser_ctx().parser_type = 'instance'
  add_to_collections('fidv1_features', fidv1_features)
  add_to_collections('fidv2_features', fidv2_features)
  add_to_collections('dense_features', dense_features)
  add_to_collections('dense_feature_shapes', dense_feature_shapes)
  add_to_collections('dense_feature_types', dense_feature_types)
  add_to_collections('extra_features', extra_features)
  add_to_collections('extra_feature_shapes', extra_feature_shapes)
  add_to_collections('variant_type', 'instance')

  names, shapes, types = [], [], []

  if not get_default_parser_ctx().enable_fused_layout:
    sparse_features = []
    if fidv1_features is not None:
      names.extend(
          [get_feature_name_and_slot(slot)[0] for slot in fidv1_features])
      if all(isinstance(feature_name, str) for feature_name in fidv1_features):
        try:
          feature_list = FeatureList.parse()
          fidv1_features = [
              feature_list.get(feature_name).slot
              for feature_name in fidv1_features
          ]
        except:
          raise RuntimeError("fidv1_features error")
      shapes.extend([-1] * len(fidv1_features))
      types.extend([tf.int64] * len(fidv1_features))

    if fidv2_features is not None:
      names.extend(fidv2_features)
      shapes.extend([-1] * len(fidv2_features))
      types.extend([tf.int64] * len(fidv2_features))

    sparse_features.extend(names)

  if dense_features is not None:
    _add_dense_features(names, shapes, types, dense_features,
                        dense_feature_shapes, dense_feature_types)

  if extra_features is not None:
    _add_extra_features(names, shapes, types, extra_features,
                        extra_feature_shapes)
  if get_default_parser_ctx().enable_fused_layout:
    if len(names) == 0:
      names.append("__FAKE_FEATURE__")
      shapes.append(1)
      types.append(tf.float32)
    out_list, instances = parse_instance_ops.parse_instances_v2(
        tensor, [], [], names, shapes, types, extra_features or [])
    features = _assemble([], names, shapes, types, out_list)
    if get_default_parser_ctx().sharding_sparse_fids_op_params is not None:
      sharding_sparse_fids_with_context(instances, features)
    else:
      features[ParserCtx.sharding_sparse_fids_sparse_features_key] = instances
    if "__FAKE_FEATURE__" in features:
      del features["__FAKE_FEATURE__"]
    return features
  else:
    types.extend([tf.int64] * len(sparse_features))
    assert len(names) == len(set(names)), "deplicate names, pls check!"
    out_list = parse_instance_ops.parse_instances(tensor, fidv1_features or [],
                                                  fidv2_features or [], names,
                                                  shapes, types,
                                                  extra_features or [])
    return _assemble(sparse_features, names, shapes, types, out_list)


@monolith_export
def parse_examples(tensor: tf.Tensor,
                   sparse_features: List[str],
                   dense_features: List[str] = None,
                   dense_feature_shapes: List[int] = None,
                   dense_feature_types: List[tf.compat.v1.dtypes.DType] = None,
                   extra_features: List[str] = None,
                   extra_feature_shapes: List[int] = None):
  """从Tensor中解析example
  
  Example格式中, 所有特征均存于feature中, 没有平铺特征. Sparse特征由于长度不定, 输出RaggedTensor, 其它特征输出Tensor
  
  Args:
    tensor (:obj:`tf.Tensor`): 输入样本
    sparse_features (:obj:`List[str]`): 稀疏特征名称, 可以有多个
    dense_features (:obj:`List[str]`): 稠密特征(或Label)名称, 可以有多个, 也可以有不同类型
    dense_feature_shapes (:obj:`List[int]`): 稠密特征名称的shape
    dense_feature_types (:obj:`List[dtype]`): 稠密特征名称的数据类型, 默认为`tf.float32`
    extra_features (:obj:`List[str]`): 主要指LineId中的字段, 可以有多个, Monolith会自动从LineId中提取数据类型
    extra_feature_shapes (:obj:`List[int]`): extra特征名称的shape
  
  Returns:
    Dict[str, Tensor] 解析出特征名到特征的字典
  
  """

  if dense_features:
    assert dense_feature_shapes is not None
    assert len(dense_feature_shapes) == len(dense_features)
    if dense_feature_types:
      assert len(dense_feature_types) == len(dense_features)
    else:
      dense_feature_types = [tf.float32] * len(dense_features)

  get_default_parser_ctx().parser_type = 'example'
  add_to_collections('sparse_features', sparse_features)
  add_to_collections('dense_features', dense_features)
  add_to_collections('dense_feature_shapes', dense_feature_shapes)
  add_to_collections('dense_feature_types', dense_feature_types)
  add_to_collections('extra_features', extra_features)
  add_to_collections('extra_feature_shapes', extra_feature_shapes)
  add_to_collections('variant_type', 'example')

  names, shapes, types = [], [], []

  if not get_default_parser_ctx().enable_fused_layout:
    assert sparse_features is not None
    names.extend(sparse_features)
    shapes.extend([-1] * len(sparse_features))
    types.extend([tf.int64] * len(sparse_features))

  if dense_features is not None:
    _add_dense_features(names, shapes, types, dense_features,
                        dense_feature_shapes, dense_feature_types)

  if extra_features is not None:
    _add_extra_features(names, shapes, types, extra_features,
                        extra_feature_shapes)

  assert len(names) == len(set(names)), "deplicate names, pls check!"
  if get_default_parser_ctx().enable_fused_layout:
    if len(names) == 0:
      names.append("__FAKE_FEATURE__")
      shapes.append(1)
      types.append(tf.float32)
    out_list, examples = parse_instance_ops.parse_examples_v2(
        tensor, names, shapes, types, extra_features or [])
    features = _assemble([], names, shapes, types, out_list)
    if get_default_parser_ctx().sharding_sparse_fids_op_params is not None:
      sharding_sparse_fids_with_context(examples, features)
    else:
      features[ParserCtx.sharding_sparse_fids_sparse_features_key] = examples
    if "__FAKE_FEATURE__" in features:
      del features["__FAKE_FEATURE__"]
    return features
  else:
    types.extend([tf.int64] * len(sparse_features))
    out_list = parse_instance_ops.parse_examples(tensor, names, shapes, types,
                                                 extra_features or [])
    return _assemble(sparse_features, names, shapes, types, out_list)


@monolith_export
def parse_example_batch(
    tensor: tf.Tensor,
    sparse_features: List[str],
    dense_features: List[str] = None,
    dense_feature_shapes: List[int] = None,
    dense_feature_types: List[tf.compat.v1.dtypes.DType] = None,
    extra_features: List[str] = None,
    extra_feature_shapes: List[int] = None):
  """从Tensor中解析example_batch
  
  Example_batch格式中, 所有特征均存于feature中, 没有平铺特征. Sparse特征由于长度不定, 输出RaggedTensor, 其它特征输出Tensor
  
  Args:
    tensor (:obj:`tf.Tensor`): 输入样本
    sparse_features (:obj:`List[str]`): 稀疏特征名称, 可以有多个
    dense_features (:obj:`List[str]`): 稠密特征(或Label)名称, 可以有多个, 也可以有不同类型
    dense_feature_shapes (:obj:`List[int]`): 稠密特征名称的shape
    dense_feature_types (:obj:`List[dtype]`): 稠密特征名称的数据类型, 默认为`tf.float32`
    extra_features (:obj:`List[str]`): 主要指LineId中的字段, 可以有多个, Monolith会自动从LineId中提取数据类型
    extra_feature_shapes (:obj:`List[int]`): extra特征名称的shape
  
  Returns:
    Dict[str, Tensor] 解析出特征名到特征的字典
  
  """

  if dense_features:
    assert dense_feature_shapes is not None
    assert len(dense_feature_shapes) == len(dense_features)
    if dense_feature_types:
      assert len(dense_feature_types) == len(dense_features)
    else:
      dense_feature_types = [tf.float32] * len(dense_features)

  get_default_parser_ctx().parser_type = 'examplebatch'
  add_to_collections('sparse_features', sparse_features)
  add_to_collections('dense_features', dense_features)
  add_to_collections('dense_feature_shapes', dense_feature_shapes)
  add_to_collections('dense_feature_types', dense_feature_types)
  add_to_collections('extra_features', extra_features)
  add_to_collections('extra_feature_shapes', extra_feature_shapes)
  add_to_collections('variant_type', 'example_batch')

  names, shapes, types = [], [], []

  if not get_default_parser_ctx().enable_fused_layout:
    assert sparse_features is not None
    names.extend(sparse_features)
    shapes.extend([-1] * len(sparse_features))
    types.extend([tf.int64] * len(sparse_features))

  if dense_features is not None:
    _add_dense_features(names, shapes, types, dense_features,
                        dense_feature_shapes, dense_feature_types)

  if extra_features is not None:
    _add_extra_features(names, shapes, types, extra_features,
                        extra_feature_shapes)
  batch_size = get_default_parser_ctx().get('batch_size')
  assert len(names) == len(set(names)), "deplicate names, pls check!"
  if get_default_parser_ctx().enable_fused_layout:
    if len(names) == 0:
      names.append("__FAKE_FEATURE__")
      shapes.append(1)
      types.append(tf.float32)
    out_list, example_batch = parse_instance_ops.parse_example_batch_v2(
        tensor, names, shapes, types, extra_features or [])
    features = _assemble([], names, shapes, types, out_list)
    if get_default_parser_ctx().sharding_sparse_fids_op_params is not None:
      sharding_sparse_fids_with_context(example_batch, features)
    else:
      features[
          ParserCtx.sharding_sparse_fids_sparse_features_key] = example_batch
    if "__FAKE_FEATURE__" in features:
      del features["__FAKE_FEATURE__"]
    return features
  else:
    types.extend([tf.int64] * len(sparse_features))
    out_list = parse_instance_ops.parse_example_batch(tensor, names, shapes,
                                                      types, extra_features or
                                                      [])
    return _assemble(sparse_features,
                     names,
                     shapes,
                     types,
                     out_list,
                     batch_size=batch_size)


@monolith_export
def sharding_sparse_fids(tensor: tf.Tensor,
                         ps_num: int,
                         feature_cfgs: FeatureConfigs,
                         unique: bool,
                         input_type: str,
                         parallel_flag: int = 0,
                         fid_list_ret_list: bool = False,
                         version: int = 5):
  assert input_type in ["example", "examplebatch", "example_batch", "instance"]
  input_type = 'examplebatch' if input_type == 'example_batch' else input_type
  table_name_list = []
  for cfg in feature_cfgs.feature_configs.values():
    if cfg.table not in table_name_list:
      table_name_list.append(cfg.table)
  table_name_list.sort()
  ps_num = 1 if ps_num == 0 else ps_num
  logging.info(
      f"num of multi_type_hashtable is {ps_num} {len(table_name_list)}: [{table_name_list}]"
  )
  table_count = len(table_name_list) * ps_num
  fid_list_emb_row_lenth = None
  fid_list_table_row_length = None
  fid_list_shard_row_lenth = None
  fid_list_row_splits_size = None
  nfl_size = None
  feature_size = None
  fid_size = None
  emb_size = None
  if version == 5:
    fid_list, fid_list_row_splits, fid_list_row_splits_size, fid_offset, feature_offset, nfl_offset, batch_size, nfl_size, feature_size, fid_size, emb_size = parse_instance_ops.sharding_sparse_fids_v5(
        pb_input=tensor,
        ps_num=ps_num,
        feature_cfgs=feature_cfgs.SerializeToString(),
        N=table_count,
        unique=unique,
        input_type=input_type,
        parallel_flag=parallel_flag)
  elif version == 4:
    fid_list, fid_list_row_splits, fid_list_table_row_length, fid_list_shard_row_lenth, fid_list_emb_row_lenth, fid_offset, feature_offset, nfl_offset, batch_size = parse_instance_ops.sharding_sparse_fids_v4(
        pb_input=tensor,
        ps_num=ps_num,
        feature_cfgs=feature_cfgs.SerializeToString(),
        unique=unique,
        input_type=input_type,
        parallel_flag=parallel_flag)
  elif version == 3:
    fid_list, fid_list_row_splits, fid_offset, feature_offset, nfl_offset, batch_size = parse_instance_ops.sharding_sparse_fids_v3(
        pb_input=tensor,
        ps_num=ps_num,
        feature_cfgs=feature_cfgs.SerializeToString(),
        N=table_count,
        unique=unique,
        input_type=input_type,
        parallel_flag=parallel_flag)
    fid_list_row_splits_size = [None] * table_count
  elif version == 2:
    fid_list, fid_list_row_splits, fid_offset, feature_offset, nfl_offset, batch_size = parse_instance_ops.sharding_sparse_fids_v2(
        pb_input=tensor,
        ps_num=ps_num,
        feature_cfgs=feature_cfgs.SerializeToString(),
        N=table_count,
        unique=unique,
        input_type=input_type,
        parallel_flag=parallel_flag)
    fid_list_row_splits_size = [None] * table_count
  else:
    fid_list, fid_offset, feature_offset, nfl_offset, batch_size = parse_instance_ops.sharding_sparse_fids(
        pb_input=tensor,
        ps_num=ps_num,
        feature_cfgs=feature_cfgs.SerializeToString(),
        N=table_count,
        unique=unique,
        input_type=input_type,
        parallel_flag=parallel_flag)
    fid_list_row_splits = [None] * table_count
    fid_list_row_splits_size = [None] * table_count
  if version != 4:
    assert len(fid_list) == table_count
    assert len(fid_list_row_splits) == table_count
  if version == 5:
    assert len(fid_list_row_splits_size) == table_count
  if fid_list_ret_list or version == 4:
    return fid_list, fid_offset, feature_offset, nfl_offset, batch_size, nfl_size, feature_size, fid_size, emb_size, fid_list_row_splits, fid_list_row_splits_size, fid_list_emb_row_lenth, fid_list_table_row_length, fid_list_shard_row_lenth
  ret = {}
  ret_row_split = {}
  ret_row_split_size = {}
  index = 0
  for table_idx in range(len(table_name_list)):
    table_name = table_name_list[table_idx]
    for ps_index in range(ps_num):
      ret[table_name + ":" + str(ps_index)] = fid_list[index]
      ret_row_split[table_name + ":" +
                    str(ps_index)] = fid_list_row_splits[index]
      ret_row_split_size[table_name + ":" +
                         str(ps_index)] = fid_list_row_splits_size[index]
      index += 1
  return ret, fid_offset, feature_offset, nfl_offset, batch_size, nfl_size, feature_size, fid_size, emb_size, ret_row_split, ret_row_split_size, fid_list_emb_row_lenth, fid_list_table_row_length, fid_list_shard_row_lenth


def sharding_sparse_fids_with_context(sparse_features: tf.Tensor,
                                      features,
                                      parser_ctx: ParserCtx = None):
  if parser_ctx is None:
    parser_ctx = get_default_parser_ctx()
  shards, fid_offset, feature_offset, nfl_offset, batch_size, nfl_size, \
  feature_size, fid_size, emb_size, shards_row_split, shards_row_split_size, \
  fid_list_emb_row_lenth, fid_list_table_row_length, fid_list_shard_row_lenth = sharding_sparse_fids(
      sparse_features,
      ps_num=parser_ctx.sharding_sparse_fids_op_params.num_ps,
      feature_cfgs=parser_ctx.sharding_sparse_fids_op_params.feature_configs,
      unique=parser_ctx.sharding_sparse_fids_op_params.unique(),
      input_type=parser_ctx.parser_type,
      fid_list_ret_list=parser_ctx.sharding_sparse_fids_op_params.enable_gpu_emb,
      parallel_flag=1 if parser_ctx.sharding_sparse_fids_op_params.use_gpu else 0,
      version=4 if parser_ctx.sharding_sparse_fids_op_params.enable_gpu_emb else 5)

  if parser_ctx.sharding_sparse_fids_op_params.enable_gpu_emb:
    '''
    table_count = len(shards) // parser_ctx.sharding_sparse_fids_op_params.num_ps

    def tensor_list_to_ragged_tensor(tensor_list):
      if len(tensor_list) == 1:
        shard_ragged_tensor = tf.RaggedTensor.from_row_starts(
            tensor_list[0], tf.constant([0], dtype=tf.int32), validate=False)
      else:
        shard_ragged_tensor = tf.ragged.stack(tensor_list)
      return shard_ragged_tensor

    shards_new_order_list = []
    for ps_i in range(parser_ctx.sharding_sparse_fids_op_params.num_ps):
      for table_i in range(table_count):
        shards_new_order_list.append(shards[ps_i + table_i * parser_ctx.sharding_sparse_fids_op_params.num_ps])
    shard_ragged_tensor = tensor_list_to_ragged_tensor(shards_new_order_list)
    shards_value = shard_ragged_tensor.values
    shards_table_row_lengths = shard_ragged_tensor.row_lengths()
    shards_row_lengths = tf.reduce_sum(tf.reshape(shards_table_row_lengths,
                                                  [parser_ctx.sharding_sparse_fids_op_params.num_ps, table_count]),
                                       axis=-1)
    '''
    shards_value = shards
    shards_row_lengths = fid_list_shard_row_lenth
    shards_table_row_lengths = fid_list_table_row_length

    parser_ctx.sharding_sparse_fids_features_insert_to_features(
        {
            "shards_value": shards_value,
            "shards_row_lengths": shards_row_lengths,
            "shards_table_row_lengths": shards_table_row_lengths,
            "fid_offset": fid_offset,
            "feature_offset": feature_offset,
            "nfl_offset": nfl_offset,
            "batch_size": batch_size,
            "fid_list_emb_row_lenth": fid_list_emb_row_lenth,
        }, features)
  else:
    features_dict = {
        "shards": shards,
        "fid_offset": fid_offset,
        "feature_offset": feature_offset,
        "nfl_offset": nfl_offset,
        "batch_size": batch_size,
        "nfl_size": nfl_size,
        "feature_size": feature_size,
        "fid_size": fid_size,
        "emb_size": emb_size,
    }
    if parser_ctx.sharding_sparse_fids_op_params.use_native_multi_hash_table:
      features_dict.update({"shards_row_split": shards_row_split})
      features_dict.update({"shards_row_split_size": shards_row_split_size})
    parser_ctx.sharding_sparse_fids_features_insert_to_features(
        features_dict, features)


def parse_example_batch_list(
    tensor: List[tf.Tensor],
    label_config: str = None,
    positive_label: float = 1.0,
    negative_label: float = 0.0,
    names: List[str] = None,
    shapes: List[int] = None,
    dtypes: List[tf.dtypes.DType] = None,
    extra_features: List[str] = None) -> Dict[str, tf.Tensor]:
  names, shapes, dtypes = list(names), list(shapes), list(dtypes)
  get_default_parser_ctx().parser_type = 'examplebatch'
  label_conf = LabelConf()
  if label_config is not None and len(label_config) > 0:
    tasks = label_config.split(';')
    names.append('label')
    shapes.append(len(tasks))
    dtypes.append(tf.float32)

    for task in tasks:
      task_conf = label_conf.conf.add()
      pos_actions, neg_actions = task.split(':')
      pos_actions_list = [
          int(pos) for pos in pos_actions.split(',') if len(pos) > 0
      ]
      neg_actions_list = [
          int(neg) for neg in neg_actions.split(',') if len(neg) > 0
      ]
      task_conf.pos_actions.extend(pos_actions_list)
      task_conf.neg_actions.extend(neg_actions_list)

  sparse_features = []
  for i, name in enumerate(names):
    if shapes[i] == -1:
      sparse_features.append(name)
      dtypes.append(tf.int64)

  assert len(names) == len(set(names)), "deplicate names, pls check!"
  out_list = parse_instance_ops.parse_example_batch_list(
      tensor,
      label_config=label_conf.SerializeToString(),
      names=names,
      shapes=shapes,
      dtypes=dtypes,
      extra_names=extra_features,
      positive_label=positive_label,
      negative_label=negative_label)
  return _assemble(sparse_features, names, shapes, dtypes, out_list)
