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
from typing import Dict, List, Iterable, Callable

import tensorflow as tf
from tensorflow.python.platform import resource_loader
from tensorflow.python.framework import load_library
from tensorflow.python.ops.ragged.row_partition import RowPartition, _row_partition_factory_key

from monolith.native_training.data.utils import get_slot_feature_name
from monolith.native_training.data.training_instance.python.parser_utils import \
  add_extra_parse_step, advanced_parse
from monolith.native_training.runtime.ops import gen_monolith_ops

parse_instance_ops = gen_monolith_ops


def _parse_instance_impl(
    serialized: tf.Tensor, fidv1_features: List[int], fidv2_features: List[str],
    float_features: List[str], float_feature_dims: List[int],
    int64_features: List[str], int64_feature_dims: List[int],
    string_features: List[str], string_feature_dims: List[int],
    misc_float_features: List[str], misc_float_dims: List[int],
    misc_int64_features: List[str], misc_int64_dims: List[int],
    misc_string_features: List[str], misc_string_dims: List[int],
    cc_op: Callable):
  fidv1_features = fidv1_features or []
  fidv2_features = fidv2_features or []
  float_features = float_features or []
  float_feature_dims = float_feature_dims or []
  int64_features = int64_features or []
  int64_feature_dims = int64_feature_dims or []
  string_features = string_features or []
  string_feature_dims = string_feature_dims or []
  misc_float_features = misc_float_features or []
  misc_float_dims = misc_float_dims or []
  misc_int64_features = misc_int64_features or []
  misc_int64_dims = misc_int64_dims or []
  misc_string_features = misc_string_features or []
  misc_string_dims = misc_string_dims or []
  (ragged_feature_splits, ragged_feature_values, float_feature_values,
   int64_feature_values, string_feature_values, misc_float_feature_values,
   misc_int64_feature_values, misc_string_feature_values) = cc_op(
       serialized,
       N=(len(fidv1_features) + len(fidv2_features)),
       M=len(float_features),
       O=len(int64_features),
       P=len(string_features),
       Q=len(misc_float_features),
       R=len(misc_int64_features),
       S=len(misc_string_features),
       fidv1_features=fidv1_features,
       fidv2_features=fidv2_features,
       float_features=float_features,
       float_feature_dims=float_feature_dims,
       string_features=string_features,
       string_feature_dims=string_feature_dims,
       int64_features=int64_features,
       int64_feature_dims=int64_feature_dims,
       misc_float_features=misc_float_features,
       misc_float_dims=misc_float_dims,
       misc_int64_features=misc_int64_features,
       misc_int64_dims=misc_int64_dims,
       misc_string_features=misc_string_features,
       misc_string_dims=misc_string_dims,
   )
  ragged_keys = [get_slot_feature_name(slot_id) for slot_id in fidv1_features
                ] + fidv2_features

  ragged_values = []
  for values, row_splits in zip(ragged_feature_values, ragged_feature_splits):
    row_partition = RowPartition(
        row_splits,
        # value_rowids=
        # nrows=
        #
        # TODO(zhuoran): Besides the "value" and "split" parsed from proto above,
        # precompute other two encodings "value_rowids" & "nrows" in Fountain also,
        # so that we could construct the ragged tensor with 4 precomputed encodings,
        # and would not need to recompute them later at training period again.
        internal=_row_partition_factory_key
        # Currently, we just compute and cache value_rowids and nrows here:
    ).with_precomputed_value_rowids().with_precomputed_nrows()
    ragged_values.append(tf.RaggedTensor(values, row_partition, internal=True))

  float_keys = float_features
  int64_keys = int64_features
  string_keys = string_features
  return dict(
      zip(
          ragged_keys + float_keys + int64_keys + string_keys +
          misc_float_features + misc_int64_features + misc_string_features,
          ragged_values + float_feature_values + int64_feature_values +
          string_feature_values + misc_float_feature_values +
          misc_int64_feature_values + misc_string_feature_values))


def parse_instances2(serialized: tf.Tensor,
                     fidv1_features: List[int] = None,
                     fidv2_features: List[str] = None,
                     float_features: List[str] = None,
                     float_feature_dims: List[int] = None,
                     int64_features: List[str] = None,
                     int64_feature_dims: List[int] = None,
                     string_features: List[str] = None,
                     string_feature_dims: List[int] = None,
                     misc_float_features: List[str] = None,
                     misc_float_dims: List[int] = None,
                     misc_int64_features: List[str] = None,
                     misc_int64_dims: List[int] = None,
                     misc_string_features: List[str] = None,
                     misc_string_dims: List[int] = None):
  """从序列化的instance Tensor中解析instance
  
  Args:
    varient_tensor (:obj:`Tensor`): 输入数据
    fidv1_features (:obj:`List[int]`): 在Instance中, fidv1_features是平铺的, 所以用slot指定, 可以是部分slot
    fidv2_features (:obj:`List[str]`): 在Instance中, fidv2_features存放于feature中, 可以用名字指定, 可以是部分特征名
    float_features (:obj:`List[str]`): 在Instance中, 连续特征存于feature中, 可以用名字指定, 可以是部分特征名
    float_feature_dims (:obj:`List[int]`): 连续特征的维度, `float_feature_dims`的长度要与`float_features`一致
    int64_features (:obj:`List[str]`): 在Instance中, int64特征(非FID)存于feature中, 可以用名字指定, 可以是部分特征名
    int64_feature_dims (:obj:`List[int]`): int64特征的维度, `int64_feature_dims`的长度要与`int64_features`一致
    string_features (:obj:`List[str]`): 在Instance中, syting特征存于feature中, 可以用名字指定, 可以是部分特征名
    string_feature_dims (:obj:`List[int]`): string特征的维度, `string_feature_dims`的长度要与`string_features`一致
    misc_float_features (:obj:`List[str]`): 在LineId中, float字段, 用名字指定, 可以有多个
    misc_float_dims (:obj:`List[int]`): 在LineId中, float字段维度, `misc_float_dims`的长度要与`misc_float_features`一致
    misc_int64_features (:obj:`List[str]`): 在LineId中, int64字段, 用名字指定, 可以有多个
    misc_int64_dims (:obj:`List[int]`): 在LineId中, int64字段维度, `misc_int64_dims`的长度要与`misc_int64_features`一致
    misc_string_features (:obj:`List[str]`): 在LineId中, string字段, 用名字指定, 可以有多个
    misc_string_dims (:obj:`List[str]`): 在LineId中, string字段维度, `misc_string_dims`的长度要与`misc_string_features`一致

  Returns:
    Dict[str, Tensor] 解析出特征名到特征的字典
  
  """

  return _parse_instance_impl(
      serialized, fidv1_features, fidv2_features, float_features,
      float_feature_dims, int64_features, int64_feature_dims, string_features,
      string_feature_dims, misc_float_features, misc_float_dims,
      misc_int64_features, misc_int64_dims, misc_string_features,
      misc_string_dims, parse_instance_ops.monolith_parse_instances)


def parse_instances(serialized: tf.Tensor,
                    fidv1_features: List[int] = None,
                    fidv2_features: List[str] = None,
                    float_features: List[str] = None,
                    float_feature_dims: List[int] = None,
                    int64_features: List[str] = None,
                    int64_feature_dims: List[int] = None,
                    string_features: List[str] = None,
                    string_feature_dims: List[int] = None,
                    misc_float_features: List[str] = ['sample_rate'],
                    misc_int64_features: List[str] = ['req_time', 'uid'],
                    misc_string_features: List[str] = None,
                    misc_repeated_float_features: List[str] = ['label'],
                    misc_repeated_float_dims: List[int] = None,
                    misc_repeated_int64_features: List[str] = None,
                    misc_repeated_int64_dims: List[int] = None,
                    misc_repeated_string_features: List[str] = None,
                    misc_repeated_string_dims: List[str] = None):
  """从序列化的instance Tensor中解析instance, 但参数较多, 请使用`parse_instances2`
  
  Args:
    varient_tensor (:obj:`Tensor`): 输入数据
    fidv1_features (:obj:`List[int]`): 在Instance中, fidv1_features是平铺的, 所以用slot指定, 可以是部分slot
    fidv2_features (:obj:`List[str]`): 在Instance中, fidv2_features存放于feature中, 可以用名字指定, 可以是部分特征名
    float_features (:obj:`List[str]`): 在Instance中, 连续特征存于feature中, 可以用名字指定, 可以是部分特征名
    float_feature_dims (:obj:`List[int]`): 连续特征的维度, `float_feature_dims`的长度要与`float_features`一致
    int64_features (:obj:`List[str]`): 在Instance中, int64特征(非FID)存于feature中, 可以用名字指定, 可以是部分特征名
    int64_feature_dims (:obj:`List[int]`): int64特征的维度, `int64_feature_dims`的长度要与`int64_features`一致
    string_features (:obj:`List[str]`): 在Instance中, syting特征存于feature中, 可以用名字指定, 可以是部分特征名
    string_feature_dims (:obj:`List[int]`): string特征的维度, `string_feature_dims`的长度要与`string_features`一致
    misc_float_features (:obj:`List[str]`): 在LineId中, 非repeated float字段, 用名字指定, 可以有多个
    misc_int64_features (:obj:`List[str]`): 在LineId中, 非repeated int64字段, 用名字指定, 可以有多个
    misc_string_features (:obj:`List[str]`): 在LineId中, 非repeated string字段, 用名字指定, 可以有多个
    misc_repeated_float_features (:obj:`List[str]`): 在LineId中, repeated float字段, 用名字指定, 可以有多个
    misc_repeated_float_dims (:obj:`List[int]`): 在LineId中, repeated float字段维度, `misc_repeated_float_dims`的长度要与`misc_repeated_float_features`一致
    misc_repeated_int64_features (:obj:`List[str]`): 在LineId中, repeated int64字段, 用名字指定, 可以有多个
    misc_repeated_int64_dims (:obj:`List[int]`): 在LineId中, repeated int64字段维度, `misc_repeated_int64_dims`的长度要与`misc_repeated_int64_features`一致
    misc_repeated_string_features (:obj:`List[str]`): 在LineId中, repeated string字段, 用名字指定, 可以有多个
    misc_repeated_string_dims (:obj:`List[str]`): 在LineId中, repeated string字段维度, `misc_repeated_string_dims`的长度要与`misc_repeated_string_features`一致

  Returns:
    Dict[str, Tensor] 解析出特征名到特征的字典
  
  """

  fidv1_features = fidv1_features or []
  fidv2_features = fidv2_features or []
  float_features = float_features or []
  float_feature_dims = float_feature_dims or []
  int64_features = int64_features or []
  int64_feature_dims = int64_feature_dims or []
  string_features = string_features or []
  string_feature_dims = string_feature_dims or []
  misc_float_features = misc_float_features or []
  misc_float_feature_dims = [1] * len(misc_float_features)
  misc_int64_features = misc_int64_features or []
  misc_int64_feature_dims = [1] * len(misc_int64_features)
  misc_string_features = misc_string_features or []
  misc_string_features_dims = [1] * len(misc_string_features)
  misc_repeated_float_features = misc_repeated_float_features or []
  misc_repeated_float_dims = misc_repeated_float_dims or [1] * len(
      misc_repeated_float_features)
  misc_repeated_int64_features = misc_repeated_int64_features or []
  misc_repeated_int64_dims = misc_repeated_int64_dims or [1] * len(
      misc_repeated_int64_features)
  misc_repeated_string_features = misc_repeated_string_features or []
  misc_repeated_string_dims = misc_repeated_string_dims or [1] * len(
      misc_repeated_string_features)

  features = parse_instances2(
      serialized, fidv1_features, fidv2_features, float_features,
      float_feature_dims, int64_features, int64_feature_dims, string_features,
      string_feature_dims, misc_float_features + misc_repeated_float_features,
      misc_float_feature_dims + misc_repeated_float_dims,
      misc_int64_features + misc_repeated_int64_features,
      misc_int64_feature_dims + misc_repeated_int64_dims,
      misc_string_features + misc_repeated_string_features,
      misc_string_features_dims + misc_repeated_string_dims)
  for key in misc_float_features + misc_int64_features:
    features[key] = tf.reshape(features[key], [-1])

  return features


# This is mainly for test purpose, DO NOT use it directly.
monolith_raw_parse_instance = parse_instance_ops.MonolithRawParseInstance
