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

from dataclasses import dataclass
from typing import List
from random import randint
import tensorflow as tf

from monolith.native_training.distribution_ops import *
from idl.matrix.proto.example_pb2 import ExampleBatch, FeatureListType, Feature, \
  FeatureConfigs, FeatureConfig, OutConfig, SliceConfig, PoolingType, OutType

batch_size = 10


@dataclass
class FeatMeta:
  name: str = None
  slot: int = None
  max_sequence_length: int = 1
  fid_version: int = 0
  slice_dims: List[int] = None
  table: str = None
  pool_type: int = 1  # 1 sum, 2 mean, 3 fristn
  fl_type: int = 0  # 0 INDIVIDUAL, 1 SHARED


@dataclass
class TableMeta:
  name: str = None
  slice_dims: List[int] = None


table1 = TableMeta(name='table1', slice_dims=[1, 4, 4, 8])
table2 = TableMeta(name='table2', slice_dims=[1, 4, 4])
table3 = TableMeta(name='table3', slice_dims=[8])

features = {
    'f_user_id':
        FeatMeta(name='f_user_id',
                 slot=1,
                 max_sequence_length=1,
                 slice_dims=table1.slice_dims,
                 table=table1.name,
                 pool_type=1,
                 fl_type=1),
    'f_user_ctx_network':
        FeatMeta(name='f_user_ctx_network',
                 slot=61,
                 max_sequence_length=1,
                 slice_dims=table1.slice_dims,
                 table=table1.name,
                 pool_type=1,
                 fl_type=1),
    'f_user_test10_array':
        FeatMeta(name='f_user_test10_array',
                 slot=549,
                 max_sequence_length=10,
                 slice_dims=table1.slice_dims,
                 table=table1.name,
                 pool_type=2,
                 fl_type=1),
    'f_user_id-f_page':
        FeatMeta(name='f_user_id-f_page',
                 slot=504,
                 max_sequence_length=10,
                 fid_version=1,
                 slice_dims=table3.slice_dims,
                 table=table3.name,
                 pool_type=3,
                 fl_type=1),
    'f_goods_id':
        FeatMeta(name='f_user_id',
                 slot=200,
                 max_sequence_length=1,
                 slice_dims=table2.slice_dims,
                 table=table2.name,
                 pool_type=1,
                 fl_type=0),
    'f_page':
        FeatMeta(name='f_page',
                 slot=305,
                 max_sequence_length=1,
                 slice_dims=table2.slice_dims,
                 table=table2.name,
                 pool_type=1,
                 fl_type=0),
}


class ServingPSTest(tf.test.TestCase):

  def test_example_gen(self):
    example_batch = ExampleBatch(batch_size=batch_size)
    for name, meta in features.items():
      named_feature_list = example_batch.named_feature_list.add()
      named_feature_list.id = meta.slot
      named_feature_list.name = name
      if meta.fl_type == 0:
        named_feature_list.type = FeatureListType.INDIVIDUAL
      else:
        named_feature_list.type = FeatureListType.SHARED
      for i in range(batch_size):
        feature = named_feature_list.feature.add()
        if meta.fid_version == 0:
          mask = (1 << 54) - 1
          feature.fid_v1_list.value.extend([
              (meta.slot << 54) | (randint(1, mask) & mask)
              for _ in range(meta.max_sequence_length)
          ])
        else:
          mask = (1 << 48) - 1
          feature.fid_v2_list.value.extend([
              (meta.slot << 48) | (randint(1, mask) & mask)
              for _ in range(meta.max_sequence_length)
          ])

        if named_feature_list.type == FeatureListType.SHARED:
          break

    print(example_batch, flush=True)

  def test_conf_gen(self):
    feature_configs = FeatureConfigs()
    for name, meta in features.items():
      feat_conf = FeatureConfig(table=meta.table)
      if meta.max_sequence_length > 1 and meta.pool_type == 3:
        max_sequence_length = meta.max_sequence_length
      feat_conf.slice_dims.extend(meta.slice_dims)
      if meta.pool_type == 1:
        feat_conf.pooling_type = PoolingType.SUM
      elif meta.pool_type == 2:
        feat_conf.pooling_type = PoolingType.MEAN
      else:
        feat_conf.pooling_type = PoolingType.FIRSTN
      feature_configs.feature_configs[name].CopyFrom(feat_conf)

    bias = OutConfig()
    bias.out_type = OutType.CONCAT
    bias_shape = (batch_size, len(features) - 1)
    sub_shape = bias.shape.add()
    sub_shape.dims.extend(bias_shape)
    for name, meta in features.items():
      if meta.pool_type != 3:
        slice_config = bias.slice_configs.add()
        slice_config.feature_name = name
        slice_config.start = 0
        slice_config.end = 1
    feature_configs.out_configs['bias'].CopyFrom(bias)

    vec = OutConfig()
    vec.out_type = OutType.CONCAT
    vec_shape = (batch_size, (len(features) - 1) * 4)
    sub_shape = vec.shape.add()
    sub_shape.dims.extend(vec_shape)
    for name, meta in features.items():
      if meta.pool_type != 3:
        slice_config = vec.slice_configs.add()
        slice_config.feature_name = name
        slice_config.start = 1
        slice_config.end = 5
    feature_configs.out_configs['vec'].CopyFrom(vec)

    uffm = OutConfig()
    uffm.out_type = OutType.NONE
    uffm_shape = (batch_size, 4)
    for name, meta in features.items():
      if meta.pool_type != 3 and 'user' in name:
        sub_shape = uffm.shape.add()
        sub_shape.dims.extend(uffm_shape)
        slice_config = uffm.slice_configs.add()
        slice_config.feature_name = name
        slice_config.start = 5
        slice_config.end = 8
    feature_configs.out_configs['uffm'].CopyFrom(uffm)

    iffm = OutConfig()
    iffm.out_type = OutType.NONE
    iffm_shape = (batch_size, 4)
    for name, meta in features.items():
      if meta.pool_type != 3 and 'user' not in name:
        sub_shape = iffm.shape.add()
        sub_shape.dims.extend(iffm_shape)
        slice_config = iffm.slice_configs.add()
        slice_config.feature_name = name
        slice_config.start = 5
        slice_config.end = 8
    feature_configs.out_configs['iffm'].CopyFrom(iffm)

    seq = OutConfig()
    seq.out_type = OutType.NONE
    meta = features['f_user_id-f_page']
    seq_shape = (batch_size, meta.slice_dims[0], meta.max_sequence_length)
    sub_shape = seq.shape.add()
    sub_shape.dims.extend(seq_shape)
    slice_config = seq.slice_configs.add()
    slice_config.feature_name = meta.name
    slice_config.start = 0
    slice_config.end = 8
    feature_configs.out_configs['seq'].CopyFrom(seq)

    user_only = OutConfig()
    user_only.out_type = OutType.STACK
    sub_shape = user_only.shape.add()
    user_only_shape = (batch_size, 8, 3)
    sub_shape.dims.extend(user_only_shape)
    for name, meta in features.items():
      if meta.pool_type != 3 and 'user' in name and '-' not in name:
        slice_config = user_only.slice_configs.add()
        slice_config.feature_name = name
        slice_config.start = 8
        slice_config.end = 16
    feature_configs.out_configs['user_only'].CopyFrom(user_only)

    print(feature_configs, flush=True)


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
