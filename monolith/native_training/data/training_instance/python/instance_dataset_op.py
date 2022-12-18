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
from enum import Enum

import tensorflow as tf

from tensorflow.python import tf2
from tensorflow.python.data.experimental.ops import matching_files
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import convert
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.platform import resource_loader
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.util.tf_export import tf_export

from monolith.native_training.distribute import distributed_dataset
from monolith.native_training.hooks import ckpt_hooks
from monolith.native_training.runner_utils import RunnerConfig
from monolith.native_training.runtime.ops import gen_monolith_ops

instance_dataset_op = gen_monolith_ops


class _PBInstanceDataset(dataset_ops.DatasetSource):

  def __init__(self, file_name, use_snappy=False, **kwargs):
    self._file_name = file_name
    self._use_snappy = use_snappy

    self._has_sort_id = kwargs.get('has_sort_id', True)
    self._kafka_dump = kwargs.get('kafka_dump', False)
    self._kafka_dump_prefix = kwargs.get('kafka_dump_prefix', False)

    variant_tensor = instance_dataset_op.instance_dataset(
        file_name=tf.convert_to_tensor(self._file_name, dtype=tf.string),
        use_snappy=tf.convert_to_tensor(self._use_snappy, dtype=tf.bool),
        has_sort_id=tf.convert_to_tensor(self._has_sort_id, dtype=tf.bool),
        kafka_dump=tf.convert_to_tensor(self._kafka_dump, dtype=tf.bool),
        kafka_dump_prefix=tf.convert_to_tensor(self._kafka_dump_prefix,
                                               dtype=tf.bool))
    logging.info("Start init of the pb instance dataset base.")
    super(_PBInstanceDataset, self).__init__(variant_tensor)

  @property
  def element_spec(self):
    return tensor_spec.TensorSpec([], dtypes.string)


class PBInstanceDatasetV2(dataset_ops.DatasetV2):
  """从标准输入/pb文件中读取序列化Instance, 不反序列化

  Args:
    file_name (:obj:`str`): 文件名, 如果为空, 则从stdin读取数据
    use_snappy (:obj:`str`): 输入文件是不否是snappy压缩的
    has_sort_id (:obj:`bool`): 输入数据中是否带8 bytes前缀标识, 表明sort_id
    kafka_dump (:obj:`bool`): 输入数据中是否带8 bytes前缀标识, 表明kafka_dump
    kafka_dump_prefix (:obj:`bool`): 输入数据中是否带8 bytes前缀标识, 表明kafka_dump_prefix
    
  Raises:
    TypeError: 如果有任何参数与类型不匹配, 则抛TypeError
    ValueError: 如果有任何值与期望不匹配, 则抛ValueError
  
  """

  def __init__(self, file_name, use_snappy=False, **kwargs):
    self._file_name = file_name
    self._use_snappy = use_snappy
    self._kwargs = kwargs

    if isinstance(file_name, str) and not file_name:
      # This is the special case that dataset uses stdin as the input.
      # In this case, we should diable the ckpt save/restore.
      ckpt_hooks.disable_iterator_save_restore()

    def creator_fn():
      return _PBInstanceDataset(file_name, use_snappy, **self._kwargs)

    self._impl = creator_fn()
    variant_tensor = self._impl._variant_tensor
    logging.info("Start init of the pb instance dataset v2")
    super(PBInstanceDatasetV2, self).__init__(variant_tensor)
    logging.info("Finish init of the pb instance dataset v2")

  def _clone(self, file_name, use_snappy=False, **kwargs):
    _kwargs = self._kwargs.copy()
    _kwargs.update(kwargs)
    return PBInstanceDatasetV2(file_name or self._file_name, use_snappy or
                               self._use_snappy, **_kwargs)

  @property
  def element_spec(self):
    return tensor_spec.TensorSpec([], dtypes.string)

  def _inputs(self):
    return []


#TODO(leqi.zou): We should rewrite this to make it more clear.
def create_instance_dataset(files_list=None,
                            use_snappy=False,
                            expand_glob_path=False,
                            cycle_length=4,
                            num_parallel_calls=4,
                            block_length=1,
                            enable_sharding: bool = False,
                            shard_index: int = None,
                            shard_num: int = None,
                            enable_dynamic_sharding=False,
                            **kwargs):
  if files_list is None:
    # use stdin
    files_list = [""]
  if len(
      files_list
  ) == 1 and not expand_glob_path and not enable_sharding and not enable_dynamic_sharding:
    if len(files_list[0]) > 0 and not tf.io.gfile.exists(files_list[0]):
      logging.fatal('File not found: {}'.format(files_list[0]))
    return PBInstanceDatasetV2(file_name=files_list[0],
                               use_snappy=use_snappy,
                               **kwargs)
  map_func = lambda file_name: PBInstanceDatasetV2(
      file_name=file_name, use_snappy=use_snappy, **kwargs)
  if enable_dynamic_sharding:
    files_list = distributed_dataset.create_dynamic_sharding_dataset(files_list)
    return files_list.flat_map(map_func)
  elif not enable_sharding:
    if expand_glob_path:
      files_list = matching_files.MatchingFilesDataset(files_list)
    else:
      files_list = tf.data.Dataset.from_tensor_slices(files_list)
  else:
    # We should use only 1 pattern for the sharded hdfs reading.
    assert len(files_list) == 1
    # List all the files via the list_files op.
    files_list = matching_files.MatchingFilesDataset(files_list)
    # Shard it according to the preallocated index.
    files_list = files_list.shard(shard_num, shard_index)
    logging.info("Shard the input files for shard {}/{}.".format(
        shard_index, shard_num))
    use_snappy = True

  dataset = files_list.interleave(map_func=map_func,
                                  cycle_length=cycle_length,
                                  block_length=block_length,
                                  num_parallel_calls=num_parallel_calls,
                                  deterministic=False)
  return dataset


PBInstanceDataset = PBInstanceDatasetV2
