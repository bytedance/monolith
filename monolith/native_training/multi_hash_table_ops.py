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
import concurrent.futures
import dataclasses
import hashlib
import os
import threading
from typing import Tuple, Union, Dict, List, Iterable, NamedTuple
import collections

from absl import logging
from google.protobuf import text_format
import tensorflow as tf
from tensorflow.python.framework import ops

from monolith.native_training import basic_restore_hook
from monolith.native_training import entry
from monolith.native_training import hash_filter_ops
from monolith.native_training.hash_table_utils import infer_dim_size
from monolith.native_training.multi_type_hash_table import BaseMultiTypeHashTable
from monolith.native_training import multi_hash_table_ops_pb2
from monolith.native_training import distributed_serving_ops
from monolith.native_training import graph_meta
from monolith.native_training.runtime.ops import gen_monolith_ops
from monolith.native_training import save_utils
from monolith.native_training.model_export.export_context import \
    is_exporting
from monolith.native_training.proto import ckpt_info_pb2
from monolith.native_training.runtime.hash_table import \
    embedding_hash_table_pb2

hash_table_ops = gen_monolith_ops

_TIMEOUT_IN_MS = 60 * 60 * 1000
_MULTI_HASH_TABLE_GRAPH_KEY = "monolith_multi_hash_tables"


class CachedConfig(NamedTuple):
  """Cache the config object to reduce the graph size."""
  # Original configs
  configs: Dict[str, entry.HashTableConfigInstance]

  # Generated data
  # The table names
  table_names: Tuple[str]
  # The multi_hash_table serialized config.
  mconfig: bytes
  # table creation may request the data from other devices.
  mconfig_tensor: tf.Tensor
  # The dim size in each config to make tf function reused.
  dims: Tuple[int]
  # This is generated from mconfig
  slot_expire_time_config: bytes


def convert_to_cached_config(configs: Dict[str, entry.HashTableConfigInstance]):
  mconfig = embedding_hash_table_pb2.MultiEmbeddingHashTableConfig()
  table_names = tuple(sorted(configs.keys()))
  slot_expire_time_config = None
  dims = []
  for table_name in table_names:
    config = configs[table_name]
    table_config = embedding_hash_table_pb2.EmbeddingHashTableConfig()
    table_config.CopyFrom(config.table_config)
    if is_exporting():
      table_config.entry_config.entry_type = embedding_hash_table_pb2.EntryConfig.EntryType.SERVING
    mconfig.names.append(table_name)
    mconfig.configs.append(table_config)
    dims.append(infer_dim_size(table_config))
    if not slot_expire_time_config:
      slot_expire_time_config = table_config.slot_expire_time_config.SerializeToString(
      )
  serialized_mconfig = mconfig.SerializeToString()
  return CachedConfig(configs=configs,
                      table_names=table_names,
                      mconfig=serialized_mconfig,
                      mconfig_tensor=tf.convert_to_tensor(serialized_mconfig),
                      dims=tuple(dims),
                      slot_expire_time_config=slot_expire_time_config)


@dataclasses.dataclass
class MultiHashTableMetadata:
  name_set: set = dataclasses.field(default_factory=set)


# TODO(leqi.zou): Add tf.function when export is fixed
def concat_1d_tensors(*args) -> tf.RaggedTensor:
  """Concat 1D tensors into a Raaged Tensor """
  values = tf.concat(args, axis=0)
  row_lengths = [tf.size(t) for t in args]
  return tf.RaggedTensor.from_row_lengths(values, row_lengths, validate=False)


# TODO(leqi.zou): Add tf.function when export is fixed
def get_list_from_flat_value(key: tf.RaggedTensor, dims: Tuple[int],
                             flat_value: tf.Tensor) -> List[tf.Tensor]:
  row_lengths = key.row_lengths()
  value_lengths = row_lengths * dims
  values = tf.split(flat_value, value_lengths)
  for i in range(len(dims)):
    values[i] = tf.reshape(values[i], [-1, dims[i]])
  return values


# TODO(leqi.zou): Add tf.function when export is fixed
def flatten_n_tensors(*args) -> tf.Tensor:
  flattened_tensors = []
  for tensor in args:
    flattened_tensors.append(tf.reshape(tensor, shape=[-1]))
  return tf.concat(flattened_tensors, axis=0)


class RawMultiTypeHashTable(abc.ABC):
  """Raw lookup API to minimize the overhead transferration between differene devices."""

  @abc.abstractmethod
  def get_ragged_id(self, slot_to_id: Dict[str, tf.Tensor]) -> tf.RaggedTensor:
    """Converts ids to a single ragged id. Graph independent."""
    pass

  @abc.abstractmethod
  def get_flat_value(self, slot_to_value: Dict[str, tf.Tensor]) -> tf.Tensor:
    """Converts values to a single float tensor. Graph independent."""
    pass

  @abc.abstractmethod
  def get_embeddings(self, ragged_id: tf.RaggedTensor,
                     value: tf.Tensor) -> Dict[str, tf.Tensor]:
    """Converts returned flat value into the dict of embeddings. Graph independent."""
    pass

  @abc.abstractmethod
  def raw_lookup(self, ragged_id: tf.RaggedTensor) -> tf.Tensor:
    pass

  @abc.abstractmethod
  def raw_apply_gradients(self, ragged_id: tf.RaggedTensor,
                          flat_grad: tf.Tensor, global_step: tf.Tensor, *args,
                          **kwargs) -> "RawMultiTypeHashTable":
    pass

  @abc.abstractclassmethod
  def raw_assign(self, ragged_id: tf.RaggedTensor, flat_value: tf.Tensor, *args,
                 **kwargs):
    pass


def _convert_to_int64(t):
  if isinstance(t, tf.Tensor):
    return t
  return tf.convert_to_tensor(t, tf.int64)


def _convert_to_float32(t):
  if isinstance(t, tf.Tensor):
    return t
  return tf.convert_to_tensor(t, tf.float32)


class MultiHashTable(BaseMultiTypeHashTable, RawMultiTypeHashTable):
  """
  It maps a int64 to a float32 embedding.
  """
  NAME_PREFIX = "MonolithMultiHashTable"

  def __init__(self,
               cc: CachedConfig = None,
               hash_filter: tf.Tensor = None,
               sync_client: tf.Tensor = None,
               learning_rate_list: List[tf.Tensor] = None,
               name_suffix: str = "",
               saver_parallel: int = -1,
               table_proto: multi_hash_table_ops_pb2.MultiHashTableProto = None,
               import_scope: str = None):
    if table_proto is not None:
      self._init_from_proto(table_proto, import_scope)
      return
    self._dims = cc.dims
    self._slot_expire_time_config = cc.slot_expire_time_config
    self._table_names = cc.table_names
    self._learning_rate = tf.concat(learning_rate_list, axis=0)
    self._saver_parallel = saver_parallel
    self._shared_name = "_".join([MultiHashTable.NAME_PREFIX, name_suffix])
    self._check_and_insert_name(self._shared_name)

    # We separate the table creation and use by using a dummy var.
    # TODO(leqi.zou): we can use register_resource mechanism to solve this problem.
    self._initializer = hash_table_ops.create_monolith_multi_hash_table(
        filter_handle=hash_filter,
        sync_client_handle=sync_client,
        config=cc.mconfig_tensor,
        shared_name=self._shared_name).op
    with tf.control_dependencies([self._initializer]):
      tf.Variable(initial_value=tf.constant(0), trainable=False)

    self._handle = hash_table_ops.read_monolith_multi_hash_table(
        shared_name=self._shared_name)

    tf.compat.v1.get_collection_ref(_MULTI_HASH_TABLE_GRAPH_KEY).append(self)

  def _init_from_proto(
      self,
      proto: multi_hash_table_ops_pb2.MultiHashTableProto = None,
      import_scope: str = None):
    assert isinstance(proto, multi_hash_table_ops_pb2.MultiHashTableProto)
    g = tf.compat.v1.get_default_graph()
    self._dims = tuple(proto.dims)
    self._slot_expire_time_config = proto.slot_expire_time_config
    self._table_names = tuple(proto.table_names)
    self._learning_rate = g.as_graph_element(
        ops.prepend_name_scope(proto.learning_rate_tensor, import_scope))
    self._saver_parallel = proto.saver_parallel
    self._shared_name = proto.shared_name
    self._initializer = g.as_graph_element(
        ops.prepend_name_scope(proto.initializer_op, import_scope))
    self._handle = g.as_graph_element(
        ops.prepend_name_scope(proto.handle_tensor, import_scope))

  @classmethod
  def from_cached_config(cls,
                         cc: CachedConfig,
                         hash_filter: tf.Tensor = None,
                         sync_client: tf.Tensor = None,
                         name_suffix: str = "",
                         saver_parallel: int = -1):
    hash_filter = hash_filter if hash_filter is not None else hash_filter_ops.create_dummy_hash_filter(
        name_suffix=name_suffix)

    sync_client = sync_client if sync_client is not None else distributed_serving_ops.create_dummy_sync_client(
    )
    dummy_sync_client = None

    learning_rate_list = []

    table_names = list(cc.configs.keys())
    for table_name in table_names:
      config = cc.configs[table_name]

      if len(config.learning_rate_fns) != len(
          config.table_config.entry_config.segments):
        raise ValueError(
            "Size of learning_rate_fns and size of segments must be equal.")
      learning_rate_list.append(config.call_learning_rate_fns())

    if tf.compat.v1.get_default_graph() != cc.mconfig_tensor.graph:
      # In this case, we can't reuse mconfig_tensor
      cc = cc._replace(mconfig_tensor=tf.convert_to_tensor(cc.mconfig))

    return cls(cc=cc,
               hash_filter=hash_filter,
               sync_client=sync_client,
               learning_rate_list=learning_rate_list,
               name_suffix=name_suffix,
               saver_parallel=saver_parallel)

  @classmethod
  def from_configs(cls, configs: Dict[str, entry.HashTableConfigInstance],
                   *args, **kwargs):
    cc = convert_to_cached_config(configs)
    return cls.from_cached_config(cc, *args, **kwargs)

  @staticmethod
  def from_proto(table_proto, import_scope=None):
    return MultiHashTable(table_proto=table_proto, import_scope=import_scope)

  def to_proto(self, export_scope=None):
    if (export_scope is not None and
        not self._handle.name.startswith(export_scope)):
      return None
    proto = multi_hash_table_ops_pb2.MultiHashTableProto()
    proto.dims.extend(self._dims)
    proto.slot_expire_time_config = self._slot_expire_time_config
    proto.table_names.extend(self._table_names)
    proto.learning_rate_tensor = ops.strip_name_scope(self._learning_rate.name,
                                                      export_scope)
    proto.saver_parallel = self._saver_parallel
    proto.shared_name = self._shared_name
    proto.initializer_op = ops.strip_name_scope(self._initializer.name,
                                                export_scope)
    proto.handle_tensor = ops.strip_name_scope(self._handle.name, export_scope)
    return proto

  @classmethod
  def _check_and_insert_name(cls, name):
    meta = graph_meta.get_meta("multi_hash_table_metadata",
                               MultiHashTableMetadata)
    if name in meta.name_set:
      raise ValueError("shared_name {} has already been used.".format(name))
    meta.name_set.add(name)

  @property
  def table_names(self):
    """Return table names."""
    return self._table_names

  @property
  def handle(self):
    return self._handle

  @property
  def shared_name(self):
    return self._shared_name

  @property
  def initializer(self):
    return self._initializer

  """Implements BaseMultiHashTable"""

  def assign(self,
             slot_to_id_and_value: Dict[str, Tuple[tf.Tensor, tf.Tensor]],
             req_time: tf.Tensor = None,
             enable_inter_table_parallelism: bool = False) -> "MultiHashTable":
    ragged_id = self.get_ragged_id(
        {k: _convert_to_int64(v[0]) for k, v in slot_to_id_and_value.items()})
    flat_value = self.get_flat_value(
        {k: _convert_to_float32(v[1]) for k, v in slot_to_id_and_value.items()})
    return self.raw_assign(ragged_id, flat_value)

  def assign_add(self,
                 slot_to_id_and_value: Dict[str, Tuple[tf.Tensor, tf.Tensor]],
                 req_time: tf.Tensor = None) -> "MultiHashTable":
    if req_time is None:
      req_time = tf.constant(0, dtype=tf.int64)
    ragged_id = self.get_ragged_id(
        {k: _convert_to_int64(v[0]) for k, v in slot_to_id_and_value.items()})
    flat_value = self.get_flat_value(
        {k: _convert_to_float32(v[1]) for k, v in slot_to_id_and_value.items()})
    new_handle = hash_table_ops.monolith_multi_hash_table_assign_add(
        mtable=self._handle,
        id=ragged_id.values,
        id_split=ragged_id.row_splits,
        value=flat_value,
        update_time=req_time)
    return self._copy_with_new_table(new_handle)

  def lookup(self, slot_to_id: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    ragged_id = self.get_ragged_id(
        {k: _convert_to_int64(v) for k, v in slot_to_id.items()})
    flat_embedding = self.raw_lookup(ragged_id)
    slot_to_embeddings = self.get_embeddings(ragged_id, flat_embedding)
    slot_to_embeddings = {
        k: v for k, v in slot_to_embeddings.items() if k in slot_to_id
    }
    return slot_to_embeddings

  def lookup_entry(
      self,
      slot_to_id: Dict[str, tf.Tensor],
      enable_inter_table_parallelism: bool = False) -> Dict[str, tf.Tensor]:
    raise NotImplementedError("")

  def apply_gradients(self,
                      slot_to_id_and_grad: Dict[str, Tuple[tf.Tensor,
                                                           tf.Tensor]],
                      global_step: tf.Tensor,
                      req_time: tf.Tensor = None) -> "MultiHashTable":
    ragged_id = self.get_ragged_id(
        {k: _convert_to_int64(v[0]) for k, v in slot_to_id_and_grad.items()})
    flat_grad = self.get_flat_value(
        {k: _convert_to_float32(v[1]) for k, v in slot_to_id_and_grad.items()})
    return self.raw_apply_gradients(ragged_id, flat_grad, global_step, req_time)

  def save(self, basename: tf.Tensor) -> "MultiHashTable":
    new_handle = hash_table_ops.monolith_multi_hash_table_save(
        mtable=self._handle,
        basename=basename,
        nshards=self._saver_parallel,
        slot_expire_time_config=self._slot_expire_time_config)
    return self._copy_with_new_table(new_handle)

  def restore(self, basename: tf.Tensor) -> "MultiHashTable":
    new_handle = hash_table_ops.monolith_multi_hash_table_restore(
        mtable=self._handle, basename=basename)
    return self._copy_with_new_table(new_handle)

  def as_op(self, *args, **kwargs):  # pylint: disable=unused-argument
    return self._handle.op

  def _copy_with_new_table(self, handle: tf.Tensor):
    copied = copy.copy(self)
    copied._handle = handle
    return copied

  def feature_stat(self, basename: tf.Tensor):
    """Only to be called after hash tables are saved."""
    features, counts = hash_table_ops.monolith_multi_hash_table_feature_stat(
        basename)
    return features, counts

  """
  Fused ops for sync training.
  """

  # This is a very concise API that supports fused lookup, without mapping the
  # IDs to its slots.
  def fused_lookup(self, ids: tf.Tensor, fused_slot_size: tf.Tensor,
                   num_of_shards: int) -> Tuple[tf.Tensor]:
    raise NotImplementedError("")

  # This is a very concise API that supports fused optimize, without mapping the
  # IDs to its slots.
  def fused_apply_gradient(
      self,
      ids: tf.Tensor,
      fused_slot_size: tf.Tensor,
      id_grads: tf.Tensor,
      id_offsets: tf.Tensor,
      grad_offsets: tf.Tensor,
      global_step: tf.Tensor,
      req_time: tf.Tensor,
      num_of_shards: int,
      enable_grad_accumulation: bool = False) -> "MultiHashTable":
    raise NotImplementedError("")

  def get_table_dim_sizes(self):
    return self._dims

  """
  RawMultiTypeHashTable APIs
  """

  def raw_lookup(self, ragged_id: tf.RaggedTensor) -> tf.Tensor:
    return hash_table_ops.monolith_multi_hash_table_lookup(
        mtable=self._handle, id=ragged_id.values, id_split=ragged_id.row_splits)

  def raw_apply_gradients(self,
                          ragged_id: tf.RaggedTensor,
                          flat_grad: tf.Tensor,
                          global_step: tf.Tensor,
                          req_time: tf.Tensor = None) -> RawMultiTypeHashTable:
    if req_time is None:
      req_time = tf.constant(0, dtype=tf.int64)
    return self._copy_with_new_table(
        hash_table_ops.monolith_multi_hash_table_optimize(
            mtable=self._handle,
            id=ragged_id.values,
            id_split=ragged_id.row_splits,
            value=flat_grad,
            learning_rate=self._learning_rate,
            update_time=req_time,
            global_step=global_step))

  def raw_assign(self,
                 ragged_id: tf.RaggedTensor,
                 flat_value: tf.Tensor,
                 req_time: tf.Tensor = None):
    logging.info(f"raw_assign {self._handle}")
    if req_time is None:
      req_time = tf.constant(0, dtype=tf.int64)
    return self._copy_with_new_table(
        hash_table_ops.monolith_multi_hash_table_assign(
            mtable=self._handle,
            id=ragged_id.values,
            id_split=ragged_id.row_splits,
            value=flat_value,
            update_time=req_time))

  def get_embeddings(self, ragged_id: tf.RaggedTensor,
                     value: tf.Tensor) -> Dict[str, tf.Tensor]:
    d = {}
    values = get_list_from_flat_value(ragged_id, self._dims, value)
    for name, value in zip(self._table_names, values):
      d[name] = value
    return d

  def get_ragged_id(self, slot_to_id: Dict[str, tf.Tensor]):
    tensors = []
    empty_id = tf.constant([], dtype=tf.int64)
    for name in self._table_names:
      tensors.append(slot_to_id.get(name, empty_id))
    return concat_1d_tensors(*tensors)

  def get_flat_value(self, slot_to_value: Dict[str, tf.Tensor]):
    tensors = []
    empty_value = tf.zeros([0, 1])
    for name in self._table_names:
      tensors.append(slot_to_value.get(name, empty_value))
    return flatten_n_tensors(*tensors)


class MultiHashTableCheckpointSaverListener(tf.estimator.CheckpointSaverListener
                                           ):
  """Saves the hash tables when saver is run."""

  def __init__(self, basename: str, write_ckpt_info: bool = True):
    """|basename| should be a file name which is same as what is passed to saver."""
    super().__init__()
    self._write_ckpt_info = write_ckpt_info
    self._helper = save_utils.SaveHelper(basename)
    self._table_id_to_placeholder = {}
    self._features_counts_tuples = []
    self._save_op = self._build_save_graph()

  def before_save(self, sess, global_step_value):
    """
    We use before save so the checkpoint file is updated after we successfully
    save the hash table.
    """
    logging.info("Starting saving MultiHashTables.")
    feed_dict = {}
    base_dir = self._helper.get_ckpt_asset_dir(
        self._helper.get_ckpt_prefix(global_step_value))
    tf.io.gfile.makedirs(base_dir)
    for table in tf.compat.v1.get_collection(_MULTI_HASH_TABLE_GRAPH_KEY):
      table_basename = base_dir + table.shared_name
      feed_dict.update(
          {self._table_id_to_placeholder[id(table)]: table_basename})
    sess.run(self._save_op,
             feed_dict=feed_dict,
             options=tf.compat.v1.RunOptions(timeout_in_ms=_TIMEOUT_IN_MS))
    logging.info("Finished saving MultiHashTables.")

    if self._write_ckpt_info:
      logging.info("Start collecting slot fid count.")
      features_counts_list = sess.run(fetches=self._features_counts_tuples,
                                      feed_dict=feed_dict)
      logging.info("Start writing CkptInfo.")
      feature_to_fid_count = collections.defaultdict(int)
      for features_counts in features_counts_list:
        features = features_counts[0].tolist()
        counts = features_counts[1].tolist()
        if not len(features) == len(counts):
          raise ValueError(
              "Number of features [{}] does not match number of fid counts [{}]"
              .format(len(features), len(counts)))
        for feature, count in zip(features, counts):
          feature_to_fid_count[feature] += count

      info = ckpt_info_pb2.CkptInfo()
      for feature, count in feature_to_fid_count.items():
        info.feature_counts[feature] = count
      ckpt_dir = os.path.dirname(self._helper._basename)
      with tf.io.gfile.GFile(
          os.path.join(ckpt_dir, f"ckpt.info-{global_step_value}"), "w") as f:
        f.write(str(info))
      logging.info("Finished writing CkptInfo.")

  def _build_save_graph(self) -> tf.Operation:
    save_ops = []
    for table in ops.get_collection(_MULTI_HASH_TABLE_GRAPH_KEY):
      table_basename = tf.compat.v1.placeholder(tf.string, shape=[])
      self._table_id_to_placeholder.update({id(table): table_basename})
      save_op = table.save(basename=table_basename).as_op()
      save_ops.append(save_op)
      if self._write_ckpt_info:
        with tf.control_dependencies([save_op]):
          self._features_counts_tuples.append(
              table.feature_stat(table_basename))
    with tf.control_dependencies(save_ops):
      return tf.no_op(name="multi_hashtable_save_all")


class MultiHashTableCheckpointRestorerListener(
    basic_restore_hook.CheckpointRestorerListener):
  """Restores the hash tables from basename"""

  def __init__(self, basename: str, ps_monitor=None):
    super().__init__()
    self._basename = basename
    self._ps_monitor = ps_monitor

    self._helper = save_utils.SaveHelper(basename)
    self._table_id_to_placeholder = {}
    self._restore_ops_per_device = self._build_restore_graph()

  def before_restore(self, session):
    """
    We use before restore so as to strictly control the order of restorer listeners.

    """
    ckpt_prefix = tf.train.latest_checkpoint(os.path.dirname(self._basename))
    if not ckpt_prefix:
      logging.info("No checkpoint found in %s. Skip the hash tables restore.",
                   self._basename)
      return

    logging.info("Restore hash tables from %s.", ckpt_prefix)
    asset_dir = self._helper.get_ckpt_asset_dir(ckpt_prefix)
    init_ops = []
    feed_dict = {}
    for mtable in tf.compat.v1.get_collection(_MULTI_HASH_TABLE_GRAPH_KEY):
      init_ops.append(mtable.initializer)
      table_basename = asset_dir + mtable.shared_name
      feed_dict.update(
          {self._table_id_to_placeholder[id(mtable)]: table_basename})

    session.run(init_ops)
    restore_ops_all = []
    for device, restore_ops in self._restore_ops_per_device.items():
      if not self._ps_monitor or self._ps_monitor.is_ps_uninitialized(
          session, device):
        restore_ops_all.extend(restore_ops)
    session.run(restore_ops_all,
                feed_dict=feed_dict,
                options=tf.compat.v1.RunOptions(timeout_in_ms=_TIMEOUT_IN_MS))
    logging.info("Finished restore.")

  def _build_restore_graph(self):
    restore_ops_per_device = collections.defaultdict(list)
    for table in ops.get_collection(_MULTI_HASH_TABLE_GRAPH_KEY):
      table_basename = tf.compat.v1.placeholder(tf.string, shape=[])
      self._table_id_to_placeholder.update({id(table): table_basename})
      restore_op = table.restore(basename=table_basename).as_op()
      restore_ops_per_device[table.handle.device].append(restore_op)
    return restore_ops_per_device


class MultiHashTableRestorerSaverListener(tf.estimator.CheckpointSaverListener):
  """Since we use restore to remove stale entries,
  we create a saver listener here."""

  def __init__(self, ckpt_prefix: str):
    self._l = MultiHashTableCheckpointRestorerListener(ckpt_prefix)

  def after_save(self, session, global_step_value):
    self._l.before_restore(session)


ops.register_proto_function(
    _MULTI_HASH_TABLE_GRAPH_KEY,
    proto_type=multi_hash_table_ops_pb2.MultiHashTableProto,
    to_proto=MultiHashTable.to_proto,
    from_proto=MultiHashTable.from_proto)
