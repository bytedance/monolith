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
from typing import Tuple, Union, Dict, List
from collections import defaultdict

from absl import logging
from google.protobuf import text_format
import tensorflow as tf
from tensorflow.python.framework import ops

from monolith.native_training import basic_restore_hook
from monolith.native_training import entry
from monolith.native_training import hash_filter_ops
from monolith.native_training import distributed_serving_ops
from monolith.native_training import graph_meta
from monolith.native_training import hash_table_ops_pb2
from monolith.native_training.runtime.ops import gen_monolith_ops
from monolith.native_training import save_utils
from monolith.native_training.hash_table_utils import infer_dim_size
from monolith.utils import get_libops_path
from monolith.native_training.model_export.export_context import \
    is_exporting
from monolith.native_training.runtime.hash_table import \
    embedding_hash_table_pb2

hash_table_ops = gen_monolith_ops

_TIMEOUT_IN_MS = 60 * 60 * 1000


class BaseHashTable(abc.ABC):
  """
  The base class for the hash table.

  For the write operation, it will return a new HashTable. This makes it easier
  to chain operations that need to use the updated tables. User can use this
  behavior to balance the parallelism and the data freshness.
  """

  @abc.abstractmethod
  def assign(self,
             ids: tf.Tensor,
             values: tf.Tensor,
             req_time: tf.Tensor = None) -> "BaseHashTable":
    """
    Assign values to |id| entry in hash table.
    ids - a 1D tensor represents which entry should be added by value
    values - a 2D tensor. The first dim should equal to ids's length, the second dim should
    equal to hash_table's dim size.

    Returns updated hash table.
    """
    pass

  @abc.abstractmethod
  def assign_add(self,
                 ids: tf.Tensor,
                 values: tf.Tensor,
                 req_time: tf.Tensor = None) -> "BaseHashTable":
    """
    Assign add values to |id| entry in hash table.
    ids - a 1D tensor represents which entry should be added by value
    values - a 2D tensor. The first dim should equal to ids's length, the second dim should
    equal to hash_table's dim size.

    Returns updated hash table.
    """
    pass

  @abc.abstractmethod
  def lookup(self,
             ids: tf.Tensor,
             use_multi_threads=False,
             enable_dedup=False) -> tf.Tensor:
    """
    Look up the embeddings in hash table. The embedding will be summed up in the same batch.

    ids - a 1D int64 tensor

    use_multi_threads - True if the caller wants to lookup using multi-threads.

    enable_dedup - True if the caller wants to lookup without duplicate ids

    Returns a 2-D tensor which maps id to embeddings.
    """
    pass

  @property
  @abc.abstractmethod
  def dim_size(self):
    pass

  @abc.abstractmethod
  def apply_gradients(self,
                      ids: tf.Tensor,
                      grads: tf.Tensor,
                      global_step: tf.Tensor,
                      use_multi_threads=False,
                      enable_dedup=False,
                      req_time: tf.Tensor = None) -> "BaseHashTable":
    """Applies the gradients with respect to the ids."""
    pass

  @abc.abstractmethod
  def as_op(self) -> Union[tf.Tensor, tf.Operation]:
    """
    Convert hash table to an op or tensor. Useful to do the dependency control.
    """
    pass


_HASH_TABLE_GRAPH_KEY = "monolith_hash_tables"


@dataclasses.dataclass
class HashTableMetadata:
  name_set: set = dataclasses.field(default_factory=set)
  tensor_table_to_obj_dict: Dict = dataclasses.field(default_factory=dict)


_BOOL_MAP = {
    None: hash_table_ops_pb2.HashTableProto.kBoolNone,
    False: hash_table_ops_pb2.HashTableProto.kFalse,
    True: hash_table_ops_pb2.HashTableProto.kTrue,
}

_BOOL_REVERSE_MAP = {v: k for k, v in _BOOL_MAP.items()}


class HashTable(BaseHashTable):
  """
  It maps a int64 to a float32 embedding.
  """

  def __init__(self,
               table: tf.Tensor = None,
               shared_name: str = None,
               dim_size: int = None,
               slot_expire_time_config: bytes = None,
               learning_rate_tensor: tf.Tensor = None,
               saver_parallel: int = -1,
               extra_restore_names=None,
               table_proto=None,
               import_scope=None):
    if table_proto is not None:
      self._init_from_proto(table_proto, import_scope)
      return
    self._table = table
    self._dim_size = dim_size
    self._init_table_name = shared_name
    self._check_and_insert_name(shared_name)
    self._slot_expire_time_config = slot_expire_time_config
    self._learning_rate_tensor = learning_rate_tensor
    self._saver_parallel = saver_parallel
    self._extra_restore_names = extra_restore_names or []
    self.export_share_embedding = None
    ops.get_collection_ref(_HASH_TABLE_GRAPH_KEY).append(self)

  def _init_from_proto(self,
                       proto: hash_table_ops_pb2.HashTableProto = None,
                       import_scope: str = None):
    g = tf.compat.v1.get_default_graph()
    self._table = g.as_graph_element(
        ops.prepend_name_scope(proto.table_tensor, import_scope))
    self._dim_size = proto.dim_size
    self._init_table_name = proto.shared_name
    self._slot_expire_time_config = proto.slot_expire_time_config
    self._learning_rate_tensor = g.as_graph_element(
        ops.prepend_name_scope(proto.learning_rate_tensor, import_scope))
    self._saver_parallel = proto.saver_parallel
    self._extra_restore_names = tuple(proto.extra_restore_names)
    self.export_share_embedding = _BOOL_REVERSE_MAP[
        proto.export_share_embedding]

  def to_proto(self, export_scope=None):
    if (export_scope is not None and
        not self._table.name.startswith(export_scope)):
      return None
    proto = hash_table_ops_pb2.HashTableProto()
    proto.table_tensor = ops.strip_name_scope(self._table.name, export_scope)
    proto.dim_size = self._dim_size
    proto.shared_name = self._init_table_name
    proto.slot_expire_time_config = self._slot_expire_time_config
    proto.learning_rate_tensor = ops.strip_name_scope(
        self._learning_rate_tensor.name, export_scope)
    proto.saver_parallel = self._saver_parallel
    proto.extra_restore_names.extend(self._extra_restore_names)
    proto.export_share_embedding = _BOOL_MAP[self.export_share_embedding]
    return proto

  @staticmethod
  def from_proto(table_proto, import_scope=None):
    return HashTable(table_proto=table_proto, import_scope=import_scope)

  @classmethod
  def get_metadata(cls) -> HashTableMetadata:
    return graph_meta.get_meta("hash_table_metadata", HashTableMetadata)

  @classmethod
  def _check_and_insert_name(cls, name):
    meta = cls.get_metadata()
    if name in meta.name_set:
      raise ValueError("shared_name {} has already been used.".format(name))
    meta.name_set.add(name)

  @property
  def table(self):
    """Returns table tensor."""
    return self._table

  @property
  def name(self):
    """Return table name."""
    return self._init_table_name

  @property
  def extra_restore_names(self):
    """Returns other possible original table names."""
    return self._extra_restore_names

  @property
  def dim_size(self):
    """Return dim size."""
    return self._dim_size

  """Implements BaseHashTable"""

  def assign(self,
             ids: tf.Tensor,
             values: tf.Tensor,
             req_time: tf.Tensor = None) -> "HashTable":
    if req_time is None:
      req_time = tf.constant(0, dtype=tf.int64)
    # Makes test easier
    ids = tf.convert_to_tensor(ids, tf.int64)
    values = tf.convert_to_tensor(values, tf.float32)
    return self._copy_with_new_table(
        hash_table_ops.monolith_hash_table_assign(self._table, ids, values,
                                                  req_time))

  def assign_add(self,
                 ids: tf.Tensor,
                 values: tf.Tensor,
                 req_time: tf.Tensor = None) -> "HashTable":
    if req_time is None:
      req_time = tf.constant(0, dtype=tf.int64)
    return self._copy_with_new_table(
        hash_table_ops.monolith_hash_table_assign_add(self._table, ids, values,
                                                      req_time))

  def lookup(self,
             ids: tf.Tensor,
             use_multi_threads=False,
             enable_dedup=False) -> tf.Tensor:
    lookup_tensor = hash_table_ops.monolith_hash_table_lookup(
        self._table, ids, self._dim_size, use_multi_threads=use_multi_threads)
    return lookup_tensor

  def lookup_entry(self, ids: tf.Tensor) -> tf.Tensor:
    lookup_tensor = hash_table_ops.monolith_hash_table_lookup_entry(
        self._table, ids)
    return lookup_tensor

  def apply_gradients(self,
                      ids: tf.Tensor,
                      grads: tf.Tensor,
                      global_step: tf.Tensor,
                      use_multi_threads=False,
                      enable_dedup=False,
                      req_time: tf.Tensor = None) -> "HashTable":
    if req_time is None:
      req_time = tf.constant(0, dtype=tf.int64)
    updated_op = hash_table_ops.monolith_hash_table_optimize(
        self._table,
        ids,
        grads,
        self._learning_rate_tensor,
        req_time,
        global_step,
        use_multi_threads=use_multi_threads,
        enable_dedup=enable_dedup)
    with tf.control_dependencies([updated_op]):
      new_table = self._copy_with_new_table(tf.identity(self._table))
    return new_table

  def as_op(self):
    return self._table

  def table_update(self, update_op: tf.Tensor) -> "HashTable":
    with tf.control_dependencies([update_op]):
      new_table = self._copy_with_new_table(tf.identity(self._table))
    return new_table

  def save(self, basename: tf.Tensor, random_sleep_ms: int = 0) -> "HashTable":
    new_table = hash_table_ops.monolith_hash_table_save(
        self._table,
        basename,
        slot_expire_time_config=self._slot_expire_time_config,
        nshards=self._saver_parallel,
        random_sleep_ms=random_sleep_ms)
    return self._copy_with_new_table(new_table)

  def restore(self, basename: tf.Tensor) -> "HashTable":
    new_table = hash_table_ops.monolith_hash_table_restore(
        self._table, basename)
    return self._copy_with_new_table(new_table)

  def _copy_with_new_table(self, new_table: tf.Tensor):
    copied = copy.copy(self)
    copied.__dict__["_table"] = new_table
    return copied

  def size(self) -> tf.Tensor:
    return hash_table_ops.monolith_hash_table_size(self._table)

  def save_as_tensor(self, shard_idx, num_shards, limit,
                     offset) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Dumps the hash table as tensors.
    Args:
    shard_idx - the idx of shard, should be within [0, num_shards)
    num_shards - the number of shards we want to have. This is helpful for dumping tensor
    in parallel.
    limit - at most, how many tensors will be output. If the output dump tensor's size is
    less than limit, it means we finish the current shard.
    offset - the offset from current shard. If we want to start from begining, set it to 0.

    Returns 2 tensors:
    1 A 0-D int64 tensor represents the new offset.
    2. A 1-D string tensor which is serialized format of monolith::EntryDump.
    """
    shard_idx = tf.convert_to_tensor(shard_idx, tf.int32)
    num_shards = tf.convert_to_tensor(num_shards, tf.int32)
    limit = tf.convert_to_tensor(limit, tf.int64)
    offset = tf.convert_to_tensor(offset, tf.int64)
    return hash_table_ops.monolith_hash_table_save_as_tensor(
        self._table,
        shard_idx,
        num_shards,
        limit,
        offset,
        name="monolith_hash_table_save_as_tensor")


def fused_lookup(tables: tf.Tensor, ids: tf.Tensor, fused_slot_size: tf.Tensor,
                 num_of_shards: int) -> Tuple[tf.Tensor]:
  """ A fused operation for lookup.

  This op takes a fused_ids, and fused_slot_sizes,
  lookup via a list of tables, and return a concatenated embedding. Several
  auxiluary results are also returned to simplify processing at later stages.

  Example: 
    tables = [{1: [1], 2: [2]}, {3: [3, 3], 4: [4, 4]}]
    ids = [1, 3, 2, 4]
    fused_slot_size = [1, 1, 1, 1]
    num_of_shards = 2

  After the op, the outputs are:
    embeddings = [1, 3, 3, 2, 4, 4]
    recv_splits = [3, 3]
    id_offsets = [0, 1, 2, 3]
    emb_offsets = [0, 1, 3, 4]
    embedding_sizes = [1, 2, 1, 2]

  For a setup of K tables, N shards:
  Args:
    tables:  A list of tables with shape [K], it is ordered by the tables' hashed_keys.
    ids:  A flattened IDs with shape [M], M=sum(fused_slot_size[i]).
    fused_slot_size: A list with shape [K*N].
    num_of_shards: a integer N.
  Returns:
    embeddings: A 1-D flattened embeddings with shape [L], L=sum(embedding_sizes[i])
    recv_splits: A 1-D flattened tensor with shape [N].
    id_offsets: A 1-D flattened tensor wih shape [K*N], and it is an artifact used by apply_gradients.
    emb_offsets: A 1-D flattened tensor with shape [K*N], and it is an artifact used by apply_gradients.
    embedding_sizes: A 1-D flattened tensor with shape [K*N], and it is an artifact used by worker side.
  """
  return hash_table_ops.monolith_hash_table_fused_lookup(
      tables, ids, fused_slot_size, num_of_shards)


def fused_apply_gradient(
    tables: List[tf.Tensor],
    ids: tf.Tensor,
    fused_slot_size: tf.Tensor,
    id_grads: tf.Tensor,
    id_offsets: tf.Tensor,
    grad_offsets: tf.Tensor,
    learning_rate_tensors: tf.Tensor,
    learning_rate_lengths: tf.Tensor,
    req_time: tf.Tensor,
    global_step: tf.Tensor,
    num_of_shards: int,
    enable_grad_accumulation: bool = False,
):
  """A fused operation for applying gradients.
  
  This op takes fused ids and fused
  gradients, and several other positional information, and applies the gradient
  updates to the list of tables.

  Example:
    tables = [{1: [1], 2: [2]}, {3: [3, 3], 4: [4, 4]}]
    ids = [1, 3, 2, 4]
    fused_slot_size = [1, 1, 1, 1]
    id_grads = [1, 2, 2, 1, 2, 2]
    id_offsets = [0, 1, 2, 3]
    grad_offsets = [0, 1, 3, 4]
    learning_rate_tensors = [1, 1]
    learning_rate_lengths = [1, 1]
    req_time = time_in_seconds
    global_step = 1
    num_of_shards = 2

  After calling the op, with SGD, the output is the updated table:
    tables = [{1: [0], 2: [1]}, {3: [1, 1], 4: [2, 2]}]

  For a setup of K tables, N shards:
  Args:
    tables:  A list of tables with shape [K], it is ordered by the tables' hashed_keys.
    ids:  A flattened IDs with shape [M], M=sum(fused_slot_size[i]).
    fused_slot_size: A list with shape [K*N].
    id_offsets: A 1-D flattened tensor wih shape [K*N], it is an intermediate artifact from fused_lookup.
    grad_offsets: A 1-D flattened tensor with shape [K*N], it is an intermediate artifact from fused_lookup.
    learning_rate_tensors: A 1-D flattened tensor wih shape [L], L=sum(learning_rate_lengths).
    learning_rate_lengths: A 1-D flattened tensor wih shape [K].
    req_time: A scalar tensor with type tf.int64.
    global_step: A scalar tensor with type tf.int64.
    num_of_shards: a integer N.
    enable_grad_accumulation: if enabled, the gradient accumulation is activated from the PS side for cross-shard gradients.
  Returns:
    An updated tables tensor.
  """
  return hash_table_ops.monolith_hash_table_fused_optimize(
      tables, ids, fused_slot_size, id_grads, id_offsets, grad_offsets,
      learning_rate_tensors, learning_rate_lengths, req_time, global_step,
      num_of_shards, enable_grad_accumulation)


def hash_table_from_config(config: entry.HashTableConfigInstance,
                           hash_filter: tf.Tensor = None,
                           name_suffix="",
                           sync_client: tf.Tensor = None,
                           saver_parallel: int = -1) -> HashTable:
  table_config = embedding_hash_table_pb2.EmbeddingHashTableConfig()
  table_config.CopyFrom(config.table_config)
  assert table_config.HasField("type")
  table_type = table_config.WhichOneof("type")
  logging.info("Hash table type: {}".format(table_type))
  use_gpu = table_type == "gpucuco"
  d = "/device:GPU:0" if use_gpu else "/device:CPU:0"

  if is_exporting():
    table_config.entry_config.entry_type = embedding_hash_table_pb2.EntryConfig.EntryType.SERVING
  dim_size = infer_dim_size(config.table_config)
  table_config_str = table_config.SerializeToString()
  slot_expire_time_config = config.table_config.slot_expire_time_config.SerializeToString(
  )
  hash_table_name = "MonolithHashTable_" + name_suffix
  if hash_filter is None or use_gpu:  # We don't have gpu filter for now, get rid of or use_gpu if added one
    with tf.device(d):
      hash_filter = hash_filter_ops.create_dummy_hash_filter(
          name_suffix=name_suffix)
  if len(config.learning_rate_fns) != len(
      config.table_config.entry_config.segments):
    raise ValueError(
        "Size of learning_rate_fns and size of segments must be equal.")
  if sync_client is None or use_gpu:  # We don't have gpu sync for now, get rid of or use_gpu if added one
    with tf.device(d):
      sync_client = distributed_serving_ops.create_dummy_sync_client()
  with tf.device(
      d
  ):  # Merged Device is essential here to avoid affecting job task placement
    table_op = hash_table_ops.monolith_hash_table(
        name=hash_table_name,
        filter_handle=hash_filter,
        sync_client_handle=sync_client,
        config=table_config_str,
        shared_name=hash_table_name)
  return HashTable(table_op,
                   shared_name=hash_table_name,
                   dim_size=dim_size,
                   slot_expire_time_config=slot_expire_time_config,
                   learning_rate_tensor=config.call_learning_rate_fns(),
                   saver_parallel=saver_parallel,
                   extra_restore_names=config.extra_restore_names)


def test_hash_table(
    dim_size,
    enable_hash_filter=False,
    name_suffix=None,
    learning_rate=1.0,
    occurrence_threshold=0,
    use_adagrad=False,
    expire_time=36500,  # For testing, the Default expire time is 100 years.
    sync_client: tf.Tensor = None,
    extra_restore_names=None,
    use_gpu=False,
) -> HashTable:
  """
  Returns a hash table which essentially is a |dim_size| float
  table with sgd optimizer.
  """
  table_config = embedding_hash_table_pb2.EmbeddingHashTableConfig()
  if use_gpu:
    table_config.gpucuco.SetInParent()
  else:
    table_config.cuckoo.SetInParent()
  segment = table_config.entry_config.segments.add()
  segment.dim_size = dim_size
  if use_adagrad:
    segment.opt_config.adagrad.SetInParent()  # use adagrad for gpu hash table
  else:
    segment.opt_config.sgd.SetInParent()

  if use_gpu:
    segment.init_config.ones.SetInParent()  # check ones
  else:
    segment.init_config.zeros.SetInParent()

  segment.comp_config.fp32.SetInParent()

  slot_occurrence_threshold_config = embedding_hash_table_pb2.SlotOccurrenceThresholdConfig(
  )
  slot_occurrence_threshold_config.default_occurrence_threshold = occurrence_threshold

  table_config.slot_expire_time_config.default_expire_time = expire_time
  config = entry.HashTableConfigInstance(
      table_config, [learning_rate], extra_restore_names=extra_restore_names)
  if not use_gpu:
    hash_filters = hash_filter_ops.create_hash_filters(
        0, enable_hash_filter,
        slot_occurrence_threshold_config.SerializeToString())
  if not name_suffix:
    name_suffix = tf.compat.v1.get_default_graph().unique_name("test")

  if not use_gpu:
    return hash_table_from_config(config=config,
                                  hash_filter=hash_filters[0],
                                  name_suffix=name_suffix,
                                  sync_client=sync_client)
  return hash_table_from_config(config=config,
                                name_suffix=name_suffix,
                                sync_client=sync_client)


def vocab_hash_table(vocab_size: int,
                     dim_size: int,
                     enable_hash_filter=False,
                     learning_rate=1.0) -> HashTable:
  """
  Returns a hash table which essentially is a [vocab_size, dim_size] float
  table with sgd optimizer.
  """
  # Here we use a hash table which is more powerful than vocab table.
  return test_hash_table(dim_size,
                         enable_hash_filter,
                         learning_rate=learning_rate)


def _all_table_tensor_prefix(table: HashTable) -> List[str]:
  all_names = [table.name] + table._extra_restore_names
  return [name.replace(":", "-").replace("/", "-") for name in all_names]


def _table_tensor_prefix(table: HashTable) -> str:
  return _all_table_tensor_prefix(table)[0]


class HashTableCheckpointSaverListener(tf.estimator.CheckpointSaverListener):
  """Saves the hash tables when saver is run."""

  def __init__(self, basename: str):
    """|basename| should be a file name which is same as what is passed to saver."""
    super().__init__()
    self._helper = save_utils.SaveHelper(basename)
    self._table_id_to_placeholder = {}
    self._save_op = self._build_save_graph()

  def before_save(self, sess, global_step_value):
    """
    We use before save so the checkpoint file is updated after we successfully
    save the hash table.
    """
    logging.info("Starting saving hash tables.")
    feed_dict = {}
    base_dir = self._helper.get_ckpt_asset_dir(
        self._helper.get_ckpt_prefix(global_step_value))
    tf.io.gfile.makedirs(base_dir)
    for table in ops.get_collection(_HASH_TABLE_GRAPH_KEY):
      table_basename = base_dir + _table_tensor_prefix(table)
      feed_dict.update(
          {self._table_id_to_placeholder[table.name]: table_basename})
    sess.run(self._save_op,
             feed_dict=feed_dict,
             options=tf.compat.v1.RunOptions(timeout_in_ms=_TIMEOUT_IN_MS))
    logging.info("Finished saving hash tables.")

  def _build_save_graph(self) -> tf.Operation:
    save_tensors = []
    # This reduces disk metadata modification pressure.
    random_sleep_ms = 15 * len(ops.get_collection(_HASH_TABLE_GRAPH_KEY))
    for table in ops.get_collection(_HASH_TABLE_GRAPH_KEY):
      table_basename = tf.compat.v1.placeholder(tf.string, shape=[])
      self._table_id_to_placeholder.update({table.name: table_basename})
      save_tensors.append(
          table.save(table_basename, random_sleep_ms=random_sleep_ms).table)
    with tf.control_dependencies(save_tensors):
      return tf.no_op()


class HashTableCheckpointRestorerListener(
    basic_restore_hook.CheckpointRestorerListener):
  """Restores the hash tables from basename"""

  def __init__(self, basename: str, ps_monitor=None):
    super().__init__()
    self._basename = basename
    self._helper = save_utils.SaveHelper(basename)
    self._table_id_to_placeholder = {}
    self._restore_ops_per_device = self._build_restore_graph()
    self._ps_monitor = ps_monitor

  def before_restore(self, session):
    """
    We use before restore so as to strictly control the order of restorer listeners.

    """
    ckpt_prefix = tf.train.latest_checkpoint(os.path.dirname(self._basename))
    if not ckpt_prefix:
      logging.info(
          "No checkpoint found in %s. Looking for assets(sparse only).",
          self._basename)
      # for sparse only ckpt converted from sail
      assets_list = tf.io.gfile.glob(
          os.path.join(os.path.dirname(self._basename), "*.assets"))
      if len(assets_list) == 0:
        logging.info("No assets(sparse only) found, skipping.")
        return
      elif len(assets_list) > 1:
        logging.info(
            f"Found {len(assets_list)} sparse assets of value {assets_list}, skipping."
        )
        return
      asset_dir = assets_list[0] + "/"
    else:
      asset_dir = self._helper.get_ckpt_asset_dir(ckpt_prefix)
    logging.info("Restore hash tables from %s.", asset_dir)
    self._restore_from_path_prefix(session, asset_dir)
    logging.info("Finished restore.")

  def _restore_from_path_prefix(self, sess, path_prefix):

    def get_restore_prefix(prefixes: List[str]):
      for prefix in prefixes:
        if len(tf.io.gfile.glob(path_prefix + prefix + "*")):
          return prefix
      raise ValueError(
          ("Unable to find table checkpoint in '%s' for table: %s. "
           "Maybe the model structure has been changed."), path_prefix,
          repr(prefixes))

    tables = tf.compat.v1.get_collection(_HASH_TABLE_GRAPH_KEY)
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
      table_to_prefix = {
          table.name: executor.submit(get_restore_prefix,
                                      _all_table_tensor_prefix(table))
          for table in tables
      }
      for table in tables:
        table_to_prefix[table.name] = table_to_prefix[table.name].result()

    feed_dict = {}
    for table in tables:
      table_basename = path_prefix + table_to_prefix[table.name]
      feed_dict.update(
          {self._table_id_to_placeholder[table.name]: table_basename})

    restore_ops_all = []
    for device, restore_ops in self._restore_ops_per_device.items():
      if not self._ps_monitor or self._ps_monitor.is_ps_uninitialized(
          sess, device):
        restore_ops_all.extend(restore_ops)
    sess.run(restore_ops_all,
             feed_dict=feed_dict,
             options=tf.compat.v1.RunOptions(timeout_in_ms=_TIMEOUT_IN_MS))

  def _build_restore_graph(self):
    restore_ops_per_device = defaultdict(list)
    for table in ops.get_collection(_HASH_TABLE_GRAPH_KEY):
      table_basename = tf.compat.v1.placeholder(tf.string, shape=[])
      self._table_id_to_placeholder.update({table.name: table_basename})
      restore_op = table.restore(table_basename).as_op()
      restore_ops_per_device[table.table.device].append(restore_op)
    return restore_ops_per_device


# This is for ByteDance internal use only
def extract_slot_from_entry(entry: tf.Tensor, fid_v2=True):
  return hash_table_ops.monolith_extract_slot_from_entry(entry, fid_v2=fid_v2)


class HashTableRestorerSaverLitsener(tf.estimator.CheckpointSaverListener):
  """Since we use restore to remove stale entries,
  we create a saver litsener here."""

  def __init__(self, ckpt_prefix: str):
    self._l = HashTableCheckpointRestorerListener(ckpt_prefix)

  def after_save(self, session, global_step_value):
    self._l.before_restore(session)


ops.register_proto_function(_HASH_TABLE_GRAPH_KEY,
                            proto_type=hash_table_ops_pb2.HashTableProto,
                            to_proto=HashTable.to_proto,
                            from_proto=HashTable.from_proto)
