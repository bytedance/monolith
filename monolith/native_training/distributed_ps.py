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

from __future__ import annotations

import re
import hashlib
import collections
import contextlib
import itertools
from contextlib import nullcontext
import copy
import os
import sys
from collections import defaultdict, namedtuple
from typing import Callable, DefaultDict, Dict, Iterable, List, Tuple, Optional, NewType

from absl import flags
from absl import logging
import tensorflow as tf
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.types.core import Value

from monolith.native_training import distribution_ops
from monolith.native_training import distributed_serving_ops
from monolith.native_training import hash_table_ops
from monolith.native_training import logging_ops
from monolith.native_training import multi_type_hash_table
from monolith.native_training import multi_hash_table_ops
from monolith.native_training import native_task_context
from monolith.native_training import tensor_utils
from monolith.native_training import utils
from monolith.native_training import entry
from monolith.native_training.hash_table_utils import infer_dim_size
from monolith.native_training.model_export import export_context
from monolith.native_training.data.parsers import sharding_sparse_fids
from idl.matrix.proto.example_pb2 import FeatureConfigs, FeatureConfig, PoolingType, OutType, OutConfig
import monolith.native_training.embedding_combiners as embedding_combiners
from monolith.native_training.data.parsers import get_default_parser_ctx, ParserCtx, ShardingSparseFidsOpParams
from monolith.native_training.prefetch_queue import \
    enqueue_dicts_with_queue_return, AsyncPushHook, EnqueueHook
from monolith.native_training import prefetch_queue

FLAGS = flags.FLAGS

enable_hvd = os.getenv("MONOLITH_WITH_HOROVOD")
if enable_hvd != None:
  import horovod.tensorflow as hvd
  from horovod.tensorflow.compression import FP16Compressor

# For mock test
remote_predict = distributed_serving_ops.remote_predict


@contextlib.contextmanager
def ps_device(i: int):
  """We need to clean the device stack first to make tf.function work properly."""
  with tf.compat.v1.get_default_graph().colocate_with(None, True), tf.device(
      utils.ps_device(i)):
    yield


class DistributedHashTable(hash_table_ops.BaseHashTable):
  """The distribution version of hash table. """

  def __init__(
      self, ps_num, config: entry.HashTableConfigInstance,
      hash_table_factory: Callable[[int, entry.HashTableConfigInstance],
                                   hash_table_ops.BaseHashTable]):
    self._ps_num = ps_num
    self._hash_tables = []
    # Build learning rate tensor on worker side
    learning_rate_tensor = config.call_learning_rate_fns()
    for i in range(self._ps_num):
      with nullcontext() if export_context.is_exporting_standalone(
      ) else ps_device(i):
        # Send learning rate tensor to ps
        learning_rate_tensor_on_ps = tf.identity(learning_rate_tensor)
        config.set_learning_rate_tensor(learning_rate_tensor_on_ps)
        self._hash_tables.append(hash_table_factory(i, config))

    self._input_lookup_tensors = {}
    self._output_lookup_tensors = set()

  @property
  def dim_size(self):
    return self._hash_tables[0].dim_size

  # Once `lookup` is edited, remember to edit `apply_gradients` too.
  def lookup(self, ids: tf.Tensor, use_multi_threads=False) -> tf.Tensor:
    unique_ids = ids
    unique_ids, idx = tf.unique(ids)
    indices = tf.math.floormod(unique_ids, self._ps_num)
    split_ids = distribution_ops.split_by_indices(indices, unique_ids,
                                                  self._ps_num)
    split_embeddings = []
    for i in range(self._ps_num):
      with nullcontext() if export_context.is_exporting_standalone(
      ) else ps_device(i), tf.name_scope("ps_{}".format(i)):
        hash_table = self._hash_tables[i]
        ids_part = split_ids[i]
        embeddings_part = hash_table.lookup(ids_part)
        self._input_lookup_tensors.update({embeddings_part: i})
        split_embeddings.append(embeddings_part)
    lookup_tensor = distribution_ops.map_id_to_embedding(
        split_ids, split_embeddings, ids)
    self._output_lookup_tensors.add(lookup_tensor)
    return lookup_tensor

  def _update(self, method_name: str, ids: tf.Tensor, values: tf.Tensor,
              req_time: tf.Tensor) -> "DistributedHashTable":
    indices = tf.math.floormod(ids, self._ps_num)
    split_ids = distribution_ops.split_by_indices(indices, ids, self._ps_num)
    split_values = distribution_ops.split_by_indices(indices, values,
                                                     self._ps_num)
    updated_tables = []
    for i in range(self._ps_num):
      with ps_device(i):
        ids_part = split_ids[i]
        values_part = split_values[i]
        updated_tables.append(
            getattr(self._hash_tables[i], method_name)(ids_part, values_part,
                                                       req_time))
    return self._copy_with_new_tables(updated_tables)

  def assign(self,
             ids: tf.Tensor,
             values: tf.Tensor,
             req_time: tf.Tensor = None) -> "DistributedHashTable":
    if req_time is None:
      req_time = tf.constant(0, dtype=tf.int64)
    return self._update("assign", ids, values, req_time)

  def assign_add(self,
                 ids: tf.Tensor,
                 values: tf.Tensor,
                 req_time: tf.Tensor = None) -> "DistributedHashTable":
    if req_time is None:
      req_time = tf.constant(0, dtype=tf.int64)
    return self._update("assign_add", ids, values, req_time)

  def apply_gradients(self,
                      ids: tf.Tensor,
                      grads: tf.Tensor,
                      global_step: tf.Tensor,
                      req_time: tf.Tensor = None) -> "DistributedHashTable":
    if req_time is None:
      req_time = tf.constant(0, dtype=tf.int64)
    unique_ids, idx = tf.unique(ids)
    indices = tf.math.floormod(unique_ids, self._ps_num)
    split_ids = distribution_ops.split_by_indices(indices, unique_ids,
                                                  self._ps_num)
    split_grads = distribution_ops.map_id_to_embedding_gradient_back_prop(
        split_ids, ids, grads)
    updated_tables = []
    for i in range(self._ps_num):
      with ps_device(i), tf.name_scope("ps_{}".format(i)):
        # TODO(leqi.zou): Think of the meaning of dedup here
        updated_tables.append(self._hash_tables[i].apply_gradients(
            split_ids[i],
            split_grads[i],
            global_step=global_step,
            enable_dedup=False,
            req_time=req_time))
    return self._copy_with_new_tables(updated_tables)

  def as_op(self, name=None) -> tf.Operation:
    name = name or "dht_ao"
    with tf.control_dependencies([table.as_op() for table in self._hash_tables
                                 ]):
      c = tf.no_op(name=("{}/done".format(name)))
    return c

  def _copy_with_new_tables(
      self, new_tables: List[tf.Tensor]) -> "DistributedHashTable":
    copied = copy.copy(self)
    copied.__dict__["_hash_tables"] = new_tables
    return copied


class DistributedMultiTypeHashTable(multi_type_hash_table.BaseMultiTypeHashTable
                                   ):

  def __init__(
      self,
      num_ps: int,
      slot_to_config: Dict[str, entry.HashTableConfigInstance],
      table_factory: Callable[[int, Dict[str, entry.HashTableConfigInstance]],
                              multi_type_hash_table.BaseMultiTypeHashTable],
      transfer_float16: bool = False,
      max_rpc_deadline_millis: int = 30):
    self._num_ps = num_ps
    self._slot_to_config = slot_to_config
    self._tables = []
    self._table_support_raw_api = True
    self.transfer_float16 = transfer_float16
    self._max_rpc_deadline_millis = max_rpc_deadline_millis
    # Build learning rate tensor on worker side
    slot_to_learning_rate_tensor = dict()
    for slot, config in slot_to_config.items():
      slot_to_learning_rate_tensor[slot] = config.call_learning_rate_fns()
    packed_slot_to_learning_rate_tensor = tensor_utils.pack_tensors(
        slot_to_learning_rate_tensor)

    def support_raw_api(table):
      return isinstance(table, multi_hash_table_ops.RawMultiTypeHashTable)

    for i in range(self._num_ps):
      if export_context.is_exporting_distributed():
        ps_graph = export_context.get_current_export_ctx().sub_graph(f"ps_{i}")
        with ps_graph.as_default():
          table = table_factory(i, slot_to_config)
          self._tables.append(table)
          # Build lookup graph on the PS side
          remote_lookup_input = {
              k: tf.compat.v1.placeholder(dtype=tf.int64, shape=(None,))
              for k in slot_to_config
          }
          remote_lookup_output = table.lookup(remote_lookup_input)
          export_context.get_current_export_ctx().add_signature(
              ps_graph, 'lookup', remote_lookup_input, remote_lookup_output)
          if support_raw_api(table):
            raw_remote_lookup_input = {
                "id":
                    tf.compat.v1.placeholder(dtype=tf.int64, shape=(None,)),
                "id_split":
                    tf.compat.v1.placeholder(dtype=tf.int64, shape=(None,)),
            }
            raw_remote_lookup_output = {
                "flat_emb":
                    table.raw_lookup(
                        tf.RaggedTensor.from_row_splits(
                            raw_remote_lookup_input["id"],
                            raw_remote_lookup_input["id_split"],
                            validate=False))
            }
            export_context.get_current_export_ctx().add_signature(
                ps_graph, 'raw_lookup', raw_remote_lookup_input,
                raw_remote_lookup_output)
      elif export_context.is_exporting_standalone():
        self._tables.append(table_factory(i, slot_to_config))
      else:
        with ps_device(i):
          # Send learning rate tensor to ps
          # TODO(leqi.zou): Here we can do some optimization to optimize raw hash table.
          slot_to_learning_rate_tensor_on_ps = tensor_utils.unpack_tensors(
              tensor_utils.get_keyed_shape(slot_to_learning_rate_tensor),
              packed_slot_to_learning_rate_tensor)
          for slot, config in slot_to_config.items():
            config.set_learning_rate_tensor(
                slot_to_learning_rate_tensor_on_ps[slot])
          self._tables.append(table_factory(i, slot_to_config))
      self._table_support_raw_api &= support_raw_api(self._tables[-1])

  def lookup(self, slot_to_id: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    with tf.name_scope("dmtht_lu"):

      def emit_lookup_timer_ops(interval):
        if not export_context.is_exporting():
          return [
              logging_ops.emit_timer(
                  "embedding_lookup",
                  tf.cast(interval, tf.float32),
                  tags={
                      "model_name": native_task_context.get().model_name,
                      "ps": str(i)
                  })
          ]
        return []

      if self._table_support_raw_api and not self.transfer_float16:
        table_0 = self._tables[0]
        dims = table_0.get_table_dim_sizes()
        ragged_id = table_0.get_ragged_id(slot_to_id)
        result = distribution_ops.unique_key_with_value_and_offset(
            ragged_id, dims)

        index = tf.math.floormod(result.unique_key.values, self._num_ps)
        splitted_ids, splitted_pos = distribution_ops.ragged_split_by_indices(
            index, result.unique_key, self._num_ps)
        filled_buffers = []
        interval_ops = []
        for i in range(self._num_ps):
          table: multi_hash_table_ops.RawMultiTypeHashTable = self._tables[i]
          splitted_id = splitted_ids[i]
          (splitted_id_values,), send_ts = logging_ops.tensors_timestamp(
              [splitted_id.values])
          splitted_id = tf.RaggedTensor.from_row_splits(splitted_id_values,
                                                        splitted_id.row_splits,
                                                        validate=False)
          if export_context.is_exporting_distributed():
            flat_emb, = remote_predict(
                ["id", "id_split"],
                [splitted_id.values, splitted_id.row_splits], ["flat_emb"],
                task=i,
                old_model_name="ps_{}".format(i),
                model_name=
                f"{native_task_context.get().model_name or ''}:ps_{i}",
                model_version=-1,
                max_rpc_deadline_millis=self._max_rpc_deadline_millis,
                output_types=[tf.float32],
                signature_name="raw_lookup")
          else:
            with nullcontext() if export_context.is_exporting_standalone(
            ) else ps_device(i):
              flat_emb = table.raw_lookup(splitted_id)
          (flat_emb,), end_ts = logging_ops.tensors_timestamp([flat_emb])
          interval_ops.extend(emit_lookup_timer_ops(end_ts - send_ts))
          filled_buffers.append(
              distribution_ops.fill_with_offset_map(splitted_pos[i], flat_emb,
                                                    result.value_offset,
                                                    result.value_buffer, dims))

        with tf.control_dependencies(interval_ops):
          flat_emb = distribution_ops.finalize_shared_tensor(filled_buffers,
                                                             dtype=tf.float32,
                                                             shape=[None])
        emb = table_0.get_embeddings(ragged_id, flat_emb)
        polished_emb = {}
        # Remove unpresented keys and make emb shape known if input shape is known.
        for k, v in emb.items():
          if k in slot_to_id:
            id = slot_to_id[k]
            if id.shape[0]:
              v = tf.reshape(v, shape=[id.shape[0], v.shape[1]])
            polished_emb[k] = v
        return polished_emb
      else:
        sharded_slot_to_id: Dict[int, Dict[
            str, tf.Tensor]] = collections.defaultdict(dict)
        slot_to_split_ids = {}
        for slot in slot_to_id:
          id = slot_to_id[slot]
          unique_id, idx = tf.unique(id)
          index = tf.math.floormod(unique_id, self._num_ps)
          split_ids = distribution_ops.split_by_indices(index, unique_id,
                                                        self._num_ps)
          slot_to_split_ids[slot] = split_ids
          for i in range(self._num_ps):
            sharded_slot_to_id[i][slot] = split_ids[i]

        sharded_slot_to_embedding: Dict[int, Dict[str, tf.Tensor]] = {}

        if export_context.is_exporting_distributed():
          slot_names = sorted(slot_to_split_ids.keys())
          slot_to_dim = [
              infer_dim_size(self._slot_to_config[slot].table_config)
              for slot in slot_names
          ]
          for i in range(self._num_ps):
            per_ps_slot_to_id = sharded_slot_to_id[i]
            # Remote call from Entry to PS
            # TODO(leqi.zou): Consider a better way to get model name.
            results = remote_predict(
                slot_names, [per_ps_slot_to_id[slot] for slot in slot_names],
                slot_names,
                task=i,
                old_model_name="ps_{}".format(i),
                model_name=
                f"{native_task_context.get().model_name or ''}:ps_{i}",
                model_version=-1,
                max_rpc_deadline_millis=self._max_rpc_deadline_millis,
                output_types=[tf.float32] * len(slot_names),
                signature_name="lookup")
            sharded_slot_to_embedding[i] = {
                slot_names[j]: tf.reshape(results[j], [-1, slot_to_dim[j]])
                for j in range(len(slot_names))
            }
        else:
          for i in range(self._num_ps):
            per_ps_slot_to_id = sharded_slot_to_id[i]
            packed_id = tensor_utils.pack_tensors(per_ps_slot_to_id)
            packed_id, send_ts = logging_ops.tensors_timestamp(packed_id)
            with nullcontext() if export_context.is_exporting_standalone(
            ) else ps_device(i):
              slot_to_id_on_ps = tensor_utils.unpack_tensors(
                  tensor_utils.get_keyed_shape(per_ps_slot_to_id), packed_id)
              slot_to_embedding_on_ps = self._tables[i].lookup(slot_to_id_on_ps)
              packed_embedding = tensor_utils.pack_tensors(
                  slot_to_embedding_on_ps)
              if self.transfer_float16:
                packed_embedding = (tf.cast(
                    packed_embedding[0],
                    dtype=tf.float16,
                    name='{}_send_{}_CastToFloat16'.format(
                        packed_embedding[0].op.name, i)), packed_embedding[1])

            packed_embedding, recv_ts = logging_ops.tensors_timestamp(
                packed_embedding)
            interval = recv_ts - send_ts
            with tf.control_dependencies(emit_lookup_timer_ops(interval)):
              packed_embedding = tf.identity_n(packed_embedding)

            if self.transfer_float16:
              packed_embedding = (tf.cast(
                  packed_embedding[0],
                  dtype=tf.float32,
                  name='{}_recv_{}_CastToFloat32'.format(
                      packed_embedding[0].op.name, i)), packed_embedding[1])
            slot_to_embedding = tensor_utils.unpack_tensors(
                tensor_utils.get_keyed_shape(slot_to_embedding_on_ps),
                packed_embedding)
            sharded_slot_to_embedding[i] = slot_to_embedding

        slot_to_split_embeddings = {}
        for slot in slot_to_id:
          slot_to_split_embeddings[slot] = [
              sharded_slot_to_embedding[i][slot] for i in range(self._num_ps)
          ]

        slot_to_embedding = {}
        for slot in slot_to_id:
          slot_to_embedding[slot] = distribution_ops.map_id_to_embedding(
              slot_to_split_ids[slot], slot_to_split_embeddings[slot],
              slot_to_id[slot])

      return slot_to_embedding

  def _update(
      self, method_name: str, name_scope: str,
      slot_to_id_and_value: Dict[str, Tuple[tf.Tensor, tf.Tensor]]
  ) -> DistributedMultiTypeHashTable:
    with tf.name_scope(name_scope):
      sharded_slot_to_id_and_value: Dict[int, Dict[str, Tuple[
          tf.Tensor, tf.Tensor]]] = collections.defaultdict(dict)
      for slot, (id, value) in slot_to_id_and_value.items():
        index = tf.math.floormod(id, self._num_ps)
        split_ids = distribution_ops.split_by_indices(index, id, self._num_ps)
        split_values = distribution_ops.split_by_indices(
            index, value, self._num_ps)
        for i in range(self._num_ps):
          sharded_slot_to_id_and_value[i][slot] = (split_ids[i],
                                                   split_values[i])
      new_tables = []
      for i in range(self._num_ps):
        new_tables.append(
            getattr(self._tables[i],
                    method_name)(sharded_slot_to_id_and_value[i]))

      return self._copy_with_new_table(new_tables)

  def assign(
      self, slot_to_id_and_value: Dict[str, Tuple[tf.Tensor, tf.Tensor]]
  ) -> DistributedMultiTypeHashTable:
    return self._update("assign", "dmtht_a", slot_to_id_and_value)

  def assign_add(
      self, slot_to_id_and_value: Dict[str, Tuple[tf.Tensor, tf.Tensor]]
  ) -> DistributedMultiTypeHashTable:
    return self._update("assign_add", "dmtht_aa", slot_to_id_and_value)

  def apply_gradients(
      self,
      slot_to_id_and_grad: Dict[str, Tuple[tf.Tensor, tf.Tensor]],
      global_step: tf.Tensor,
      req_time: Optional[tf.Tensor] = None) -> DistributedMultiTypeHashTable:
    if req_time is None:
      req_time = tf.constant(0, dtype=tf.int64)
    with tf.name_scope("dmtht_ag"):
      if self._table_support_raw_api and not self.transfer_float16:
        slot_to_id = {k: v[0] for k, v in slot_to_id_and_grad.items()}
        slot_to_grad = {k: v[1] for k, v in slot_to_id_and_grad.items()}
        table_0 = self._tables[0]
        dims = table_0.get_table_dim_sizes()
        ragged_id = table_0.get_ragged_id(slot_to_id)
        result = distribution_ops.unique_key_with_value_and_offset(
            ragged_id, dims, generate_buffer=False)
        index = tf.math.floormod(result.unique_key.values, self._num_ps)
        splitted_ids, splitted_pos = distribution_ops.ragged_split_by_indices(
            index, result.unique_key, self._num_ps)
        flat_grad = table_0.get_flat_value(slot_to_grad)
        new_tables = []
        for i in range(self._num_ps):
          splitted_id = splitted_ids[i]
          splitted_flat_grad = distribution_ops.fill_with_offset_map_gradient(
              splitted_pos[i], flat_grad, result.value_offset, dims)
          table: multi_hash_table_ops.RawMultiTypeHashTable = self._tables[i]

          with ps_device(i):
            new_tables.append(
                table.raw_apply_gradients(splitted_id,
                                          splitted_flat_grad,
                                          global_step=global_step,
                                          req_time=req_time))
        return self._copy_with_new_table(new_tables)
      else:
        sharded_slot_to_id_and_grad: Dict[int, Dict[str, Tuple[
            tf.Tensor, tf.Tensor]]] = collections.defaultdict(dict)
        for slot, (id, grad) in slot_to_id_and_grad.items():
          unique_id, _ = tf.unique(id)
          index = tf.math.floormod(unique_id, self._num_ps)
          split_ids = distribution_ops.split_by_indices(index, unique_id,
                                                        self._num_ps)
          split_grads = distribution_ops.map_id_to_embedding_gradient_back_prop(
              split_ids, id, grad)
          for i in range(self._num_ps):
            sharded_slot_to_id_and_grad[i][slot] = (split_ids[i],
                                                    split_grads[i])

        new_tables = []
        for i in range(self._num_ps):
          keyed_id = {
              k: v[0] for k, v in sharded_slot_to_id_and_grad[i].items()
          }
          keyed_grad = {
              k: v[1] for k, v in sharded_slot_to_id_and_grad[i].items()
          }
          packed_list = tensor_utils.pack_typed_keyed_tensors(
              [keyed_id, keyed_grad])
          if self.transfer_float16:
            packed_list[1] = tf.cast(packed_list[1],
                                     dtype=tf.float16,
                                     name='{}_send_{}_CastToFloat16'.format(
                                         packed_list[1].op.name, i))
          with ps_device(i):
            if self.transfer_float16:
              packed_list[1] = tf.cast(packed_list[1],
                                       dtype=tf.float32,
                                       name='{}_recv_{}_CastToFloat32'.format(
                                           packed_list[1].op.name, i))
            keyed_list_on_ps = tensor_utils.unpack_packed_tensors(
                tensor_utils.get_typed_keyed_shape([keyed_id, keyed_grad]),
                packed_list)
            keyed_id_on_ps = keyed_list_on_ps[0]
            keyed_grad_on_ps = keyed_list_on_ps[1]
            slot_to_id_and_grad_on_ps = {
                slot: (keyed_id_on_ps[slot], keyed_grad_on_ps[slot])
                for slot in keyed_id_on_ps
            }
            new_tables.append(self._tables[i].apply_gradients(
                slot_to_id_and_grad_on_ps, global_step, req_time=req_time))
        return self._copy_with_new_table(new_tables)

  def as_op(self, name=None):
    name = name or "dmtht_ao"
    ops = []
    for i in range(self._num_ps):
      with ps_device(i):
        ops.append(self._tables[i].as_op(name="{}/sub_{}".format(name, i)))
    with tf.control_dependencies(ops):
      return tf.no_op(name=("{}/done".format(name)))

  def get_table_dim_sizes(self):
    return self._cc.dims

  def _copy_with_new_table(
      self, new_tables: List[tf.Tensor]) -> DistributedMultiTypeHashTable:
    copied = copy.copy(self)
    copied._tables = new_tables
    return copied


def get_sub_table_name(strs: List[str]):
  concat = ",".join(strs)
  return concat, hashlib.md5(concat.encode()).hexdigest()


Partition = NewType("Partition", int)
TableName = NewType("TableName", str)
Fids = NewType("Fids", tf.Tensor)
Emb = NewType("Emb", tf.Tensor)
EmbGrad = NewType("EmbGrad", tf.Tensor)

FidEmbPair = NewType("FidEmbPair", Tuple[Fids, Emb])
FidEmbGradPair = NewType("FidEmbGradPair", Tuple[Fids, EmbGrad])
LookupData = NewType("LookupData", Dict[Partition, Dict[TableName, Fids]])
UpdateData = NewType("UpdateData", Dict[Partition, Dict[TableName,
                                                        FidEmbGradPair]])
AssignData = NewType("UpdateData", Dict[Partition, Dict[TableName, FidEmbPair]])
TableFactory = NewType(
    "TableFactory",
    Callable[[Partition, Dict[TableName, entry.HashTableConfigInstance]],
             multi_type_hash_table.BaseMultiTypeHashTable])
FeatureInfo = namedtuple('FeatureInfo', 'slice_dims combiner sub_table')

from monolith.native_training.distributed_ps_sync import enable_custom_optimized_hvd, enable_hvd_fid_g2g, \
  enable_hvd_fwd_g2g, enable_hvd_bwd_g2g, enable_bps, enable_bps_fid, enable_bps_fwd, enable_bps_bwd, \
  enable_bps_bwd_cast, enable_bps_bwd_fake_cast, enable_bps_fwd_gdr, enable_bps_fwd_gdr_g2g, \
  enable_bps_bwd_gdr, enable_bps_bwd_gdr_g2g


class PartitionedHashTable(object):
  # Allow pipelined graph execution.
  _local_queue_hooks: List[prefetch_queue.EnqueueHook |
                           prefetch_queue.AsyncPushHook]
  _native_multi_hash_table_fake_table = "native_multi_hash_table_fake_table"

  @classmethod
  def gen_feature_configs(
      cls,
      num_ps: int,
      feature_name_to_config: Dict[str, entry.HashTableConfigInstance],
      layout_configs: Dict[str, OutConfig],
      feature_to_unmerged_slice_dims: Dict[str, List[int]],
      feature_to_combiner: Dict[str, embedding_combiners.Combiner],
      use_native_multi_hash_table: bool,
      transfer_float16: bool = False,
      unique: Callable = None,
      enable_gpu_emb: bool = False,
      use_gpu: bool = False,
  ):
    _num_ps: int = 1 if num_ps == 0 else num_ps
    _use_native_multi_hash_table = use_native_multi_hash_table and not transfer_float16
    # feature/slot -> sub_hashtable_name
    if _use_native_multi_hash_table:
      _sub_table_name_to_config, feature_to_sub_table = cls.no_merge_feature_config(
          feature_name_to_config, use_same_table=not enable_gpu_emb)
    else:
      _sub_table_name_to_config, feature_to_sub_table = cls.merge_feature_config(
          feature_name_to_config)

    feature_info: Dict[str, FeatureInfo] = {}
    for feature_name, sub_table in feature_to_sub_table.items():
      feature_info[feature_name] = FeatureInfo(
          feature_to_unmerged_slice_dims[feature_name],
          feature_to_combiner[feature_name], sub_table)

    feature_configs = FeatureConfigs()

    # fill feature config for feature_configs
    for feature_name, info in feature_info.items():
      combiner = info.combiner
      if isinstance(combiner, embedding_combiners.ReduceSum):
        pooling_type = PoolingType.SUM
      elif isinstance(combiner, embedding_combiners.ReduceMean):
        pooling_type = PoolingType.MEAN
      elif isinstance(combiner, embedding_combiners.FirstN):
        pooling_type = PoolingType.FIRSTN
      else:
        raise Exception("pooling_type error!")

      max_sequence_length = combiner.max_seq_length
      fc = FeatureConfig(table=info.sub_table,
                         pooling_type=pooling_type,
                         max_sequence_length=max_sequence_length)
      fc.slice_dims.extend(info.slice_dims)
      feature_configs.feature_configs[feature_name].CopyFrom(fc)

    for out_name, oc in layout_configs.items():
      feature_configs.out_configs[out_name].CopyFrom(oc)

    return ShardingSparseFidsOpParams(_num_ps, _use_native_multi_hash_table,
                                      unique, transfer_float16,
                                      _sub_table_name_to_config,
                                      feature_configs, enable_gpu_emb, use_gpu)

  @classmethod
  def no_merge_feature_config(
      cls, feature_name_to_config: Dict[str, entry.HashTableConfigInstance],
      use_same_table: bool):
    sub_table_name_to_config, feature_to_sub_table = {}, {}
    for feature_name in sorted(feature_name_to_config):
      feature_to_sub_table[
          feature_name] = cls._native_multi_hash_table_fake_table if use_same_table else feature_name
    return feature_name_to_config, feature_to_sub_table

  @classmethod
  def merge_feature_config(
      cls, feature_name_to_config: Dict[str, entry.HashTableConfigInstance]):
    # create merged config
    config_to_feature_name_list: Dict[
        str, List[entry.HashTableConfigInstance]] = defaultdict(list)
    for feature_name in sorted(feature_name_to_config):
      config = feature_name_to_config[feature_name]
      config_to_feature_name_list[str(config)].append(feature_name)

    sub_table_name_to_config, feature_to_sub_table = {}, {}  # merged config
    for config_str, feature_name_list in config_to_feature_name_list.items():
      _, sub_table_name = get_sub_table_name(feature_name_list)
      sub_table_config = copy.copy(feature_name_to_config[feature_name_list[0]])

      # replace "fc_slot_*" to "slot_*"
      old_feature_name_list = [
          feature_name[3:]
          if re.match("^fc_slot_[0-9]*$", feature_name) else feature_name
          for feature_name in feature_name_list
      ]
      _, old_sub_table_name = get_sub_table_name(old_feature_name_list)
      if old_sub_table_name != sub_table_name:
        sub_table_config.extra_restore_names.append(old_sub_table_name)

      sub_table_name_to_config[sub_table_name] = sub_table_config
      for feature_name in feature_name_list:
        feature_to_sub_table[feature_name] = sub_table_name

    return sub_table_name_to_config, feature_to_sub_table

  def __init__(self,
               num_ps: int,
               table_factory: TableFactory,
               use_native_multi_hash_table: bool,
               max_rpc_deadline_millis: int = 30,
               queue_configs: Dict[str, int] = None,
               parser_ctx=None):
    self._local_ps: bool = True if num_ps == 0 else False
    self._max_rpc_deadline_millis = max_rpc_deadline_millis
    self._queue_configs = queue_configs or {}

    if parser_ctx is None:
      parser_ctx = get_default_parser_ctx()
    self._inner_data_type = parser_ctx.parser_type
    assert parser_ctx.sharding_sparse_fids_op_params is not None
    self._num_ps = parser_ctx.sharding_sparse_fids_op_params.num_ps
    self._use_native_multi_hash_table = parser_ctx.sharding_sparse_fids_op_params.use_native_multi_hash_table
    self._unique = parser_ctx.sharding_sparse_fids_op_params.unique
    self.transfer_float16 = parser_ctx.sharding_sparse_fids_op_params.transfer_float16
    self._sub_table_name_to_config = parser_ctx.sharding_sparse_fids_op_params.sub_table_name_to_config
    self._feature_configs = parser_ctx.sharding_sparse_fids_op_params.feature_configs
    self._enable_gpu_emb = parser_ctx.sharding_sparse_fids_op_params.enable_gpu_emb
    self._use_gpu = parser_ctx.sharding_sparse_fids_op_params.use_gpu

    sub_table_to_learning_rate_tensor = {
        sub_table_name: config.call_learning_rate_fns()
        for sub_table_name, config in self._sub_table_name_to_config.items()
    }
    packed_sub_table_to_learning_rate_tensor = tensor_utils.pack_tensors(
        sub_table_to_learning_rate_tensor)
    if self._use_native_multi_hash_table:
      self._sub_table_names = [self._native_multi_hash_table_fake_table]
    else:
      self._sub_table_names = sorted(self._sub_table_name_to_config)

    if self._enable_gpu_emb:
      self._shard_num = self._num_ps
      if enable_bps:
        import byteps.tensorflow as bps
        assert bps.size() == self._shard_num
        self._index = bps.rank()
      else:
        assert hvd.size() == self._shard_num
        self._index = hvd.rank()
      self._table = table_factory(self._index, self._sub_table_name_to_config)
      self._output_dims = self._table.get_table_dim_sizes()
      self._dependency_ops = []
      table_name_list = []
      feautres_name_list = []
      for feature_name, cfg in self._feature_configs.feature_configs.items():
        if cfg.table not in table_name_list:
          table_name_list.append(cfg.table)
        if feature_name not in feautres_name_list:
          feautres_name_list.append(feature_name)
      table_name_list.sort()
      self._num_table = len(table_name_list)
      self._num_feature = len(feautres_name_list)
    else:
      self._tables = []
      for i in range(self._num_ps):
        if export_context.is_exporting_distributed():
          ps_graph = export_context.get_current_export_ctx().sub_graph(
              f"ps_{i}")
          with ps_graph.as_default():
            table = table_factory(i, self._sub_table_name_to_config)
            self._tables.append(table)
            # Build lookup graph on the PS side
            remote_lookup_input = {
                sub_table_name: tf.compat.v1.placeholder(dtype=tf.int64,
                                                         shape=(None,))
                for sub_table_name in self._sub_table_name_to_config
            }
            remote_lookup_output = table.lookup(remote_lookup_input)
            export_context.get_current_export_ctx().add_signature(
                ps_graph, 'lookup', remote_lookup_input, remote_lookup_output)
            if use_native_multi_hash_table:
              raw_remote_lookup_input = {
                  "id":
                      tf.compat.v1.placeholder(dtype=tf.int64, shape=(None,)),
                  "id_split":
                      tf.compat.v1.placeholder(dtype=tf.int64, shape=(None,)),
              }
              raw_remote_lookup_output = {
                  "flat_emb":
                      table.raw_lookup(
                          tf.RaggedTensor.from_row_splits(
                              raw_remote_lookup_input["id"],
                              raw_remote_lookup_input["id_split"],
                              validate=False))
              }
              export_context.get_current_export_ctx().add_signature(
                  ps_graph, 'raw_lookup', raw_remote_lookup_input,
                  raw_remote_lookup_output)
        elif export_context.is_exporting_standalone():
          self._tables.append(table_factory(i, self._sub_table_name_to_config))
        else:
          with nullcontext() if self._local_ps else ps_device(i):
            # Send learning rate tensor to ps
            sub_table_to_learning_rate_tensor_on_ps = tensor_utils.unpack_tensors(
                tensor_utils.get_keyed_shape(sub_table_to_learning_rate_tensor),
                packed_sub_table_to_learning_rate_tensor)
            for sub_table_name, config in self._sub_table_name_to_config.items(
            ):
              config.set_learning_rate_tensor(
                  sub_table_to_learning_rate_tensor_on_ps[sub_table_name])
            self._tables.append(table_factory(i,
                                              self._sub_table_name_to_config))

  @property
  def slot_mapping(self):
    """Returns slot mapping."""
    return self._slot_mapping

  def _native_hash_table_lookup_raw(self, lookup_data_on_wk: LookupData,
                                    lookup_data_on_wk_row_split: LookupData):
    ps_idx_to_multi_type_resp = {}

    def emit_lookup_timer_ops(i, interval):
        return [
            logging_ops.emit_timer(
                "embedding_lookup",
                tf.cast(interval, tf.float32),
                tags={
                    "model_name": native_task_context.get().model_name,
                    "ps": str(i)
                })
        ]

    interval_ops = []
    for i in range(self._num_ps):
      table: multi_hash_table_ops.RawMultiTypeHashTable = self._tables[i]
      (splitted_id_values,), send_ts = logging_ops.tensors_timestamp(
          [lookup_data_on_wk[i][self._native_multi_hash_table_fake_table]])
      splitted_id = tf.RaggedTensor.from_row_splits(
          splitted_id_values,
          lookup_data_on_wk_row_split[i][
              self._native_multi_hash_table_fake_table],
          validate=False)
      is_standalone = export_context.is_exporting_standalone() or self._local_ps
      with nullcontext() if is_standalone else ps_device(i):
        flat_emb = table.raw_lookup(splitted_id)
      (flat_emb,), end_ts = logging_ops.tensors_timestamp([flat_emb])
      interval_ops.extend(emit_lookup_timer_ops(i, end_ts - send_ts))
      ps_idx_to_multi_type_resp[i] = {
          self._native_multi_hash_table_fake_table: flat_emb
      }

    ret = {}
    with tf.control_dependencies(interval_ops):
      for i, sub_item in ps_idx_to_multi_type_resp.items():
        ret[i] = {}
        for tname, ts in sub_item.items():
          ret[i][tname] = tf.identity(ts)
    return ret

  def _lookup_raw(self, lookup_data_on_wk: LookupData):
    ps_idx_to_multi_type_resp = {}

    def emit_lookup_timer_ops(i, interval):
        return [
            logging_ops.emit_timer(
                "embedding_lookup",
                tf.cast(interval, tf.float32),
                tags={
                    "model_name": native_task_context.get().model_name,
                    "ps": str(i)
                })
        ]

    interval_ops = []
    for i in range(self._num_ps):
      # sub_table_name -> fids tensor
      multi_type_query: Dict[str, tf.Tensor] = lookup_data_on_wk[i]
      # Note: this is a python object, use it outside this device context required no data transfer
      multi_type_query_shape: Dict[
          str, List[int]] = tensor_utils.get_keyed_shape(multi_type_query)

      # to reduce the number of rpc (send/recv ops) call, we pack fids by concat
      packed_fids_on_worker, send_ts = logging_ops.tensors_timestamp(
          tensor_utils.pack_tensors(multi_type_query))

      is_standalone = export_context.is_exporting_standalone() or self._local_ps
      with nullcontext() if is_standalone else ps_device(i):
        packed_fids_on_ps = packed_fids_on_worker  # data transfer in logic
        unpacked_multi_type_query = tensor_utils.unpack_tensors(
            multi_type_query_shape, packed_fids_on_ps)
        multi_type_resp_on_ps = self._tables[i].lookup(
            unpacked_multi_type_query)
        # Note: this is a python object, use it outside this device context required no data transfer
        multi_type_resp_shape = tensor_utils.get_keyed_shape(
            multi_type_resp_on_ps)

        # to reduce rpc (send/recv ops), we pack embeddings by concat
        packed_embeddings, emb_sizes = tensor_utils.pack_tensors(
            multi_type_resp_on_ps)
        if self.transfer_float16:
          packed_embeddings_fp16 = tf.cast(
              packed_embeddings,
              dtype=tf.float16,
              name='{}_send_{}_CastToFloat16'.format(packed_embeddings.op.name,
                                                     i))
          packed_embedding_on_ps = (packed_embeddings_fp16, emb_sizes)
        else:
          packed_embedding_on_ps = (packed_embeddings, emb_sizes)

      packed_embedding_on_worker, end_ts = logging_ops.tensors_timestamp(
          packed_embedding_on_ps)  # data transfer in logic
      interval_ops.extend(emit_lookup_timer_ops(i, end_ts - send_ts))

      # on worker, uppack
      if self.transfer_float16:
        packed_embeddings_fp16_recv, emb_sizes_recv = packed_embedding_on_worker
        packed_embeddings_fp32 = tf.cast(packed_embeddings_fp16_recv,
                                         dtype=tf.float32,
                                         name='{}_recv_{}_CastToFloat32'.format(
                                             packed_embeddings_fp16.op.name, i))
        packed_embedding_on_worker = (packed_embeddings_fp32, emb_sizes)

      multi_type_resp_on_worker = tensor_utils.unpack_tensors(
          multi_type_resp_shape, packed_embedding_on_worker)
      ps_idx_to_multi_type_resp[i] = multi_type_resp_on_worker

    ret = {}
    with tf.control_dependencies(interval_ops):
      for i, sub_item in ps_idx_to_multi_type_resp.items():
        ret[i] = {}
        for tname, ts in sub_item.items():
          ret[i][tname] = tf.identity(ts)
    return ret

  def _native_hash_table_lookup_with_remote_predict(
      self, lookup_data_on_wk: LookupData,
      lookup_data_on_wk_row_split: LookupData):
    ps_idx_to_multi_type_resp = {}
    for i in range(self._num_ps):
      flat_emb, = remote_predict(
          ["id", "id_split"], [
              lookup_data_on_wk[i][self._native_multi_hash_table_fake_table],
              lookup_data_on_wk_row_split[i][
                  self._native_multi_hash_table_fake_table]
          ], ["flat_emb"],
          task=i,
          old_model_name="ps_{}".format(i),
          model_name=f"{native_task_context.get().model_name or ''}:ps_{i}",
          model_version=-1,
          max_rpc_deadline_millis=self._max_rpc_deadline_millis,
          output_types=[tf.float32],
          signature_name="raw_lookup")
      ps_idx_to_multi_type_resp[i] = {
          self._native_multi_hash_table_fake_table: flat_emb
      }
    return ps_idx_to_multi_type_resp

  def _lookup_with_remote_predict(self, lookup_data_on_wk: LookupData):
    sub_table_to_dim = [
        infer_dim_size(
            self._sub_table_name_to_config[sub_table_name].table_config)
        for sub_table_name in self._sub_table_names
    ]

    ps_idx_to_multi_type_resp = {}
    for i in range(self._num_ps):
      multi_type_query = lookup_data_on_wk[i]
      # Remote call from Entry to PS
      # TODO(leqi.zou): Consider a better way to get model name.
      results = remote_predict(
          input_tensor_alias=self._sub_table_names,
          input_tensors=[
              multi_type_query[sub_table_name]
              for sub_table_name in self._sub_table_names
          ],
          output_tensor_alias=self._sub_table_names,
          task=i,
          old_model_name="ps_{}".format(i),
          model_name=f"{native_task_context.get().model_name or ''}:ps_{i}",
          model_version=-1,
          max_rpc_deadline_millis=self._max_rpc_deadline_millis,
          output_types=[tf.float32] * len(self._sub_table_names),
          signature_name="lookup")
      ps_idx_to_multi_type_resp[i] = {
          sub_table_name: tf.reshape(results[j], [-1, sub_table_to_dim[j]])
          for j, sub_table_name in enumerate(self._sub_table_names)
      }

    return ps_idx_to_multi_type_resp

  def lookup(
      self,
      features: Dict[str, tf.Tensor],
      auxiliary_bundle: Dict[str, tf.Tensor] = None,
      ret_fused_layout_callable_fn=False,
      ret_lookup_callable_fn=False,
      embedding_prefetch_capacity=0,
  ) -> Dict[str, Union[tf.Tensor, List[tf.Tensor]]]:
    if self._enable_gpu_emb:
      ret = self._lookup_gpu(features, auxiliary_bundle)
      if ret_fused_layout_callable_fn or ret_lookup_callable_fn:
        def lookup_callable_fn(auxiliary_bundle_, features_):
          return ret
        return lookup_callable_fn
      else:
        return ret

    with tf.name_scope("pht_lookup"):
      if ParserCtx.sharding_sparse_fids_sparse_features_key in features:
        #assert False, "not support, please use sharding_sparse_fids_with_context before call lookup"
        # only support for cpu training(without dsworker)
        shards, fid_offset, feature_offset, nfl_offset, batch_size, nfl_size, \
          feature_size, fid_size, emb_size, shards_row_split, shards_row_split_size, \
          fid_list_emb_row_lenth, fid_list_table_row_length, fid_list_shard_row_lenth = \
          sharding_sparse_fids(
              features[ParserCtx.sharding_sparse_fids_sparse_features_key],
              ps_num=self._num_ps,
              feature_cfgs=self._feature_configs,
              unique=self._unique(),
              input_type=self._inner_data_type,
              parallel_flag=0)
      else:
        sharding_features = ParserCtx.sharding_sparse_fids_features_parse_from_features(
            features)
        shards, fid_offset, feature_offset, nfl_offset, batch_size, nfl_size, \
        feature_size, fid_size, emb_size, shards_row_split, shards_row_split_size = \
          sharding_features.get("shards"), sharding_features.get("fid_offset") ,\
          sharding_features.get("feature_offset"), sharding_features.get("nfl_offset") ,\
          sharding_features.get("batch_size"), sharding_features.get("nfl_size", None), \
          sharding_features.get("feature_size", None), sharding_features.get("fid_size", None), \
          sharding_features.get("emb_size", None), sharding_features.get("shards_row_split", None), \
          sharding_features.get("shards_row_split_size", None)
      if auxiliary_bundle is None:
        auxiliary_bundle = {}
      auxiliary_bundle['__sharding_sparse_fids__fid_offset'] = fid_offset
      auxiliary_bundle[
          '__sharding_sparse_fids__feature_offset'] = feature_offset
      auxiliary_bundle['__sharding_sparse_fids__nfl_offset'] = nfl_offset
      auxiliary_bundle['__sharding_sparse_fids__batch_size'] = tf.identity(
          batch_size, name="batch_size")
      if nfl_size is not None:
        auxiliary_bundle['__sharding_sparse_fids__nfl_size'] = nfl_size
      if feature_size is not None:
        auxiliary_bundle['__sharding_sparse_fids__feature_size'] = feature_size
      if fid_size is not None:
        auxiliary_bundle['__sharding_sparse_fids__fid_size'] = fid_size
      if emb_size is not None:
        auxiliary_bundle['__sharding_sparse_fids__emb_size'] = emb_size
      logging.info(f"sharding_sparse_fids done, {shards}")

      if ret_lookup_callable_fn:
        for key, shard_fids in shards.items():
          sub_table_name, ps_idx = key.split(':')
          ps_idx = int(ps_idx)
          name = '__sharding_sparse_fids__shards@{}@{}'.format(ps_idx, sub_table_name)
          auxiliary_bundle[name] = shard_fids
          if self._use_native_multi_hash_table:
            name = '__sharding_sparse_fids__shards_row_split@{}@{}'.format(ps_idx, sub_table_name)
            auxiliary_bundle[name] = shards_row_split[key]
            if shards_row_split_size is not None and shards_row_split_size[key] is not None:
              name = '__sharding_sparse_fids__shards_row_split_size@{}@{}'.format(ps_idx, sub_table_name)
              auxiliary_bundle[name] = shards_row_split_size[key]

    def fused_layout_callable_fn(auxiliary_bundle_, features_):
      flattened_embs = []
      assert auxiliary_bundle_ is not None
      for sub_table_name in self._sub_table_names:
        for ps_idx in range(self._num_ps):
          flattened_embs.append(auxiliary_bundle_[
              f'__sharding_sparse_fids__{sub_table_name}:{ps_idx}:embs'])

      nfl_offset_ = auxiliary_bundle_['__sharding_sparse_fids__nfl_offset']
      feature_offset_ = auxiliary_bundle_[
          '__sharding_sparse_fids__feature_offset']
      fid_offset_ = auxiliary_bundle_['__sharding_sparse_fids__fid_offset']
      batch_size_ = auxiliary_bundle_['__sharding_sparse_fids__batch_size']
      nfl_size_ = auxiliary_bundle_.get('__sharding_sparse_fids__nfl_size', None)
      feature_size_ = auxiliary_bundle_.get('__sharding_sparse_fids__feature_size', None)
      fid_size_ = auxiliary_bundle_.get('__sharding_sparse_fids__fid_size', None)
      emb_size_ = auxiliary_bundle_.get('__sharding_sparse_fids__emb_size', None)

      if export_context.is_exporting():
        fused_layout_use_gpu = export_context.get_current_export_ctx().with_remote_gpu
      else:
        fused_layout_use_gpu = self._use_gpu

      with tf.device("/device:GPU:0" if fused_layout_use_gpu else "/device:CPU:0"):
        layout_tensors = distribution_ops.fused_embedding_to_layout(
            flattened_embs,
            None,  #self.fids_list_row_split, v3 not need fids_list_row_split
            fid_offset=fid_offset_,
            feature_offset=feature_offset_,
            nfl_offset=nfl_offset_,
            batch_size=batch_size_,
            nfl_size=nfl_size_,
            feature_size=feature_size_,
            fid_size=fid_size_,
            emb_size=emb_size_,
            variant_type=self._inner_data_type,
            feature_cfgs=self._feature_configs,
            ps_num=self._num_ps,
            version=5)
        layout_embeddings = self.nest_layout(layout_tensors)
      '''
      if not self._use_gpu:
        # embedding_prefetch
        logging.info(
            f"PartitionedHashTable lookup fused_layout enqueue: {auxiliary_bundle_} {features_}"
        )
        (deq_layout_embeddings, deq_auxiliary_bundle,
         deq_features), queue = enqueue_dicts_with_queue_return(
             (layout_embeddings, auxiliary_bundle_, features_),
             capacity=embedding_prefetch_capacity)
        if queue:
          self.add_queue_hook(EnqueueHook(queue))
        features_.update(deq_features)
        auxiliary_bundle_.update(deq_auxiliary_bundle)
        logging.info(
            f"PartitionedHashTable lookup fused_layout dequeue: {auxiliary_bundle_} {features_}"
        )
      else:
        deq_layout_embeddings = layout_embeddings
      '''
      logging.info("fused_embedding_to_layout done!")
      return layout_embeddings  #deq_layout_embeddings

    def call_lookup(lookup_data_on_wk: LookupData,
                    lookup_data_on_wk_row_split: LookupData,
                    auxiliary_bundle_, features_):
      with tf.name_scope("pht_lookup"):
        # ps_idx_to_multi_type_resp: Dict[int, Dict[str, tf.Tensor]] = {}
        with tf.device("/device:CPU:0"):
          if export_context.is_exporting_distributed():
            if self._use_native_multi_hash_table:
              ps_idx_to_multi_type_resp = self._native_hash_table_lookup_with_remote_predict(
                  lookup_data_on_wk, lookup_data_on_wk_row_split)
            else:
              ps_idx_to_multi_type_resp = self._lookup_with_remote_predict(
                  lookup_data_on_wk)
          else:
            if self._use_native_multi_hash_table:
              ps_idx_to_multi_type_resp = self._native_hash_table_lookup_raw(
                  lookup_data_on_wk, lookup_data_on_wk_row_split)
            else:
              ps_idx_to_multi_type_resp = self._lookup_raw(lookup_data_on_wk)

        for sub_table_name in self._sub_table_names:
          for ps_idx in range(self._num_ps):
            embeddings_tensor = ps_idx_to_multi_type_resp[ps_idx][sub_table_name]
            auxiliary_bundle_[
                f'__sharding_sparse_fids__{sub_table_name}:{ps_idx}:embs'] = embeddings_tensor
            if not export_context.is_exporting():
              fids_tensor = lookup_data_on_wk[ps_idx][sub_table_name]
              auxiliary_bundle_[
                  f'__sharding_sparse_fids__{sub_table_name}:{ps_idx}:fids'] = fids_tensor
              if self._use_native_multi_hash_table:
                fids_tensor_row_split = lookup_data_on_wk_row_split[ps_idx][
                    sub_table_name]
                auxiliary_bundle_[
                    f'__sharding_sparse_fids__{sub_table_name}:{ps_idx}:fids_row_split'] = fids_tensor_row_split

      if self._use_gpu:
        logging.info(
            f"PartitionedHashTable lookup gpu fused_layout tensor to gpu before: {auxiliary_bundle} {features}"
        )
        self.tensor_move_to_gpu(
            ((auxiliary_bundle_, ["__sharding_sparse_fids__batch_size",
                                  "__sharding_sparse_fids__nfl_size",
                                  "__sharding_sparse_fids__feature_size",
                                  "__sharding_sparse_fids__fid_size",
                                  "__sharding_sparse_fids__emb_size"]),
             (features_, ["req_time"])))
      logging.info(
          f"PartitionedHashTable lookup fused_layout enqueue before: {auxiliary_bundle} {features}"
      )
      (dequeued_features,
       deq_auxiliary_bundle), queue = enqueue_dicts_with_queue_return(
           (features_, auxiliary_bundle_), capacity=embedding_prefetch_capacity)
      if queue:
        self.add_queue_hook(EnqueueHook(queue))
      features_.update(dequeued_features)
      auxiliary_bundle_.update(deq_auxiliary_bundle)
      logging.info(
          f"PartitionedHashTable lookup fused_layout dequeue: {auxiliary_bundle} {features}"
      )

    def lookup_callable_fn(auxiliary_bundle_, features_):
      with tf.name_scope("pht_lookup"):
        lookup_data_on_wk: LookupData = {}
        lookup_data_on_wk_row_split: LookupData = {}
        for sub_table_name in self._sub_table_names:
          for ps_idx in range(self._num_ps):
            key = '__sharding_sparse_fids__shards@{}@{}'.format(ps_idx, sub_table_name)
            if ps_idx not in lookup_data_on_wk:
              lookup_data_on_wk[ps_idx] = {}
            lookup_data_on_wk[ps_idx][sub_table_name] = auxiliary_bundle_[key]

            if self._use_native_multi_hash_table:
              key = '__sharding_sparse_fids__shards_row_split@{}@{}'.format(ps_idx, sub_table_name)
              size_key = '__sharding_sparse_fids__shards_row_split_size@{}@{}'.format(ps_idx, sub_table_name)
              if ps_idx not in lookup_data_on_wk_row_split:
                lookup_data_on_wk_row_split[ps_idx] = {}
              if size_key not in auxiliary_bundle_:
                lookup_data_on_wk_row_split[ps_idx][sub_table_name] = auxiliary_bundle_[key]
              else:
                lookup_data_on_wk_row_split[ps_idx][sub_table_name] = distribution_ops.normalize_merged_split(
                    auxiliary_bundle_[key], auxiliary_bundle_[size_key])

      call_lookup(lookup_data_on_wk, lookup_data_on_wk_row_split,
                  auxiliary_bundle_, features_)
      return fused_layout_callable_fn(auxiliary_bundle_, features_)

    if ret_lookup_callable_fn:
      return lookup_callable_fn

    with tf.name_scope("pht_lookup"):
      lookup_data_on_wk: LookupData = {}
      lookup_data_on_wk_row_split: LookupData = {}
      for key, shard_fids in shards.items():
        sub_table_name, ps_idx = key.split(':')
        ps_idx = int(ps_idx)
        if self._use_native_multi_hash_table:
          shards_row_split_part = shards_row_split[key]
        else:
          shards_row_split_part = None

        if ps_idx in lookup_data_on_wk:
          lookup_data_on_wk[ps_idx][sub_table_name] = shard_fids
          lookup_data_on_wk_row_split[ps_idx][
              sub_table_name] = shards_row_split_part
        else:
          lookup_data_on_wk[ps_idx] = {sub_table_name: shard_fids}
          lookup_data_on_wk_row_split[ps_idx] = {
              sub_table_name: shards_row_split_part
          }

    call_lookup(lookup_data_on_wk, lookup_data_on_wk_row_split,
                auxiliary_bundle, features)

    if ret_fused_layout_callable_fn:
      return fused_layout_callable_fn
    else:
      return fused_layout_callable_fn(auxiliary_bundle, features)

  def apply_gradients(
      self,
      layout_grads_and_vars: List[Tuple[tf.Tensor, tf.Tensor]],
      global_step: tf.Tensor,
      req_time: Optional[tf.Tensor] = None,
      auxiliary_bundle: Dict[str, tf.Tensor] = None,
      async_function_mgr: prefetch_queue.AsyncFunctionMgr = None,
      async_push: bool = False,
      grad_scale: tf.Tensor = None) -> PartitionedHashTable:
    logging.info(
        f"PartitionedHashTable apply_gradients {async_push} {async_function_mgr}"
    )
    with tf.device(
        "/device:GPU:0" if self._enable_gpu_emb else "/device:CPU:0"):
      if req_time is None:
        req_time = tf.constant(0, dtype=tf.int64)
      else:
        req_time = tf.reduce_max(req_time)
    assert auxiliary_bundle is not None
    if self._enable_gpu_emb:
      assert not async_push
      return self._apply_gradients_gpu(layout_grads_and_vars, global_step,
                                       req_time, auxiliary_bundle, grad_scale=grad_scale)
    with tf.name_scope("pht_apply_gradients"):
      layout_grad, layout = zip(*layout_grads_and_vars)
      flattened_fids, flattened_fids_row_split, flattened_embs = [], [], []
      for sub_table_name in self._sub_table_names:
        for ps_idx in range(self._num_ps):
          flattened_fids.append(auxiliary_bundle[
              f'__sharding_sparse_fids__{sub_table_name}:{ps_idx}:fids'])
          if self._use_native_multi_hash_table:
            flattened_fids_row_split.append(auxiliary_bundle[
                f'__sharding_sparse_fids__{sub_table_name}:{ps_idx}:fids_row_split']
                                           )
          flattened_embs.append(auxiliary_bundle[
              f'__sharding_sparse_fids__{sub_table_name}:{ps_idx}:embs'])
      nfl_offset = auxiliary_bundle['__sharding_sparse_fids__nfl_offset']
      feature_offset = auxiliary_bundle[
          '__sharding_sparse_fids__feature_offset']
      fid_offset = auxiliary_bundle['__sharding_sparse_fids__fid_offset']
      batch_size = auxiliary_bundle['__sharding_sparse_fids__batch_size']

      with tf.device("/device:GPU:0" if self._use_gpu else "/device:CPU:0"):
        embeddings_grad = distribution_ops.fused_embedding_to_layout_grad(
            nfl_offset=nfl_offset,
            feature_offset=feature_offset,
            fid_offset=fid_offset,
            batch_size=batch_size,
            embeddings_list=flattened_embs,
            fid_list_row_split=None,  #flattened_fids_row_split, v3 no need
            layout_tensors_grad=layout_grad,
            layout_tensors_grad_scale=grad_scale,
            variant_type=self._inner_data_type,
            feature_cfgs=self._feature_configs,
            ps_num=self._num_ps)

      def hash_table_apply_gradients(flattened_fids_, flattened_fids_row_split_,
                                     embeddings_grad_, global_step_, req_time_):
        if self._use_gpu:
          logging.info(
              f"PartitionedHashTable apply_gradients fused_layout before tensor_move_to_cpu: \
              {flattened_fids_}, {flattened_fids_row_split_}, {embeddings_grad_}, \
              {global_step_}, {req_time_}")

          def tensor_move_to_cpu(*inputs):
            inputs_info = []
            to_cpu_value_list = []
            for tensor_ in inputs:
              to_cpu_dict = defaultdict(int)
              if isinstance(tensor_, List):
                for idx in range(len(tensor_)):
                  part_tensor = tensor_[idx]
                  to_cpu_dict[idx] = len(to_cpu_value_list)
                  to_cpu_value_list.append(part_tensor)
              elif tensor_ is not None:
                to_cpu_dict[-1] = len(to_cpu_value_list)
                to_cpu_value_list.append(tensor_)
              inputs_info.append(to_cpu_dict)
            with tf.device("/device:CPU:0"):
              gpu_value_list = tf.identity_n(to_cpu_value_list)
            outputs = []
            for input_idx in range(len(inputs)):
              to_cpu_dict = inputs_info[input_idx]
              part_input = inputs[input_idx]
              for k, idx in to_cpu_dict.items():
                if k == -1:
                  part_input = gpu_value_list[idx]
                  continue
                else:
                  part_input[k] = gpu_value_list[idx]
              outputs.append(part_input)
            return tuple(outputs)

          flattened_fids_, flattened_fids_row_split_, embeddings_grad_, \
            global_step_, req_time_ = tensor_move_to_cpu(
              flattened_fids_, flattened_fids_row_split_, embeddings_grad_,
              global_step_, req_time_)
          logging.info(
              f"PartitionedHashTable apply_gradients fused_layout tensor_move_to_cpu: \
              {flattened_fids_}, {flattened_fids_row_split_}, {embeddings_grad_}, \
              {global_step_}, {req_time_}")
        with tf.device("/device:CPU:0"):
          if self._use_native_multi_hash_table:
            new_tables = []
            for i in range(self._num_ps):
              splitted_id = tf.RaggedTensor.from_row_splits(
                  flattened_fids_[i],
                  flattened_fids_row_split_[i],
                  validate=False)
              splitted_flat_grad = embeddings_grad_[i]
              table: multi_hash_table_ops.RawMultiTypeHashTable = self._tables[
                  i]
              with nullcontext() if self._local_ps else ps_device(i):
                new_tables.append(
                    table.raw_apply_gradients(splitted_id,
                                              splitted_flat_grad,
                                              global_step=global_step,
                                              req_time=req_time_))
          else:
            fids_and_embgrad_pairs = list(zip(flattened_fids_,
                                              embeddings_grad_))
            logging.info(
                f"PartitionedHashTable apply_gradients fused_embedding_to_layout_grad done, {fids_and_embgrad_pairs}"
            )
            # reconstruct update data
            offset = 0
            update_data_on_worker: UpdateData = {
                ps_idx: {} for ps_idx in range(self._num_ps)
            }
            for sub_table_name in self._sub_table_names:
              for ps_idx in range(self._num_ps):
                update_data_on_worker[ps_idx][
                    sub_table_name] = fids_and_embgrad_pairs[offset]
                offset += 1
            new_tables = []
            for i in range(self._num_ps):
              keyed_fids, keyed_grads = {}, {}
              for tbname, (fids, emb_grad) in update_data_on_worker[i].items():
                keyed_fids[tbname] = fids
                keyed_grads[tbname] = emb_grad
              packed_list = tensor_utils.pack_typed_keyed_tensors(
                  [keyed_fids, keyed_grads])
              (packed_fids_on_wk, packed_emb_grad_on_wk,
               packed_sizes_on_wk) = packed_list
              typed_keyed_shape = tensor_utils.get_typed_keyed_shape(
                  [keyed_fids, keyed_grads])

              if self.transfer_float16:
                packed_emb_grad_on_wk = tf.cast(
                    packed_emb_grad_on_wk,
                    dtype=tf.float16,
                    name='{}_send_{}_CastToFloat16'.format(
                        packed_emb_grad_on_wk.op.name, i))

              with nullcontext() if self._local_ps else ps_device(i):
                packed_fids_on_ps = packed_fids_on_wk
                packed_emb_grad_on_ps = packed_emb_grad_on_wk
                packed_sizes_on_ps = packed_sizes_on_wk
                if self.transfer_float16:
                  packed_emb_grad_on_ps = tf.cast(
                      packed_emb_grad_on_ps,
                      dtype=tf.float32,
                      name='{}_recv_{}_CastToFloat32'.format(
                          packed_emb_grad_on_ps.op.name, i))
                keyed_fids_on_ps, keyed_grads_on_ps = tensor_utils.unpack_packed_tensors(
                    typed_keyed_shape,
                    packed_list=[
                        packed_fids_on_ps, packed_emb_grad_on_ps,
                        packed_sizes_on_ps
                    ])
                partitioned_update_data_on_ps = {
                    tbname:
                    (keyed_fids_on_ps[tbname], keyed_grads_on_ps[tbname])
                    for tbname in keyed_fids_on_ps
                }
                new_tables.append(self._tables[i].apply_gradients(
                    partitioned_update_data_on_ps,
                    global_step_,
                    req_time=req_time_))
          return self._copy_with_new_table(new_tables).as_op()

      if async_function_mgr is None or not async_push:
        return hash_table_apply_gradients(flattened_fids,
                                          flattened_fids_row_split,
                                          embeddings_grad, global_step,
                                          req_time)
      else:
        return async_function_mgr.add_async_function(
            hash_table_apply_gradients,
            (flattened_fids, flattened_fids_row_split, embeddings_grad,
             global_step, req_time),
            is_async=async_push,
            queue_name="postpush_queue")

  def tensor_move_to_gpu(self, inputs):
    inputs_info = []
    to_gpu_value_list = []
    for tensor_dict, except_list in inputs:
      to_gpu_dict = defaultdict(int)
      for k, v in tensor_dict.items():
        if k in except_list:
          continue
        to_gpu_dict[k] = len(to_gpu_value_list)
        to_gpu_value_list.append(v)
      inputs_info.append((to_gpu_dict, tensor_dict))
    with tf.device("/device:GPU:0"):
      gpu_value_list = tf.identity_n(to_gpu_value_list)
    for to_gpu_dict, tensor_dict in inputs_info:
      for k, idx in to_gpu_dict.items():
        tensor_dict[k] = gpu_value_list[idx]

  def as_op(self, name=None):
    name = name or "pht_as_op"
    if self._enable_gpu_emb:
      with tf.control_dependencies(self._dependency_ops):
        return self._table.as_op(name)
    ops = []
    for i in range(self._num_ps):
      with nullcontext() if self._local_ps else ps_device(i):
        ops.append(self._tables[i].as_op(name="{}/sub_{}".format(name, i)))
    with tf.control_dependencies(ops):
      return tf.no_op(name=("{}/done".format(name)))

  def _lookup_gpu(
      self,
      features: Dict[str, tf.Tensor],
      auxiliary_bundle: Dict[str, tf.Tensor] = None,
  ) -> Dict[str, Union[tf.Tensor, List[tf.Tensor]]]:
    if enable_bps:
      import byteps.tensorflow as bps
    with tf.name_scope("pht_lookup_gpu"):
      logging.info(
          f"PartitionedHashTable lookup_gpu fused_layout tensor to gpu before: {features}"
      )

      slot_num = self._num_table
      recv_emb_splits_tmp = tf.reshape(
          tf.matmul(
              tf.reshape(
                  features["__sharding_sparse_fids__shards_table_row_lengths"],
                  [self._shard_num, slot_num]),
              tf.expand_dims(tf.constant(self._output_dims, dtype=tf.int32),
                             -1)  # [slot_num, 1]
          ),
          [-1]  # flatten
      )
      features["__sharding_sparse_fids__recv_emb_splits"] = recv_emb_splits_tmp

      #self.tensor_move_to_gpu(
      #    ((features, ["__sharding_sparse_fids__batch_size"]),))
      logging.info(
          f"PartitionedHashTable lookup_gpu fused_layout enqueue before: {features}"
      )

      sharding_features = ParserCtx.sharding_sparse_fids_features_parse_from_features(
          features)
      shards_value, shards_row_lengths, shards_table_row_lengths, fid_offset, feature_offset, \
      nfl_offset, batch_size, fid_list_emb_row_lenth, recv_emb_splits = \
        sharding_features.get("shards_value"), sharding_features.get("shards_row_lengths"), \
        sharding_features.get("shards_table_row_lengths"), sharding_features.get("fid_offset") ,\
        sharding_features.get("feature_offset"), sharding_features.get("nfl_offset") ,\
        sharding_features.get("batch_size"), sharding_features.get("fid_list_emb_row_lenth"), \
        sharding_features.get("recv_emb_splits")

      if auxiliary_bundle is None:
        auxiliary_bundle = {}
      auxiliary_bundle['__sharding_sparse_fids__fid_offset'] = fid_offset
      auxiliary_bundle[
          '__sharding_sparse_fids__feature_offset'] = feature_offset
      auxiliary_bundle['__sharding_sparse_fids__nfl_offset'] = nfl_offset
      auxiliary_bundle['__sharding_sparse_fids__batch_size'] = batch_size
      auxiliary_bundle[
          "__sharding_sparse_fids__recv_emb_splits"] = recv_emb_splits
      auxiliary_bundle[
          "__sharding_sparse_fids__fid_list_emb_row_lenth"] = fid_list_emb_row_lenth

      logging.info(
          f"sharding_sparse_fids done, {shards_value} {shards_row_lengths}")

      all_fids = shards_value
      shard_sizes = shards_row_lengths
      sharded_slot_sizes = shards_table_row_lengths

      # We exchange the flattened IDs and their splits.
      # M: num_of_ids,
      # N: num_of_shards,
      # K: num_of_merged_tables,
      # E: num_of_total_embedding_dim.

      # id_flat_t: [M], id_flat_split_t: [N]
      # id_size_flat_t: [K*N], id_size_flat_split_t: [N]
      if enable_bps and enable_bps_fid:
        logging.info('Enabled BPS for fid alltoall')
        id_flat_t, id_flat_split_t = bps.alltoall(all_fids,
                                                  splits=shard_sizes,
                                                  with_size=True,
                                                  name='fid_data')
        # We also add the flat_t sizes.
        id_size_flat_t = bps.alltoall(sharded_slot_sizes,
                                      splits=[slot_num] * self._shard_num,
                                      recv_splits=([slot_num] *
                                                   self._shard_num),
                                      name='fid_size')
      elif enable_custom_optimized_hvd:
        id_flat_t, id_flat_split_t = hvd.alltoall(all_fids,
                                                  splits=shard_sizes,
                                                  with_size=True)
        # We also add the flat_t sizes.
        id_size_flat_t = hvd.alltoall(sharded_slot_sizes,
                                      splits=[slot_num] * self._shard_num,
                                      recv_splits=[slot_num] * self._shard_num)
      elif enable_hvd:
        if enable_hvd_fid_g2g:
          logging.info('Enabled hvd for fid alltoall g2g')
          with tf.device("/device:GPU:0"):
            id_flat_t = hvd.alltoall(all_fids, splits=shard_sizes)
            id_flat_split_t = hvd.alltoall(shard_sizes)
            id_size_flat_t = hvd.alltoall(sharded_slot_sizes,
                                          splits=[slot_num] * self._shard_num)
        else:
          id_flat_t = hvd.alltoall(all_fids, splits=shard_sizes)
          id_flat_split_t = hvd.alltoall(shard_sizes)
          id_size_flat_t = hvd.alltoall(sharded_slot_sizes,
                                        splits=[slot_num] * self._shard_num)

      auxiliary_bundle["__sharding_sparse_fids__shard_sizes"] = shard_sizes
      auxiliary_bundle["__sharding_sparse_fids__id_flat_t"] = id_flat_t
      auxiliary_bundle[
          "__sharding_sparse_fids__id_size_flat_t"] = id_size_flat_t

      # fused_embeddings: [E], fused_splits: [N]
      # id_offsets: [K*N], emb_offsets: [K*N]
      req_time = features.get("req_time", None)
      with tf.device(
          "/device:GPU:0" if self._enable_gpu_emb else "/device:CPU:0"):
        if req_time is None:
          logging.warning(f"fused_embedding_to_layout use default req_time")
          req_time = tf.constant(0, dtype=tf.int64)
        else:
          req_time = tf.reduce_max(req_time)

      with tf.device("/GPU:0"):
        fused_embeddings, embedding_splits, id_offsets, emb_offsets, indices = \
            self._table.fused_lookup(id_flat_t, id_size_flat_t, self._shard_num, req_time)
      if FLAGS.enable_alltoall_metrics:
        with tf.device("/CPU:0"):
          tf.compat.v1.summary.histogram("fused_embedding_splits",
                                         embedding_splits)

      auxiliary_bundle[
          "__sharding_sparse_fids__fused_embeddings"] = fused_embeddings
      auxiliary_bundle[
          "__sharding_sparse_fids__embedding_splits"] = embedding_splits
      auxiliary_bundle["__sharding_sparse_fids__id_offsets"] = id_offsets
      auxiliary_bundle["__sharding_sparse_fids__emb_offsets"] = emb_offsets
      auxiliary_bundle["__sharding_sparse_fids__indices"] = indices

      deq_auxiliary_bundle, queue = enqueue_dicts_with_queue_return(
          auxiliary_bundle,
          capacity=int(self._queue_configs.get("enable_pipelined_fwda2a", 0)),
          queue_name="queue_lookup_to_fusedEmbA2A")
      if queue:
        self.add_queue_hook(EnqueueHook(queue))
      auxiliary_bundle.update(deq_auxiliary_bundle)

      fused_embeddings = auxiliary_bundle.pop(
          "__sharding_sparse_fids__fused_embeddings")
      embedding_splits = auxiliary_bundle[
          "__sharding_sparse_fids__embedding_splits"]
      recv_emb_splits = auxiliary_bundle[
          "__sharding_sparse_fids__recv_emb_splits"]

      # recv_embeddings: [E'], recv_embedding_sizes: [N]
      if enable_bps and enable_bps_fwd:
        if enable_bps_fwd_gdr:
          if enable_bps_fwd_gdr_g2g:
            logging.info('Enabled BPS for fwd embed alltoall GDR (G2G)')
            with tf.device("/device:GPU:0"):
              fused_embeddings_gpu = fused_embeddings
            with tf.device("/device:GPU:0"):
              recv_embeddings = bps.alltoall(fused_embeddings_gpu,
                                             embedding_splits,
                                             recv_splits=recv_emb_splits,
                                             name="fwd_alltoall_g2g")
          else:
            logging.info('Enabled BPS for fwd embed alltoall GDR (C2G)')
            with tf.device("/device:GPU:0"):
              recv_embeddings = bps.alltoall_cpu2gpu(
                  fused_embeddings,
                  embedding_splits,
                  recv_splits=recv_emb_splits,
                  name="fwd_alltoall_c2g")
        else:
          logging.info('Enabled BPS for fwd embed alltoall')
          recv_embeddings = bps.alltoall(fused_embeddings,
                                         embedding_splits,
                                         recv_splits=recv_emb_splits,
                                         name="fwd_alltoall")
      elif enable_custom_optimized_hvd:
        if enable_hvd_fwd_g2g:
          logging.info('Enabled optimized hvd for fwd embed alltoall g2g')
          with tf.device("/device:GPU:0"):
            recv_embeddings = hvd.alltoall(
                fused_embeddings,
                embedding_splits,
                recv_splits=recv_emb_splits,
            )
        else:
          logging.info('Enabled optimized hvd for fwd embed alltoall')
          recv_embeddings = hvd.alltoall(
              fused_embeddings,
              embedding_splits,
              recv_splits=recv_emb_splits,
          )
      elif enable_hvd:
        if enable_hvd_fwd_g2g:
          logging.info('Enabled hvd for fwd embed alltoall g2g')
          with tf.device("/device:GPU:0"):
            recv_embeddings = hvd.alltoall(fused_embeddings,
                                           embedding_splits,
                                           name='hvd_fwd_a2a_g2g')
        else:
          logging.info('Enabled hvd for fwd embed alltoall')
          recv_embeddings = hvd.alltoall(fused_embeddings,
                                         embedding_splits,
                                         name='hvd_fwd_a2a')

      auxiliary_bundle[
          "__sharding_sparse_fids__recv_embeddings"] = recv_embeddings
      #TODO enable embedding_prefetch_capacity train will slow down
      '''
      deq_auxiliary_bundle, queue = enqueue_dicts_with_queue_return(
          auxiliary_bundle,
          capacity=int(self._queue_configs.get("embedding_prefetch_capacity",
                                               0)),
          queue_name="queue_fusedEmbA2A_to_fusedGather")
      if queue:
        self.add_queue_hook(EnqueueHook(queue))
      auxiliary_bundle.update(deq_auxiliary_bundle)
      '''

      recv_embeddings = auxiliary_bundle[
          "__sharding_sparse_fids__recv_embeddings"]

      fid_offset = auxiliary_bundle['__sharding_sparse_fids__fid_offset']
      feature_offset = auxiliary_bundle[
          '__sharding_sparse_fids__feature_offset']
      nfl_offset = auxiliary_bundle['__sharding_sparse_fids__nfl_offset']
      batch_size = auxiliary_bundle['__sharding_sparse_fids__batch_size']
      fid_list_emb_row_lenth = auxiliary_bundle[
          '__sharding_sparse_fids__fid_list_emb_row_lenth']

      with tf.device("/device:GPU:0"):
        '''
        recv_embeddings_split = tf.split(recv_embeddings,
                                         fid_list_emb_row_lenth)

        flattened_embs = [None] * (self._num_ps * self._num_table)
        recv_embeddings_split_index = 0
        for ps_index in range(self._num_ps):
          for table_idx in range(self._num_table):
            flattened_embs[
                table_idx * self._num_ps +
                ps_index] = recv_embeddings_split[recv_embeddings_split_index]
            recv_embeddings_split_index += 1
        '''

        layout_tensors = distribution_ops.fused_embedding_to_layout(
            [recv_embeddings],  #flattened_embs,
            None,  #self.fids_list_row_split, v3 not need fids_list_row_split
            fid_list_emb_row_lenth=fid_list_emb_row_lenth,
            fid_offset=fid_offset,
            feature_offset=feature_offset,
            nfl_offset=nfl_offset,
            batch_size=batch_size,
            variant_type=self._inner_data_type,
            feature_cfgs=self._feature_configs,
            ps_num=self._num_ps,
            version=4)
      #auxiliary_bundle[
      #    '__sharding_sparse_fids__flattened_embs'] = flattened_embs
      logging.info("fused_embedding_to_layout done!")
      return self.nest_layout(layout_tensors)

  def _apply_gradients_gpu(
      self,
      layout_grads_and_vars: List[Tuple[tf.Tensor, tf.Tensor]],
      global_step: tf.Tensor,
      req_time: Optional[tf.Tensor] = None,
      auxiliary_bundle: Dict[str, tf.Tensor] = None,
      grad_scale: tf.Tensor = None) -> PartitionedHashTable:
    with tf.name_scope("pht_apply_gradients_gpu"):

      auxiliary_bundle['__sharding_sparse_fids__global_step'] = global_step
      auxiliary_bundle["__sharding_sparse_fids__req_time"] = req_time

      layout_grad, layout = zip(*layout_grads_and_vars)
      assert auxiliary_bundle is not None

      fid_offset = auxiliary_bundle.pop('__sharding_sparse_fids__fid_offset')
      feature_offset = auxiliary_bundle.pop(
          '__sharding_sparse_fids__feature_offset')
      nfl_offset = auxiliary_bundle.pop('__sharding_sparse_fids__nfl_offset')
      batch_size = auxiliary_bundle.pop('__sharding_sparse_fids__batch_size')

      recv_embeddings = auxiliary_bundle.pop(
          "__sharding_sparse_fids__recv_embeddings")
      fid_list_emb_row_lenth = auxiliary_bundle.pop(
          '__sharding_sparse_fids__fid_list_emb_row_lenth')

      #flattened_embs = auxiliary_bundle.pop(
      #    '__sharding_sparse_fids__flattened_embs')

      with tf.device("/device:GPU:0"):
        embeddings_grad = distribution_ops.fused_embedding_to_layout_grad(
            nfl_offset=nfl_offset,
            feature_offset=feature_offset,
            fid_offset=fid_offset,
            batch_size=batch_size,
            embeddings_list=[recv_embeddings],  #flattened_embs,
            fid_list_row_split=None,  #flattened_fids_row_split, v3 no need
            fid_list_emb_row_lenth=fid_list_emb_row_lenth,
            layout_tensors_grad=layout_grad,
            layout_tensors_grad_scale=grad_scale,
            variant_type=self._inner_data_type,
            feature_cfgs=self._feature_configs,
            ps_num=self._num_ps,
            version=4)
        '''
        embeddings_grad_reorder = [None] * (self._num_ps * self._num_table)
        embeddings_grad_index = 0
        for table_idx in range(self._num_table):
          for ps_index in range(self._num_ps):
            embeddings_grad_reorder[
                table_idx * self._num_ps +
                ps_index] = embeddings_grad[embeddings_grad_index]
            embeddings_grad_index += 1
        grad_flat = tf.concat(embeddings_grad_reorder, axis=0)
        '''
        grad_flat = embeddings_grad[0]

      auxiliary_bundle['__sharding_sparse_fids__grad_flat'] = grad_flat

      deq_auxiliary_bundle, async_optimize_queue = enqueue_dicts_with_queue_return(
          auxiliary_bundle,
          capacity=int(self._queue_configs.get("enable_async_optimize", 0)),
          queue_name="queue_fusedGatherGrad_to_fusedEmbGradA2A")
      auxiliary_bundle.update(deq_auxiliary_bundle)

      grad_flat = auxiliary_bundle.pop("__sharding_sparse_fids__grad_flat")
      recv_emb_splits = auxiliary_bundle.pop(
          "__sharding_sparse_fids__recv_emb_splits")
      embedding_splits = auxiliary_bundle.pop(
          "__sharding_sparse_fids__embedding_splits")

      if enable_bps and enable_bps_bwd:
        import byteps.tensorflow as bps
        from byteps.tensorflow.compression import FP16Compressor as BPSFP16Compressor
        if enable_bps_bwd_gdr:
          with tf.device("/device:GPU:0"), tf.name_scope("bps_bwd_alltoall"):
            if enable_bps_bwd_gdr_g2g:
              logging.info('Enabled BPS for bwd embed alltoall GDR (G2G)')
              bwd_tensor_name = "bwd_alltoall_g2g"
              grad_flat_t = bps.alltoall(grad_flat,
                                         recv_emb_splits,
                                         recv_splits=embedding_splits,
                                         name=bwd_tensor_name)
              if enable_bps_bwd_cast == 16:
                # do cast on GPU
                if enable_bps_bwd_fake_cast:
                  grad_flat_t = grad_flat_t * 0.0
                grad_flat_t = tf.cast(grad_flat_t, tf.float32)
              with tf.device("/device:CPU:0"):
                grad_flat_t = tf.identity(grad_flat_t)
            else:
              logging.info('Enabled BPS for bwd embed alltoall GDR (G2C)')
              bwd_tensor_name = "bwd_alltoall_g2c"
              grad_flat_t = bps.alltoall_gpu2cpu(grad_flat,
                                                 recv_emb_splits,
                                                 recv_splits=embedding_splits,
                                                 name=bwd_tensor_name)
              with tf.device("/device:CPU:0"):
                # grad_flat_t<tensor>.device = <tensor>._op.device (bps.alltoall_gpu2cpu as GPU op)
                # However the tensor is on CPU, so we fixed the tensor placement info with an identity.
                grad_flat_t = tf.identity(grad_flat_t)
              if enable_bps_bwd_cast == 16:
                if enable_bps_bwd_fake_cast:
                  grad_flat_t = grad_flat_t * 0.0
                grad_flat_t = tf.cast(grad_flat_t, tf.float32)
        else:
          logging.info(
              'Enabled BPS for bwd embed alltoall with cast optimization')
          grad_flat_t = bps.alltoall(grad_flat,
                                     recv_emb_splits,
                                     recv_splits=embedding_splits,
                                     name="bwd_alltoall")
          if enable_bps_bwd_cast == 16:
            if enable_bps_bwd_fake_cast:
              grad_flat_t = grad_flat_t * 0.0
            grad_flat_t = tf.cast(grad_flat_t, tf.float32)

        sent_grad_split_size = embedding_splits
      elif enable_custom_optimized_hvd:
        if enable_hvd_bwd_g2g:
          logging.info('Enabled optimized hvd for bwd embed alltoall g2g')
          with tf.device("/device:GPU:0"):
            grad_flat_t, sent_grad_split_size = hvd.alltoall(
                grad_flat,
                recv_emb_splits,
                recv_splits=embedding_splits,
                with_size=True,
                compression=FP16Compressor)
            with tf.device("/device:CPU:0"):
              grad_flat_t = tf.identity(grad_flat_t)
        else:
          logging.info('Enabled optimized hvd for bwd embed alltoall')
          grad_flat_t, sent_grad_split_size = hvd.alltoall(
              grad_flat,
              recv_emb_splits,
              recv_splits=embedding_splits,
              with_size=True,
              compression=FP16Compressor)
        if FLAGS.enable_alltoall_metrics and (self._shard_num > 1):
          # There is some issue with tf.compat.v1.summary on the horovod alltoall input,
          # using output instead. They are almost equivalent.
          shard_sizes = auxiliary_bundle["__sharding_sparse_fids__shard_sizes"]
          shard_sizes = tf.slice(shard_sizes, [1], [self._shard_num - 1])
          total_alltoall_id_size = tf.reduce_sum(shard_sizes)
          recv_emb_splits = tf.slice(tf.identity(recv_emb_splits), [1],
                                     [self._shard_num - 1])
          total_alltoall_emb_size = tf.reduce_sum(tf.identity(recv_emb_splits))
          tmp_result = tf.reshape(total_alltoall_id_size, [1])
          tmp_result2 = tf.reshape(total_alltoall_emb_size, [1])
          total_id_dist = hvd.allgather(tmp_result)
          total_emb_dist = hvd.allgather(tmp_result2)
          with tf.device("/CPU:0"):
            tf.compat.v1.summary.histogram("Alltoall_id_dist", total_id_dist)
            tf.compat.v1.summary.histogram("Alltoall_emb_dist", total_emb_dist)
          if self._index == 0:
            min_idx = tf.math.argmin(shard_sizes)
            max_idx = tf.math.argmax(shard_sizes)
            min_idx_size = tf.reshape(
                tf.slice(shard_sizes, tf.reshape(min_idx, [-1]), [1]), [])
            max_idx_size = tf.reshape(
                tf.slice(shard_sizes, tf.reshape(max_idx, [-1]), [1]), [])
            with tf.device("/CPU:0"):
              tf.compat.v1.summary.histogram("Alltoall_id_splits", shard_sizes)
              tf.compat.v1.summary.scalar("Alltoall_id_sizes",
                                          total_alltoall_id_size)
              tf.compat.v1.summary.scalar("Alltoall_id_min_idx", min_idx)
              tf.compat.v1.summary.scalar("Alltoall_id_max_idx", max_idx)
              tf.compat.v1.summary.scalar("Alltoall_id_min_size", min_idx_size)
              tf.compat.v1.summary.scalar("Alltoall_id_max_size", max_idx_size)
            sent_grad_split_size = tf.slice(sent_grad_split_size, [1],
                                            [self._shard_num - 1])
            total_alltoall_grad_size = tf.reduce_sum(sent_grad_split_size)
            with tf.device("/CPU:0"):
              tf.compat.v1.summary.histogram("Alltoall_grad_splits",
                                             sent_grad_split_size)
              tf.compat.v1.summary.scalar("Alltoall_grad_sizes",
                                          total_alltoall_grad_size)
            min_emb = tf.math.argmin(recv_emb_splits)
            max_emb = tf.math.argmax(recv_emb_splits)
            min_emb_size = tf.reshape(
                tf.slice(recv_emb_splits, tf.reshape(min_emb, [-1]), [1]), [])
            max_emb_size = tf.reshape(
                tf.slice(recv_emb_splits, tf.reshape(max_emb, [-1]), [1]), [])
            with tf.device("/CPU:0"):
              tf.compat.v1.summary.histogram("Alltoall_emb_splits",
                                             tf.identity(recv_emb_splits))
              tf.compat.v1.summary.scalar("Alltoall_emb_sizes",
                                          tf.identity(total_alltoall_emb_size))
              tf.compat.v1.summary.scalar("Alltoall_emb_min_idx", min_emb)
              tf.compat.v1.summary.scalar("Alltoall_emb_max_idx", max_emb)
              tf.compat.v1.summary.scalar("Alltoall_emb_min_size", min_emb_size)
              tf.compat.v1.summary.scalar("Alltoall_emb_max_size", max_emb_size)
      elif enable_hvd:
        if enable_hvd_bwd_g2g:
          logging.info('Enabled hvd for bwd embed alltoall g2g')
          with tf.device("/device:GPU:0"):
            grad_flat_t = hvd.alltoall(grad_flat,
                                       recv_emb_splits,
                                       name='hvd_bwd_a2a_g2g')
            #with tf.device("/device:CPU:0"):
            #  grad_flat_t = tf.identity(grad_flat_t)
        else:
          logging.info('Enabled hvd for bwd embed alltoall')
          grad_flat_t = hvd.alltoall(grad_flat, recv_emb_splits)
      else:
        grad_flat_t = grad_flat

      auxiliary_bundle["__sharding_sparse_fids__grad_flat_t"] = grad_flat_t
      auxiliary_bundle.pop("__sharding_sparse_fids__shard_sizes")

      deq_auxiliary_bundle, q = enqueue_dicts_with_queue_return(
          auxiliary_bundle,
          capacity=int(self._queue_configs.get("enable_pipelined_bwda2a", 0)),
          queue_name="queue_fusedEmbGradA2A_to_sparseOptimize")
      if q:
        self.add_queue_hook(EnqueueHook(q))
      auxiliary_bundle.update(deq_auxiliary_bundle)

      with tf.device("/GPU:0"):
        updated_table = self._table.fused_apply_gradient(
            auxiliary_bundle.pop("__sharding_sparse_fids__id_flat_t"),
            auxiliary_bundle.pop("__sharding_sparse_fids__indices"),
            auxiliary_bundle.pop("__sharding_sparse_fids__id_size_flat_t"),
            auxiliary_bundle.pop("__sharding_sparse_fids__grad_flat_t"),
            auxiliary_bundle.pop("__sharding_sparse_fids__id_offsets"),
            auxiliary_bundle.pop("__sharding_sparse_fids__emb_offsets"),
            auxiliary_bundle.pop("__sharding_sparse_fids__global_step"),
            auxiliary_bundle.pop("__sharding_sparse_fids__req_time"),
            self._shard_num)

      update_op = self._copy_with_new_table_gpu(updated_table)
      # TODO(zouxuan): add better tests to test the async optimize.
      if async_optimize_queue:
        self.add_queue_hook(
            AsyncPushHook(async_optimize_queue, update_op.as_op()))
        self._dependency_ops.append(async_optimize_queue.enqueue_op)
        # return self essentially means to call dependency_ops
        return self.as_op()
      else:
        return update_op.as_op()

  def _copy_with_new_table(self,
                           new_tables: List[tf.Tensor]) -> PartitionedHashTable:
    copied = copy.copy(self)
    copied._tables = new_tables
    return copied

  def _copy_with_new_table_gpu(self,
                               new_table: tf.Tensor) -> PartitionedHashTable:
    copied = copy.copy(self)
    copied._dependency_ops = copy.copy(self._dependency_ops)
    copied._table = new_table
    return copied

  def _native_hash_table_update(
      self, method_name: str, name_scope: str,
      update_data: AssignData) -> PartitionedHashTable:
    with tf.name_scope(name_scope):
      sharded_slot_to_id_and_value: Dict[int, Dict[str, Tuple[
          tf.Tensor, tf.Tensor]]] = collections.defaultdict(dict)
      for slot, (id, value) in update_data.items():
        index = tf.math.floormod(id, self._num_ps)
        split_ids = distribution_ops.split_by_indices(index, id, self._num_ps)
        split_values = distribution_ops.split_by_indices(
            index, value, self._num_ps)
        for i in range(self._num_ps):
          sharded_slot_to_id_and_value[i][slot] = (split_ids[i],
                                                   split_values[i])
      new_tables = []
      for i in range(self._num_ps):
        new_tables.append(
            getattr(self._tables[i],
                    method_name)(sharded_slot_to_id_and_value[i]))

      return self._copy_with_new_table(new_tables)

  def _update(self, method_name: str, name_scope: str,
              update_data: AssignData) -> PartitionedHashTable:
    if self._enable_gpu_emb:
      raise NotImplementedError
    with tf.name_scope(name_scope):
      new_tables = []
      for i in range(self._num_ps):
        new_tables.append(getattr(self._tables[i], method_name)(update_data[i]))
      return self._copy_with_new_table(new_tables)

  def assign(self, data: AssignData) -> PartitionedHashTable:
    if self._enable_gpu_emb:
      raise NotImplementedError
    if self._use_native_multi_hash_table:
      return self._update("assign", "dmtht_a", data)
    else:
      return self._update("assign", "pht_assign", data)

  def assign_add(self, data: AssignData) -> PartitionedHashTable:
    if self._enable_gpu_emb:
      raise NotImplementedError
    if self._use_native_multi_hash_table:
      return self._update("assign_add", "dmtht_aa", data)
    else:
      return self._update("assign_add", "pht_assign_add", data)

  def flatten_layout(
      self, nested: Dict[str, Union[tf.Tensor,
                                    List[tf.Tensor]]]) -> List[tf.Tensor]:
    result = []
    for name in sorted(self._feature_configs.out_configs):
      value = nested[name]
      if isinstance(value, (list, tuple)):
        assert all(isinstance(v, tf.Tensor) for v in value)
        result.extend(value)
      else:
        assert isinstance(value, tf.Tensor)
        result.append(value)
    return result

  def nest_layout(
      self,
      tensors: List[tf.Tensor]) -> Dict[str, Union[tf.Tensor, List[tf.Tensor]]]:
    offset, result = 0, {}
    for name in sorted(self._feature_configs.out_configs):
      conf = self._feature_configs.out_configs[name]
      if conf.out_type == OutType.NONE:
        sub_list = []
        for _ in range(len(conf.slice_configs)):
          sub_list.append(tensors[offset])
          offset += 1
        result[name] = sub_list
      else:
        result[name] = tensors[offset]
        offset += 1

    return result

  def add_queue_hook(self, hook):
    # Allow pipelined graph execution.
    if not getattr(self, "_local_queue_hooks", None):
      self._local_queue_hooks = []
    self._local_queue_hooks.append(hook)

  def get_queue_hooks(self):
    hooks = copy.copy(getattr(self, "_local_queue_hooks", []))
    if getattr(self, "_tables", None):
      hooks.extend(
          itertools.chain.from_iterable(
              [t.get_queue_hooks() for t in self._tables]))
    return hooks
