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
from monolith.native_training.data.parsers import get_default_parser_ctx

FLAGS = flags.FLAGS

enable_hvd = os.getenv("MONOLITH_WITH_HOROVOD")
enable_custom_optimized_hvd = os.getenv("MONOLITH_WITH_OPTIMIZED_HOROVOD")
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


class PartitionedHashTable(object):
  # Allow pipelined graph execution.
  _local_queue_hooks: List[prefetch_queue.EnqueueHook |
                           prefetch_queue.AsyncPushHook]
  _native_multi_hash_table_fake_table = "native_multi_hash_table_fake_table"

  @classmethod
  def gen_feature_configs(
      cls,
      feature_info: Dict[str, FeatureInfo],
      layout_configs: Dict[str, OutConfig],
  ):
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

    return feature_configs

  @classmethod
  def no_merge_feature_config(
      cls, feature_name_to_config: Dict[str, entry.HashTableConfigInstance]):
    sub_table_name_to_config, feature_to_sub_table = {}, {}
    for feature_name in sorted(feature_name_to_config):
      feature_to_sub_table[
          feature_name] = cls._native_multi_hash_table_fake_table
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
               feature_name_to_config: Dict[str, entry.HashTableConfigInstance],
               table_factory: TableFactory,
               layout_configs: Dict[str, OutConfig],
               feature_to_unmerged_slice_dims: Dict[str, List[int]],
               feature_to_combiner: Dict[str, embedding_combiners.Combiner],
               use_native_multi_hash_table: bool,
               max_rpc_deadline_millis: int = 30,
               unique: bool = False,
               transfer_float16: bool = False):
    self._num_ps: int = 1 if num_ps == 0 else num_ps
    self._local_ps: bool = True if num_ps == 0 else False
    self._feature_name_to_config = feature_name_to_config  # unmerged, feature_name -> htconf
    self._max_rpc_deadline_millis = max_rpc_deadline_millis
    self._unique = unique
    self.transfer_float16 = transfer_float16

    self._use_native_multi_hash_table = use_native_multi_hash_table and not self.transfer_float16
    self.fids_list = []
    self.fids_list_row_split = []
    self.embeddings_list = []

    parser_ctx = get_default_parser_ctx()
    self._inner_data_type = parser_ctx.parser_type

    # feature/slot -> sub_hashtable_name
    if self._use_native_multi_hash_table:
      self._sub_table_name_to_config, feature_to_sub_table = self.no_merge_feature_config(
          feature_name_to_config)
    else:
      self._sub_table_name_to_config, feature_to_sub_table = self.merge_feature_config(
          feature_name_to_config)

    feature_info = {}
    for feature_name, sub_table in feature_to_sub_table.items():
      feature_info[feature_name] = FeatureInfo(
          feature_to_unmerged_slice_dims[feature_name],
          feature_to_combiner[feature_name], sub_table)
    self._feature_configs: FeatureConfigs = self.gen_feature_configs(
        feature_info, layout_configs)

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

    self._tables = []
    for i in range(self._num_ps):
      if export_context.is_exporting_distributed():
        ps_graph = export_context.get_current_export_ctx().sub_graph(f"ps_{i}")
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
          for sub_table_name, config in self._sub_table_name_to_config.items():
            config.set_learning_rate_tensor(
                sub_table_to_learning_rate_tensor_on_ps[sub_table_name])
          self._tables.append(table_factory(i, self._sub_table_name_to_config))

  @property
  def slot_mapping(self):
    """Returns slot mapping."""
    return self._slot_mapping

  def _native_hash_table_lookup_raw(self, lookup_data_on_wk: LookupData,
                                    lookup_data_on_wk_row_split: LookupData):
    ps_idx_to_multi_type_resp = {}
    for i in range(self._num_ps):
      table: multi_hash_table_ops.RawMultiTypeHashTable = self._tables[i]
      splitted_id = tf.RaggedTensor.from_row_splits(
          lookup_data_on_wk[i][self._native_multi_hash_table_fake_table],
          lookup_data_on_wk_row_split[i][
              self._native_multi_hash_table_fake_table],
          validate=False)
      is_standalone = export_context.is_exporting_standalone() or self._local_ps
      with nullcontext() if is_standalone else ps_device(i):
        flat_emb = table.raw_lookup(splitted_id)
      ps_idx_to_multi_type_resp[i] = {
          self._native_multi_hash_table_fake_table: flat_emb
      }
    return ps_idx_to_multi_type_resp

  def _lookup_raw(self, lookup_data_on_wk: LookupData):
    ps_idx_to_multi_type_resp = {}
    for i in range(self._num_ps):
      # sub_table_name -> fids tensor
      multi_type_query: Dict[str, tf.Tensor] = lookup_data_on_wk[i]
      # Note: this is a python object, use it outside this device context required no data transfer
      multi_type_query_shape: Dict[
          str, List[int]] = tensor_utils.get_keyed_shape(multi_type_query)

      # to reduce the number of rpc (send/recv ops) call, we pack fids by concat
      packed_fids_on_worker = tensor_utils.pack_tensors(multi_type_query)

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

      packed_embedding_on_worker = packed_embedding_on_ps  # data transfer in logic

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

    return ps_idx_to_multi_type_resp

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
      sparse_features: tf.Tensor,
      auxiliary_bundle: Dict[str, tf.Tensor] = None
  ) -> Dict[str, Union[tf.Tensor, List[tf.Tensor]]]:
    with tf.name_scope("pht_lookup"):
      shards, fid_offset, feature_offset, nfl_offset, batch_size, shards_row_split = sharding_sparse_fids(
          sparse_features,
          ps_num=self._num_ps,
          feature_cfgs=self._feature_configs,
          unique=self._unique,
          input_type=self._inner_data_type,
          parallel_flag=4 if self._use_native_multi_hash_table else 0)
      auxiliary_bundle['fid_offset'] = fid_offset
      auxiliary_bundle['feature_offset'] = feature_offset
      auxiliary_bundle['nfl_offset'] = nfl_offset
      auxiliary_bundle['batch_size'] = batch_size
      logging.info(f"sharding_sparse_fids done, {shards}")
      lookup_data_on_wk: LookupData = {}
      lookup_data_on_wk_row_split: LookupData = {}
      for key, shard_fids in shards.items():
        sub_table_name, ps_idx = key.split(':')
        ps_idx = int(ps_idx)
        if ps_idx in lookup_data_on_wk:
          lookup_data_on_wk[ps_idx][sub_table_name] = shard_fids
          lookup_data_on_wk_row_split[ps_idx][
              sub_table_name] = shards_row_split[key]
        else:
          lookup_data_on_wk[ps_idx] = {sub_table_name: shard_fids}
          lookup_data_on_wk_row_split[ps_idx] = {
              sub_table_name: shards_row_split[key]
          }

      # ps_idx_to_multi_type_resp: Dict[int, Dict[str, tf.Tensor]] = {}
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
          fids_tensor = lookup_data_on_wk[ps_idx][sub_table_name]
          fids_tensor_row_split = lookup_data_on_wk_row_split[ps_idx][
              sub_table_name]
          embeddings_tensor = ps_idx_to_multi_type_resp[ps_idx][sub_table_name]
          self.fids_list.append(fids_tensor)
          self.fids_list_row_split.append(fids_tensor_row_split)
          self.embeddings_list.append(embeddings_tensor)
          if auxiliary_bundle:
            auxiliary_bundle[f'{sub_table_name}:{ps_idx}:fids'] = fids_tensor
            auxiliary_bundle[
                f'{sub_table_name}:{ps_idx}:fids_row_split'] = fids_tensor_row_split
            auxiliary_bundle[
                f'{sub_table_name}:{ps_idx}:embs'] = embeddings_tensor

      logging.info("before fused_embedding_to_layout")
      layout_tensors = distribution_ops.fused_embedding_to_layout(
          self.embeddings_list,
          self.fids_list_row_split,
          fid_offset=fid_offset,
          feature_offset=feature_offset,
          nfl_offset=nfl_offset,
          batch_size=batch_size,
          variant_type=self._inner_data_type,
          feature_cfgs=self._feature_configs,
          ps_num=self._num_ps)
      logging.info("fused_embedding_to_layout done!")
      return self.nest_layout(layout_tensors)

  def apply_gradients(
      self,
      layout_grads_and_vars: List[Tuple[tf.Tensor, tf.Tensor]],
      global_step: tf.Tensor,
      req_time: Optional[tf.Tensor] = None,
      auxiliary_bundle: Dict[str, tf.Tensor] = None) -> PartitionedHashTable:
    with tf.name_scope("pht_apply_gradients"):
      layout_grad, layout = zip(*layout_grads_and_vars)
      flattened_fids, flattened_fids_row_split, flattened_embs = [], [], []
      assert auxiliary_bundle is not None
      for sub_table_name in self._sub_table_names:
        for ps_idx in range(self._num_ps):
          flattened_fids.append(
              auxiliary_bundle[f'{sub_table_name}:{ps_idx}:fids'])
          flattened_fids_row_split.append(
              auxiliary_bundle[f'{sub_table_name}:{ps_idx}:fids_row_split'])
          flattened_embs.append(
              auxiliary_bundle[f'{sub_table_name}:{ps_idx}:embs'])
      nfl_offset = auxiliary_bundle['nfl_offset']
      feature_offset = auxiliary_bundle['feature_offset']
      fid_offset = auxiliary_bundle['fid_offset']
      batch_size = auxiliary_bundle['batch_size']
      embeddings_grad = distribution_ops.fused_embedding_to_layout_grad(
          nfl_offset=nfl_offset,
          feature_offset=feature_offset,
          fid_offset=fid_offset,
          batch_size=batch_size,
          embeddings_list=flattened_embs,
          fid_list_row_split=flattened_fids_row_split,
          layout_tensors_grad=layout_grad,
          variant_type=self._inner_data_type,
          feature_cfgs=self._feature_configs,
          ps_num=self._num_ps)
      fids_and_embgrad_pairs = list(zip(flattened_fids, embeddings_grad))
      logging.info(
          f"fused_embedding_to_layout_grad done, {fids_and_embgrad_pairs}")
      if req_time is None:
        req_time = tf.constant(0, dtype=tf.int64)
      else:
        req_time = tf.reduce_max(req_time)

      if self._use_native_multi_hash_table:
        new_tables = []
        for i in range(self._num_ps):
          splitted_id = tf.RaggedTensor.from_row_splits(
              flattened_fids[i], flattened_fids_row_split[i], validate=False)
          splitted_flat_grad = embeddings_grad[i]
          table: multi_hash_table_ops.RawMultiTypeHashTable = self._tables[i]
          with nullcontext() if self._local_ps else ps_device(i):
            new_tables.append(
                table.raw_apply_gradients(splitted_id,
                                          splitted_flat_grad,
                                          global_step=global_step,
                                          req_time=req_time))
      else:
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
                    packed_fids_on_ps, packed_emb_grad_on_ps, packed_sizes_on_ps
                ])
            partitioned_update_data_on_ps = {
                tbname: (keyed_fids_on_ps[tbname], keyed_grads_on_ps[tbname])
                for tbname in keyed_fids_on_ps
            }
            new_tables.append(self._tables[i].apply_gradients(
                partitioned_update_data_on_ps, global_step, req_time=req_time))
      return self._copy_with_new_table(new_tables)

  def as_op(self, name=None):
    name = name or "pht_as_op"
    ops = []
    for i in range(self._num_ps):
      with nullcontext() if self._local_ps else ps_device(i):
        ops.append(self._tables[i].as_op(name="{}/sub_{}".format(name, i)))
    with tf.control_dependencies(ops):
      return tf.no_op(name=("{}/done".format(name)))

  def _copy_with_new_table(self,
                           new_tables: List[tf.Tensor]) -> PartitionedHashTable:
    copied = copy.copy(self)
    copied._tables = new_tables
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
    with tf.name_scope(name_scope):
      new_tables = []
      for i in range(self._num_ps):
        new_tables.append(getattr(self._tables[i], method_name)(update_data[i]))
      return self._copy_with_new_table(new_tables)

  def assign(self, data: AssignData) -> PartitionedHashTable:
    # TODO
    if self._use_native_multi_hash_table:
      return self._update("assign", "dmtht_a", data)
    else:
      return self._update("assign", "pht_assign", data)

  def assign_add(self, data: AssignData) -> PartitionedHashTable:
    # TODO
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
