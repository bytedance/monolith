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

import abc
from collections import defaultdict
import copy
import dataclasses
import hashlib
import itertools
import re
from typing import Callable, Dict, Iterable, List, Tuple

from absl import logging
import tensorflow as tf

from monolith.native_training import device_utils
from monolith.native_training import distribution_ops
from monolith.native_training import entry
from monolith.native_training import hash_table_ops
from monolith.native_training import utils
from monolith.native_training import prefetch_queue
from monolith.native_training.hash_table_utils import infer_dim_size


class BaseMultiTypeHashTable(abc.ABC):

  # https://www.python.org/dev/peps/pep-0526/#class-and-instance-variable-annotations
  # Allow nested instances.
  _table: BaseMultiTypeHashTable
  _tables: List[BaseMultiTypeHashTable]
  # Allow pipelined graph execution.
  _local_queue_hooks: List[prefetch_queue.EnqueueHook |
                           prefetch_queue.AsyncPushHook]

  @abc.abstractmethod
  def lookup(self, slot_to_id: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    pass

  @abc.abstractmethod
  def assign(
      self, slot_to_id_and_value: Dict[str, Tuple[tf.Tensor, tf.Tensor]]
  ) -> BaseMultiTypeHashTable:
    pass

  @abc.abstractmethod
  def assign_add(
      self, slot_to_id_and_value: Dict[str, Tuple[tf.Tensor, tf.Tensor]]
  ) -> BaseMultiTypeHashTable:
    pass

  @abc.abstractmethod
  def apply_gradients(self, slot_to_id_and_grad: Dict[str, Tuple[tf.Tensor,
                                                                 tf.Tensor]],
                      *args) -> BaseMultiTypeHashTable:
    pass

  @abc.abstractmethod
  def as_op(self, name=None) -> tf.Operation:
    pass

  def add_queue_hook(self, hook):
    # Allow pipelined graph execution.
    if not getattr(self, "_local_queue_hooks", None):
      self._local_queue_hooks = []
    self._local_queue_hooks.append(hook)

  def get_queue_hooks(self):
    hooks = copy.copy(getattr(self, "_local_queue_hooks", []))
    if getattr(self, "_table", None):
      hooks.extend(self._table.get_queue_hooks())
    if getattr(self, "_tables", None):
      hooks.extend(
          itertools.chain.from_iterable(
              [t.get_queue_hooks() for t in self._tables]))
    return hooks

  @abc.abstractmethod
  def get_table_dim_sizes(self) -> List[int]:
    pass


# TODO(leqi.zou): Makes this have a better name.
class MultiTypeHashTable(BaseMultiTypeHashTable):
  """
  A hash tables that support different types of embeddings (they may have different dimensions/ optimizers). 
  Different types are distinguished by "Slot". Slot is the type of ids, the embeddings in the same
  slot has the same dimension.

  The functionality are same as BaseHashTable. The only difference is that now the input is a map, which
  maps slot to ids/values.

  hash_table_factory has two params: name_suffix & hash_table_config
  """

  def __init__(
      self, slot_to_config: Dict[str, entry.HashTableConfigInstance],
      hash_table_factory: Callable[[str, entry.HashTableConfigInstance],
                                   hash_table_ops.BaseHashTable]):
    self._slot_to_config = slot_to_config
    self._hash_tables = {}
    self._hash_table_resources = []
    learning_rate_tensors = []
    learning_rate_lengths = []
    for slot in sorted(self._slot_to_config.keys()):
      # We need to keep the order here.
      config = self._slot_to_config[slot]
      self._hash_tables[slot] = hash_table_factory(slot, config)
      # Here we setup the hashtable resources based on
      # self._slot_to_config.keys()
      self._hash_table_resources.append(self._hash_tables[slot].as_op())
      learning_rate_tensors.append(config.call_learning_rate_fns())
      learning_rate_lengths.append(tf.size(learning_rate_tensors[-1]))

    # Build flattened learning rate tensor for fused apply gradient.
    self._learning_rate_tensors = tf.concat(learning_rate_tensors, 0)
    self._learning_rate_lengths = tf.stack(learning_rate_lengths)

  def lookup(self, slot_to_id: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    slot_to_embedding = {}
    for slot, id in slot_to_id.items():
      embedding = self._hash_tables[slot].lookup(id)
      slot_to_embedding[slot] = embedding
    return slot_to_embedding

  def assign(
      self, slot_to_id_and_value: Dict[str, Tuple[tf.Tensor, tf.Tensor]]
  ) -> MultiTypeHashTable:
    return self._update("assign", slot_to_id_and_value)

  def assign_add(
      self, slot_to_id_and_value: Dict[str, Tuple[tf.Tensor, tf.Tensor]]
  ) -> MultiTypeHashTable:
    return self._update("assign_add", slot_to_id_and_value)

  def apply_gradients(
      self,
      slot_to_id_and_grad: Dict[str, Tuple[tf.Tensor, tf.Tensor]],
      *args,
      **kwargs,
  ) -> MultiTypeHashTable:
    return self._update("apply_gradients", slot_to_id_and_grad, *args, **kwargs)

  def _update(
      self,
      method_name: str,
      slot_to_id_and_tensor: Dict[str, Tuple[tf.Tensor, tf.Tensor]],
      *args,
      **kwargs,
  ) -> MultiTypeHashTable:
    updated_tables = dict(self._hash_tables)
    for slot, (id, tensor) in slot_to_id_and_tensor.items():
      updated_tables[slot] = getattr(self._hash_tables[slot],
                                     method_name)(id, tensor, *args, **kwargs)
    return self._copy_with_new_tables(updated_tables)

  def as_op(self, name=None) -> tf.Operation:
    name = name or "mtht_ao"
    with tf.control_dependencies(
        [table.as_op() for table in self._hash_tables.values()]):
      c = tf.no_op(name=("{}/done".format(name)))
    return c

  def _copy_with_new_tables(
      self, tables: Dict[int, tf.Tensor]) -> "MultiTypeHashTable":
    copied = copy.copy(self)
    # Update the hash_table_resources everytime when there is a table update.
    hash_table_resources = []
    for slot in sorted(self._slot_to_config.keys()):
      hash_table_resources.append(tables[slot].as_op())
    copied.__dict__["_hash_tables"] = tables
    copied.__dict__["_hash_table_resources"] = hash_table_resources
    return copied

  # This is a very concise API that supports fused lookup, without mapping the
  # IDs to its slots.
  def fused_lookup(self, ids: tf.Tensor, fused_slot_size: tf.Tensor,
                   num_of_shards: int) -> Tuple[tf.Tensor]:
    return hash_table_ops.fused_lookup(self._hash_table_resources, ids,
                                       fused_slot_size, num_of_shards)

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
      enable_grad_accumulation: bool = False) -> MultiTypeHashTable:
    table_handles_output = hash_table_ops.fused_apply_gradient(
        self._hash_table_resources, ids, fused_slot_size, id_grads, id_offsets,
        grad_offsets, self._learning_rate_tensors, self._learning_rate_lengths,
        req_time, global_step, num_of_shards, enable_grad_accumulation)
    copied = copy.copy(self)
    updated_tables = dict(self._hash_tables)
    for slot, handle in zip(sorted(self._slot_to_config.keys()),
                            table_handles_output):
      updated_tables[slot] = self._hash_tables[slot].table_update(handle)
    copied.__dict__["_hash_tables"] = updated_tables
    copied.__dict__["_hash_table_resources"] = table_handles_output
    return copied

  def get_table_dim_sizes(self) -> List[int]:
    return [
        self._hash_tables[slot].dim_size
        for slot in sorted(self._slot_to_config.keys())
    ]


@dataclasses.dataclass
class _IndexedValues:
  """
  _IndexedValues represents tensors merged from multiple slots.
  slots are a list of string represents where values are coming from
  indices are a tensor of 1-D int64 ranged in [0, len(slots)) represents the slot of that value is slots[index]
  values are a tensor represents a list tensors which merged from multiple slots.
  """
  slots: List[tf.Tensor]
  index: tf.Tensor
  value: tf.Tensor


class MergedMultiTypeHashTable(BaseMultiTypeHashTable):
  """A decorator that merge slots which have the same embedding config.
  This helps reduce the size of graph. However, the caller need to make sure 
  ids in different slots are different."""

  def __init__(self, slot_to_config: Dict[str, entry.HashTableConfigInstance],
               factory: Callable[[Dict[str, entry.HashTableConfigInstance]],
                                 BaseMultiTypeHashTable]):
    self._slot_to_config = slot_to_config
    logging.info(
        "Create MergedMultiTypeHashTable: 1) reverse feature_name -> config into config -> feature_name_list"
    )
    self._slot_mapping: Dict[str, str] = {}  # feature/slot -> merged_slot
    deduped_config_to_slots = defaultdict(list)
    for slot in sorted(slot_to_config):
      config = slot_to_config[slot]
      # Use str of config as the key for merging slots.
      deduped_config_to_slots[str(config)].append(slot)

    logging.info(
        "Create MergedMultiTypeHashTable: 2) gen merged_slot and map merged_slot -> config"
    )
    merged_slot_to_config = {}
    for key, slots in deduped_config_to_slots.items():

      def get_merged_str(strs: List[str]):
        concat = ",".join(strs)
        return concat, hashlib.md5(concat.encode()).hexdigest()

      slots_str, merged_slot = get_merged_str(slots)
      logging.info("Merged '{}' into '{}'".format(slots_str, merged_slot))
      merged_config = copy.copy(slot_to_config[slots[0]])
      # replace "fc_slot_*" to "slot_*"
      old_slots = [
          slot[3:] if re.match("^fc_slot_[0-9]*$", slot) else slot
          for slot in slots
      ]
      _, old_merged_slot = get_merged_str(old_slots)
      if old_merged_slot != merged_slot:
        merged_config.extra_restore_names.append(old_merged_slot)

      merged_slot_to_config[merged_slot] = merged_config
      for slot in slots:
        self._slot_mapping[slot] = merged_slot

    self._merged_slot_to_config = merged_slot_to_config

    logging.info(
        f"Create MergedMultiTypeHashTable: 3) sub hash tables {factory}")
    self._table = factory(merged_slot_to_config)

  @property
  def slot_mapping(self):
    """Returns slot mapping."""
    return self._slot_mapping

  def lookup(self,
             slot_to_id: Dict[str, tf.Tensor],
             auxiliary_bundle=None,
             early_reorder_indicies_res_pack=None):
    if auxiliary_bundle is None:
      auxiliary_bundle = {}

    if early_reorder_indicies_res_pack:
      merged_slot_to_sizes, res_pack = early_reorder_indicies_res_pack
      merged_slot_to_id = {
          k: None for k in merged_slot_to_sizes.keys()
          # None to keep interface, we will only use the keys
      }
      auxiliary_bundle['merged_slot_to_sizes'] = merged_slot_to_sizes
      merged_slot_to_embedding, auxiliary_bundle = self._table.lookup(
          merged_slot_to_id, auxiliary_bundle, res_pack)
    else:
      logging.info(
          "Lookup MergedMultiTypeHashTable: 1) merged the features ids belong the same hash table"
      )
      merged_slot_to_id, merged_slot_to_sizes = self._get_merged_to_indexed_tensor(
          slot_to_id)
      auxiliary_bundle['merged_slot_to_sizes'] = merged_slot_to_sizes
      logging.info(
          f"Lookup MergedMultiTypeHashTable: 2) lookup sub hash table {self._table} for embeddings"
      )
      merged_slot_to_embedding = self._table.lookup(merged_slot_to_id)

    merged_slot_to_slots = defaultdict(list)
    for k in sorted(self._slot_mapping.keys()):
      if k in slot_to_id:
        merged_slot_to_slots[self._slot_mapping[k]].append(k)

    merged_slot_to_sizes = auxiliary_bundle.pop('merged_slot_to_sizes')
    logging.info(
        "Lookup MergedMultiTypeHashTable: 3) split the lookuped embeddings into feature_name -> embedding"
    )
    slot_to_embedding = {}
    for merged_slot, emb in merged_slot_to_embedding.items():
      sizes = merged_slot_to_sizes[merged_slot]
      slots = merged_slot_to_slots[merged_slot]
      with device_utils.maybe_device_if_allowed("/device:GPU:0"):
        embs = tf.split(emb, sizes, axis=0, num=len(slots))
      for slot, emb in zip(slots, embs):
        slot_to_embedding[slot] = emb

    if auxiliary_bundle:
      return slot_to_embedding, auxiliary_bundle
    return slot_to_embedding

  def assign(
      self, slot_to_id_and_value: Dict[str, Tuple[tf.Tensor, tf.Tensor]]
  ) -> MergedMultiTypeHashTable:
    return self._update(self._table.assign, slot_to_id_and_value)

  def assign_add(
      self, slot_to_id_and_value: Dict[str, Tuple[tf.Tensor, tf.Tensor]]
  ) -> MergedMultiTypeHashTable:
    return self._update(self._table.assign_add, slot_to_id_and_value)

  def apply_gradients(self, slot_to_id_and_grad: Dict[str, Tuple[tf.Tensor,
                                                                 tf.Tensor]],
                      *args, **kwargs) -> MergedMultiTypeHashTable:
    return self._update(self._table.apply_gradients, slot_to_id_and_grad, *args,
                        **kwargs)

  def _update(self, method, slot_to_id_and_tensor: Dict[str, Tuple[tf.Tensor,
                                                                   tf.Tensor]],
              *args, **kwargs):
    if kwargs.pop("skip_merge_id", False):
      # To avoid redundant cpu usage, in MergedMultiTypeHashTable
      # sync training only passes the slot_to_grad for apply_gradient
      slot_to_grad = slot_to_id_and_tensor
      with device_utils.maybe_device_if_allowed("/device:GPU:0"):
        merged_slot_to_grad, _ = self._get_merged_to_indexed_tensor(
            slot_to_grad)
      return self._copy_with_new_table(
          method(merged_slot_to_grad, *args, **kwargs))

    slot_to_id = {k: v[0] for k, v in slot_to_id_and_tensor.items()}
    merged_slot_to_id, _ = self._get_merged_to_indexed_tensor(slot_to_id)
    slot_to_tensor = {k: v[1] for k, v in slot_to_id_and_tensor.items()}
    with device_utils.maybe_device_if_allowed("/device:GPU:0"):
      merged_slot_to_tensor, _ = self._get_merged_to_indexed_tensor(
          slot_to_tensor)
    merged_slot_to_id_and_tensor = {}
    for slot in merged_slot_to_id:
      merged_slot_to_id_and_tensor[slot] = (merged_slot_to_id[slot],
                                            merged_slot_to_tensor[slot])
    return self._copy_with_new_table(
        method(merged_slot_to_id_and_tensor, *args, **kwargs))

  def as_op(self, name=None) -> tf.Operation:
    return self._table.as_op(name)

  def _get_merged_to_indexed_tensor(
      self, slot_to_tensor: Dict[str, tf.Tensor]
  ) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
    merged_slot_to_tensors = defaultdict(list)
    merged_slot_to_sizes = defaultdict(list)
    for slot in sorted(slot_to_tensor.keys()):
      # We sorted the merged slot keys here to guarantee, the merging order.
      tensor = slot_to_tensor[slot]
      merged_slot = self._slot_mapping[slot]
      merged_slot_to_sizes[merged_slot].append(tf.shape(tensor)[0])
      merged_slot_to_tensors[merged_slot].append(tensor)

    return {k: tf.concat(v, axis=0) for k, v in merged_slot_to_tensors.items()}, \
           {k: tf.stack(v) for k, v in merged_slot_to_sizes.items()}

  def _copy_with_new_table(
      self, table: BaseMultiTypeHashTable) -> MergedMultiTypeHashTable:
    copied = copy.copy(self)
    copied._table = table
    return copied

  def get_table_dim_sizes(self) -> List[int]:
    return [
        infer_dim_size(self._merged_slot_to_config[slot].table_config)
        for slot in sorted(self._merged_slot_to_config.keys())
    ]
