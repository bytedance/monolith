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

"""Several methods to create hash tables."""
from typing import Dict, Iterable, List, Tuple

import tensorflow as tf

from idl.matrix.proto.example_pb2 import OutConfig
from monolith.native_training import distributed_ps
from monolith.native_training import distributed_ps_sync
from monolith.native_training import entry
from monolith.native_training import hash_filter_ops
from monolith.native_training import hash_table_ops
from monolith.native_training import multi_type_hash_table
from monolith.native_training import multi_hash_table_ops
import monolith.native_training.embedding_combiners as embedding_combiners


class MultiHashTableFactory:

  def __init__(self, hash_filters, sync_clients):
    self._cc_dict = {}
    self.hash_filters = hash_filters
    self.sync_clients = sync_clients

  def __call__(self, idx: int, slot_to_config):
    k = id(slot_to_config)
    cc = self._cc_dict.get(k, None)
    if cc is None:
      cc = multi_hash_table_ops.convert_to_cached_config(slot_to_config)
      self._cc_dict[k] = cc
    return multi_hash_table_ops.MultiHashTable.from_cached_config(
        cc=cc,
        hash_filter=self.hash_filters[idx],
        sync_client=self.sync_clients[idx],
        name_suffix=str(idx))


def create_in_worker_multi_type_hash_table(
    shard_num: int,
    slot_to_config: Dict[str, entry.HashTableConfigInstance],
    hash_filter: tf.Tensor,
    sync_client: tf.Tensor = None,
    queue_configs: Dict[str, int] = None,
) -> multi_type_hash_table.BaseMultiTypeHashTable:
  """
  Creates a in worker multi-type hash table factory.
  Args:
  shard_num: the number of shards for distributing hash tables.
  """

  # The logic here is
  # merged_slots -> distributed_fused_multitype_table -> alltoall -> hash_table
  def distributed_multi_type_table_factory(merged_slot_to_config):

    def multi_type_table_factory(idx):

      def table_factory(name_suffix, config):
        return hash_table_ops.hash_table_from_config(
            config=config,
            hash_filter=hash_filter,
            name_suffix="_".join([name_suffix, str(idx)]),
            sync_client=sync_client)

      return multi_type_hash_table.MultiTypeHashTable(merged_slot_to_config,
                                                      table_factory)

    return distributed_ps_sync.DistributedMultiTypeHashTableMpi(
        shard_num, multi_type_table_factory, queue_configs)

  return multi_type_hash_table.MergedMultiTypeHashTable(
      slot_to_config, distributed_multi_type_table_factory)


def create_multi_type_hash_table(
    num_ps: int,
    slot_to_config: Dict[str, entry.HashTableConfigInstance],
    hash_filters: List[tf.Tensor],
    sync_clients: List[tf.Tensor] = None,
    reduce_network_packets: bool = False,
    max_rpc_deadline_millis: int = 30,
) -> multi_type_hash_table.BaseMultiTypeHashTable:
  """Create a distributed multi type hash table.
  Args:
  reduce_network_packets - if True, it will compact all tensors locally so ps will get less load.
  Useful when there are a lot of workers.
  """
  if num_ps and sync_clients:
    assert num_ps == len(
        sync_clients
    ), "Number of PS should be equal to number of sync clients, while got {} vs {}".format(
        num_ps, len(sync_clients))
  if not sync_clients:
    sync_clients = [None] * max(num_ps, 1)

  if num_ps == 0:

    def factory(name_suffix, config):
      return hash_table_ops.hash_table_from_config(config,
                                                   hash_filter=hash_filters[0],
                                                   name_suffix=name_suffix,
                                                   sync_client=sync_clients[0])

    def multi_type_factory(merged_slot_to_config):
      return multi_type_hash_table.MultiTypeHashTable(merged_slot_to_config,
                                                      factory)

    return multi_type_hash_table.MergedMultiTypeHashTable(
        slot_to_config, multi_type_factory)
  elif not reduce_network_packets:
    # The logic here is
    # dedup_slots -> multi hash table -> distributed_hash_table -> hash_table
    # |                          worker                         |     ps     |
    def multi_type_factory(merged_slot_to_config):

      def distributed_factory(name_suffix, config):

        def factory(idx, config_on_ps):
          return hash_table_ops.hash_table_from_config(
              config_on_ps,
              hash_filter=hash_filters[idx],
              name_suffix="_".join([name_suffix, str(idx)]),
              sync_client=sync_clients[idx])

        return distributed_ps.DistributedHashTable(num_ps, config, factory)

      return multi_type_hash_table.MultiTypeHashTable(merged_slot_to_config,
                                                      distributed_factory)

    return multi_type_hash_table.MergedMultiTypeHashTable(
        slot_to_config, multi_type_factory)
  else:
    # The logic here is
    # dedup_slots -> distributed multi hash table -> multi hash table -> hash table
    # |                worker                     |              ps                |
    def distributed_multi_type_factory(merged_slot_to_config):

      def multi_type_factory(idx: int, slot_to_config_on_ps):

        def factory(name_suffix, config):
          return hash_table_ops.hash_table_from_config(
              config,
              hash_filter=hash_filters[idx],
              name_suffix="_".join([name_suffix, str(idx)]),
              sync_client=sync_clients[idx])

        return multi_type_hash_table.MultiTypeHashTable(slot_to_config_on_ps,
                                                        factory)

      return distributed_ps.DistributedMultiTypeHashTable(
          num_ps,
          merged_slot_to_config,
          multi_type_factory,
          max_rpc_deadline_millis=max_rpc_deadline_millis)

    return multi_type_hash_table.MergedMultiTypeHashTable(
        slot_to_config, distributed_multi_type_factory)


def create_native_multi_hash_table(
    num_ps: int,
    slot_to_config: Dict[str, entry.HashTableConfigInstance],
    hash_filters: List[tf.Tensor],
    sync_clients: List[tf.Tensor] = None,
    max_rpc_deadline_millis: int = 30,
) -> multi_type_hash_table.BaseMultiTypeHashTable:
  """Create a distributed native multi hash table."""
  if num_ps and sync_clients:
    assert num_ps == len(
        sync_clients
    ), "Number of PS should be equal to number of sync clients, while got {} vs {}".format(
        num_ps, len(sync_clients))
  if not sync_clients:
    sync_clients = [None] * max(num_ps, 1)

  if num_ps == 0:
    return multi_hash_table_ops.MultiHashTable.from_configs(
        configs=slot_to_config,
        hash_filter=hash_filters[0],
        sync_client=sync_clients[0])
  else:
    # The logic here is
    # slots -> distributed multi hash table -> multi hash table
    # |              worker                 |       ps        |
    return distributed_ps.DistributedMultiTypeHashTable(
        num_ps,
        slot_to_config,
        MultiHashTableFactory(hash_filters, sync_clients),
        max_rpc_deadline_millis=max_rpc_deadline_millis)


def create_partitioned_hash_table(
    num_ps: int,
    feature_name_to_config: Dict[str, entry.HashTableConfigInstance],
    layout_configs: Dict[str, OutConfig],
    feature_to_combiner: Dict[str, embedding_combiners.Combiner],
    feature_to_unmerged_slice_dims: Dict[str, List[int]],
    use_native_multi_hash_table: bool,
    max_rpc_deadline_millis: int = 30,
    unique: bool = False,
    transfer_float16: bool = False,
    hash_filters: List[tf.Tensor] = None,
    sync_clients: List[tf.Tensor] = None
) -> distributed_ps.PartitionedHashTable:
  num_ps_tmp = num_ps if num_ps > 0 else 1
  if hash_filters is None:
    hash_filters = [None] * num_ps_tmp
  if sync_clients is None:
    sync_clients = [None] * num_ps_tmp

  if use_native_multi_hash_table:
    multi_type_factory = MultiHashTableFactory(hash_filters, sync_clients)
  else:

    def multi_type_factory(idx: int, slot_to_config_on_ps):

      def factory(name_suffix, config):
        name_suffix = name_suffix if num_ps == 0 else "_".join(
            [name_suffix, str(idx)])
        return hash_table_ops.hash_table_from_config(
            config,
            hash_filter=hash_filters[idx],
            name_suffix=name_suffix,
            sync_client=sync_clients[idx])

      return multi_type_hash_table.MultiTypeHashTable(slot_to_config_on_ps,
                                                      factory)

  return distributed_ps.PartitionedHashTable(
      num_ps, feature_name_to_config, multi_type_factory, layout_configs,
      feature_to_unmerged_slice_dims, feature_to_combiner,
      use_native_multi_hash_table, max_rpc_deadline_millis, unique,
      transfer_float16)
