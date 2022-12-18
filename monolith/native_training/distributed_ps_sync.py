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

import copy
import collections
import os
from typing import Callable, Dict, Tuple, List

from absl import flags, logging
import tensorflow as tf

from monolith.native_training import multi_type_hash_table
from monolith.native_training import distributed_ps
from monolith.native_training import distribution_ops
from monolith.native_training.prefetch_queue import \
    enqueue_dicts_with_queue_return, AsyncPushHook, EnqueueHook

enable_hvd = os.getenv("MONOLITH_WITH_HOROVOD")
enable_custom_optimized_hvd = os.getenv("MONOLITH_WITH_OPTIMIZED_HOROVOD")
if enable_hvd != None:
  import horovod.tensorflow as hvd
  from horovod.tensorflow.compression import FP16Compressor

enable_hvd_fid_g2g = int(os.getenv("MONOLITH_WITH_HOROVOD_FID_G2G", 1))
enable_hvd_fwd_g2g = int(os.getenv("MONOLITH_WITH_HOROVOD_FWD_G2G", 1))
enable_hvd_bwd_g2g = int(os.getenv("MONOLITH_WITH_HOROVOD_BWD_G2G", 1))
enable_bps = int(os.getenv("MONOLITH_WITH_BYTEPS", "0"))
enable_bps_fid = int(os.getenv("MONOLITH_WITH_BYTEPS_FID", "1"))
enable_bps_fwd = int(os.getenv("MONOLITH_WITH_BYTEPS_FWD", "1"))
enable_bps_bwd = int(os.getenv("MONOLITH_WITH_BYTEPS_BWD", "1"))
# MONOLITH_WITH_BYTEPS_BWD_CAST
# 32: fp32 for embed grad (default)
# 16: fp16 for embed grad
enable_bps_bwd_cast = int(os.getenv("MONOLITH_WITH_BYTEPS_BWD_CAST", "32"))
enable_bps_bwd_fake_cast = int(
    os.getenv("MONOLITH_WITH_BYTEPS_BWD_FAKE_CAST", "0"))
# enable forward alltoall with GDR
enable_bps_fwd_gdr = int(os.getenv("MONOLITH_WITH_BYTEPS_FWD_GDR", "0"))
enable_bps_fwd_gdr_g2g = int(os.getenv("MONOLITH_WITH_BYTEPS_FWD_GDR_G2G", "0"))
# enable backward alltoall with GDR
enable_bps_bwd_gdr = int(os.getenv("MONOLITH_WITH_BYTEPS_BWD_GDR", "0"))
enable_bps_bwd_gdr_g2g = int(os.getenv("MONOLITH_WITH_BYTEPS_BWD_GDR_G2G", "0"))

FLAGS = flags.FLAGS
flags.DEFINE_bool("enable_alltoall_metrics",
                  default=False,
                  help=("Whether to turn on alltoall detailed stats."))
flags.DEFINE_string(
    "enable_alltoall_metrics_for_slot",
    default=None,
    help="ID of the merged slot to summary alltoall stats. For example:"
    "(af17bbdba2be72580bf5c8c43975078c for merged slot of fc_clk_ads_4d)")


def _alltoall_flats(flats: List[tf.Tensor]) -> List[tf.Tensor]:
  """All to all a list of 1-D tensors."""
  flat = tf.concat(flats, 0)
  size = tf.stack([tf.size(flat) for flat in flats])
  if enable_custom_optimized_hvd:
    flat_t, flat_t_size = hvd.alltoall(flat, size, with_size=True)
  else:
    flat_t = hvd.alltoall(flat, size)
    flat_t_size = hvd.alltoall(size)
  flats_t = tf.split(flat_t, flat_t_size, 0, len(flats))
  return flats_t


def _get_keyed_shape_with_unknown_first_dim(
    keyed_tensors: Dict[str, tf.Tensor]) -> Dict[str, List[int]]:
  keyed_shape = distributed_ps._get_keyed_shape(keyed_tensors)
  # In all to all, we don't know other hosts' shape. The assumption here
  # is that the first dim is unknown.
  for k, v in keyed_shape.items():
    v[0] = -1
  return keyed_shape


def _alltoall_sharded_map(
    sharded_maps: List[Dict[str, tf.Tensor]]) -> List[Dict[str, tf.Tensor]]:
  flats = []
  flats_sizes = []
  for i in range(len(sharded_maps)):
    flat, flat_size = distributed_ps._pack_tensors(sharded_maps[i])
    flats.append(flat)
    flats_sizes.append(flat_size)
  flats_t = _alltoall_flats(flats)
  flats_sizes_t = _alltoall_flats(flats_sizes)
  keyed_shape = _get_keyed_shape_with_unknown_first_dim(sharded_maps[0])

  sharded_maps_t = []
  for i in range(len(sharded_maps)):
    sharded_map_t = distributed_ps._unpack_tensors(
        keyed_shape, (flats_t[i], flats_sizes_t[i]))
    sharded_maps_t.append(sharded_map_t)
  return sharded_maps_t


class DistributedMultiTypeHashTableMpi(
    multi_type_hash_table.BaseMultiTypeHashTable):

  def __init__(
      self,
      shard_num: int,
      table_factory: Callable[[int],
                              multi_type_hash_table.BaseMultiTypeHashTable],
      queue_configs: Dict[str, int] = None):

    self._shard_num = shard_num
    if enable_bps:
      import byteps.tensorflow as bps
      assert bps.size() == self._shard_num
      self._index = bps.rank()
    else:
      assert hvd.size() == self._shard_num
      self._index = hvd.rank()
    self._table = table_factory(self._index)
    self._output_dims = self._table.get_table_dim_sizes()
    self._queue_configs = queue_configs or {}
    self._dependency_ops = []

  def lookup(self,
             slot_to_id: Dict[str, tf.Tensor],
             auxiliary_bundle: Dict[str, tf.Tensor | tf.RaggedTensor],
             early_reorder_indicies_res_pack=None) -> Dict[str, tf.Tensor]:
    if enable_bps:
      import byteps.tensorflow as bps
    sorted_slot_keys = sorted(slot_to_id.keys())
    slot_num = len(sorted_slot_keys)
    if early_reorder_indicies_res_pack:
      all_fids, shard_sizes, sharded_slot_sizes, fused_embedding_offsets = early_reorder_indicies_res_pack
      if FLAGS.enable_alltoall_metrics:
        slot_name = FLAGS.enable_alltoall_metrics_for_slot
        if slot_name and slot_name in sorted_slot_keys:
          m = sorted_slot_keys.index(slot_name)
          with tf.device("/CPU:0"):
            tf.compat.v1.summary.scalar(
                "{}_size".format(slot_name),
                tf.reduce_sum(
                    tf.gather(
                        sharded_slot_sizes,
                        [m + i * slot_num for i in range(self._shard_num)])))
        with tf.device("/CPU:0"):
          tf.compat.v1.summary.scalar("all_fids_size", tf.size(all_fids))
          tf.compat.v1.summary.histogram("shard_sizes", shard_sizes)
          tf.compat.v1.summary.histogram("sharded_slot_sizes",
                                         sharded_slot_sizes)
    else:
      sorted_input = [slot_to_id[k] for k in sorted_slot_keys]
      all_fids, shard_sizes, sharded_slot_sizes, fused_embedding_offsets = \
          distribution_ops.fused_reorder_by_indices(
              sorted_input, self._shard_num, self._output_dims
          )

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
                                    recv_splits=([slot_num] * self._shard_num),
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

    auxiliary_bundle["shard_sizes"] = shard_sizes
    auxiliary_bundle["fused_embedding_offsets"] = fused_embedding_offsets
    auxiliary_bundle["id_flat_t"] = id_flat_t
    # Note: id_flat_split_t is not being used in later computation.
    auxiliary_bundle["id_size_flat_t"] = id_size_flat_t

    # fused_embeddings: [E], fused_splits: [N]
    # id_offsets: [K*N], emb_offsets: [K*N]
    with tf.device("/GPU:0"):
      fused_embeddings, embedding_splits, id_offsets, emb_offsets, fused_emb_sizes = \
          self._table.fused_lookup(id_flat_t, id_size_flat_t, self._shard_num)
    if FLAGS.enable_alltoall_metrics:
      with tf.device("/CPU:0"):
        tf.compat.v1.summary.histogram("fused_embedding_splits",
                                       embedding_splits)

    auxiliary_bundle["fused_embeddings"] = fused_embeddings
    auxiliary_bundle["embedding_splits"] = embedding_splits
    auxiliary_bundle["id_offsets"] = id_offsets
    auxiliary_bundle["emb_offsets"] = emb_offsets
    auxiliary_bundle["recv_emb_splits"] = tf.reshape(
        tf.matmul(
            tf.reshape(sharded_slot_sizes, [self._shard_num, slot_num]),
            tf.expand_dims(tf.constant(self._output_dims, dtype=tf.int32),
                           -1)  # [slot_num, 1]
        ),
        [-1]  # flatten
    )

    auxiliary_bundle, queue = enqueue_dicts_with_queue_return(
        auxiliary_bundle,
        capacity=int(self._queue_configs.get("enable_pipelined_fwda2a", 0)),
        queue_name="queue_lookup_to_fusedEmbA2A")
    if queue:
      self.add_queue_hook(EnqueueHook(queue))

    fused_embeddings = auxiliary_bundle.pop("fused_embeddings")
    embedding_splits = auxiliary_bundle["embedding_splits"]
    recv_emb_splits = auxiliary_bundle["recv_emb_splits"]
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
            recv_embeddings = bps.alltoall_cpu2gpu(fused_embeddings,
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

    auxiliary_bundle["recv_embeddings"] = recv_embeddings

    with tf.device("/device:GPU:0"):
      # GPUQueue: Pass to GPU at Enqueue
      auxiliary_bundle["recv_embeddings"] = tf.identity(
          auxiliary_bundle["recv_embeddings"])
      _size = tf.size(auxiliary_bundle["recv_embeddings"])
      with tf.device("/device:CPU:0"):
        # Size output on HostMemory in kernel. Ensure it's CPU device.
        auxiliary_bundle["recv_embeddings_size"] = tf.identity(_size)
    # Keep on CPU
    auxiliary_bundle["fused_embedding_offsets"] = [
        tf.identity(t)
        for t in list(auxiliary_bundle["fused_embedding_offsets"])
    ]

    auxiliary_bundle, queue = enqueue_dicts_with_queue_return(
        auxiliary_bundle,
        capacity=int(self._queue_configs.get("embedding_prefetch_capacity", 0)),
        queue_name="queue_fusedEmbA2A_to_fusedGather")
    if queue:
      self.add_queue_hook(EnqueueHook(queue))

    # auxiliary_bundle includes all dequeued tensors, if a prefetch queue in-between.
    recv_embeddings = auxiliary_bundle.pop("recv_embeddings")
    fused_embedding_offsets = auxiliary_bundle["fused_embedding_offsets"]

    with tf.device("/device:GPU:0"):
      outputs = distribution_ops.fused_gather_embeddings_by_input(
          recv_embeddings, fused_embedding_offsets, self._output_dims)

    # a.k.a merged_slot_to_embedding
    slot_to_embedding = {k: outputs[i] for i, k in enumerate(sorted_slot_keys)}
    return slot_to_embedding, auxiliary_bundle

  # TODO(zouxuan): assign is broken.
  def assign(
      self, slot_to_id_and_value: Dict[str, Tuple[tf.Tensor, tf.Tensor]]
  ) -> DistributedMultiTypeHashTableMpi:
    raise NotImplementedError

  # TODO(zouxuan): assign_add is broken.
  def assign_add(
      self, slot_to_id_and_value: Dict[str, Tuple[tf.Tensor, tf.Tensor]]
  ) -> DistributedMultiTypeHashTableMpi:
    raise NotImplementedError

  # Apply_gradients uses fused update.
  def apply_gradients(
      self,
      slot_to_grad: Dict[str, Tensor],
      auxiliary_bundle: Dict[str, tf.Tensor],
      global_step: tf.Tensor,
      req_time: tf.Tensor = None) -> DistributedMultiTypeHashTableMpi:

    auxiliary_bundle['global_step'] = global_step
    if req_time is None:
      req_time = tf.constant(0, dtype=tf.int64)
    auxiliary_bundle["req_time"] = req_time

    sorted_slot_keys = sorted(slot_to_grad.keys())
    sorted_grads = [slot_to_grad[k] for k in sorted_slot_keys]

    recv_embeddings_size = auxiliary_bundle.pop("recv_embeddings_size")
    fused_embedding_offsets = auxiliary_bundle.pop("fused_embedding_offsets")
    with tf.device("/device:GPU:0"):
      grad_flat = distribution_ops.fused_gather_embeddings_by_input_gradient(
          recv_embeddings_size, sorted_grads, fused_embedding_offsets,
          self._output_dims)

    with tf.device("/device:GPU:0"):
      if enable_bps_bwd_cast == 16:
        auxiliary_bundle['grad_flat'] = tf.cast(grad_flat, tf.float16)
      else:
        auxiliary_bundle['grad_flat'] = tf.identity(grad_flat)

    # Here we add a queue to let the optimize stage non-blocking and
    # interleaving at the next round of update.
    auxiliary_bundle, async_optimize_queue = enqueue_dicts_with_queue_return(
        auxiliary_bundle,
        capacity=int(self._queue_configs.get("enable_async_optimize", 0)),
        queue_name="queue_fusedGatherGrad_to_fusedEmbGradA2A")
    grad_flat = auxiliary_bundle.pop("grad_flat")
    recv_emb_splits = auxiliary_bundle.pop("recv_emb_splits")
    embedding_splits = auxiliary_bundle.pop("embedding_splits")

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
        shard_sizes = auxiliary_bundle["shard_sizes"]
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

    auxiliary_bundle["grad_flat_t"] = grad_flat_t
    auxiliary_bundle.pop("shard_sizes")
    auxiliary_bundle, q = enqueue_dicts_with_queue_return(
        auxiliary_bundle,
        capacity=int(self._queue_configs.get("enable_pipelined_bwda2a", 0)),
        queue_name="queue_fusedEmbGradA2A_to_sparseOptimize")
    if q:
      self.add_queue_hook(EnqueueHook(q))

    with tf.device("/GPU:0"):
      updated_table = self._table.fused_apply_gradient(
          auxiliary_bundle.pop("id_flat_t"),
          auxiliary_bundle.pop("id_size_flat_t"),
          auxiliary_bundle.pop("grad_flat_t"),
          auxiliary_bundle.pop("id_offsets"),
          auxiliary_bundle.pop("emb_offsets"),
          auxiliary_bundle.pop("global_step"), auxiliary_bundle.pop("req_time"),
          self._shard_num)

    update_op = self._copy_with_new_table(updated_table)
    # TODO(zouxuan): add better tests to test the async optimize.
    if async_optimize_queue:
      self.add_queue_hook(AsyncPushHook(async_optimize_queue,
                                        update_op.as_op()))
      self._dependency_ops.append(async_optimize_queue.enqueue_op)
      # return self essentially means to call dependency_ops
      return self
    else:
      return update_op

  def _update(self, method: str,
              slot_to_id_and_value: Dict[str, Tuple[tf.Tensor, tf.Tensor]],
              *args, **kwargs):
    raise NotImplementedError

  def as_op(self, name=None):
    with tf.control_dependencies(self._dependency_ops):
      return self._table.as_op(name)

  def get_table_dim_sizes(self):
    return self._tables[0].get_table_dim_sizes()

  def _copy_with_new_table(self,
                           table: multi_type_hash_table.BaseMultiTypeHashTable):
    copied = copy.copy(self)
    copied._dependency_ops = copy.copy(self._dependency_ops)
    copied._table = table
    return copied
