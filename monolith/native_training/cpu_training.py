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

"""This module defines how to run a native task in CPU training environment.

CpuTraining defines those conversion.
"""
import contextlib
import copy
import dataclasses
import getpass
import json
import os
import platform
import socket
import threading
import timeit

import sys
import traceback
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Set, Tuple
from urllib.parse import urlparse

import time
from absl import logging
from absl import flags
import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow.python.training.summary_io import SummaryWriterCache

from monolith.agent_service.agent_service_pb2 import ServerType
from monolith.agent_service.backends import SyncBackend
from monolith.core.hyperparams import InstantiableParams
from monolith.native_training import barrier_ops
from monolith.native_training import basic_restore_hook
from monolith.native_training import cluster_manager
from monolith.native_training import device_utils
from monolith.native_training import distributed_ps_factory
from monolith.native_training import distribution_ops
from monolith.native_training import embedding_combiners
from monolith.native_training import entry
from monolith.native_training import feature
from monolith.native_training import hash_filter_ops
from monolith.native_training import hash_table_ops
from monolith.native_training import multi_hash_table_ops
from monolith.native_training import logging_ops
from monolith.native_training import monolith_checkpoint_state_pb2
from monolith.native_training import multi_type_hash_table
from monolith.native_training import native_task
from monolith.native_training import native_task_context
from monolith.native_training import net_utils
from monolith.native_training import ps_benchmark
from monolith.native_training import save_utils
from monolith.native_training import session_run_hooks
from monolith.native_training import sync_hooks
from monolith.native_training import sync_training_hooks
from monolith.native_training import tensor_utils
from monolith.native_training import utils
from monolith.native_training import variables
from monolith.native_training import distributed_serving_ops
from monolith.native_training import yarn_runtime
from monolith.native_training.alert import alert_manager
from monolith.native_training.hash_table_utils import infer_dim_size
from monolith.native_training.distributed_serving_ops import ParameterSyncClient
from monolith.native_training.hash_filter_ops import FilterType
from monolith.native_training.hooks import controller_hooks
from monolith.native_training.hooks import ckpt_hooks
from monolith.native_training.hooks import ckpt_info
from monolith.native_training.hooks import ps_check_hooks
from monolith.native_training.hooks import hook_utils
from monolith.native_training.hooks import session_hooks
from monolith.native_training.hooks import feature_engineering_hooks
from monolith.native_training.metric import cli
from monolith.native_training.metric.metric_hook import Tf2ProfilerHook, Tf2ProfilerCaptureOnceHook, NVProfilerCaptureOnceHook
from monolith.native_training.metric.metric_hook import ByteCCLTelemetryHook
from monolith.native_training.metric.metric_hook import ThroughputMetricHook
from monolith.native_training.model_export import export_hooks
from monolith.native_training.model_export import export_utils
from monolith.native_training.model_export import saved_model_exporters
from monolith.native_training.model_export import export_context
from monolith.native_training.model_export.export_context import \
    is_exporting, is_exporting_distributed, ExportMode
from monolith.native_training.native_task import NativeTask
from monolith.native_training.prefetch_queue import \
    enqueue_dicts_with_queue_return, EnqueueHook
from monolith.native_training import prefetch_queue
from monolith.native_training.proto import debugging_info_pb2
from monolith.native_training.runtime.hash_table import \
    embedding_hash_table_pb2
from monolith.native_training.runtime.parameter_sync import \
    parameter_sync_pb2
from monolith.native_training.service_discovery import ServiceDiscovery
from monolith.native_training.service_discovery import TfConfigServiceDiscovery
from monolith.native_training.data.training_instance.python import parser_utils
from monolith.native_training.model_dump.dump_utils import DumpUtils, DRY_RUN
from monolith.native_training.data.parsers import ParserCtx, get_default_parser_ctx
from monolith.native_training.dense_reload_utils import CustomRestoreListenerKey, CustomRestoreListener
from monolith.native_training.data.item_pool_hook import ItemPoolSaveRestoreHook, POOL_KEY

flags.DEFINE_string(
    "monolith_chief_alert_proto", "",
    "The text format of alert proto. Will only be activated by chief.")

FLAGS = flags.FLAGS


def _combine_slices_as_table(
    slices: List[feature.SliceConfig],
    hashtable_config: entry.HashTableConfig) -> entry.HashTableConfigInstance:
  table_config = embedding_hash_table_pb2.EmbeddingHashTableConfig()
  entry_config = table_config.entry_config
  learning_rate_fns = list()

  if is_exporting():
    entry_config.entry_type = embedding_hash_table_pb2.EntryConfig.EntryType.SERVING
  for s in slices:
    entry_config.segments.append(s.segment)
    learning_rate_fns.append(s.learning_rate_fn)
  hashtable_config.mutate_table(table_config)
  return entry.HashTableConfigInstance(table_config, learning_rate_fns)


def _lookup_embedding_ids(
    hash_table: multi_type_hash_table.BaseMultiTypeHashTable,
    name_to_embedding_ids: Dict[str, tf.RaggedTensor]) -> Dict[str, tf.Tensor]:
  name_to_ids = {k: v.values for k, v in name_to_embedding_ids.items()}
  return hash_table.lookup(name_to_ids)


def create_exporter(task,
                    model_dir,
                    warmup_file,
                    export_dir_base,
                    dense_only,
                    include_graphs=None):
  if task._params.serving.export_mode == ExportMode.STANDALONE:
    exporter = saved_model_exporters.StandaloneExporter(
        task.create_model_fn(),
        model_dir=model_dir,
        export_dir_base=export_dir_base,
        shared_embedding=task._params.serving.shared_embedding,
        warmup_file=warmup_file)
  elif task._params.serving.export_mode == ExportMode.DISTRIBUTED:
    exporter = saved_model_exporters.DistributedExporter(
        task.create_model_fn(),
        model_dir=model_dir,
        export_dir_base=export_dir_base,
        shared_embedding=task._params.serving.shared_embedding,
        warmup_file=warmup_file,
        dense_only=dense_only,
        allow_gpu=task._params.serving.export_with_gpu_allowed,
        clear_entry_devices=task._params.serving.
        export_with_cleared_entry_devices,
        include_graphs=include_graphs,
        global_step_as_timestamp=task.config.enable_sync_training,
        with_remote_gpu=task._params.serving.with_remote_gpu,
    )
  else:
    raise ValueError("Invalid export_mode: {}".format(
        task._params.serving.export_mode))

  return exporter


class _CpuFeatureFactory(feature.FeatureFactoryFromEmbeddings):

  def __init__(self,
               hash_table: multi_type_hash_table.BaseMultiTypeHashTable,
               embedding_ids: Dict[str, tf.RaggedTensor],
               embeddings: Dict[str, tf.Tensor],
               embedding_slices: Dict[str, tf.Tensor],
               req_time: tf.Tensor,
               async_function_mgr: prefetch_queue.AsyncFunctionMgr,
               async_push: bool = False):
    super().__init__(embeddings, embedding_slices)
    self._hash_table = hash_table
    self._req_time = req_time
    self._embedding_ids = embedding_ids
    self._embeddings = embeddings
    self._async_function_mgr = async_function_mgr
    self._async_push = async_push

  def _push(self, slot_to_emb_ids, slot_to_emb_grads, global_step, req_time):
    # TODO(leqi.zou): This the hack. Finally we need to figure out a way to
    # resolve this var leaking issue.
    slot_to_ids_and_grads = {
        k: (v, slot_to_emb_grads[k])
        for k, v in slot_to_emb_ids.items()
        if slot_to_emb_grads[k] is not None
    }
    return self._hash_table.apply_gradients(slot_to_ids_and_grads,
                                            global_step,
                                            req_time=req_time).as_op()

  def apply_gradients(self,
                      grads_and_vars: Iterable[Tuple[tf.Tensor, tf.Tensor]],
                      req_time: tf.Tensor = None):
    if req_time is None:
      req_time = self._req_time
    emb_grads = utils.propagate_back_gradients(grads_and_vars,
                                               self._embeddings.values())
    slot_to_emb_ids = {k: v.values for k, v in self._embedding_ids.items()}
    slot_to_emb_grads = dict(zip(self._embeddings.keys(), emb_grads))
    global_step = tf.identity(tf.compat.v1.train.get_or_create_global_step())
    return self._async_function_mgr.add_async_function(
        self._push, (slot_to_emb_ids, slot_to_emb_grads, global_step, req_time),
        is_async=self._async_push,
        queue_name="postpush_queue")


class _FusedCpuFeatureFactory(feature.FeatureFactoryFromEmbeddings):

  def __init__(self, hash_table: multi_type_hash_table.BaseMultiTypeHashTable,
               name_to_embeddings: Dict[str, tf.Tensor],
               name_to_embedding_slices: Dict[str, tf.Tensor],
               req_time: tf.Tensor, auxiliary_bundle: Dict[str,
                                                           tf.RaggedTensor]):
    super().__init__(name_to_embeddings, name_to_embedding_slices)
    self._hash_table = hash_table
    self._embeddings = name_to_embeddings
    self._auxiliary_bundle = auxiliary_bundle
    self._req_time = req_time

  def apply_gradients(self,
                      grads_and_vars: Iterable[Tuple[tf.Tensor, tf.Tensor]],
                      req_time: tf.Tensor = None):
    if req_time is None:
      req_time = self._req_time
    with tf.device("/device:GPU:0"):
      emb_grads = utils.propagate_back_gradients(grads_and_vars,
                                                 self._embeddings.values())
    slot_to_emb_grads = dict(zip(self._embeddings.keys(), emb_grads))
    global_step = tf.identity(tf.compat.v1.train.get_or_create_global_step())
    # Restore back to ID/Embedding mapping and flatten.
    apply_op = self._hash_table.apply_gradients(slot_to_emb_grads,
                                                self._auxiliary_bundle,
                                                global_step,
                                                req_time=req_time,
                                                skip_merge_id=True).as_op()
    return apply_op


@dataclasses.dataclass
class CpuTrainingConfig:
  """The CPU training config.

  attributes:
    :param server_type: The type of this process. Can be 'ps' or 'worker'.
    :param index: The index of the current process in servers.
    :param model_name: The model name. If empty, will be overridden by deepinsight_name.
    :param num_ps: The number of ps.
    :param num_workers: The number of worker.
    :param enable_variable_prefetch: Whether enable variable prefetch.
    :param filter_capacity: Sliding hash filter capacity.
    :param filter_split_num: Number of hash filter.
    :param filter_equal_probability: Probabilistic modeling type.
    :param filter_type: Sliding hash filter or probabilistic filter.
    :param hashtable_init_capacity: hashtable init capacity.
    :param use_native_multi_hash_table: Use native MultiHashTable.
    :param embedding_prefetch_capacity: The queue capacity to prefetch lookuped embeddings.
    :param enable_embedding_postpush: Whether enable embedding post push.
    :param enable_variable_postpush: Whether enable variable post push.
    :param enable_sync_training: Whether use MPI or RPC for the distributed training.
    :param merge_sync_training_ckpt: Whether merge sync ckpt and skip non-worker0 dense variable save.
    :param mode: The running mode, must be train/eval/infer.
    :param partial_recovery: Whether enable partial recovery when failover.
    :param tide_start_hour: tide start hour
    :param tide_start_minute: tide start minute
    :param tide_end_hour: tide end hour
    :param tide_end_minute: tide end minute
    :param tide_save_secs: tide save seconds
    :param enable_async_optimize: Whether enable async optimize.
    :param enable_pipelined_fwda2a: Whether enable async fwd a2a after lookup at prefetch.
    :param enable_pipelined_bwda2a: Whether enable async bwd a2a before sparse optimize stage.
    :param profile_some_steps_from: Whether profile to save timeline from step to step+10.
    :param profile_with_nvprof_from_to: Whether to profile with nvprof at certain step in-between
    :param enable_realtime_training: Whether enable realtime training. Some default value will be changed if enabled.
    :param reorder_fids_in_data_pipeline: reorder fids in data pipeline.
    :param chief_timeout_secs: chief timeout in secs
    :param save_checkpoints_secs: Save checkpoint every save_checkpoints_secs
    :param save_checkpoints_steps: Save checkpoint every save_checkpoints_steps
    :param warmup_file: The warmup file name.
    :param max_rpc_deadline_millis: Timeout for remote predict op in millisenconds.
    :param dense_only_save_checkpoints_secs: Save dense checkpoint every save_checkpoints_secs
    :param dense_only_save_checkpoints_steps: Save dense checkpoint every save_checkpoints_steps
    :param dense_only_stop_training_when_save: If barrier will be put when chief saves the dense params.
    :param checkpoints_max_to_keep: The maximum number of recent checkpoint files to keep.
    :param submit_time_secs: The time when job submitted.
    :param containers_ready_time_secs: The time when all containers are ready.
    :param max_slow_start_wait_minute: max slow start waiting time, default 10 min.
    :param cluster_type: Type of cluster. Can be 'stable', 'tide' or 'k8s'
    :param enable_model_ckpt_info: If we generate the model ckpt info when save checkpoint.
    :param feature_eviction_on_save: If we remove stale hash table entries when do the save.
    :param only_feature_engineering: only run feature engineering
    :param enable_variable_partition: Enable Variable Partition, default True.
    :param enable_fused_layout: Enable Fused Layout, default False.
    :param force_shutdown_ps: Sometimes should shutdown ps even with enable realtime training.
    :param clear_nn: Whether clean dense part of model
    :param continue_training: Whether the global step continue increase when clear nn
    :param reload_alias_map: A dict map the old variable name to the new one
    :param enable_alias_map_auto_gen: Whether enable alias_map auto generate
    :param enable_model_dump: Whether enable model dump for scurelty purpose
  """

  server_type: str = "worker"
  index: int = 0
  num_ps: int = 0
  num_workers: int = 1
  model_name: str = ""
  filter_capacity: int = 300000000
  filter_split_num: int = 7
  filter_type: str = FilterType.SLIDING_HASH_FILTER
  filter_equal_probability: bool = True
  hashtable_init_capacity: int = 0
  use_native_multi_hash_table: bool = None
  embedding_prefetch_capacity: int = 0
  enable_embedding_postpush: bool = False
  enable_variable_prefetch: bool = False
  enable_variable_postpush: bool = False
  enable_sync_training: bool = False
  merge_sync_training_ckpt: bool = True
  mode: str = tf.estimator.ModeKeys.TRAIN
  partial_recovery: bool = None
  tide_start_hour: int = None
  tide_start_minute: int = None
  tide_end_hour: int = None
  tide_end_minute: int = None
  tide_save_secs: int = None
  enable_realtime_training: bool = False
  enable_async_optimize: bool = False
  enable_pipelined_fwda2a: bool = False
  enable_pipelined_bwda2a: bool = False
  profile_some_steps_from: int = None
  profile_with_nvprof_from_to: str = None
  # Sync training optimization
  reorder_fids_in_data_pipeline: bool = False
  chief_timeout_secs: int = 1800
  save_checkpoints_secs: int = None
  save_checkpoints_steps: int = None
  dense_only_save_checkpoints_secs: int = None
  dense_only_save_checkpoints_steps: int = None
  dense_only_stop_training_when_save: bool = False
  warmup_file: str = './warmup_file'
  max_rpc_deadline_millis: int = 30000
  checkpoints_max_to_keep: int = 10
  submit_time_secs: int = None
  containers_ready_time_secs: int = None
  cluster_type: str = "stable"
  max_slow_start_wait_minute: int = 10  # 10 min
  enable_model_ckpt_info: bool = False
  feature_eviction_on_save: bool = False
  only_feature_engineering: bool = False
  enable_variable_partition: bool = True
  enable_fused_layout: bool = False
  force_shutdown_ps: bool = False
  clear_nn: bool = False
  continue_training: bool = False
  reload_alias_map: Dict[str, int] = None
  enable_alias_map_auto_gen: bool = None
  enable_model_dump: bool = False


def _make_serving_config_from_training_config(
    training_config: CpuTrainingConfig):
  serving_config = copy.deepcopy(training_config)
  serving_config.embedding_prefetch_capacity = 0
  serving_config.enable_embedding_postpush = False
  serving_config.enable_variable_prefetch = False
  serving_config.enable_variable_postpush = False
  serving_config.reorder_fids_in_data_pipeline = False
  serving_config.enable_model_ckpt_info = False
  if serving_config.enable_sync_training:
    serving_config.enable_sync_training = False
    serving_config.num_ps = training_config.num_workers
  return serving_config


def _make_serving_feature_configs_from_training_configs(feature_configs):
  serving_feature_configs = copy.deepcopy(feature_configs)
  for config in serving_feature_configs[0].values():
    # config: entry.HashTableConfigInstance
    config.table_config.entry_config.entry_type = embedding_hash_table_pb2.EntryConfig.EntryType.SERVING
  return serving_feature_configs


def make_native_task_context(config: CpuTrainingConfig,
                             sync_backend: SyncBackend = None):
  return native_task_context.NativeTaskContext(
      num_ps=config.num_ps,
      ps_index=config.index if config.server_type == 'ps' else 0,
      num_workers=config.num_workers,
      worker_index=config.index if config.server_type == 'worker' else 0,
      model_name=config.model_name,
      sync_backend=sync_backend,
      server_type=config.server_type)


def is_chief(config: CpuTrainingConfig):
  return config.server_type == "worker" and config.index == 0


class CpuTraining:
  """Wraps a native task to be runnable on CPU."""

  def __init__(self,
               config: CpuTrainingConfig,
               task: native_task.NativeTask,
               sync_backend: SyncBackend = None):
    if config.server_type != "worker":
      raise ValueError("server_type in CpuTraining must be `worker`")
    if not config.model_name:
      if isinstance(task, InstantiableParams):
        default_name = f'di_name_{task.cls.__name__}'
      else:
        default_name = f'di_name_{task.__class__.__name__}'
      config.model_name = task.p.metrics.deep_insight_name or default_name
    if config.enable_realtime_training:
      # Set some default value in streaming training if not set.
      if config.partial_recovery is None:
        config.partial_recovery = True
      if config.dense_only_save_checkpoints_secs is None and config.dense_only_save_checkpoints_steps is None:
        config.dense_only_save_checkpoints_secs = 30 * 60
    if config.use_native_multi_hash_table is None:
      if config.enable_sync_training:
        config.use_native_multi_hash_table = False
      else:
        config.use_native_multi_hash_table = True

    get_default_parser_ctx().enable_fused_layout = config.enable_fused_layout

    self._config_do_not_refer_directly = copy.deepcopy(config)
    self._serving_config_do_not_refer_directly = _make_serving_config_from_training_config(
        self._config_do_not_refer_directly)
    self._task = task
    self._params = task.p
    self._enable_hash_filter = False
    self._slot_to_occurrence_threshold = {}
    self._slot_to_expire_time = {}
    self._sync_backend = sync_backend

    # Gather extra configs for initialization earlier here.
    with native_task_context.with_ctx(
        make_native_task_context(self.config, sync_backend)):
      dump_utils = DumpUtils()
      if hasattr(self._task, 'is_dumped'):
        assert dump_utils.has_collected
        feature_name_config: Dict[
            str, entry.HashTableConfigInstance] = dump_utils.table_configs
        feature_to_unmerged_slice_dims: Dict[
            str, List[int]] = dump_utils.feature_slice_dims
        feature_to_combiner = dump_utils.feature_combiners  # Dict[str, embedding_combiners.Combiner]
        self._slot_to_occurrence_threshold = dump_utils.get_slot_to_occurrence_threshold(
        )
        self._slot_to_expire_time = dump_utils.get_slot_to_expire_time()
        self._feature_configs_do_not_refer_directly = (
            feature_name_config, feature_to_unmerged_slice_dims,
            feature_to_combiner)
      else:
        self._feature_configs_do_not_refer_directly = self._collect_feature_name_to_table_config(
        )
        if dump_utils.enable:
          dump_utils.table_configs = self._feature_configs_do_not_refer_directly[
              0]
          dump_utils.feature_slice_dims = self._feature_configs_do_not_refer_directly[
              1]
          dump_utils.feature_combiners = self._feature_configs_do_not_refer_directly[
              2]
      self._serving_feature_configs_do_not_refer_directly = _make_serving_feature_configs_from_training_configs(
          self._feature_configs_do_not_refer_directly)

  @property
  def config(self) -> CpuTrainingConfig:
    if export_context.is_exporting():
      return self._serving_config_do_not_refer_directly
    return self._config_do_not_refer_directly

  @property
  def feature_configs(
      self
  ) -> Tuple[Dict[str, entry.HashTableConfigInstance], Dict[str, List[int]],
             Dict[str, embedding_combiners.Combiner]]:
    if export_context.is_exporting():
      return self._serving_feature_configs_do_not_refer_directly
    return self._feature_configs_do_not_refer_directly

  def create_input_fn(self, mode):

    def _create_parse_fn():
      """Create a parse fn for reorder_fids_in_data_pipeline."""
      # Get merging info
      # TODO: Create real hash table earlier before inputfn, instead of dummy table here
      logging.info(
          'Wrapping parser to dedup and reorder fids in data pipeline...')
      feature_name_config, _, feature_to_combiner = self.feature_configs
      embedding_feature_names = feature_name_config.keys()

      def dummy_factory(*args, **kwargs):
        pass

      dummy_merged_table = multi_type_hash_table.MergedMultiTypeHashTable(
          feature_name_config, dummy_factory)

      def wrapped_parse_fn(*args):
        # features = parse_fn(*args, **kwargs)
        features = args[0]
        # CpuTraining.create_model_fn: def model_fn
        embedding_ragged_ids = {
            k: v for k, v in features.items() if k in embedding_feature_names
        }
        dense_features = {
            k: v
            for k, v in features.items()
            if k not in embedding_feature_names
        }
        name_to_embedding_ids = {
            k: v.values for k, v in embedding_ragged_ids.items()
        }
        # merged_multi_type_hash_table.fused_lookup
        merged_slot_to_id, merged_slot_to_sizes = dummy_merged_table._get_merged_to_indexed_tensor(
            name_to_embedding_ids)
        merged_slot_dims = dummy_merged_table.get_table_dim_sizes()
        # DistributedMultiTypeHashTableMpi.lookup
        sorted_slot_keys = sorted(merged_slot_to_id.keys())
        sorted_input = [merged_slot_to_id[k] for k in sorted_slot_keys]
        all_fids, shard_sizes, sharded_slot_sizes, fused_embbedding_offsets = \
        distribution_ops.fused_reorder_by_indices(
            sorted_input, self.config.num_workers, merged_slot_dims
        )
        # Results include the following intermediate tensors
        res = (
            dense_features,  # Dense features
            # CpuTraining.create_model_fn: def model_fn
            (
                parser_utils.RaggedEncodingHelper.expand(
                    embedding_ragged_ids,  # Sparse Features
                    with_precomputed_nrows=True,
                    with_precomputed_value_rowids=False
                    # Because most GPU-downstream poolings
                    # are not using value_rowids anymore,
                    # we choose not to precompute it here.
                ),
                # merged_multi_type_hash_table.lookup
                (
                    merged_slot_to_sizes,
                    # DistributedMultiTypeHashTableMpi.lookup
                    (all_fids, shard_sizes, sharded_slot_sizes,
                     tuple(fused_embbedding_offsets)))))
        # None here to prevent tf.Estimator from automatically treating the second in the return tuple as labels
        return res, None

      return wrapped_parse_fn

    if (mode == tf.estimator.ModeKeys.TRAIN and
        self.config.reorder_fids_in_data_pipeline):
      # Assumption: use_reduced_alltoall
      parser_utils.add_extra_parse_step(_create_parse_fn())

    def wrapped_input_fn():
      input_fn = self._task.create_input_fn(mode)
      with native_task_context.with_ctx(
          make_native_task_context(self.config, self._sync_backend)):
        return input_fn()

    return wrapped_input_fn

  def create_model_fn(self):

    def create_hash_table_and_filters_fn():
      (feature_name_config, feature_to_unmerged_slice_dims,
       feature_to_combiner) = self.feature_configs

      logging.vlog(
          1, "feature_to_unmerged_slice_dims: {}".format(
              feature_to_unmerged_slice_dims))
      slot_occurrence_threshold_config = embedding_hash_table_pb2 \
        .SlotOccurrenceThresholdConfig()

      for slot, occurrence_threshold in self._slot_to_occurrence_threshold.items(
      ):
        slot_occurrence_threshold = slot_occurrence_threshold_config.slot_occurrence_thresholds.add(
        )
        slot_occurrence_threshold.slot = slot
        slot_occurrence_threshold.occurrence_threshold = occurrence_threshold
        if occurrence_threshold > 0:
          self._enable_hash_filter = True

      # In the sync training, hash filter and hashtables are inside worker.
      if is_exporting():
        hash_filters = [None] * max(1, self.config.num_ps)
      else:
        with device_utils.maybe_device_if_allowed('/device:GPU:0'):
          hash_filters = hash_filter_ops.create_hash_filters(
              self.config.num_ps,
              # disable hash filter (only use dummy filter) for GPU table for now
              False if self._params.train.use_gpu_emb_table else
              self._enable_hash_filter,
              config=slot_occurrence_threshold_config.SerializeToString(),
              filter_capacity=self.config.filter_capacity,
              filter_split_num=self.config.filter_split_num,
              filter_type=self.config.filter_type)

        slot_to_expire_time_config = embedding_hash_table_pb2.SlotExpireTimeConfig(
        )

        for slot, expire_time in self._slot_to_expire_time.items():
          slot_expire_time = slot_to_expire_time_config.slot_expire_times.add()
          slot_expire_time.slot = slot
          slot_expire_time.expire_time = expire_time

        for config in feature_name_config.values():
          config.table_config.slot_expire_time_config.CopyFrom(
              slot_to_expire_time_config)

      unique = False if is_exporting() else True
      sync_clients = [None] * max(1, self.config.num_ps)
      if self.config.enable_realtime_training and not is_exporting():
        sync_clients = distributed_serving_ops.create_parameter_sync_clients(
            self.config.num_ps)
      if not self.config.enable_sync_training:
        if self.config.enable_fused_layout:
          return distributed_ps_factory.create_partitioned_hash_table(
              num_ps=self.config.num_ps,
              feature_name_to_config=feature_name_config,
              layout_configs=self._task.layout_dict,
              feature_to_combiner=feature_to_combiner,
              feature_to_unmerged_slice_dims=feature_to_unmerged_slice_dims,
              use_native_multi_hash_table=self.config.
              use_native_multi_hash_table,
              max_rpc_deadline_millis=self.config.max_rpc_deadline_millis,
              unique=unique,
              transfer_float16=False,
              hash_filters=hash_filters,
              sync_clients=sync_clients), hash_filters
        elif self.config.use_native_multi_hash_table:
          return distributed_ps_factory.create_native_multi_hash_table(
              self.config.num_ps,
              feature_name_config,
              hash_filters,
              sync_clients=sync_clients,
              max_rpc_deadline_millis=self.config.max_rpc_deadline_millis,
          ), hash_filters
        else:
          return distributed_ps_factory.create_multi_type_hash_table(
              self.config.num_ps,
              feature_name_config,
              hash_filters,
              sync_clients=sync_clients,
              reduce_network_packets=True,
              max_rpc_deadline_millis=self.config.max_rpc_deadline_millis,
          ), hash_filters
      else:
        queue_configs = {
            k: int(getattr(self.config, k))
            for k in ("embedding_prefetch_capacity", "enable_async_optimize",
                      "enable_pipelined_fwda2a", "enable_pipelined_bwda2a")
        }

        if self.config.enable_fused_layout:
          return distributed_ps_factory.create_partitioned_hash_table(
              num_ps=self.config.num_ps,
              feature_name_to_config=feature_name_config,
              layout_configs=self._task.layout_dict,
              feature_to_combiner=feature_to_combiner,
              feature_to_unmerged_slice_dims=feature_to_unmerged_slice_dims,
              use_native_multi_hash_table=self.config.
              use_native_multi_hash_table,
              max_rpc_deadline_millis=self.config.max_rpc_deadline_millis,
              unique=unique,
              transfer_float16=False,
              hash_filters=hash_filters,
              sync_clients=sync_clients), hash_filters
        elif self.config.use_native_multi_hash_table:
          raise NotImplementedError(
              "Multi hash table for sync training is not implemented yet.")
        else:
          return distributed_ps_factory.create_in_worker_multi_type_hash_table(
              self.config.num_workers,
              feature_name_config,
              hash_filters[0],
              sync_client=sync_clients[0],
              queue_configs=queue_configs), hash_filters

    with native_task_context.with_ctx(
        make_native_task_context(self.config, self._sync_backend)):
      return self._get_pipelined_model_fn(create_hash_table_and_filters_fn)

  def _generate_valid_features(self) -> Dict[str, tf.Tensor]:
    """Generates a valid feature dict which can be fed into model_fn in TRAIN mode."""
    input_fn = self._task.create_input_fn(tf.estimator.ModeKeys.TRAIN)
    dataset = input_fn()
    return tf.data.experimental.get_single_element(dataset)

  def _collect_feature_name_to_table_config(
      self
  ) -> Tuple[Dict[str, entry.HashTableConfigInstance], Dict[str, List[int]],
             Dict[str, embedding_combiners.Combiner]]:
    per_replica_batch_size = self._params.train.per_replica_batch_size
    with tf.Graph().as_default() as g, ParserCtx(enable_fused_layout=False):
      setattr(g, DRY_RUN, True)
      feature_factory = feature.DummyFeatureFactory(per_replica_batch_size)
      self._task.ctx.feature_factory = feature_factory
      self._task.ctx.layout_factory = None
      self._task.ctx.async_function_mgr = prefetch_queue.AsyncFunctionMgr(
          is_async=False)
      global_step = tf.compat.v1.train.get_or_create_global_step()
      model_fn = self._task.create_model_fn()
      features = self._generate_valid_features()
      model_fn(features=features,
               mode=tf.estimator.ModeKeys.TRAIN,
               config=tf.estimator.RunConfig())

    table_to_config = feature_factory.get_table_name_to_table_config()
    feature_to_config: Dict[str, entry.HashTableConfigInstance] = {}
    feature_to_unmerged_slice_dims: Dict[str, List[int]] = {}
    feature_to_combiner: Dict[str, embedding_combiners.Combiner] = {}
    for k, table_config in table_to_config.items():
      for feature_name in table_config.feature_names:
        assert not feature_name in feature_to_config, "Feature must only belongs to one table."
        feature_to_config.update({
            feature_name:
                _combine_slices_as_table(table_config.slice_configs,
                                         table_config.hashtable_config)
        })
        feature_to_unmerged_slice_dims[
            feature_name] = table_config.unmerged_slice_dims
        feature_to_combiner[feature_name] = table_config.feature_to_combiners[
            feature_name]

    if self.config.hashtable_init_capacity > 0:
      for conf in feature_to_config.values():
        conf.table_config.initial_capacity = self.config.hashtable_init_capacity

    self._slot_to_occurrence_threshold = feature_factory.slot_to_occurrence_threshold
    self._slot_to_expire_time = feature_factory.slot_to_expire_time

    if self.config.enable_sync_training:
      # To improve hash table performance
      for config in feature_to_config.values():
        config.table_config.entry_type = embedding_hash_table_pb2.EmbeddingHashTableConfig.RAW

    return feature_to_config, feature_to_unmerged_slice_dims, feature_to_combiner

  # TODO(leqi.zou): Add a function to disable pipelining.
  def _get_pipelined_model_fn(self, create_hash_table_and_filters_fn: Callable[
      [], Tuple[multi_type_hash_table.MultiTypeHashTable, List[tf.Tensor]]]):
    (feature_name_config, feature_to_unmerged_slice_dims,
     feature_to_combiner) = self.feature_configs
    embedding_feature_names: Iterable[str] = feature_name_config.keys()
    if not embedding_feature_names:
      # We need to skip pipeline phase since dequeue might never be called if
      # embedding feature is not used.
      self._task.ctx.feature_factory = None
      return self._task.create_model_fn()

    def get_hooks_for_restore(model_dir: str, hash_filters: List[tf.Tensor],
                              ps_monitor: save_utils.PsMonitor):
      if not model_dir:
        return ()
      basename = os.path.join(model_dir, "model.ckpt")

      restore_listeners = [
          hash_table_ops.HashTableCheckpointRestorerListener(
              basename, ps_monitor),
          multi_hash_table_ops.MultiHashTableCheckpointRestorerListener(
              basename, ps_monitor),
          hash_filter_ops.HashFilterCheckpointRestorerListener(
              basename,
              hash_filters,
              self._enable_hash_filter,
              enable_save_restore=(not self.config.enable_sync_training)),
          CustomRestoreListener(
              self.config.reload_alias_map,
              self.config.clear_nn,
              self.config.continue_training,
              model_dir=model_dir,
              enable_alias_map_auto_gen=self.config.enable_alias_map_auto_gen)
      ]
      return (basic_restore_hook.CheckpointRestorerHook(
          listeners=restore_listeners),)

    def get_saver_listeners_for_exporting(save_path: str,
                                          export_dir_base: str = None,
                                          dense_only=False,
                                          exempt_checkpoint_paths=None,
                                          include_graphs=None):
      # TODO(leqi.zou): Add a test for this when graceful shutdown is implemented.
      if is_exporting():
        # Safety check. To prevent infinite recursion.
        raise ValueError(
            "Logic corrupted. Try to call exporting listeners inside exporting."
        )
      if dense_only and self._params.serving.export_mode is not ExportMode.DISTRIBUTED:
        raise ValueError(
            "Please set params.serving.export_mode = ExportMode.DISTRIBUTED. "
            "Only DISTRIBUTED mode is allowed when dense_only=True, got",
            self._params.serving.export_mode)
      if not self._params.serving.export_when_saving:
        return []
      model_dir = os.path.dirname(save_path)
      serving_input_receiver_fn = self.create_serving_input_receiver_fn()
      if not serving_input_receiver_fn:
        raise ValueError("A valid serving_input_receiver_fn must be provided ",
                         "if exporting is enabled. Got ",
                         serving_input_receiver_fn)
      if not export_dir_base:
        export_dir_base = os.path.join(model_dir,
                                       self._params.serving.export_dir_base)
      # TODO(leqi.zou): Needs to do lifecycle management for exported model.
      exporter = create_exporter(self,
                                 model_dir=model_dir,
                                 warmup_file=self.config.warmup_file,
                                 export_dir_base=export_dir_base,
                                 dense_only=dense_only,
                                 include_graphs=include_graphs)
      barrier_listeners = []
      if self.config.enable_sync_training and not dense_only:
        barrier_listeners.append(
            sync_training_hooks.SyncTrainingBarrierSaverListener())
      return [
          export_hooks.ExportSaverListener(
              save_path,
              serving_input_receiver_fn,
              exporter,
              exempt_checkpoint_paths=exempt_checkpoint_paths,
              dense_only=dense_only)
      ]

    def get_hooks_for_save(model_dir: str, hash_filters: List[tf.Tensor],
                           barrier_op: barrier_ops.BarrierOp,
                           ps_monitor: save_utils.PsMonitor):
      logging.info("get_hooks_for_save model_dir is " + model_dir)
      if not model_dir:
        raise ValueError("model_dir must be provided")
      hooks = []

      exempt_checkpoint_paths = list()
      monolith_ckpt_state = save_utils.get_monolith_checkpoint_state(
          model_dir, remove_invalid_path=True)
      ckpt_state = tf.train.get_checkpoint_state(model_dir)
      if monolith_ckpt_state and monolith_ckpt_state.exempt_model_checkpoint_paths:
        exempt_checkpoint_paths = [
            os.path.basename(p)
            for p in monolith_ckpt_state.exempt_model_checkpoint_paths
        ]
        logging.info(
            'Exempt checkpoint paths: {}'.format(exempt_checkpoint_paths))
        existing_checkpoint_paths = set([
            os.path.basename(p) for p in ckpt_state.all_model_checkpoint_paths
        ])
        assert all(
            [p in existing_checkpoint_paths for p in exempt_checkpoint_paths])

      is_root_node = not self.config.enable_sync_training or self.config.index == 0

      def create_saver():
        return save_utils.PartialRecoverySaver(
            sharded=not self.config.enable_sync_training,
            max_to_keep=self.config.checkpoints_max_to_keep,
            keep_checkpoint_every_n_hours=24,
            ps_monitor=ps_monitor,
            exempt_checkpoint_paths=exempt_checkpoint_paths,
            skip_save=not is_root_node,
            model_dir=model_dir)

      saver = create_saver()

      tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.SAVERS, saver)
      basename = os.path.join(model_dir, "model.ckpt")

      save_listeners = [
          hash_table_ops.HashTableCheckpointSaverListener(basename),
          multi_hash_table_ops.MultiHashTableCheckpointSaverListener(
              basename, write_ckpt_info=is_root_node),
          hash_filter_ops.HashFilterCheckpointSaverListener(
              basename,
              hash_filters,
              self._enable_hash_filter,
              enable_save_restore=(not self.config.enable_sync_training))
      ]

      include_graphs = None
      if self.config.enable_sync_training:
        include_graphs = [f"ps_{self.config.index}"]
        if is_root_node:
          include_graphs.append("entry")

      save_listeners += get_saver_listeners_for_exporting(
          basename,
          exempt_checkpoint_paths=exempt_checkpoint_paths,
          include_graphs=include_graphs)

      if self.config.enable_model_ckpt_info and is_root_node:
        save_listeners.append(ckpt_info.FidSlotCountSaverListener(model_dir))

      if self.config.feature_eviction_on_save:
        save_listeners.extend([
            hash_table_ops.HashTableRestorerSaverLitsener(basename),
            multi_hash_table_ops.MultiHashTableRestorerSaverListener(basename),
        ])


      save_checkpoints_secs = self.config.save_checkpoints_secs or self._params.train.save_checkpoints_secs
      save_checkpoints_steps = self.config.save_checkpoints_steps or self._params.train.save_checkpoints_steps
      if save_checkpoints_secs is None and save_checkpoints_steps is None:
        save_checkpoints_steps = 100000000
      if (save_checkpoints_secs is not None) and (save_checkpoints_steps
                                                  is not None):
        raise ValueError(
            "Can not provide both save_checkpoints_secs and save_checkpoints_steps."
        )

      # We do not use barrier for the sync training.
      barrier_before_save_listeners = []
      barrier_after_save_listeners = []
      if not self.config.enable_sync_training:
        l = ckpt_hooks.BarrierSaverListener(
            barrier_op,
            max_pending_seconds=self._params.train.
            max_pending_seconds_for_barrier)
        barrier_before_save_listeners.append(hook_utils.BeforeSaveListener(l))
        barrier_after_save_listeners.append(hook_utils.AfterSaveListener(l))
      save_listeners = barrier_before_save_listeners + save_listeners
      finally_after_save_listeners = barrier_after_save_listeners

      # In the rare case, we need to do the first save.
      # Otherwise, partial_recovery won't work and will go through initialization phase.
      # It is supposed to be
      # should_do_first_save = self.config.partial_recovery and ckpt_state is None
      # Here we just make it false because there are issues with uninitialized iterator.
      should_do_first_save = False
      saver_hook = save_utils.NoFirstSaveCheckpointSaverHook(
          model_dir,
          save_secs=save_checkpoints_secs,
          save_steps=save_checkpoints_steps,
          saver=saver,
          listeners=save_listeners,
          finally_after_save_listeners=finally_after_save_listeners,
          save_graph_def=is_root_node,
          tide_start_hour=self.config.tide_start_hour,
          tide_start_minute=self.config.tide_start_minute,
          tide_end_hour=self.config.tide_end_hour,
          tide_end_minute=self.config.tide_end_minute,
          tide_save_secs=self.config.tide_save_secs,
          ignore_save_errors=self.config.enable_realtime_training,
          is_dense_only=False,
          use_native_multi_hash_table=self.config.use_native_multi_hash_table,
          no_first_save=not should_do_first_save)

      if self.config.enable_sync_training and self.config.enable_realtime_training:
        hooks.append(
            sync_training_hooks.SyncTrainingForceDumpHook(
                model_dir, saver_hook.timer))

      hooks.append(saver_hook)

      if not self.config.enable_sync_training:
        ctrl_hook = controller_hooks.ControllerHook(self.config.num_ps,
                                                    barrier_op,
                                                    saver_hook.timer.reset)
        query_hook = controller_hooks.QueryActionHook(model_dir, ctrl_hook)
        hooks.extend([ctrl_hook, query_hook])

      dense_only_save_checkpoints_steps = self.config.dense_only_save_checkpoints_steps or self._params.train.dense_only_save_checkpoints_steps
      dense_only_save_checkpoints_secs = self.config.dense_only_save_checkpoints_secs or self._params.train.dense_only_save_checkpoints_secs
      if (dense_only_save_checkpoints_steps or
          dense_only_save_checkpoints_secs) and is_root_node:
        dense_saver = create_saver()

        dense_model_dir = os.path.join(model_dir, "dense_only")
        stats = tf.train.get_checkpoint_state(dense_model_dir)
        if stats:
          dense_saver.recover_last_checkpoints(stats.all_model_checkpoint_paths)
        dense_basename = os.path.join(dense_model_dir, "model.ckpt")
        export_dir_base = os.path.join(model_dir,
                                       self._params.serving.export_dir_base)
        save_utils.NoFirstSaveCheckpointSaverHook._has_dense_only = True
        dense_barrier_before_save_listeners = barrier_before_save_listeners if self.config.dense_only_stop_training_when_save else []
        dense_barrier_after_save_listeners = barrier_after_save_listeners if self.config.dense_only_stop_training_when_save else []

        dense_saver_hook = save_utils.NoFirstSaveCheckpointSaverHook(
            dense_model_dir,
            save_secs=dense_only_save_checkpoints_secs,
            save_steps=dense_only_save_checkpoints_steps,
            saver=dense_saver,
            listeners=dense_barrier_before_save_listeners +
            get_saver_listeners_for_exporting(
                dense_basename,
                export_dir_base=export_dir_base,
                dense_only=True,
                exempt_checkpoint_paths=exempt_checkpoint_paths),
            finally_after_save_listeners=dense_barrier_after_save_listeners,
            tide_start_hour=self.config.tide_start_hour,
            tide_start_minute=self.config.tide_start_minute,
            tide_end_hour=self.config.tide_end_hour,
            tide_end_minute=self.config.tide_end_minute,
            tide_save_secs=self.config.tide_save_secs,
            ignore_save_errors=self.config.enable_realtime_training,
            is_dense_only=True)

        if self.config.enable_sync_training and self.config.enable_realtime_training:
          hooks.append(
              sync_training_hooks.SyncTrainingSaverControlHook(
                  model_dir, dense_saver_hook.timer))

        hooks.append(dense_saver_hook)

      return tuple(hooks)

    def get_slow_start_hook(slow_start_steps: int):

      if slow_start_steps:
        return (session_run_hooks.CustomGlobalStepWaiterHook(
            int(slow_start_steps * np.log(1 + self.config.index)),
            max_non_tide_wait_minute=self.config.max_slow_start_wait_minute),)
      return ()

    def get_tide_stopping_hook():
      if self.config.tide_start_hour is not None and self.config.tide_end_hour is not None:
        return (session_run_hooks.TideStoppingHook(
            self.config.tide_start_hour, self.config.tide_start_minute,
            self.config.tide_end_hour, self.config.tide_end_minute),)
      return ()

    def get_hooks_for_metrics(model_dir: str, save_steps: int):
      hooks = []
      if self._params.metrics.enable_tf2_profiler_hook:
        start_step = self.config.profile_some_steps_from
        end_step = None if start_step is None else start_step + 10
        hooks.append(
            Tf2ProfilerCaptureOnceHook(
                logdir=model_dir, capture_step_range=[start_step, end_step]))

      if self._params.metrics.enable_throughput_hook and is_chief(self.config):
        hooks.append(
            ThroughputMetricHook(
                model_name=self.config.model_name,
                start_time_secs=self.config.containers_ready_time_secs,
                cluster_type=self.config.cluster_type))
      return tuple(hooks)

    def variable_prefetch_enabled():
      return not self.config.enable_sync_training and self.config.enable_variable_prefetch

    def get_cached_variable_context():
      if variable_prefetch_enabled():
        return tf.variable_creator_scope(variables.cached_variable_creator)
      return contextlib.nullcontext()

    def get_partitioner_variable_context():
      if not is_exporting() and self.config.enable_variable_partition:
        # TODO(leqi.zou): This only works for tf.compat.v1.get_variable,
        # but not for tf.Variable.
        #
        # Finally, we can use something similar to PSStrategy to solve
        # this problem.
        return tf.compat.v1.variable_scope(
            "",
            partitioner=tf.compat.v1.variable_axis_size_partitioner(
                max_shard_bytes=1 << 17, max_shards=self.config.num_ps))
      return contextlib.nullcontext()

    def get_variable_prefetch_hooks():
      if variable_prefetch_enabled():
        return (variables.FetchAllCachedVariablesHook(),)
      return ()

    def get_req_time(features):
      if "req_time" in features:
        return features["req_time"][0]
      else:
        return None

    def get_itempool_hook(model_dir, mode):
      pools = tf.compat.v1.get_collection(POOL_KEY)
      if pools and mode != tf.estimator.ModeKeys.PREDICT:
        logging.info("append itempool_save_restore_hook in training_hooks")
        save_checkpoints_secs = self.config.save_checkpoints_secs or self._params.train.save_checkpoints_secs
        save_checkpoints_steps = self.config.save_checkpoints_steps or self._params.train.save_checkpoints_steps
        if save_checkpoints_secs is None and save_checkpoints_steps is None:
          save_checkpoints_steps = 100000000
        item_pool_hook = ItemPoolSaveRestoreHook(
            model_dir=model_dir, save_steps=save_checkpoints_steps, mode=mode)
        if hasattr(self._task, 'add_training_hook'):
          self._task.add_training_hook(item_pool_hook)

    def model_fn(features: Dict[str, tf.Tensor], mode: str,
                 config: tf.estimator.RunConfig):
      hash_table, hash_filters = create_hash_table_and_filters_fn()
      logging.info(
          f'\n> hash_table: {hash_table}\n> hash_filters: {hash_filters}')
      # For prefetch queue, collect auxiliary tensors

      get_itempool_hook(config.model_dir, mode=mode)

      auxiliary_bundle = {}
      async_function_mgr = prefetch_queue.AsyncFunctionMgr(
          is_async=self.config.enable_variable_postpush)
      self._task.ctx.async_function_mgr = async_function_mgr

      def call_raw_model_fn(features):
        raw_model_fn = self._task.create_model_fn()
        with get_cached_variable_context(), get_partitioner_variable_context():
          spec = raw_model_fn(features=features, mode=mode, config=config)

        return spec

      if self.config.enable_fused_layout:
        self._task.ctx.feature_factory = None
        sparse_features = features.pop('sparse_features')
        auxiliary_bundle.update(features)
        layout_embeddings = hash_table.lookup(sparse_features, auxiliary_bundle)
        # embedding_prefetch
        (deq_layout_embeddings,
         deq_auxiliary_bundle), queue = enqueue_dicts_with_queue_return(
             (layout_embeddings, auxiliary_bundle),
             capacity=self.config.embedding_prefetch_capacity)
        if queue:
          hash_table.add_queue_hook(EnqueueHook(queue))

        dequeued_features = {key: deq_auxiliary_bundle[key] for key in features}
        # args are data we will transfer to remote deivce if needed.
        args = (deq_layout_embeddings, dequeued_features)

        def call_model_fn(args):
          # set layout_factory, this step must after embedding_prefetch
          self._task.ctx.layout_factory = feature.EmbeddingLayoutFactory(
              hash_table, args[0], auxiliary_bundle=deq_auxiliary_bundle)
          return call_raw_model_fn(args[1])
      else:
        if self.config.reorder_fids_in_data_pipeline:
          features, res_pack = features
          features = {
              k: v
              for k, v in features.items()
              if not isinstance(v, tf.RaggedTensor)
          }
          embedding_ragged_ids, res_pack = res_pack
          name_to_ids = {
              k: None for k in embedding_ragged_ids.keys()
              # None to keep interface, we will only use the keys
          }
          auxiliary_bundle["features"] = features
          auxiliary_bundle[
              "embedding_ragged_ids"] = parser_utils.RaggedEncodingHelper.contract(
                  embedding_ragged_ids)
          embeddings, auxiliary_bundle = hash_table.lookup(
              name_to_ids, auxiliary_bundle, res_pack)
          dequeued_embeddings = embeddings
        else:
          embedding_ragged_ids: Dict[str, tf.RaggedTensor] = {
              k: v for k, v in features.items() if k in embedding_feature_names
          }
          features: Dict[str, tf.Tensor] = {  # Dense features or labels
              k: v
              for k, v in features.items()
              if not isinstance(v, tf.RaggedTensor)
          }
          auxiliary_bundle["features"] = features
          auxiliary_bundle["embedding_ragged_ids"] = embedding_ragged_ids
          # 'feature_name' -> <tf.Tensor 'RaggedFromVariant:1' shape=(None,) dtype=int64>
          name_to_ids: Dict[str, tf.Tensor] = {
              k: v.values for k, v in embedding_ragged_ids.items()
          }
          # for MergedMultiTypeHashTable, lookup returns:
          embeddings: Dict[str, tf.Tensor] = hash_table.lookup(name_to_ids)
          (dequeued_embeddings,
           auxiliary_bundle), q = enqueue_dicts_with_queue_return(
               (embeddings, auxiliary_bundle),
               capacity=self.config.embedding_prefetch_capacity)
          if q:
            hash_table.add_queue_hook(EnqueueHook(q))

        dequeued_embedding_ragged_ids = auxiliary_bundle.pop(
            "embedding_ragged_ids")
        dequeued_features = auxiliary_bundle.pop("features")

        # record dequeued features, primarily for evaluation purposes
        for name, tensor in dequeued_embedding_ragged_ids.items():
          tensor.feature_name = name
          tf.compat.v1.add_to_collection('dequeued_sparse_features', tensor)

        for name, tensor in dequeued_features.items():
          try:
            tensor.feature_name = name
            tf.compat.v1.add_to_collection('dequeued_features', tensor)
          except Exception as e:
            logging.error(f'tensor name is {name}')
            logging.error(f'exception is {str(e)}')

        embedding_slices = feature.create_embedding_slices(
            dequeued_embeddings, dequeued_embedding_ragged_ids,
            feature_to_combiner, feature_to_unmerged_slice_dims)

        args = (
            dequeued_features,
            embedding_slices,
        )

        # To enable remote inference, we need to transfer all tensors
        # which are needed by using remote predict. All tensors used should be
        # listed in the parameter, otherwise export graph will complain that
        # tensor may come from another graph.
        #
        # Notice this is only for the inference. In the training, TensorFlow
        # will automatically add send/recv if tensors are on the different graph.
        def call_model_fn(args):
          self._task.ctx.layout_factory = None
          if self.config.enable_sync_training:
            # TODO(zouxuan): enable this for async training later on.
            self._task.ctx.feature_factory = _FusedCpuFeatureFactory(
                hash_table,
                dequeued_embeddings,
                args[1],
                get_req_time(args[0]),
                auxiliary_bundle=auxiliary_bundle)
          else:
            self._task.ctx.feature_factory = _CpuFeatureFactory(
                hash_table,
                dequeued_embedding_ragged_ids,
                dequeued_embeddings,
                args[1],
                get_req_time(args[0]),
                async_function_mgr=async_function_mgr,
                async_push=self.config.enable_embedding_postpush)
          return call_raw_model_fn({**args[0], **dequeued_embeddings})

      spec = None
      if export_context.is_exporting(
      ) and export_context.get_current_export_ctx().with_remote_gpu:

        def remote_call(tensors):
          with tf.device("/device:GPU:0"):
            spec = call_model_fn(tensors)
          return spec.predictions

        g = export_context.get_current_export_ctx().sub_graph("dense_0")
        with g.as_default():
          helper = export_utils.RemotePredictHelper("gpu_remote_call", args,
                                                    remote_call)
        predictions = helper.call_remote_predict(
            model_name=f"{native_task_context.get().model_name or ''}:dense_0",
            old_model_name="dense_0")
        spec = tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
      else:
        spec = call_model_fn(args)

      if not is_exporting():
        ps_monitor = save_utils.PsMonitor(
            self.config.num_ps
        ) if self.config.partial_recovery and self.config.num_ps > 1 else None

        training_hooks = ()
        training_chief_hooks = ()
        ckpt_helper = ckpt_hooks.WorkerCkptHelper(config.model_dir,
                                                  self.config.index)
        stop_helper = controller_hooks.StopHelper()
        # SetCurrentSessionHook must present first.
        training_hooks += (session_hooks.SetCurrentSessionHook(),)
        barrier_op = None
        enable_sync_hook = self.config.num_workers > 1 and not self.config.enable_sync_training
        sync_hook_helper = sync_hooks.TrainingHooksHelper(
            enable_sync_hook,
            self.config.num_workers,
            self.config.index,
            chief_timeout_seconds=self.config.chief_timeout_secs)
        if not self.config.enable_sync_training:
          barrier_op = barrier_ops.BarrierOp(
              self.config.num_workers,
              is_chief=is_chief(self.config),
              barrier_callbacks=[
                  ckpt_helper.create_save_iterator_callback(),
                  stop_helper.create_barrier_callback()
              ])
          training_hooks += sync_hook_helper.training_hooks + (
              barrier_ops.BarrierHook(self.config.index, barrier_op),)
          if self._params.mode == tf.estimator.ModeKeys.TRAIN:
            training_hooks += get_slow_start_hook(
                self._params.train.slow_start_steps)
          training_hooks += (ckpt_helper.create_restorer_hook(),
                             stop_helper.create_stop_hook())

          training_chief_hooks += (ps_check_hooks.PsHealthCheckerHook(
              ps_check_hooks.Config(barrier_op=barrier_op,
                                    num_ps=self.config.num_ps)),)

        # Prefetch hooks should be put after control hooks (like slow start hook).
        # Just in case, we have shuffle in dataset and we start reading too much
        # data before we actually start the training.
        training_hooks += tuple(hash_table.get_queue_hooks())
        training_hooks += get_variable_prefetch_hooks()
        training_hooks += tuple(self._task.ctx.async_function_mgr.hooks)
        """
        Make sure sync hook running after restore hook and before save hook.
        The running order is similar to:
          'after_create_session' : restore, chief start, worker start
          'end'                  : worker end, save, chief end
        """
        training_chief_hooks += (
            get_hooks_for_restore(config.model_dir, hash_filters, ps_monitor) +
            sync_hook_helper.training_chief_hooks + get_hooks_for_save(
                config.model_dir, hash_filters, barrier_op, ps_monitor) +
            get_hooks_for_metrics(config.model_dir, config.save_summary_steps))

        predicting_hooks = (get_hooks_for_restore(config.model_dir,
                                                  hash_filters, ps_monitor))

        spec = spec._replace(
            training_chief_hooks=training_chief_hooks +
            spec.training_chief_hooks,
            training_hooks=training_hooks + spec.training_hooks,
            prediction_hooks=predicting_hooks + spec.prediction_hooks)

        logging.info("Training Chief Hooks: {}".format(
            spec.training_chief_hooks))
        logging.info("Training Hooks: {}".format(spec.training_hooks))

      return spec

    def wrapped_model_fn(features: Dict[str, tf.Tensor], mode: str,
                         config: tf.estimator.RunConfig):
      with native_task_context.with_ctx(
          make_native_task_context(self.config, self._sync_backend)):
        return model_fn(features, mode, config)

    return wrapped_model_fn

  def create_serving_input_receiver_fn(self):
    return self._task.create_serving_input_receiver_fn()


@dataclasses.dataclass
class DistributedCpuTrainingConfig(CpuTrainingConfig):
  """The training config for distributed training.

  attributes:
    :param model_dir: The directory where the model is load/saved.
    :param intra_op_parallelism_threads: intra_op parallelism threads.
    :param inter_op_parallelism_threads: inter_op parallelism threads.
    :param num_extra_ps: The number of extra ps for ps benchmark.
    :param num_redundant_ps: The number of redundant ps for quickly starting.
                             We will pick |num_ps| from |num_ps + num_extra_ps + num_redundant_ps| ps
    :param uuid: uuid of cpu training.
    :param operation_timeout_in_ms: Global timeout for all blocking operations in this session.
    :param session_creation_timeout_secs: Max time workers should wait for a session to become available.
    :param enable_gpu_training: Whethere to also use GPU for training besides CPU.
    :param processes_per_gpu: Integer number of mpi processes on GPU (for sync training).
    :param max_retry_times: Maximum retry times for workers to start train.
    :param retry_wait_in_secs: Sleep time interval to wait for worker retry.
    :param fountain_zk_host: zk_host for fountain service.
    :param fountain_model_name: model_name for fountain service.
    :param dc_aware: data-center aware or not.
  """

  model_dir: str = ""
  intra_op_parallelism_threads: int = 8
  inter_op_parallelism_threads: int = 16
  num_extra_ps: int = 0
  num_redundant_ps: int = 0
  uuid: str = ""

  operation_timeout_in_ms: int = -1
  session_creation_timeout_secs: int = 7200

  enable_gpu_training: bool = False
  processes_per_gpu: int = 1

  max_retry_times: int = 0
  retry_wait_in_secs: int = 30

  fountain_zk_host: str = ""
  fountain_model_name: str = ""
  dc_aware: bool = False


def _prepare_server(target: str, config: DistributedCpuTrainingConfig):
  """Do some preparation before we register the server to the server discovery"""
  if config.server_type == "ps":
    session_config = cluster_manager.generate_session_config()
    with tf.compat.v1.Session(target, config=session_config) as sess:
      # Creates machine info so the following access won't create new machine info.
      sess.run(
          logging_ops.machine_info(
              shared_name=ps_check_hooks.get_ps_machine_info_shared_name(
                  config.index)))


def _shutdown_ps(target, cluster, task, num_ps):
  cluster = copy.deepcopy(cluster)
  # Worker has already shutdowned.
  if "worker" in cluster:
    del cluster["worker"]
  session_config = cluster_manager.generate_session_config((cluster, task))
  with tf.compat.v1.Session(target, config=session_config) as sess:
    for i in range(num_ps):
      logging.info('Try to shutdown ps {}'.format(i))
      with tf.device(utils.ps_device(i)):
        queue = tf.queue.FIFOQueue(1,
                                   tf.int32,
                                   shared_name="ps_queue_" + str(i))
      sess.run(queue.enqueue(1))
      logging.info('Shutdown ps {} successfully!'.format(i))


def _join_ps(target, ps_index, sync_backend: SyncBackend = None):
  session_config = cluster_manager.generate_session_config()
  with tf.compat.v1.Session(target, config=session_config) as sess:
    queue = tf.queue.FIFOQueue(1,
                               tf.int32,
                               shared_name="ps_queue_" + str(ps_index))
    finished = False
    t = None
    if sync_backend is not None:
      sync_client = ParameterSyncClient(
          distributed_serving_ops.parameter_sync_client_from_config(
              name_suffix=str(ps_index)))
      sync_config_str = tf.compat.v1.placeholder(tf.string,
                                                 shape=(),
                                                 name="sync_config_str")
      sync_run_step = sync_client.create_sync_op(sync_config_str)

      def parameter_sync_job(sess, sync_run_step: tf.Tensor):
        with sess.graph.as_default(
        ):  # To make sure the graphs in/out thread are same
          nonlocal finished
          while not finished:
            start = timeit.default_timer()
            try:
              sess.run(sync_run_step,
                       feed_dict={
                           sync_config_str:
                               distributed_serving_ops.refresh_sync_config(
                                   sync_backend, ps_index)
                       },
                       options=tf.compat.v1.RunOptions(timeout_in_ms=1000 * 60))
            except tf.errors.OpError as e:
              logging.error('Error occurred when synchronizing parameter: %s',
                            str(e))
              exc_type, exc_value, exc_traceback_obj = sys.exc_info()
              logging.error(f"exc_type: {exc_type}")
              logging.error(f"exc_value: {exc_value}")
              traceback.print_tb(exc_traceback_obj, limit=10)
            total_cost = timeit.default_timer() - start
            # Synchronizing parameter per 10 seconds
            time.sleep(max(0, 10 - total_cost))
          logging.info(
              "Ps {} received chief's shutdown signal...".format(ps_index))

      t = threading.Thread(target=parameter_sync_job,
                           args=(sess, sync_run_step))
      t.start()
      logging.info(
          'Ps {} started a thread for parameter sync!'.format(ps_index))

    # Try to dequeue, if success means chief will finish soon.
    sess.run(queue.dequeue())
    finished = True
    if t:
      t.join()
    logging.info("Ps {} shutdown successfully!".format(ps_index))


def _get_blocked_addrs(cluster: Dict, ignored_jobs: Set = {}):
  cluster_spec = tf.train.ClusterSpec(cluster)
  addrs = set()
  for job in cluster_spec.jobs:
    if job not in ignored_jobs:
      for addr in cluster_spec.job_tasks(job):
        addrs.add(addr)
  return list(addrs)


class NodeAliveCheckerError(Exception):

  def __init__(self, msg):
    super(NodeAliveCheckerError, self).__init__(self)
    self.msg = msg

  def __str__(self):
    return self.msg


def _do_worker_train(config: DistributedCpuTrainingConfig,
                     params: InstantiableParams, cluster: Dict, task: Dict):
  params.mode = config.mode
  native_task = params.instantiate()
  if not isinstance(native_task, NativeTask):
    raise ValueError(
        "distributed train only support NativeTask. Got {}".format(native_task))
  if params.serving.with_remote_gpu and config.enable_model_dump:
    # 当with_remote_gpu=True时，export时会在dense subgraph中运行model_fn
    # 导致dump下来的infer model不完整，缺少serving_input_receiver_fn信息
    raise ValueError("unsupport enable_model_dump while with_remote_gpu=True")

  session_config = cluster_manager.generate_session_config((cluster, task))
  session_config.operation_timeout_in_ms = config.operation_timeout_in_ms
  check_addrs = _get_blocked_addrs(cluster=cluster, ignored_jobs={'worker'})
  alive_checker = net_utils.NodeAliveChecker(check_addrs, timeout=60)
  if not alive_checker.all_nodes_alive():
    raise NodeAliveCheckerError("{} is unreachable".format(','.join(
        alive_checker.get_dead_nodes())))
  os.environ["TF_CONFIG"] = json.dumps({"cluster": cluster, "task": task})
  try:
    run_config = tf.estimator.RunConfig(
        model_dir=config.model_dir,
        session_config=session_config,
        save_summary_steps=100 * config.num_workers,
        log_step_count_steps=100 * config.num_workers,
        session_creation_timeout_secs=config.session_creation_timeout_secs)

    training = CpuTraining(config, native_task)
    estimator = tf.estimator.Estimator(training.create_model_fn(),
                                       config=run_config)

    if is_chief(config):
      _save_debugging_info(config, cluster, training)

    estimator.train(training.create_input_fn(config.mode),
                    max_steps=params.train.max_steps)
  finally:
    # TODO(leqi.zou): we have some thread safety issue in the test.
    if "TF_CONFIG" in os.environ:
      del os.environ["TF_CONFIG"]

  return estimator


_EXTRA_PS_BENCHMARK_SECS = 120


def _run_ps_benchmark(config: DistributedCpuTrainingConfig,
                      num_ps_required: int, cluster: dict, task: dict):
  config = copy.deepcopy(config)
  cluster = copy.deepcopy(cluster)
  bm_params = ps_benchmark.PsBenchMarkTask.params()
  ps_list = copy.copy(cluster["ps"])
  bm_params.bm_config = ps_benchmark.BenchmarkConfig(
      ps_list=ps_list,
      num_ps_required=num_ps_required,
      num_workers=config.num_workers,
      index=config.index,
      benchmark_secs=_EXTRA_PS_BENCHMARK_SECS)
  config.num_ps += config.num_extra_ps
  config.model_dir = os.path.join(config.model_dir, "benchmark_dir")
  config.operation_timeout_in_ms = int(_EXTRA_PS_BENCHMARK_SECS * 1000 +
                                       30 * 1000)
  logging.info("Run PS benchmark")
  _do_worker_train(config, bm_params, cluster, task)
  cluster["ps"] = ps_list
  return cluster


def _save_debugging_info(config: DistributedCpuTrainingConfig, cluster: dict,
                         training: CpuTraining):
  debugging_info = debugging_info_pb2.DebuggingInfo()
  debugging_info.cluster.chief_addr = cluster["chief"][0]
  for addr in cluster["ps"]:
    debugging_info.cluster.ps_addrs.append(addr)
  debugging_info.num_workers = config.num_workers
  for k, v in training.feature_configs[0].items():
    feature_name_config = debugging_info.feature_name_configs.add()
    feature_name_config.feature_name = k
    feature_name_config.config_str = str(v)

  debugging_info_file_name = utils.get_debugging_info_file_name(
      config.model_dir)
  tf.io.gfile.makedirs(os.path.dirname(debugging_info_file_name))
  file_io.atomic_write_string_to_file(debugging_info_file_name,
                                      debugging_info.SerializeToString())


def _get_replica_device_setter(config):
  if config.task_type:
    worker_device = '/job:%s/task:%d' % (config.task_type, config.task_id)
  else:
    worker_device = '/job:worker'

  if config.num_ps_replicas > 0:
    from tensorflow.python.training import device_setter
    return tf.compat.v1.train.replica_device_setter(
        ps_tasks=config.num_ps_replicas,
        worker_device=worker_device,
        merge_devices=True,
        ps_ops=list(device_setter.STANDARD_PS_OPS),
        cluster=config.cluster_spec)
  else:
    return None


def _do_worker_feature_engineering(target, config: DistributedCpuTrainingConfig,
                                   params: InstantiableParams, cluster: Dict,
                                   task: Dict):
  logging.info("do worker feature engineering. mode: %s.", config.mode)
  params.mode = config.mode
  native_task = params.instantiate()
  if not isinstance(native_task, NativeTask):
    raise ValueError(
        "distributed train only support NativeTask. Got {}".format(native_task))

  session_config = cluster_manager.generate_session_config((cluster, task))
  session_config.operation_timeout_in_ms = config.operation_timeout_in_ms
  check_addrs = _get_blocked_addrs(cluster=cluster, ignored_jobs={'worker'})
  alive_checker = net_utils.NodeAliveChecker(check_addrs, timeout=60)
  if not alive_checker.all_nodes_alive():
    raise NodeAliveCheckerError("{} is unreachable".format(','.join(
        alive_checker.get_dead_nodes())))
  os.environ["TF_CONFIG"] = json.dumps({"cluster": cluster, "task": task})

  run_config = tf.estimator.RunConfig(
      model_dir=config.model_dir,
      session_config=session_config,
      save_summary_steps=100 * config.num_workers,
      log_step_count_steps=100 * config.num_workers,
      session_creation_timeout_secs=config.session_creation_timeout_secs)

  device_fn = _get_replica_device_setter(run_config)

  with tf.Graph().as_default() as g, g.device(device_fn):
    dataset = native_task.input_fn(config.mode)
    itr = tf.compat.v1.data.make_initializable_iterator(dataset)
    nxt_elem = itr.get_next()

    # TODO(ltli): 后续转 ExampleBatch 的逻辑转移到 C++，效率更高
    fe_save_hook = feature_engineering_hooks.FeatureEngineeringSaveHook(
        config, nxt_elem)
    with tf.compat.v1.train.MonitoredTrainingSession(
        target, hooks=[fe_save_hook], config=session_config) as sess:
      sess.run(itr.initializer)
      while not sess.should_stop():
        sess.run(nxt_elem)

  if "TF_CONFIG" in os.environ:
    del os.environ["TF_CONFIG"]

  logging.info("finish worker feature engineering. mode: %s.", config.mode)
  return None


def make_config_backward_compatible(model_dir: str, config: CpuTrainingConfig):
  # Will remove this compatible logic after 1/1/2023
  if config.use_native_multi_hash_table is None:
    monolith_ckpt = save_utils.get_monolith_checkpoint_state(model_dir)
    if monolith_ckpt is not None and monolith_ckpt.builtin_hash_table_type in (
        monolith_checkpoint_state_pb2.MonolithCheckpointState.UNKNOWN,
        monolith_checkpoint_state_pb2.MonolithCheckpointState.CUCKOO_HASH_MAP):
      config.use_native_multi_hash_table = False


def distributed_train(config: DistributedCpuTrainingConfig,
                      discovery: ServiceDiscovery,
                      params: InstantiableParams,
                      sync_backend: SyncBackend = None):
  """Trains the server in a distributed fashion."""
  if config.index is None:
    raise ValueError("Index can't be none.")
  if config.num_ps is None:
    raise ValueError("Num ps can't be none.")
  if config.num_workers is None:
    raise ValueError("Num workers can't be none.")
  if not config.server_type in ["ps", "worker"]:
    raise ValueError("Unknown server type. type: {}".format(config.server_type))
  if not config.model_dir:
    raise ValueError("model dir can't be empty.")

  if is_chief(config):
    FLAGS.monolith_alert_proto = FLAGS.monolith_chief_alert_proto

  make_config_backward_compatible(config.model_dir, config)

  server_config = tf.compat.v1.ConfigProto(
      intra_op_parallelism_threads=config.intra_op_parallelism_threads,
      inter_op_parallelism_threads=config.inter_op_parallelism_threads)
  if isinstance(discovery, TfConfigServiceDiscovery):
    addr = discovery.addr
    server = tf.distribute.Server({"local": [addr]}, config=server_config)
  else:
    assert isinstance(discovery, ServiceDiscovery)
    # We prefer to use another nic since the default client will always use eth0.
    ip = yarn_runtime.get_local_host()
    server = tf.distribute.Server({"local": [ip + ":0"]}, config=server_config)
    addr = urlparse(server.target).netloc

  _prepare_server(server.target, config)
  discovery.register(config.server_type, config.index, addr)
  logging.info("Started %s %d at %s.", config.server_type, config.index, addr)

  estimator = None

  if config.server_type == "ps":
    if not config.model_name:
      if isinstance(params, InstantiableParams):
        default_name = f'di_name_{params.cls.__name__}'
      else:
        default_name = f'di_name_{params.__class__.__name__}'
      config.model_name = params.metrics.deep_insight_name or default_name
    with native_task_context.with_ctx(
        make_native_task_context(config, sync_backend)):
      _join_ps(server.target, config.index, sync_backend)
  elif config.server_type == "worker":
    num_retries, worker_failover_cnt = 0, 0
    max_retries = config.max_retry_times or (6
                                             if config.partial_recovery else 0)
    cluster, task = {}, {}

    num_required_ps = config.num_ps + config.num_extra_ps

    def _get_cluster_and_task():
      cluster, task = cluster_manager.get_training_cluster(
          discovery, addr, config.index, config.num_redundant_ps,
          num_required_ps, config.num_workers, config.model_dir, config.uuid,
          params.metrics.deep_insight_name, config.cluster_type)
      filtered_cluster = copy.copy(cluster)
      if config.submit_time_secs and config.index == 0 and params.metrics.deep_insight_name:
        container_ready_elapsed_time = int(
            time.time()) - config.submit_time_secs
        logging.info(
            "Containers ready took {}s.".format(container_ready_elapsed_time))
        tags = {
            "model_name": config.model_name or params.metrics.deep_insight_name,
            "cluster_type": config.cluster_type
        }
        cli.get_cli(utils.get_metric_prefix()).emit_timer(
            "container_ready_elapsed_time.all", container_ready_elapsed_time,
            tags)
        config.containers_ready_time_secs = int(time.time())
      if config.num_extra_ps:
        filtered_cluster = _run_ps_benchmark(config, config.num_ps,
                                             filtered_cluster, task)
      return filtered_cluster, task

    cluster, task = _get_cluster_and_task()
    captured_exception = None
    start_ts = datetime.timestamp(datetime.now())
    logging.info("Worker Start %s", str(start_ts))
    logging.info("only_feature_engineering: {}.".format(
        config.only_feature_engineering))
    try:
      while True:
        try:
          if config.only_feature_engineering:
            estimator = _do_worker_feature_engineering(server.target, config,
                                                       params, cluster, task)
          else:
            estimator = _do_worker_train(config, params, cluster, task)
          break
        except (tf.errors.DeadlineExceededError, tf.errors.UnavailableError,
                NodeAliveCheckerError) as e:
          worker_failover_cnt += 1
          tags = {
              "model_name": config.model_name,
              "worker_index": str(config.index)
          }
          cli.get_cli(utils.get_metric_prefix()).emit_timer(
              "worker_failover_cnt",
              f'worker_failover_cnt: {worker_failover_cnt}, msg: {e}', tags)
          time.sleep(config.retry_wait_in_secs)
          old_cluster = cluster
          cluster, task = _get_cluster_and_task()
          if cluster == old_cluster:
            logging.info('Temporary error: %s. Retrying...', str(e))
            continue
          num_retries += 1
          if num_retries <= max_retries:
            logging.error(
                'error is "{}", we try to the {}-th retry, sleep for {} seconds!'
                .format(e, num_retries, config.retry_wait_in_secs))
          else:
            captured_exception = e
            raise e
        except Exception as e:
          captured_exception = e
          raise e
    finally:
      if is_chief(config):
        try:
          if config.num_redundant_ps or config.num_extra_ps:
            num_required_ps += config.num_redundant_ps
            # Query the total ps cluster for shutdown.
            cluster, task = cluster_manager.get_training_cluster(
                discovery, addr, config.index, config.num_redundant_ps,
                num_required_ps, config.num_workers, config.model_dir,
                config.uuid)
          # In the realtime training, we want to keep ps alive so we can
          # restart chief without side effect.
          if not config.enable_realtime_training or config.force_shutdown_ps:
            _shutdown_ps(server.target, cluster, task, num_required_ps)
        finally:
          if captured_exception is None:
            yarn_runtime.maybe_finish_application()
          else:
            success = yarn_runtime.maybe_kill_application(
                str(captured_exception))
    end_ts = datetime.timestamp(datetime.now())
    logging.info("Worker End %s, Cost: %s(s)", str(end_ts),
                 str(end_ts - start_ts))

  logging.info("Finished %s %d.", config.server_type, config.index)
  return estimator


def distributed_sync_train(config: DistributedCpuTrainingConfig,
                           params: InstantiableParams,
                           sync_backend: SyncBackend = None):
  """
  This is the entry point for synchronous distributed training.
  This system allows the model to train in a half sync manner as well, when set
  embedding_prefetch_capacity value > 0.

  All the dense parameters are synced via allreduce and no asynchronicity is
  allowed for dense paramters.

  No Worker num is needed, the system derives the number of workers via MPI API.

  Args:
    config: the configs for monolith cpu training.
    params: the parameters for the model and other modules.

  """

  enable_bps = int(os.getenv("MONOLITH_WITH_BYTEPS", "0"))
  # Import horovod/byteps on demand, which should've already been initialized.
  if enable_bps:
    import byteps.tensorflow as hvd
  else:
    import horovod.tensorflow as hvd
  assert hvd.rank(
  ) == config.index, "Given RunConfig.index should be consistent with hvd.rank()."

  # To remove this contraint future
  if config.enable_gpu_training:
    params.train.use_gpu_emb_table = True

  task = params.instantiate()
  training = CpuTraining(config, task)
  session_config = tf.compat.v1.ConfigProto(allow_soft_placement=False,
                                            log_device_placement=False)
  # CPU Configs
  session_config.intra_op_parallelism_threads = config.intra_op_parallelism_threads
  session_config.inter_op_parallelism_threads = config.inter_op_parallelism_threads
  # GPU Configs
  if config.enable_gpu_training:
    device_utils.enable_gpu_training()
    if os.environ.get('MONOLITH_FORCE_GPU_COMPATIBLE', '1') == '1':
      session_config.gpu_options.force_gpu_compatible = True
      logging.info("set force_gpu_compatible=True")
    if enable_bps and (os.environ.get('MONOLITH_WITH_BYTEPS_FWD_GDR', '0') == '1' or \
       os.environ.get('MONOLITH_WITH_BYTEPS_BWD_GDR', '0') == '1'):
      # if GDR alltoall is enabled, GPU memory need to be registered for UCX
      # ahead of time. Therefore, we disable the allow_growth option for GPU.
      # The cuda visible devices are also limited to one device only.
      session_config.gpu_options.allow_growth = False
      session_config.gpu_options.per_process_gpu_memory_fraction = 0.4
      session_config.gpu_options.visible_device_list = '0'
    else:
      session_config.gpu_options.allow_growth = True
      session_config.gpu_options.visible_device_list = '0'
  # By default the grappler (meta_optimizer) is enabled.
  # session_config.graph_options.rewrite_options.disable_meta_optimizer = True
  session_config.graph_options.rewrite_options.memory_optimization = 1

  # We reduce the frequency of saving to HDFS summary, otherwise it slows down
  # the training.
  # TODO(zouxuan): always use the TF v2 summary with flush will fix this issue.

  class Nop(object):

    def nop(*args, **kwargs):
      pass

    def __getattr__(self, _):
      return self.nop

  # only rank 0 writes events
  if config.index != 0:
    SummaryWriterCache.get = lambda _: Nop()

  run_config = tf.estimator.RunConfig(
      model_dir=config.model_dir,
      device_fn=device_utils.default_device_fn,
      session_config=session_config,
      save_summary_steps=None if config.index != 0 else int(
          os.environ.get('MONOLITH_SAVE_SUMMARY_INTERVAL', '1000000')),
      log_step_count_steps=params.train.max_steps if config.index != 0 else int(
          os.environ.get('MONOLITH_ROOT_LOG_INTERVAL', '100')))
  if not isinstance(task, NativeTask):
    raise ValueError(
        "distributed train only support NativeTask. Got {}".format(task))

  estimator = tf.estimator.Estimator(training.create_model_fn(),
                                     config=run_config)
  enable_bps_bcast = int(os.getenv("MONOLITH_WITH_BYTEPS_BCAST", "1"))
  if enable_bps and enable_bps_bcast == -1:
    run_hooks = []
  elif enable_bps and enable_bps_bcast:
    import byteps.tensorflow as bps
    logging.info('Enabled BPS for bcast')
    run_hooks = [bps.BroadcastGlobalVariablesHook(0)]
    run_hooks.append(ByteCCLTelemetryHook(50))
  else:
    import horovod.tensorflow as hvd
    run_hooks = [hvd.BroadcastGlobalVariablesHook(0)]

  # When we use distributed training, we always use rank 1 to profile
  # because rank 0 might not get embedding shard due to partition logic
  # for np >= 4.
  if hvd.size() == 1 or hvd.rank() == 1:
    tf.profiler.experimental.server.start(6666)
    if config.profile_some_steps_from:
      start_step = config.profile_some_steps_from
      end_step = start_step + 10
      options = tf.profiler.experimental.ProfilerOptions(
          host_tracer_level=2,
          python_tracer_level=1,
          # CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED:
          # CUPTI doesn't allow multiple callback subscribers.
          # Only a single subscriber can be registered at a time.
          device_tracer_level=0 if config.profile_with_nvprof_from_to else 1)
      prof_hook = Tf2ProfilerCaptureOnceHook(logdir=config.model_dir,
                                             capture_step_range=(start_step,
                                                                 end_step),
                                             options=options)
      run_hooks.append(prof_hook)

    if config.profile_with_nvprof_from_to:
      s, e = config.profile_with_nvprof_from_to.split(',')
      run_hooks.append(
          NVProfilerCaptureOnceHook(capture_step_range=[int(s), int(e)]))

  if sync_backend is not None:
    run_hooks.append(
        sync_training_hooks.ParameterSyncHook(sync_backend, config.index))
  run_hooks.append(sync_training_hooks.SyncTrainingInfoHook())

  estimator.train(training.create_input_fn(tf.estimator.ModeKeys.TRAIN),
                  hooks=run_hooks,
                  max_steps=params.train.max_steps)

  logging.info("Finished worker %d.", config.index)
  return estimator


def local_train_internal(params: InstantiableParams,
                         conf: CpuTrainingConfig,
                         model_dir: str,
                         steps: int = 100,
                         profiling: bool = False) -> tf.estimator.Estimator:
  """Do a local training. Especially useful in the local demo."""
  if tf.compat.v1.executing_eagerly():
    raise EnvironmentError(
        "Local train is not supported in the eager mode. Please call `tf.compat.v1.disable_eager_execution()`"
    )

  task = params.instantiate()

  if conf.num_ps <= 0:
    session_config = tf.compat.v1.ConfigProto()
    training = CpuTraining(conf, task)
    if "TF_CONFIG" in os.environ:
      del os.environ["TF_CONFIG"]
  else:
    training = CpuTraining(conf, task)
    ps_servers = []
    for _ in range(conf.num_ps):
      ps_servers.append(tf.distribute.Server.create_local_server())
    master = tf.distribute.Server.create_local_server()

    def get_addr(server: tf.distribute.Server):
      return server.target[len('grpc://'):]

    cluster = {
        "chief": [get_addr(master)],
        "ps": [get_addr(server) for server in ps_servers],
    }
    os.environ["TF_CONFIG"] = json.dumps({
        "cluster": cluster,
        "task": {
            "type": "chief",
            "index": 0
        }
    })

    spec = tf.train.ClusterSpec(cluster)
    session_config = tf.compat.v1.ConfigProto(
        cluster_def=spec.as_cluster_def(),
        allow_soft_placement=True,
        share_cluster_devices_in_session=True)

    session_config.experimental.share_session_state_in_clusterspec_propagation = True
  # grappler doesn't really understand RaggedTensor.
  session_config.graph_options.rewrite_options.disable_meta_optimizer = True

  config = tf.estimator.RunConfig(model_dir=model_dir,
                                  session_config=session_config)
  estimator = tf.estimator.Estimator(training.create_model_fn(), config=config)
  estimator.train(training.create_input_fn(tf.estimator.ModeKeys.TRAIN),
                  steps=steps)

  if "TF_CONFIG" in os.environ:
    del os.environ["TF_CONFIG"]

  return estimator


def local_feature_engineering_internal(
    params: InstantiableParams,
    conf: CpuTrainingConfig,
    model_dir: str,
    profiling: bool = False) -> tf.estimator.Estimator:
  """Do a local feature engineer. Especially useful in the local demo."""
  if tf.compat.v1.executing_eagerly():
    raise EnvironmentError(
        "Local train is not supported in the eager mode. Please call `tf.compat.v1.disable_eager_execution()`"
    )

  task = params.instantiate()

  if conf.num_ps <= 0:
    session_config = tf.compat.v1.ConfigProto()
    if "TF_CONFIG" in os.environ:
      del os.environ["TF_CONFIG"]
  else:
    ps_servers = []
    for _ in range(conf.num_ps):
      ps_servers.append(tf.distribute.Server.create_local_server())
    master = tf.distribute.Server.create_local_server()

    def get_addr(server: tf.distribute.Server):
      return server.target[len('grpc://'):]

    cluster = {
        "chief": [get_addr(master)],
        "ps": [get_addr(server) for server in ps_servers],
    }
    os.environ["TF_CONFIG"] = json.dumps({
        "cluster": cluster,
        "task": {
            "type": "chief",
            "index": 0
        }
    })

    spec = tf.train.ClusterSpec(cluster)
    session_config = tf.compat.v1.ConfigProto(
        cluster_def=spec.as_cluster_def(),
        allow_soft_placement=True,
        share_cluster_devices_in_session=True)

    session_config.experimental.share_session_state_in_clusterspec_propagation = True
  # grappler doesn't really understand RaggedTensor.
  session_config.graph_options.rewrite_options.disable_meta_optimizer = True

  if not model_dir:
    model_dir = "/tmp/{}/{}".format(getpass.getuser(), params.name)
  run_config = tf.estimator.RunConfig(model_dir=model_dir,
                                      session_config=session_config)

  device_fn = _get_replica_device_setter(run_config)

  if profiling:
    tf.profiler.experimental.start(model_dir)

  with tf.Graph().as_default() as g, g.device(device_fn):
    dataset = task.input_fn(conf.mode)
    itr = tf.compat.v1.data.make_initializable_iterator(dataset)
    nxt_elem = itr.get_next()

    fe_save_hook = feature_engineering_hooks.FeatureEngineeringSaveHook(
        conf, nxt_elem)
    with tf.compat.v1.train.MonitoredTrainingSession(
        master.target, hooks=[fe_save_hook], config=session_config) as sess:
      sess.run(itr.initializer)
      while not sess.should_stop():
        sess.run(nxt_elem)

  if profiling:
    tf.profiler.experimental.stop()

  if "TF_CONFIG" in os.environ:
    del os.environ["TF_CONFIG"]

  return None


def local_train(params: InstantiableParams,
                num_ps=0,
                model_dir: str = None,
                steps=100,
                save_checkpoints_steps=50,
                profiling=False,
                enable_embedding_prefetch: bool = True,
                enable_embedding_postpush: bool = True,
                remove_model_dir_if_exists: bool = True,
                only_feature_engineering: bool = False):
  embedding_prefetch_capacity = 1 if enable_embedding_prefetch else 0
  conf = CpuTrainingConfig(
      model_name=params.name,
      num_ps=num_ps,
      embedding_prefetch_capacity=embedding_prefetch_capacity,
      enable_embedding_postpush=enable_embedding_postpush,
      save_checkpoints_steps=save_checkpoints_steps)
  if not model_dir:
    model_dir = "/tmp/{}/{}".format(getpass.getuser(), params.name)
  if remove_model_dir_if_exists:
    try:
      tf.io.gfile.rmtree(model_dir)
    except tf.errors.NotFoundError:
      pass
  make_config_backward_compatible(model_dir, conf)
  if only_feature_engineering:
    return local_feature_engineering_internal(params, conf, model_dir,
                                              profiling)
  else:
    return local_train_internal(params, conf, model_dir, steps, profiling)
