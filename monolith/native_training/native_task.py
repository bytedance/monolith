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
import dataclasses
from typing import Any, Callable, Dict, Iterable, Tuple, Union

import tensorflow as tf

from monolith.core import hyperparams
from monolith.core.base_task import BaseTask
from monolith.native_training import feature
from monolith.native_training import prefetch_queue
from monolith.native_training.model_export.export_context import ExportMode
from idl.matrix.proto.example_pb2 import OutConfig, OutType, TensorShape


class NativeContext:
  """Provides the context of the NativeTask."""

  def __init__(self,
               feature_factory: feature.FeatureFactory = None,
               async_function_mgr: prefetch_queue.AsyncFunctionMgr = None,
               layout_factory: feature.EmbeddingLayoutFactory = None):
    self.feature_factory = feature_factory
    self.async_function_mgr = async_function_mgr
    self.layout_factory = layout_factory
    if layout_factory and feature_factory:
      raise ValueError(
          "Cannot set feature_factory and layout_factory in the same time")

  # Provides some convinient functions

  def create_feature_slot(
      self, config: feature.FeatureSlotConfig) -> feature.FeatureSlot:
    """Creates a feature slot."""
    # No TensorFlow op is created at this function call.
    if self.layout_factory:
      return self.layout_factory.create_feature_slot(config)
    else:
      return self.feature_factory.create_feature_slot(config)

  def apply_embedding_gradients(self,
                                grads_and_vars: Iterable[Tuple[tf.Tensor,
                                                               tf.Tensor]]):
    """
    Apply gradients for embeddings. Notice vars must be coming from FeatureColumn's
    get_all_embeddings_concatenated.
    """
    if self.layout_factory:
      return self.layout_factory.apply_gradients(grads_and_vars)
    else:
      return self.feature_factory.apply_gradients(grads_and_vars)

  def add_async_function(
      self,
      target: Callable,
      args: Tuple = None,
      kwargs: Dict = None,
      is_async: bool = None,
      queue_name: str = "async_queue") -> Union[tf.Operation, Any]:
    """Adds async func.
    Returns an enqueue op if is_async. Otherwise, returns calling result of target.

    Args:
      is_async - if not specified, will use default value in async_function_mgr.

    Requirements: 
    - target should return ops/tensors which can be added to session.run
    All tensors used by |async_function| should *ONLY* come from arguments passed in.
    Otherwise, we may use updated value in the async function.
    TODO(leqi.zou): Adds a check for this."""
    return self.async_function_mgr.add_async_function(target,
                                                      args,
                                                      kwargs,
                                                      is_async=is_async,
                                                      queue_name=queue_name)


class NativeTask(BaseTask, abc.ABC):
  """
  A task is supported to be train/eval/serving in multiple devices with native tensorflow
  code.
  """

  @classmethod
  def params(cls):
    p = super(NativeTask, cls).params()
    # metrics
    p.define("metrics", hyperparams.Params(), "Metric parameters.")
    p.metrics.define("enable_deep_insight", False,
                     'Whether enable deep insight.')
    p.metrics.define("deep_insight_target", "ctr_head", "Deep insight target.")
    p.metrics.define('deep_insight_name', None, 'str')
    p.metrics.define('deep_insight_sample_ratio', 0.01, 'float')
    p.metrics.define('extra_fields_keys', [],
                     'extra_fields_keys for deepinsight, List[str]')

    # [todo] (fitz) the mode will remove when the estimator is ready
    p.define("mode", tf.estimator.ModeKeys.TRAIN, "run mode")

    p.metrics.define("enable_throughput_hook", True,
                     "If enables throughput hook.")
    p.metrics.define("enable_kafka_metrics", False, "enable_kafka_metrics")
    p.metrics.define(
        "enable_tf2_profiler_hook", False,
        "If enables tf profiler hook. When enabled, remeber to increase worker's memory."
    )

    p.metrics.define("enable_file_metrics", False, "enable_file_metrics")
    p.metrics.define("file_base_name", '/vepfs/jaguar_deepinsight_results',
                     "file_base_name")
    p.metrics.define("file_ext", 'txt', "file_ext")
    p.metrics.define("parse_fn", None, "parse_fn")
    p.metrics.define("key_fn", None, "key_fn")
    p.metrics.define("layout_fn", None, "layout_fn")

    p.train.define(
        'max_pending_seconds_for_barrier', 30,
        'Maximum waiting time for barrier block. Used for testing in most cases.'
    )
    p.train.define(
        "slow_start_steps", 0,
        ("How many steps will worker wait before they start to train."
         " The formula of wait is `slow_start_steps * log(1 + index)`"))
    p.train.define(
        "sample_bias", 0.,
        "Sample bias is a float scalar which acts as compensation for ads "
        "realtime training (FastEmit training instance).")
    p.train.define("use_gpu_emb_table", False,
                   "Use GPU embedding table for sync training if enabled.")
    p.train.define("use_fountain", False,
                   "Use fountain data service if enabled.")
    p.train.define("fountain_zk_host", "", "zk_host for fountain service.")
    p.train.define("fountain_model_name", "",
                   "model_name for fountain service.")
    p.train.define("fountain_parse_on_server", False,
                   "Parsing logic on fountain server.")
    p.train.define("fountain_precompute_value_rowids", False,
                   "Parsing logic on fountain server.")

    p.define("serving", hyperparams.Params(), "Serving parameters.")
    p.serving.define(
        "export_with_gpu_allowed", False,
        "When true it allows cpu/gpu training to export model graph "
        "with specified gpu device placement contexts.")
    p.serving.define(
        "export_with_cleared_entry_devices", False,
        "When true it clears the devices in the exported model graph"
        "for entry only at DistributedExporter Mode.")
    p.serving.define(
        "export_when_saving", False,
        "When true, a valid create_serving_input_fn must be provided. The "
        "framework will do export when saving. ")
    p.serving.define(
        "export_dir_base", "exported_models",
        "The base dir (either relative to model_dir or an absolute path) When "
        "exporting models.")
    p.serving.define("export_mode", ExportMode.DISTRIBUTED,
                     "standalone or distributed.")
    p.serving.define(
        "shared_embedding", True,
        "If true, instead of exporting a hermetic SavedModel, we will use the "
        "embedding in checkpoints instead of copying it.")
    p.serving.define("with_remote_gpu", False,
                     "If true, the whole dense will be put on the GPU.")

    return p

  def __init__(self, params):
    super().__init__(params)
    self._ctx = NativeContext()
    self.p = params

  @property
  def ctx(self) -> NativeContext:
    """Returns task ctx."""
    return self._ctx

  @abc.abstractmethod
  def create_input_fn(self, mode):
    """
    Same as BaseTask.create_input_fn
    """

  @abc.abstractmethod
  def create_model_fn(self):
    """
    For the child class, returned model_fn must follow the signature of
    (features, mode, config) -> SomeEstimatorSpec
    """

  def create_serving_input_receiver_fn(self):
    """Returns a serving input fn for serving. 
    See https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#export_saved_model
    for the possible return values for this method.
    By default, None is provided (which is invalid if we enable serving).
    """
    return None
