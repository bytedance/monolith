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

from posixpath import split
from monolith.native_training.distributed_serving_ops import remote_predict
from monolith.native_training.utils import with_params
from absl import logging, flags
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
from functools import partial
import os, math, time
import hashlib
from typing import Tuple, Dict, Iterable, Union, Optional
import numpy as np

import tensorflow as tf
from tensorflow.estimator.export import ServingInputReceiver
from tensorflow.python.data.ops.dataset_ops import DatasetV2
from tensorflow.python.framework import ops
from tensorflow.python.saved_model.signature_constants import DEFAULT_SERVING_SIGNATURE_DEF_KEY

from monolith.core import hyperparams
from monolith.native_training.entry import *
from monolith.native_training.feature import *
from monolith.core.base_layer import get_layer_loss
from monolith.core.hyperparams import update_params

from monolith.native_training import distribution_ops
from monolith.native_training import file_ops
from monolith.native_training import hash_table_ops
from monolith.native_training.native_task_context import get
import monolith.native_training.feature_utils as feature_utils
from monolith.native_training.estimator import EstimatorSpec
from monolith.native_training.embedding_combiners import FirstN
from monolith.native_training.layers import LogitCorrection
from monolith.native_training.native_task import NativeTask, NativeContext
from monolith.native_training.metric import utils as metric_utils
from monolith.native_training.model_export import export_context
from monolith.native_training.model_export.export_context import is_exporting, is_exporting_distributed
from monolith.native_training.data.feature_list import FeatureList, get_feature_name_and_slot
from monolith.native_training.monolith_export import monolith_export
from monolith.native_training.runtime.hash_table import \
    embedding_hash_table_pb2
from monolith.native_training.data.utils import get_slot_feature_name, enable_to_env
from monolith.native_training.utils import add_to_collections
from monolith.native_training.model_dump.dump_utils import DumpUtils
from monolith.native_training.dense_reload_utils import CustomRestoreListener, CustomRestoreListenerKey
from monolith.native_training.layers.utils import dim_size
from monolith.native_training.metric.metric_hook import KafkaMetricHook, FileMetricHook, vepfs_key_fn, vepfs_layout_fn
from idl.matrix.proto.example_pb2 import OutConfig, OutType, TensorShape
from monolith.native_training.data.datasets import POOL_KEY
from monolith.native_training.model_dump.graph_utils import _node_name

FLAGS = flags.FLAGS
dump_utils = DumpUtils(enable=False)


@monolith_export
def get_sigmoid_loss_and_pred(
    name,
    logits,
    label,
    batch_size: int,
    sample_rate: Union[tf.Tensor, float] = 1.0,
    sample_bias: bool = False,
    mode: tf.estimator.ModeKeys = tf.estimator.ModeKeys.TRAIN,
    instance_weight: tf.Tensor = None,
    mask: tf.Tensor = None,
    logit_clip_threshold: Optional[float] = None,
    predict_before_correction: bool = True):
  """对二分类, 基于sigmoid计算loss和predict

  由于负例采样, fast_emit等原因, 需要对logit进进较正, 在get_sigmoid_loss_and_pred会透明地进行

  Args:
    name (:obj:`str`): 名称
    logits (:obj:`tf.Tensor`): 样本logits(无偏logit), 可用于直接predict, 但是不能用于直接计算loss
    label (:obj:`tf.Tensor`): 样本标签
    batch_size (:obj:`int`): 批大小
    sample_rate (:obj:`tf.Tensor`): 负例采样的采样率
    sample_bias (:obj:`bool`): 是否有开启fast_emit
    mode (:obj:`str`): 运行模式, 可以是train/eval/predict等
    mask (:obj:`tf.Tensor`): Apply boolean mask to loss before reduce_sum

  """

  logits = tf.reshape(logits, shape=(-1,))
  batch_size = dim_size(logits, 0)
  if mode != tf.estimator.ModeKeys.PREDICT:
    if sample_rate is not None and isinstance(sample_rate, float):
      sample_rate = tf.fill(dims=(batch_size,), value=sample_rate)
    if sample_rate is None:
      sample_rate = tf.fill(dims=(batch_size,), value=1.0)
    src = LogitCorrection(activation=None,
                          sample_bias=sample_bias,
                          name='sample_rate_correction')
    logits_biased = src((logits, sample_rate))
    if predict_before_correction:
      pred = tf.nn.sigmoid(logits, name='{name}_sigmoid_pred'.format(name=name))
    else:
      pred = tf.nn.sigmoid(logits_biased,
                           name='{name}_sigmoid_pred'.format(name=name))

    if logit_clip_threshold is not None:
      assert 0 < logit_clip_threshold < 1
      threshold = math.log((1 - logit_clip_threshold) / logit_clip_threshold)
      logits_biased = tf.clip_by_value(logits_biased, -threshold, threshold)

    losses = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.reshape(label, shape=(-1,)),
        logits=logits_biased,
        name='{name}_sigmoid_loss'.format(name=name))
    if instance_weight is not None:
      instance_weight = tf.reshape(instance_weight, shape=(-1,))
    if mask is not None:
      mask = tf.reshape(mask, shape=(-1,))
      losses = tf.boolean_mask(losses, mask)
      if instance_weight is not None:
        instance_weight = tf.boolean_mask(instance_weight, mask)
    if instance_weight is not None:
      losses = tf.multiply(losses, instance_weight)
    loss = tf.reduce_sum(losses)
  else:
    loss = None
    pred = tf.nn.sigmoid(logits, name='{name}_sigmoid_pred'.format(name=name))

  return loss, pred


@monolith_export
def get_softmax_loss_and_pred(name, logits, label, mode):
  """对多分类, 基于softmax计算loss和predict

  Args:
    name (:obj:`str`): 名称
    logits (:obj:`tf.Tensor`): 样本logits
    label (:obj:`tf.Tensor`): 样本标签
    mode (:obj:`str`): 运行模式, 可以是train/eval/predict等

  """

  pred = tf.argmax(tf.nn.softmax(logits,
                                 name='{name}_softmax_pred'.format(name=name)),
                   axis=1)
  if mode != tf.estimator.ModeKeys.PREDICT:
    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=label,
        logits=logits,
        name='{name}_softmax_loss'.format(name=name))
  else:
    loss = None

  return loss, pred




@monolith_export
class MonolithBaseModel(NativeTask, ABC):
  """模型开发的基类"""

  @classmethod
  def params(cls):
    p = super(MonolithBaseModel, cls).params()
    p.define("output_path", None, "The output path of predict/eval")
    p.define("output_fields", None, "The output fields")
    p.define("delimiter", '\t', "The delimiter of output file")
    p.define('file_name', '', 'the test input file name')
    p.define('enable_grads_and_vars_summary', False,
             'enable_grads_and_vars_summary')
    p.define('dense_weight_decay', 0.0, 'dense_weight_decay')
    p.define("clip_norm", 1000.0, "float, clip_norm")
    p.define('default_occurrence_threshold', 0, 'int')
    return p

  def __init__(self, params):
    super(MonolithBaseModel, self).__init__(params)
    enable_to_env()
    self.fs_dict = {}
    self.fc_dict = {}
    # feature_name -> slice_name -> FeatureSlice(feature_slot, start, end)
    self.slice_dict = {}
    self._layout_dict = {}
    self.valid_label_threshold = 0
    self._occurrence_threshold = {}

  def __getattr__(self, name):
    if "p" in self.__dict__:
      if hasattr(self.p, name):
        return getattr(self.p, name)
      elif name == 'batch_size':
        if self.p.mode == tf.estimator.ModeKeys.EVAL:
          return self.p.eval.per_replica_batch_size
        else:
          return self.p.train.per_replica_batch_size

    if (hasattr(type(self), name) and
        isinstance(getattr(type(self), name), property)):
      return getattr(type(self), name).fget(self)
    else:
      return super(MonolithBaseModel, self).__getattr__(name)

  def __setattr__(self, key, value):
    if 'p' in self.__dict__:
      if hasattr(self.p, key):
        setattr(self.p, key, value)
        return value
      elif key == 'batch_size':
        self.p.eval.per_replica_batch_size = value
        self.p.train.per_replica_batch_size = value
        return value

    super(MonolithBaseModel, self).__setattr__(key, value)
    return value

  def __deepcopy__(self, memo):
    cls = self.__class__
    result = cls.__new__(cls)
    memo[id(self)] = result
    for name, value in self.__dict__.items():
      if name == 'dump_utils':
        result.__dict__[name] = value
      else:
        result.__dict__[name] = deepcopy(value)
    return result

  def _get_file_ops(self, features, pred):
    assert self.p.output_fields is not None
    output_path = os.path.join(self.p.output_path,
                               f"part-{get().worker_index:05d}")
    op_file = file_ops.WritableFile(output_path)
    op_fields = [features[field] for field in self.p.output_fields.split(',')]
    if isinstance(pred, (tuple, list)):
      op_fields.extend(pred)
    else:
      op_fields.append(pred)
    fmt = self.p.delimiter.join(["{}"] * len(op_fields)) + "\n"
    result = tf.map_fn(fn=lambda t: tf.strings.format(fmt, t),
                       elems=tuple(op_fields),
                       fn_output_signature=tf.string)
    write_op = op_file.append(tf.strings.reduce_join(result))
    return op_file, write_op

  def _get_real_mode(self, mode: tf.estimator.ModeKeys):
    if mode == tf.estimator.ModeKeys.PREDICT:
      return mode
    elif mode == tf.estimator.ModeKeys.TRAIN:
      return self.mode
    else:
      raise ValueError('model error!')

  def is_fused_layout(self) -> bool:
    return self.ctx.layout_factory is not None

  def instantiate(self):
    """实例化对像"""
    return self

  def add_loss(self, losses):
    """用于追加辅助loss, 如layer loss等

    Args:
      losses (:obj:`List[tf.Tensor]`): 辅助loss列表

    """

    if losses:
      if isinstance(losses, (list, tuple)):
        self.losses.extend(losses)
      else:
        self.losses.append(losses)

  @property
  def losses(self):
    graph = tf.compat.v1.get_default_graph()
    if hasattr(graph, '__losses'):
      return getattr(graph, '__losses')
    else:
      setattr(graph, '__losses', [])
      return graph.__losses

  @losses.setter
  def losses(self, losses):
    graph = tf.compat.v1.get_default_graph()
    if hasattr(graph, '__losses'):
      graph.__losses = losses
    else:
      setattr(graph, '__losses', losses)

  @property
  def _global_step(self):
    return tf.compat.v1.train.get_or_create_global_step()

  @property
  def _training_hooks(self):
    graph = tf.compat.v1.get_default_graph()
    if hasattr(graph, '__training_hooks'):
      return getattr(graph, '__training_hooks')
    else:
      setattr(graph, '__training_hooks', [])
      return graph.__training_hooks

  @_training_hooks.setter
  def _training_hooks(self, hooks):
    graph = tf.compat.v1.get_default_graph()
    if hasattr(graph, '__training_hooks'):
      graph.__training_hooks = hooks
    else:
      setattr(graph, '__training_hooks', hooks)

  def clean(self):
    # update fs_dict, fc_dict, slice_dict
    self.fs_dict = {}
    self.fc_dict = {}
    self.slice_dict = {}  # slot_id -> Dict[slot_id, slice]
    self.valid_label_threshold = 0
    self._occurrence_threshold = {}

  def create_input_fn(self, mode):
    """生成input_fn"""
    return partial(self.input_fn, mode)

  def create_model_fn(self):
    """生成model_fn"""
    self.clean()

    def model_fn_internal(
        features: Dict[str, tf.Tensor], mode: tf.estimator.ModeKeys,
        config: tf.estimator.RunConfig) -> tf.estimator.EstimatorSpec:

      real_mode = self._get_real_mode(mode)
      local_spec = self.model_fn(features, real_mode)

      # get label, loss, pred and head_name from model_fn result
      if isinstance(local_spec, EstimatorSpec):
        label, loss, pred = local_spec.label, local_spec.loss, local_spec.pred
        if isinstance(pred, dict):
          assert label is None or isinstance(label, dict)
          head_name, pred = list(zip(*pred.items()))
        else:
          head_name = local_spec.head_name or self.metrics.deep_insight_target.split(
              ',')
        is_classification = local_spec.classification
      elif isinstance(local_spec, (tuple, list)):
        label, loss, pred = local_spec
        if isinstance(pred, dict):
          assert label is None or isinstance(label, dict)
          head_name, pred = list(zip(*pred.items()))
        else:
          head_name = self.metrics.deep_insight_target
        assert head_name is not None
        is_classification = True
        logging.warning(
            'if this is not a classification task, pls. return EstimatorSpec in model_fn and specify it'
        )
      else:
        raise Exception("EstimatorSpec Error!")

      # check label/pred/head_name
      if isinstance(pred, (list, tuple, dict)):
        assert isinstance(head_name, (list, tuple))
        assert isinstance(pred, (list, tuple))
        if label is not None:
          assert len(head_name) == len(label)
          assert len(label) == len(pred)
      else:
        if isinstance(head_name, (list, tuple)):
          assert len(head_name) == 1
          head_name = head_name[0]
        assert isinstance(head_name, str)
        if label is not None:
          assert isinstance(label, tf.Tensor)
        if isinstance(pred, (list, tuple)):
          assert len(pred) == 1
          pred = pred[0]
          assert isinstance(pred, tf.Tensor)

      if label is not None:
        if isinstance(label, dict):
          label = {
              key: None if value is None else tf.identity(value, name=key)
              for key, value in label.items()
          }
        elif isinstance(label, (list, tuple)):
          label = [
              None if l is None else tf.identity(
                  l, name=f'label_{_node_name(l.name)}') for l in label
          ]
        else:
          label = label if label is None else tf.identity(
              label, name=f'label_{_node_name(label.name)}')

      dump_utils.add_model_fn(self, mode, features, label, loss, pred,
                              head_name, is_classification)

      if self.losses:
        loss = loss + tf.add_n(self.losses)

      if real_mode == tf.estimator.ModeKeys.PREDICT:
        if isinstance(pred, (list, tuple)):
          assert isinstance(head_name,
                            (list, tuple)) and len(pred) == len(head_name)
          predictions = dict(zip(head_name, pred))
        else:
          predictions = pred

        if is_exporting() or self.p.output_path is None:
          spec = tf.estimator.EstimatorSpec(real_mode,
                                            predictions=predictions,
                                            training_hooks=self._training_hooks)
        else:
          op_file, write_op = self._get_file_ops(features, pred)
          close_hook = file_ops.FileCloseHook([op_file])
          with tf.control_dependencies(control_inputs=[write_op]):
            if isinstance(pred, dict):
              predictions = {k: tf.identity(v) for k, v in predictions.items()}
            else:
              predictions = tf.identity(predictions)
            spec = tf.estimator.EstimatorSpec(mode,
                                              training_hooks=[close_hook] +
                                              self._training_hooks,
                                              predictions=predictions)
        if is_exporting() and self._export_outputs:
          self._export_outputs.update(spec.export_outputs)
          return spec._replace(export_outputs=self._export_outputs)
        else:
          return spec

      train_ops = []
      targets, labels_list, preds_list = [], [], []
      if isinstance(pred, (list, tuple, dict)):
        assert isinstance(label,
                          (list, tuple, dict)) and len(pred) == len(label)
        assert isinstance(head_name,
                          (list, tuple)) and len(pred) == len(head_name)
        if isinstance(is_classification, (tuple, list, dict)):
          assert len(pred) == len(is_classification)
        else:
          is_classification = [is_classification] * len(pred)

        for i, name in enumerate(head_name):
          label_tensor = label[i] if isinstance(label,
                                                (list, tuple)) else label[name]
          pred_tensor = pred[i] if isinstance(pred,
                                              (list, tuple)) else pred[name]
          head_classification = is_classification[i] if isinstance(
              is_classification, (list, tuple)) else is_classification[name]

          targets.append(name)
          labels_list.append(label_tensor)
          preds_list.append(pred_tensor)

          mask = tf.greater_equal(label_tensor, self.valid_label_threshold)
          l = tf.boolean_mask(label_tensor, mask)
          p = tf.boolean_mask(pred_tensor, mask)

          if head_classification:
            auc_per_core, auc_update_op = tf.compat.v1.metrics.auc(
                labels=l, predictions=p, name=name)
            tf.compat.v1.summary.scalar("{}_auc".format(name), auc_per_core)
            train_ops.append(auc_update_op)
          else:
            mean_squared_error, mse_update_op = tf.compat.v1.metrics.mean_squared_error(
                labels=l, predictions=p, name=name)
            tf.compat.v1.summary.scalar("{}_mse".format(name),
                                        mean_squared_error)
            train_ops.append(mse_update_op)
      else:
        targets.append(head_name)
        labels_list.append(label)
        preds_list.append(pred)

        if is_classification:
          auc_per_core, auc_update_op = tf.compat.v1.metrics.auc(
              labels=label, predictions=pred, name=head_name)
          tf.compat.v1.summary.scalar(f"{head_name}_auc", auc_per_core)
          train_ops.append(auc_update_op)
        else:
          mean_squared_error, mse_update_op = tf.compat.v1.metrics.mean_squared_error(
              labels=label, predictions=pred, name=head_name)
          tf.compat.v1.summary.scalar("{}_mse".format(head_name),
                                      mean_squared_error)
          train_ops.append(mse_update_op)

      enable_metrics = self.metrics.enable_kafka_metrics or self.metrics.enable_file_metrics or self.metrics.enable_deep_insight
      if enable_metrics and self.metrics.deep_insight_sample_ratio > 0:
        model_name = self.metrics.deep_insight_name
        sample_ratio = self.metrics.deep_insight_sample_ratio
        extra_fields_keys = self.metrics.extra_fields_keys

        deep_insight_op = metric_utils.write_deep_insight(
            features=features,
            sample_ratio=self.metrics.deep_insight_sample_ratio,
            labels=label,
            preds=pred,
            model_name=model_name or "model_name",
            target=self.metrics.deep_insight_target,
            targets=targets,
            labels_list=labels_list,
            preds_list=preds_list,
            extra_fields_keys=extra_fields_keys,
            enable_kafka_metrics=self.metrics.enable_kafka_metrics or
            self.metrics.enable_file_metrics)
        logging.info("model_name: {}, target: {}.".format(
            model_name, self.metrics.deep_insight_target))
        train_ops.append(deep_insight_op)
        tf.compat.v1.add_to_collection("deep_insight_op", deep_insight_op)
        if self.metrics.enable_kafka_metrics:
          self.add_training_hook(KafkaMetricHook(deep_insight_op))
        elif self.metrics.enable_file_metrics:
          self.add_training_hook(
              FileMetricHook(deep_insight_op,
                             worker_id=get().worker_index,
                             parse_fn=self.metrics.parse_fn,
                             key_fn=self.metrics.key_fn or vepfs_key_fn,
                             layout_fn=self.metrics.layout_fn or
                             vepfs_layout_fn,
                             base_name=self.metrics.file_base_name,
                             file_ext=self.metrics.file_ext))
        logging.info("model_name: {}, target {}".format(model_name, head_name))

      if real_mode == tf.estimator.ModeKeys.EVAL:
        if is_exporting() or self.output_path is None:
          if isinstance(pred, (list, tuple)):
            train_ops.extend(pred)
          else:
            train_ops.append(pred)
          return tf.estimator.EstimatorSpec(mode,
                                            loss=loss,
                                            train_op=tf.group(train_ops),
                                            training_hooks=self._training_hooks)
        else:
          op_file, write_op = self._get_file_ops(features, pred)
          close_hook = file_ops.FileCloseHook([op_file])
          with tf.control_dependencies(control_inputs=[write_op]):
            if isinstance(pred, (list, tuple)):
              train_ops.extend([tf.identity(p) for p in pred])
            else:
              train_ops.append(tf.identity(pred))
            return tf.estimator.EstimatorSpec(mode,
                                              loss=loss,
                                              train_op=tf.group(train_ops),
                                              training_hooks=[close_hook] +
                                              self._training_hooks)
      else:  # training
        if hasattr(local_spec, 'optimizer'):
          dense_optimizer = local_spec.optimizer
        elif hasattr(self, '_default_dense_optimizer'):
          dense_optimizer = self._default_dense_optimizer
        else:
          raise Exception("dense_optimizer not found!")
        dump_utils.add_optimizer(dense_optimizer)

        if self.is_fused_layout():
          train_ops.append(
              feature_utils.apply_gradients(
                  self.ctx,
                  dense_optimizer,
                  loss,
                  clip_type=feature_utils.GradClipType.ClipByGlobalNorm,
                  clip_norm=self.clip_norm,
                  dense_weight_decay=self.dense_weight_decay,
                  global_step=self._global_step))
        else:
          train_ops.append(
              feature_utils.apply_gradients_with_var_optimizer(
                  self.ctx,
                  self.fc_dict.values(),
                  dense_optimizer,
                  loss,
                  clip_type=feature_utils.GradClipType.ClipByGlobalNorm,
                  clip_norm=self.clip_norm,
                  dense_weight_decay=self.dense_weight_decay,
                  global_step=self._global_step,
                  grads_and_vars_summary=self.enable_grads_and_vars_summary))

        return tf.estimator.EstimatorSpec(mode,
                                          loss=loss,
                                          train_op=tf.group(train_ops),
                                          training_hooks=self._training_hooks)

    return model_fn_internal

  def create_serving_input_receiver_fn(self):
    """生在Serving数据流, serving_input_receiver_fn"""
    return dump_utils.record_receiver(self.serving_input_receiver_fn)

  @abstractmethod
  def input_fn(self, mode: tf.estimator.ModeKeys) -> DatasetV2:
    """抽象方法, 定义数据流

    Args:
      mode (:obj:`str`): 训练模式, train/eval/predict等

    Returns:
      DatasetV2, TF数据集

    """

    raise NotImplementedError('input_fn() not Implemented')

  @abstractmethod
  def model_fn(
      self, features: Dict[str, tf.Tensor], mode: tf.estimator.ModeKeys
  ) -> Union[EstimatorSpec, Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    """抽象方法, 定义模型

    Args:
      features (:obj:`Dict[str, tf.Tensor]`): 特征
      mode (:obj:`str`): 训练模式, train/eval/predict等

    Returns:
      Union[EstimatorSpec, Tuple[tf.Tensor, tf.Tensor, tf.Tensor]], 可以是tuple, 包括(loss, label, predict),
                                                                    也可以是EstimatorSpec
    """

    raise NotImplementedError('generate_model() not Implemented')

  @abstractmethod
  def serving_input_receiver_fn(self) -> ServingInputReceiver:
    """Serving数据流, 训练数据流与Serving数据流或能不一样

    Returns:
      ServingInputReceiver

    """

    raise NotImplementedError('serving_input_receiver_fn() not Implemented')

  @property
  def _export_outputs(self):
    graph = tf.compat.v1.get_default_graph()
    if hasattr(graph, '__export_outputs'):
      return getattr(graph, '__export_outputs')
    else:
      setattr(graph, '__export_outputs', {})
      return graph.__export_outputs

  def add_extra_output(self,
                       name: str,
                       outputs: Union[tf.Tensor, Dict[str, tf.Tensor]],
                       head_name: str = None,
                       head_type: str = None):
    """如果有出多输出, 可以用add_extra_output, 每个输出会成为Serving中的一个Signature

    Args:
      name (:obj:`str`): 签名的名称
      outputs (:obj:`Union[tf.Tensor, Dict[str, tf.Tensor]]`): 输出, 可以是一个Tensor, 也可以是一个Dict[str, tf.Tensor]
      head_name (:obj:`str`): output对应的head的名称
      head_name (:obj:`str`): output对应的head的类型, 如user, item, context等 

    """

    add_to_collections('signature_name', name)
    if is_exporting():
      exported_outputs = self._export_outputs
      if name not in exported_outputs:
        exported_outputs[name] = tf.estimator.export.PredictOutput(outputs)
      else:
        raise KeyError("key {name} exists!".format(name))

  def add_training_hook(self, hook):
    if isinstance(hook, KafkaMetricHook):
      if any(isinstance(h, KafkaMetricHook) for h in self._training_hooks):
        return
    elif isinstance(hook, FileMetricHook):
      if any(isinstance(h, FileMetricHook) for h in self._training_hooks):
        return
    self._training_hooks.append(hook)

  def add_layout(self, name: str, slice_list: list, out_type: str,
                 shape_list: list):
    if out_type == 'concat':
      out_conf = OutConfig(out_type=OutType.CONCAT)
    elif out_type == 'stack':
      out_conf = OutConfig(out_type=OutType.STACK)
    elif out_type == 'addn':
      out_conf = OutConfig(out_type=OutType.ADDN)
    else:
      out_conf = OutConfig(out_type=OutType.NONE)

    for slice_conf in slice_list:
      slice_config = out_conf.slice_configs.add()
      if len(slice_conf.feature_slot.get_feature_columns()) > 1:
        raise Exception(
            "There are multi feature columns on a slot, not support yet!")
      slice_config.feature_name = slice_conf.feature_slot.name
      slice_config.start = slice_conf.start
      slice_config.end = slice_conf.end

    for shape in shape_list:
      shape_dims = out_conf.shape.add()
      for i, dim in enumerate(shape):
        if i == 0:
          shape_dims.dims.append(-1)
        else:
          if isinstance(dim, int):
            shape_dims.dims.append(dim)
          else:
            assert hasattr(dim, 'value')
            shape_dims.dims.append(dim.value)

    self._layout_dict[name] = out_conf

  @property
  def layout_dict(self):
    return self._layout_dict

  @layout_dict.setter
  def layout_dict(self, layouts):
    self._layout_dict = layouts


@monolith_export
class MonolithModel(MonolithBaseModel):
  '''模型开发的基类

  Args:
      params (:obj:`Params`): 配置参数, 默认为None
  '''

  @classmethod
  def params(cls):
    p = super(MonolithModel, cls).params()
    p.define("feature_list", None, "The feature_list conf file.")
    return p

  def __init__(self, params=None):
    params = params or type(self).params()
    super(MonolithModel, self).__init__(params)
    dump_utils.enable = FLAGS.enable_model_dump

  def _get_fs_conf(self, shared_name: str, slot: int, occurrence_threshold: int,
                   expire_time: int) -> FeatureSlotConfig:
    return FeatureSlotConfig(name=shared_name,
                             has_bias=False,
                             slot_id=slot,
                             occurrence_threshold=occurrence_threshold,
                             expire_time=expire_time)

  def _embedding_slice_lookup(self, fc: Union[str, FeatureColumn],
                              slice_name: str, slice_dim: int,
                              initializer: Initializer, optimizer: Optimizer,
                              compressor: Compressor, learning_rate_fn,
                              slice_list: list) -> FeatureSlice:
    assert not self.is_fused_layout()
    if isinstance(fc, str):
      fc = self.fc_dict[fc]

    feature_slot = fc.feature_slot
    feature_name = fc.feature_name

    if feature_name in self.slice_dict:
      if slice_name in self.slice_dict[feature_name]:
        fc_slice = self.slice_dict[feature_name][slice_name]
      else:
        fc_slice = feature_slot.add_feature_slice(slice_dim, initializer,
                                                  optimizer, compressor,
                                                  learning_rate_fn)
        self.slice_dict[feature_name][slice_name] = fc_slice
    else:
      fc_slice = feature_slot.add_feature_slice(slice_dim, initializer,
                                                optimizer, compressor,
                                                learning_rate_fn)
      self.slice_dict[feature_name] = {slice_name: fc_slice}

    slice_list.append(fc_slice)
    return fc.embedding_lookup(fc_slice)

  @dump_utils.record_feature
  def create_embedding_feature_column(self,
                                      feature_name,
                                      occurrence_threshold: int = None,
                                      expire_time: int = 36500,
                                      max_seq_length: int = 0,
                                      shared_name: str = None) -> FeatureColumn:
    """创建嵌入特征列(embedding feature column)

    Args:
      feature_name (:obj:`Any`): 特征列的名字
      occurrence_threshold (:obj:`int`): 用于低频特征过滤, 如果出现次数小于`occurrence_threshold`, 则这个特征将大概率不会进入模型
      expire_time (:obj:`int`): 特征过期时间, 如果一个特征在`expire_time`之内没有更新了, 则这个特征可能从hash表中移除
      max_seq_length (:obj:`int`): 如果设为0, 表示非序列特征, 如果设为正数, 则表示序列特征的长度
      shared_name (:obj:`str`): 共享embedding. 如果本feature与另一个feature共享embedding, 则可以将被共享feature设为`shared_name`

    Returns:
     FeatureColumn, 特征列

    """

    feature_name, slot = get_feature_name_and_slot(feature_name)

    if feature_name in self.fc_dict:
      return self.fc_dict[feature_name]
    else:
      if shared_name is not None and len(shared_name) > 0:
        if shared_name in self.fs_dict:
          fs = self.fs_dict[shared_name]
        elif shared_name in self.fc_dict:
          fs = self.fc_dict[shared_name].feature_slot
        else:
          try:
            feature_list = FeatureList.parse()
            shared_slot = feature_list[shared_name].slot
            shared_fs = self.ctx.create_feature_slot(
                self._get_fs_conf(shared_name, shared_slot,
                                  occurrence_threshold, expire_time))
            self.fs_dict[shared_name] = shared_fs
            fs = shared_fs
          except:
            raise Exception(
                f"{feature_name} shared embedding with {shared_name}, so {shared_name} should create first!"
            )
      else:
        fs = self.ctx.create_feature_slot(
            self._get_fs_conf(feature_name, slot, occurrence_threshold,
                              expire_time))
      if max_seq_length > 0:
        combiner = FeatureColumn.first_n(max_seq_length)
      else:
        combiner = FeatureColumn.reduce_sum()
      fc = FeatureColumn(fs, feature_name, combiner=combiner)
      self.fc_dict[feature_name] = fc
      return fc

  @dump_utils.record_slice
  def lookup_embedding_slice(self,
                             features,
                             slice_name,
                             slice_dim=None,
                             initializer: Initializer = None,
                             optimizer: Optimizer = None,
                             compressor: Compressor = None,
                             learning_rate_fn=None,
                             group_out_type: str = 'add_n',
                             out_type: str = None) -> tf.Tensor:
    """Monolith中embedding是分切片的, 每个切片可以有独立的初始化器, 优化器, 压缩器, 学习率等. 切片的引入使Embedding更加强大. 如某些情况
    下要共享Embedding, 另一些情况下要独立Embedding, 与一些域交叉要用一种Embedding, 与另一些域交叉用另一种Embedding等. 切片的引入可以方便
    解上以上问题. 切片与完整Embedding的关系由Monolith自动维护, 对用户透明.

    Args:
      slice_name (:obj:`str`): 切片名称
      features (:obj:`List[str], Dict[str, int]`): 支持三种形式
        1) 特征名列表, 此时每个切片的长度相同, 由`slice_dim`确定, 不能为None
        2) 特征 (特征名, 切片长度) 列表, 此时每个切片的长度可以不同, 全局的`slice_dim`必须为None
        3) 特征字典, 特征名 -> 切片长度, 此时每个切片的长度可以不同, 全局的`slice_dim`必须为None
      slice_dim (:obj:`int`): 切片长度
      initializer (:obj:`Initializer`): 切片的初始化器, Monolith中的初始化器,  不能是TF中的
      optimizer (:obj:`Optimizer`): 切片的优化器, Monolith中的优化器,  不能是TF中的
      compressor (:obj:`Compressor`): 切片的压缩器, 用于在Servering模型加载时将模型压缩
      learning_rate_fn (:obj:`tf.Tensor`): 切片的学习率

    """
    concat = ",".join(sorted(map(str, features)))
    layout_name = f'{slice_name}_{hashlib.md5(concat.encode()).hexdigest()}'
    if self.is_fused_layout():
      if isinstance(features, (list, tuple)) and isinstance(slice_dim, int):
        if all(isinstance(ele, (tuple, list)) for ele in features):
          raise ValueError("group pool is not support when fused_layout")
      return self.ctx.layout_factory.get_layout(layout_name)

    feature_embeddings, slice_list = [], []
    if isinstance(features, dict):
      for fc_name, sdim in features.items():
        fc_name, _ = get_feature_name_and_slot(fc_name)
        feature_embeddings.append(
            self._embedding_slice_lookup(fc_name, slice_name, sdim, initializer,
                                         optimizer, compressor,
                                         learning_rate_fn, slice_list))
    elif isinstance(features, (list, tuple)) and isinstance(slice_dim, int):
      if all(isinstance(ele, (str, int, FeatureColumn)) for ele in features):
        # a list of feature with fixed dim
        for fc_name in features:
          fc_name, _ = get_feature_name_and_slot(fc_name)
          feature_embeddings.append(
              self._embedding_slice_lookup(fc_name, slice_name, slice_dim,
                                           initializer, optimizer, compressor,
                                           learning_rate_fn, slice_list))
      elif all(isinstance(ele, (tuple, list)) for ele in features):
        assert group_out_type in {'concat', 'add_n'}
        for group_name in features:
          assert all(isinstance(ele, int) for ele in group_name)
          local_embeddings = []
          for fc_name in group_name:
            fc_name, _ = get_feature_name_and_slot(fc_name)
            local_embeddings.append(
                self._embedding_slice_lookup(fc_name, slice_name, slice_dim,
                                             initializer, optimizer, compressor,
                                             learning_rate_fn, slice_list))
          if group_out_type == 'add_n':
            feature_embeddings.append(tf.add_n(local_embeddings))
          else:
            feature_embeddings.append(tf.concat(local_embeddings, axis=1))
      else:
        raise ValueError("ValueError for features")
    elif isinstance(features, (list, tuple)):
      if all([
          isinstance(ele, (tuple, list)) and len(ele) == 2 for ele in features
      ]):
        for fc_name, sdim in features:
          fc_name, _ = get_feature_name_and_slot(fc_name)
          feature_embeddings.append(
              self._embedding_slice_lookup(fc_name, slice_name, sdim,
                                           initializer, optimizer, compressor,
                                           learning_rate_fn, slice_list))
      else:
        raise ValueError("ValueError for features")
    else:
      raise ValueError("ValueError for features")

    if out_type is None:
      shape_list = [emb.shape for emb in feature_embeddings]
      self.add_layout(layout_name, slice_list, out_type, shape_list)
      return feature_embeddings
    else:
      assert out_type in {'concat', 'stack', 'add_n', 'addn'}
      if out_type == 'concat':
        out = tf.concat(feature_embeddings, axis=1, name=layout_name)
        self.add_layout(layout_name,
                        slice_list,
                        out_type,
                        shape_list=[out.shape])
        return out
      elif out_type == 'stack':
        out = tf.stack(feature_embeddings, axis=1, name=layout_name)
        self.add_layout(layout_name,
                        slice_list,
                        out_type,
                        shape_list=[out.shape])
        return out
      else:
        out = tf.add_n(feature_embeddings, name=layout_name)
        self.add_layout(layout_name, slice_list, 'addn', shape_list=[out.shape])
        return out


