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

from absl import logging
from contextlib import nullcontext
import os
from typing import List

import tensorflow as tf
from tensorflow.python.framework import ops

from monolith.native_training import basic_restore_hook
from monolith.native_training import save_utils
from monolith.native_training import utils
from monolith.native_training.runtime.ops import gen_monolith_ops
from monolith.utils import get_libops_path
from monolith.native_training.model_export.export_context import is_exporting_standalone

HASH_FILTER_CAPACITY = 300000000
HASH_FILTER_SPLIT_NUM = 7

filter_ops = gen_monolith_ops

filter_save_op = gen_monolith_ops

filter_restore_op = gen_monolith_ops

_TIMEOUT_IN_MS = 30 * 60 * 1000


class FilterType(object):

  SLIDING_HASH_FILTER = 'sliding_hash_filter'
  PROBABILISTIC_FILTER = 'probabilistic_filter'


def create_hash_filter(capacity: int,
                       split_num: int,
                       config: bytes = b"",
                       name_suffix: str = "") -> tf.Tensor:
  """Creates a hash filter"""
  return filter_ops.MonolithHashFilter(capacity=capacity,
                                       split_num=split_num,
                                       config=config,
                                       shared_name="MonolithHashFilter" +
                                       name_suffix)


def create_probabilistic_filter(equal_probability,
                                config: bytes = b"",
                                name_suffix: str = "") -> tf.Tensor:
  """Creates a probabilistic filter"""
  return filter_ops.MonolithProbabilisticFilter(
      equal_probability=equal_probability,
      config=config,
      shared_name="MonolithProbabilisticFilter" + name_suffix)


def create_dummy_hash_filter(name_suffix: str = "0") -> tf.Tensor:
  """Creates a dummy hash filter"""
  return filter_ops.MonolithDummyHashFilter(shared_name="DummyHashFilter" +
                                            name_suffix)


def _create_hash_filter(
    enable_hash_filter: bool,
    config: bytes = b"",
    name_suffix: str = "",
    filter_capacity: int = HASH_FILTER_CAPACITY,
    filter_split_num: int = HASH_FILTER_SPLIT_NUM,
    filter_equal_probability: bool = False,
    filter_type: FilterType = FilterType.SLIDING_HASH_FILTER) -> tf.Tensor:
  if enable_hash_filter is True:
    if filter_type == FilterType.SLIDING_HASH_FILTER:
      return create_hash_filter(filter_capacity, filter_split_num, config,
                                name_suffix)
    elif filter_type == FilterType.PROBABILISTIC_FILTER:
      return create_probabilistic_filter(filter_equal_probability, config,
                                         name_suffix)
    else:
      raise ValueError("Invalid filter type, please investigate and retry!")
  else:
    return create_dummy_hash_filter(name_suffix)


def create_hash_filters(
    ps_num: int,
    enable_hash_filter: bool,
    config: bytes = b"",
    filter_capacity: int = HASH_FILTER_CAPACITY,
    filter_split_num: int = HASH_FILTER_SPLIT_NUM,
    filter_equal_probability: bool = False,
    filter_type: FilterType = FilterType.SLIDING_HASH_FILTER
) -> List[tf.Tensor]:
  logging.info(
      "Create hash fitlers, enable_hash_filter:{}.".format(enable_hash_filter))
  if ps_num == 0:
    return [
        _create_hash_filter(enable_hash_filter,
                            config,
                            "",
                            filter_capacity,
                            filter_split_num,
                            filter_equal_probability=filter_equal_probability,
                            filter_type=filter_type)
    ]
  else:
    hash_filters = []
    for i in range(ps_num):
      ps_device_name = utils.ps_device(i)
      with nullcontext() if is_exporting_standalone() else tf.device(
          ps_device_name):
        hash_filters.append(
            _create_hash_filter(
                enable_hash_filter,
                config,
                "_" + str(i),
                filter_capacity,
                filter_split_num,
                filter_equal_probability=filter_equal_probability,
                filter_type=filter_type))
    return hash_filters


def save_hash_filter(hash_filter: tf.Tensor,
                     hash_filter_basename: tf.Tensor,
                     enable_hash_filter: bool = False) -> tf.Operation:
  if enable_hash_filter is True:
    return filter_save_op.monolith_hash_filter_save(hash_filter,
                                                    hash_filter_basename)
  else:
    return tf.no_op()


def restore_hash_filter(hash_filter: tf.Tensor,
                        hash_filter_base_name: tf.Tensor,
                        enable_hash_filter: bool = False) -> tf.Operation:
  if enable_hash_filter is True:
    return filter_restore_op.monolith_hash_filter_restore(
        hash_filter, hash_filter_base_name)
  else:
    return tf.no_op()


def intercept_gradient(filter_tensor: tf.Tensor, ids: tf.Tensor,
                       embeddings: tf.Tensor):
  """
  If id is supposed to be filtered, the graident will be intercepted. Output the same 
  embeddings.
  Args:
    ids - a 1-D int64 tensor.
    embeddings - a N-d embedding tensor whose the first dimention is corresponding to ids. 
  """
  return filter_ops.MonolithHashFilterInterceptGradient(
      filter_handle=filter_tensor, ids=ids, embeddings=embeddings)


class HashFilterCheckpointSaverListener(tf.estimator.CheckpointSaverListener):
  """
  Saves the hash filters when saver is run.
  """

  def __init__(self,
               basename: str,
               hash_filters: [tf.Tensor],
               enable_hash_filter: bool = False,
               enable_save_restore: bool = True):
    """
    |basename| should be a file name which is same as what is passed to saver.
    |hash_filters| hash filters to save in checkpoint.
    |enable_hash_filter| whether use real hash filters. If true, will save
                         hash filters in checkpoint. If false, will skip save
                         logic internally.
    enable_hash_filter: TODO(zouxuan) Whether to use save and restore on the
                        hash filter.
                        Hash filter is broken for save restore during sync
                        training right now.
    """
    super().__init__()
    self._helper = save_utils.SaveHelper(basename)
    self._hash_filters = hash_filters
    self._enable_hash_filter = enable_hash_filter
    self._enable_save_restore = enable_save_restore
    self._hash_filter_id_to_placeholder = {}
    self._save_op = self._build_save_graph()

  def before_save(self, sess, global_step_value):
    """
    We use before save so the checkpoint file is updated after we successfully
    save the hash filter.
    """
    if self._enable_hash_filter is False or self._enable_save_restore is False:
      return

    feed_dict = {}
    hash_filter_names = []
    asset_dir = self._helper.get_ckpt_asset_dir(
        self._helper.get_ckpt_prefix(global_step_value))
    tf.io.gfile.makedirs(asset_dir)
    for ps_idx, hash_filter in enumerate(self._hash_filters):
      hash_filter_basename = asset_dir + "hash_filter_{}".format(ps_idx)
      hash_filter_names.append(hash_filter_basename)
      feed_dict.update({
          self._hash_filter_id_to_placeholder[id(hash_filter)]:
              hash_filter_basename
      })

    sess.run(self._save_op,
             feed_dict=feed_dict,
             options=tf.compat.v1.RunOptions(timeout_in_ms=_TIMEOUT_IN_MS))
    logging.info("Finished saving hash filters.")

  def _build_save_graph(self) -> tf.Operation:
    if self._enable_hash_filter is False or self._enable_save_restore is False:
      return tf.no_op()

    last_op = tf.no_op()
    for ps_idx, hash_filter in enumerate(self._hash_filters):
      hash_filter_basename = tf.compat.v1.placeholder(tf.string, shape=[])
      self._hash_filter_id_to_placeholder.update(
          {id(hash_filter): hash_filter_basename})
      with tf.control_dependencies([last_op]):
        last_op = save_hash_filter(hash_filter, hash_filter_basename, True)
    return last_op


class HashFilterCheckpointRestorerListener(
    basic_restore_hook.CheckpointRestorerListener):
  """Restores the hash filters from basename"""

  def __init__(self,
               basename: str,
               hash_filters: [tf.Tensor],
               enable_hash_filter: bool = False,
               enable_save_restore: bool = True):
    """
    |basename| should be a file name which is same as what is passed to saver.
    |hash_filters| hash filters to save in checkpoint.
    |enable_hash_filter| whether use real hash filters. If true, will save
                         hash filters in checkpoint. If false, will skip save
                         logic internally.
    enable_hash_filter: TODO(zouxuan) Whether to use save and restore on the
                        hash filter.
                        Hash filter is broken for save restore during sync
                        training right now.
    """
    super().__init__()
    self._basename = basename
    self._helper = save_utils.SaveHelper(self._basename)
    self._hash_filters = hash_filters
    self._enable_hash_filter = enable_hash_filter
    self._enable_save_restore = enable_save_restore
    self._hash_filter_id_to_placeholder = {}
    self._restore_op = self._build_restore_graph()

  def before_restore(self, session):
    """
    We use before restore so as to strictly control the order of restorer listeners.

    """
    ckpt_prefix = tf.train.latest_checkpoint(os.path.dirname(self._basename))
    if not ckpt_prefix:
      logging.info("No checkpoint found in %s. Skip the hash filters restore.",
                   self._basename)
      return
    logging.info("Restore hash filter from %s", ckpt_prefix)
    asset_dir = self._helper.get_ckpt_asset_dir(ckpt_prefix)
    if tf.io.gfile.exists(asset_dir):
      self._restore_from_path_prefix(session, asset_dir)
    else:
      # This is the legacy behavior and should be removed soon.
      self._restore_from_path_prefix(session, ckpt_prefix)

  def _restore_from_path_prefix(self, sess, path_prefix):
    if self._enable_hash_filter is False or self._enable_save_restore is False:
      return

    feed_dict = {}
    hash_filter_names = []
    for ps_idx, hash_filter in enumerate(self._hash_filters):
      hash_filter_basename = path_prefix + "hash_filter_{}".format(ps_idx)
      hash_filter_names.append(hash_filter_basename)
      feed_dict.update({
          self._hash_filter_id_to_placeholder[id(hash_filter)]:
              hash_filter_basename
      })

    sess.run(self._restore_op,
             feed_dict=feed_dict,
             options=tf.compat.v1.RunOptions(timeout_in_ms=_TIMEOUT_IN_MS))

  def _build_restore_graph(self) -> tf.Operation:
    if self._enable_hash_filter is False or self._enable_save_restore is False:
      return tf.no_op()

    restore_ops = []
    for ps_idx, hash_filter in enumerate(self._hash_filters):
      hash_filter_basename = tf.compat.v1.placeholder(tf.string, shape=[])
      self._hash_filter_id_to_placeholder.update(
          {id(hash_filter): hash_filter_basename})
      restore_ops.append(
          restore_hash_filter(hash_filter, hash_filter_basename, True))
    return tf.group(restore_ops)


@ops.RegisterGradient("MonolithHashFilterInterceptGradient")
def _intercept_gradient_gradient(op: tf.Operation, grad: tf.Tensor):
  filter_tensor = op.inputs[0]
  ids = op.inputs[1]
  filtered_grad = filter_ops.MonolithHashFilterInterceptGradientGradient(
      filter_handle=filter_tensor, ids=ids, grad=grad)
  return None, None, filtered_grad
