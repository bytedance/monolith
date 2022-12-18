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

import dataclasses
import os
import time

from absl import logging
import tensorflow as tf
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.experimental.ops import distribute_options  # Should be removed after tf2.5

from monolith.native_training.hooks import ckpt_hooks_pb2
from monolith.native_training import basic_restore_hook
from monolith.native_training import barrier_ops
from monolith.native_training import graph_meta


@dataclasses.dataclass
class Meta:
  info_var: tf.Variable
  info_var_placeholder: tf.compat.v1.placeholder
  info_var_assign_op: tf.Operation
  enable_iter_save_restore: bool = True


SAVE_ACTION = "Save"


def _get_meta() -> Meta:

  def factory():
    info_var = tf.compat.v1.get_local_variable("WorkerCkptMetaInfo",
                                               dtype=tf.string,
                                               initializer="")
    info_var_placeholder = tf.compat.v1.placeholder(tf.string, [])
    info_var_assign_op = info_var.assign(info_var_placeholder)
    return Meta(info_var=info_var,
                info_var_placeholder=info_var_placeholder,
                info_var_assign_op=info_var_assign_op)

  return graph_meta.get_meta("worker_ckpt_meta", factory)


def assign_ckpt_info(session: tf.compat.v1.Session,
                     info: ckpt_hooks_pb2.WorkerCkptInfo):
  meta = _get_meta()
  session.run(meta.info_var_assign_op,
              feed_dict={meta.info_var_placeholder: info.SerializeToString()})


def get_ckpt_info(
    session: tf.compat.v1.Session) -> ckpt_hooks_pb2.WorkerCkptInfo:
  ckpt_info = ckpt_hooks_pb2.WorkerCkptInfo()
  ckpt_info.ParseFromString(session.run(_get_meta().info_var))
  return ckpt_info


class BarrierSaverListener(tf.estimator.CheckpointSaverListener):
  """During saving, set up barrier condition to block worker for chief."""

  def __init__(self,
               barrier_op: barrier_ops.BarrierOp,
               wait_seconds=1,
               max_pending_seconds=30):
    self._barrier_op = barrier_op
    self._wait_seconds = wait_seconds
    self._max_pending_seconds = max_pending_seconds
    # Make sure meta is created.
    self._meta = _get_meta()

  def before_save(self, session, global_step_value):
    assign_ckpt_info(
        session, ckpt_hooks_pb2.WorkerCkptInfo(global_step=global_step_value))
    logging.info("Place barrier for saving.")
    start_time = time.time()
    self._barrier_op.place_barrier(session, action=SAVE_ACTION)
    while not self._barrier_op.is_all_blocked(session):
      time.sleep(self._wait_seconds)
      if time.time() - start_time > self._max_pending_seconds:
        break

    unblocked_indices = self._barrier_op.get_unblocked_indices(session)
    if unblocked_indices:
      logging.info("Unblocked worker indices: {}.".format(
          str(unblocked_indices)))
    else:
      logging.info("All workers have been blocked.")

  def after_save(self, session, global_step_value):
    logging.info("Remove barrier for saving.")
    start_time = time.time()
    self._barrier_op.remove_barrier(session)
    while not self._barrier_op.is_none_blocked(session):
      time.sleep(self._wait_seconds)
      if time.time() - start_time > self._max_pending_seconds:
        break

    blocked_indices = self._barrier_op.get_blocked_indices(session)
    if blocked_indices:
      logging.info("Blocked worker indices: {}.".format(str(blocked_indices)))
    else:
      logging.info("None worker has been blocked.")


class _WorkerCkptRestorerHook(tf.estimator.SessionRunHook):

  def __init__(self, saver: tf.compat.v1.train.Saver, model_dir: str,
               latest_filename: str):
    self._saver = saver
    self._model_dir = model_dir
    self._latest_filename = latest_filename

  def after_create_session(self, session, coord):
    latest_ckpt = tf.train.latest_checkpoint(self._model_dir,
                                             self._latest_filename)
    if latest_ckpt is not None and self._saver:
      self._saver.restore(session, latest_ckpt)
    else:
      logging.info("Skipped worker ckpt restore.")


class WorkerCkptHelper:

  def __init__(self, model_dir: str, index: int):
    # Here we try to keep them as similar as tf.data.experimental.CheckpointInputPipelineHook
    self._model_dir = model_dir
    self._index = index
    checkpoint_prefix = "input_worker_{}".format(index)
    self._checkpoint_basename = checkpoint_prefix + ".ckpt"
    self._latest_filename = "checkpoint_" + checkpoint_prefix
    iterators = tf.compat.v1.get_collection(iterator_ops.GLOBAL_ITERATORS)
    saveables = []
    if _get_meta().enable_iter_save_restore:
      saveables.extend([
          iterator_ops._IteratorSaveable(
              i,
              i.name,
              external_state_policy=distribute_options.ExternalStatePolicy.
              IGNORE) for i in iterators
      ])
    else:
      logging.info("The iterator save is disabled.")
    if saveables:
      self._saver = tf.compat.v1.train.Saver(var_list=saveables, sharded=True)
    else:
      # Saver will throw error if we try to saveables is an empty list.
      self._saver = None

  def create_save_iterator_callback(self):

    def callback(action: str, sess: tf.compat.v1.Session):
      if not action == SAVE_ACTION:
        return
      ckpt_info = get_ckpt_info(sess)
      try:
        if self._saver:
          self._saver.save(sess,
                           os.path.join(self._model_dir,
                                        self._checkpoint_basename),
                           global_step=ckpt_info.global_step,
                           latest_filename=self._latest_filename,
                           write_meta_graph=False)
      except tf.errors.UnimplementedError as e:
        logging.warning(
            "Current dataset iterators don't support save. This might be expected. %s",
            str(e))

    return callback

  def create_restorer_hook(self):
    return _WorkerCkptRestorerHook(self._saver, self._model_dir,
                                   self._latest_filename)


def disable_iterator_save_restore():
  """In some situations (like in ByteDance we feed data via stdin), the 
  input progress is not trackable by tensorflow. In this case, we should
  disable iterator restore since its state is inaccurate.
  
  NOTICE: this function should be called before any creation of classes
  in this module.
  """
  _get_meta().enable_iter_save_restore = False
