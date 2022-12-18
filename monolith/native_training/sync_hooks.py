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

import time

from absl import logging

import tensorflow as tf


class SyncHelper:

  #TODO(leqi.zou): maybe in the future, we want to support the dynamic number of workers.
  def __init__(self,
               num_workers: int,
               is_chief,
               var_device="/job:chief/task:0"):
    self._num_workers = num_workers
    with tf.name_scope("monolith_sync_helper"):
      # In distributed training, the var is local variable for chief, but global variable for worker.
      collections = [tf.compat.v1.GraphKeys.LOCAL_VARIABLES
                    ] if is_chief else [tf.compat.v1.GraphKeys.VARIABLES]
      with tf.device(var_device):
        # For idx 0, represents the current restore status
        # For idx >0, represents if the current worker is alive
        self._var = tf.compat.v1.get_variable(
            "monolith_sync_helper/control_var",
            initializer=[False] * num_workers,
            dtype=tf.bool,
            trainable=False,
            collections=collections)

      self._idx_ph = tf.compat.v1.placeholder(tf.int32, shape=[], name="idx_ph")
      self._val_ph = tf.compat.v1.placeholder(tf.bool,
                                              shape=[],
                                              name="value_ph")
      self._read_value = self._var[self._idx_ph]
      self._assign_value = self._var[self._idx_ph].assign(self._val_ph)
      self._workers_status = self._var[1:]
      self._alive_workers = tf.where(self._workers_status) + 1
      self._num_alive_workers = tf.math.reduce_sum(
          tf.cast(self._workers_status, tf.int32))

  @property
  def num_workers(self):
    return self._num_workers

  def mark_restore_done(self, sess):
    sess.run(self._assign_value,
             feed_dict={
                 self._idx_ph: 0,
                 self._val_ph: True
             })

  def get_restore_status(self, sess):
    return sess.run(self._read_value, feed_dict={self._idx_ph: 0})

  def start_worker(self, sess, idx: int):
    assert idx > 0 and idx < self._num_workers, f"Index {idx} is out range "
    sess.run(self._assign_value,
             feed_dict={
                 self._idx_ph: idx,
                 self._val_ph: True
             })

  def finish_worker(self, sess, idx: int):
    assert idx > 0 and idx < self._num_workers, f"Index {idx} is out range "
    sess.run(self._assign_value,
             feed_dict={
                 self._idx_ph: idx,
                 self._val_ph: False
             })

  def get_alive_workers(self, sess):
    return sess.run(self._alive_workers).flatten()

  def get_num_alive_workers(self, sess):
    return sess.run(self._num_alive_workers)


_CHIEF_TIMEOUT_SECONDS = 1800


class ChiefSyncHook(tf.estimator.SessionRunHook):
  """
  A hook that used for chief and woker sync at the begining and at the end.
  Args:
    has_chief - if cluster has chief, place queue in chief.
                Otherwise, place queue in localhost.
  """

  def __init__(self,
               sync_helper: SyncHelper,
               timeout_seconds=_CHIEF_TIMEOUT_SECONDS):
    self._timeout_seconds = timeout_seconds
    self._helper = sync_helper

  def after_create_session(self, session, coord):
    self._helper.mark_restore_done(session)

  def end(self, session):
    # At the end, chief waits for other workers performing an enqueue op.
    start_time = time.time()
    while True:
      num_alive_workers = self._helper.get_num_alive_workers(session)
      if time.time(
      ) - start_time > self._timeout_seconds or num_alive_workers == 0:
        break
      logging.log_every_n_seconds(
          logging.INFO,
          "Total worker count: {}, remaining count: {}.\nRemaining workers: %s".
          format(self._helper.num_workers, num_alive_workers), 60,
          self._helper.get_alive_workers(session))
      time.sleep(1)

    if num_alive_workers > 0:
      logging.info("Reach timeout seconds! Remaining worker count: {}.".format(
          num_alive_workers))
    else:
      logging.info("All other workers had been finished.")


class WorkerSyncHook(tf.estimator.SessionRunHook):
  """
  A hook that used for chief and woker sync at the begining and at the end.
  Args:
    has_chief - if cluster has chief, place queue in chief.
                Otherwise, place queue in localhost.
  """

  def __init__(self, worker_index, sync_helper: SyncHelper):
    self._worker_index = worker_index
    self._helper = sync_helper

  def after_create_session(self, session, coord):
    if self._worker_index > 0:
      self._helper.start_worker(session, self._worker_index)
      while not self._helper.get_restore_status(session):
        logging.log_every_n_seconds(
            logging.INFO,
            "The worker {} waits for start signal of chief.".format(
                self._worker_index), 60)
        time.sleep(1)

  def end(self, session):
    if self._worker_index > 0:
      self._helper.finish_worker(session, self._worker_index)


class TrainingHooksHelper:

  def __init__(self,
               enable_sync: bool,
               num_workers: int,
               worker_idx: int,
               chief_timeout_seconds: int = _CHIEF_TIMEOUT_SECONDS):
    self._enable_sync = enable_sync
    self._training_chief_hooks = []
    self._training_hooks = []
    if self._enable_sync:
      sync_helper = SyncHelper(num_workers, worker_idx == 0)
      self._training_chief_hooks.append(
          ChiefSyncHook(sync_helper, timeout_seconds=chief_timeout_seconds))
      self._training_hooks.append(WorkerSyncHook(worker_idx, sync_helper))

  @property
  def training_chief_hooks(self):
    return tuple(self._training_chief_hooks)

  @property
  def training_hooks(self):
    return tuple(self._training_hooks)
