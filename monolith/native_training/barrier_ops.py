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

#coding:utf-8
import time
import threading

from absl import logging

import tensorflow as tf

from monolith.native_training import basic_restore_hook


class BarrierOp:
  """
  A barrier operation that used to blocking worker by chief.
  Thread safe.
  """

  def __init__(self,
               capacity,
               is_chief=True,
               wait_seconds=1,
               name_prefix="default",
               barrier_callbacks=None):
    self._capacity = capacity
    self._wait_seconds = wait_seconds
    with tf.name_scope(name_prefix + "_barrier_op"):
      # For non-chief workers, barrier vars are treated as global variables.
      collections = [tf.compat.v1.GraphKeys.LOCAL_VARIABLES
                    ] if is_chief else [tf.compat.v1.GraphKeys.VARIABLES]
      self._barrier_vars = tf.compat.v1.get_variable("barrier_var",
                                                     initializer=[False] *
                                                     capacity,
                                                     collections=collections)
      self._idx_ph = tf.compat.v1.placeholder(tf.int32,
                                              shape=[],
                                              name="index_placeholder")
      self._place_op = self._barrier_vars[self._idx_ph].assign(True)
      self._remove_op = self._barrier_vars[self._idx_ph].assign(False)
      self._barrier_placed_tensor = self._barrier_vars[0]
      self._barrier_callbacks = barrier_callbacks or []
      self._action = tf.compat.v1.get_variable(
          "barrier_op_action",
          dtype=tf.string,
          initializer="",
          trainable=False,
          collections=[tf.compat.v1.GraphKeys.LOCAL_VARIABLES])
      self._action_placeholder = tf.compat.v1.placeholder(
          tf.string, [], "barrier_op_action_placeholder")
      self._action_assign = self._action.assign(self._action_placeholder,
                                                read_value=False)
      self._lock = threading.Lock()

  def place_barrier(self, session, action: str = ""):
    with self._lock:
      session.run([self._place_op, self._action_assign],
                  feed_dict={
                      self._action_placeholder: action,
                      self._idx_ph: 0
                  })
      self._run_barrier_callbacks(action, session)

  def remove_barrier(self, session):
    session.run(self._remove_op, feed_dict={self._idx_ph: 0})

  def is_barrier_placed(self, session):
    return session.run(self.barrier_placed_tensor)

  @property
  def barrier_placed_tensor(self):
    return self._barrier_placed_tensor

  def is_barrier_removed(self, session):
    qsize = session.run(self.barrier_placed_tensor)
    return not qsize

  def wait_until_barrier_removed(self, session, index):
    with self._lock:
      if index <= 0 or index > self._capacity:
        raise ValueError(
            "Index [{}] must be non-negative and less than capacity [{}]. ".
            format(index, self._capacity))
      session.run(self._place_op, feed_dict={self._idx_ph: index})
      action = session.run(self._action).decode()
      self._run_barrier_callbacks(action, session)

      while not self.is_barrier_removed(session):
        logging.log_every_n_seconds(
            logging.INFO,
            "The worker {} waits until barrier removed.".format(index), 60)
        time.sleep(self._wait_seconds)

      session.run(self._remove_op, feed_dict={self._idx_ph: index})

  def is_all_blocked(self, session):
    barriers = session.run(self._barrier_vars)
    count = sum(barriers)
    return count == self._capacity

  def is_none_blocked(self, session):
    barriers = session.run(self._barrier_vars)
    count = sum(barriers)
    return count == 0

  def get_unblocked_indices(self, session):
    barriers = session.run(self._barrier_vars)
    return [i for i in range(self._capacity) if not barriers[i]]

  def get_blocked_indices(self, session):
    barriers = session.run(self._barrier_vars)
    return [i for i in range(self._capacity) if barriers[i]]

  def _run_barrier_callbacks(self, action: str, session: tf.compat.v1.Session):
    for callback in self._barrier_callbacks:
      callback(action, session)


class BarrierHook(tf.estimator.SessionRunHook):
  """During training, check the barrier condition for worker."""

  def __init__(self, index, barrier_op: BarrierOp):
    self._index = index
    self._barrier_op = barrier_op

  def before_run(self, run_context):
    return tf.estimator.SessionRunArgs(self._barrier_op.barrier_placed_tensor)

  def after_run(self, run_context, run_values):
    barrier_placed_value = run_values.results
    if barrier_placed_value:
      self._barrier_op.wait_until_barrier_removed(run_context.session,
                                                  self._index)
