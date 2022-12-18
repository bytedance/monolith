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

import contextlib
import os
import threading
import time
import traceback
from typing import Callable

from absl import logging
from google.protobuf import text_format
import tensorflow as tf

from monolith.native_training.hooks import controller_hooks_pb2
from monolith.native_training import barrier_ops
from monolith.native_training import utils

STOP_ACTION = "Stop"


class ControllerHook(tf.estimator.SessionRunHook):

  def __init__(self,
               num_ps=0,
               barrier_op: barrier_ops.BarrierOp = None,
               trigger_save: Callable = None):
    self._barrier_op = barrier_op
    self._trigger_save = trigger_save
    device_ctx = tf.device(
        utils.ps_device(0)) if num_ps > 0 else contextlib.nullcontext()
    with tf.name_scope("monolith_controller_hook"), device_ctx:
      self._control_var = tf.compat.v1.get_local_variable(
          "control_var", initializer=[False, False], trainable=False)
      self._stop_op = self._control_var[0].assign(True)
      self._trigger_save_op = self._control_var[1].assign(True)
      self._reset_trigger_save_op = self._control_var[1].assign(False)

  @property
  def stop_op(self):
    return self._stop_op

  @property
  def trigger_save_op(self):
    return self._trigger_save_op

  def before_run(self, run_context):
    return tf.estimator.SessionRunArgs(self._control_var)

  def after_run(self, run_context, run_values):
    if run_values.results[0]:
      if self._barrier_op:
        self._barrier_op.place_barrier(run_context.session, action=STOP_ACTION)
        logging.info("Trying to stop all workers.")
        start_time = time.time()
        while time.time(
        ) - start_time < 30 and not self._barrier_op.is_all_blocked(
            run_context.session):
          time.sleep(2)
        self._barrier_op.remove_barrier(run_context.session)
    elif run_values.results[1]:
      run_context.session.run(self._reset_trigger_save_op)
      if self._trigger_save:
        self._trigger_save()


class _StopHook(tf.estimator.SessionRunHook):

  def __init__(self, should_stop_fn):
    self._should_stop_fn = should_stop_fn

  def after_run(self, run_context, run_values):
    if self._should_stop_fn():
      run_context.request_stop()


class StopHelper:

  def __init__(self):
    self._should_stop = False

  def create_barrier_callback(self):

    def callback(action: str, sess: tf.compat.v1.Session):
      if action != STOP_ACTION:
        return
      self._should_stop = True
      logging.info("Receive the request to stop the training.")

    return callback

  def create_stop_hook(self):

    def should_stop():
      return self._should_stop

    return _StopHook(should_stop)


QUERY_INTERVAL = 60


class QueryActionHook(tf.estimator.SessionRunHook):

  def __init__(self, model_dir: str, hook: ControllerHook):
    self._query_path = os.path.join(model_dir, "monolith_action")
    self._resp_path = os.path.join(model_dir, "monolith_action_response")
    self._hook = hook
    self._session = None
    self._th = None
    self._close = threading.Event()

  def after_create_session(self, session, coord):
    self._session = session
    self._th = threading.Thread(name="QuertActionHookThread",
                                target=self._query_loop,
                                daemon=True)
    self._th.start()

  def end(self, session):
    self._close.set()
    if self._th:
      self._th.join()

  def _query_loop(self):
    while True:
      if self._close.wait(timeout=QUERY_INTERVAL):
        break
      try:
        self._query()
      except:
        logging.error(traceback.format_exc())

  def _query(self):
    if not tf.io.gfile.exists(self._query_path):
      return
    with tf.io.gfile.GFile(self._query_path, "r") as f:
      text_proto = f.read()
    try:
      proto = controller_hooks_pb2.ControllerHooksProto()
      try:
        text_format.Parse(text_proto, proto)
      except text_format.ParseError as e:
        self._write_resp(str(e))
        return
      if proto.action == controller_hooks_pb2.ControllerHooksProto.TRIGGER_SAVE:
        self._session.run(self._hook.trigger_save_op)
      elif proto.action == controller_hooks_pb2.ControllerHooksProto.STOP:
        self._session.run(self._hook.stop_op)
      else:
        self._write_resp("Unknown action: ", text_proto)
        return
      self._write_resp("OK")
    finally:
      tf.io.gfile.remove(self._query_path)

  def _write_resp(self, content: str):
    with tf.io.gfile.GFile(self._resp_path, "w") as f:
      f.write(content)
