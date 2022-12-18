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
from cProfile import run
import os
import uuid
import tempfile
import time
from datetime import datetime

from absl import logging
import grpc
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.training import training_util
from tensorflow.python.training import session_run_hook

from tensorflow_serving.apis import predict_pb2, get_model_metadata_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from monolith.agent_service import backends
try:
  from monolith.coordinator.utils import token_utils
except ImportError:
  pass
from monolith.native_training import distributed_serving_ops
from monolith.native_training import hash_table_ops
from monolith.native_training.distributed_serving_ops import ParameterSyncClient, refresh_sync_config
from monolith.utils import find_main

enable_hvd = os.getenv("MONOLITH_WITH_HOROVOD")
if enable_hvd != None:
  import horovod.tensorflow as hvd
  from horovod.tensorflow import mpi_ops


class SyncTrainingBarrierSaverListener(tf.estimator.CheckpointSaverListener):

  def begin(self):
    self._barrier_op = None
    if enable_hvd != None:
      self._barrier_var = tf.compat.v1.placeholder(dtype=tf.int64,
                                                   shape=[],
                                                   name="hvd_export_barrier_ph")
      self._barrier_op = mpi_ops.broadcast(tf.identity(self._barrier_var), 0)

  def after_save(self, session, global_step_value):
    if enable_hvd:
      logging.info(f"exporter barrier begin {hvd.rank()}")
      try:
        barrier_val = session.run(
            self._barrier_op, feed_dict={self._barrier_var: global_step_value})
        logging.info(f"exporter barrier end {hvd.rank()} value: {barrier_val}")
      except Exception as ex:
        logging.error(f"barrier error: {ex}")


class ParameterSyncHook(session_run_hook.SessionRunHook):
  """
  sync parameter sync to online ps
  """

  def __init__(self, sync_backend, ps_index, refresh_interval=100):
    self._sync_backend = sync_backend
    self._ps_index = ps_index
    self._refresh_interval = refresh_interval
    self._last_sync_time = 0
    self._last_refresh_time = 0
    self._sync_config = None
    logging.info(
        f"sync hook for ps_{self._ps_index} with refresh_interval={self._refresh_interval}"
    )

  def begin(self):
    self._config_ph = tf.compat.v1.placeholder(tf.string,
                                               shape=(),
                                               name="sync_config_str")
    sync_client = ParameterSyncClient(
        distributed_serving_ops.parameter_sync_client_from_config(
            name_suffix=str(self._ps_index)))
    self._sync_run_step = sync_client.create_sync_op(self._config_ph)

  def before_run(self, run_context):
    cur_time = time.time()
    if cur_time - self._last_refresh_time >= self._refresh_interval:
      self._sync_config = refresh_sync_config(self._sync_backend,
                                              self._ps_index)
      self._last_refresh_time = cur_time

    return session_run_hook.SessionRunArgs(
        fetches=self._sync_run_step,
        feed_dict={self._config_ph: self._sync_config})


class SyncTrainingForceDumpHook(tf.estimator.SessionRunHook):

  def __init__(self, model_dir, target_timer, step_interval=100):
    self._model_dir = model_dir
    self._target_timer = target_timer
    self._step_interval = step_interval

  def begin(self):
    self._global_step_tensor = training_util._get_or_create_global_step_read()  # pylint: disable=protected-access
    self._ctrl_ph = tf.compat.v1.placeholder(tf.int16,
                                             shape=(3,),
                                             name='hvd_dump_ctrl')
    self._broadcast_op = mpi_ops.broadcast(self._ctrl_ph, 0)

  def after_run(self, run_context, run_values):
    global_step = run_context.session.run(self._global_step_tensor)
    if global_step % self._step_interval == 0:
      utc_hour = datetime.utcnow().hour
      should_dump, should_stop, timer_enabled = 0, 0, 0
      if hvd.rank() == 0:
        timer_enabled = int(utc_hour >= 18 and utc_hour <= 20)
        logging.info(f"utc_hour: {utc_hour} time_enabled: {timer_enabled}")
        dump_path = os.path.join(self._model_dir, f"dump_{global_step}")
        stop_path = os.path.join(self._model_dir, f"stop_{global_step}")
        should_stop = int(tf.io.gfile.exists(stop_path))
        logging.info(f"checked stop {stop_path} {should_stop}")
        should_dump = int(tf.io.gfile.exists(dump_path))
        logging.info(f"checked dump {dump_path} {should_dump}")

      try:
        should_stop, should_dump, timer_enabled = run_context.session.run(
            self._broadcast_op,
            feed_dict={
                self._ctrl_ph: [should_stop, should_dump, timer_enabled]
            },
            options=tf.compat.v1.RunOptions(timeout_in_ms=1000 * 10))
      except (RuntimeError, TypeError, ValueError, tf.errors.OpError) as ex:
        logging.error('Error occurred in syncing control flags: %s', str(ex))

      if timer_enabled:
        logging.info(f"enable timer with utc_hour: {utc_hour}")
        self._target_timer.enable()
      else:
        logging.info(f"disable timer with utc_hour: {utc_hour}")
        self._target_timer.disable()

      if should_dump or should_stop:
        logging.info(f"reset and enable timer for dump at step {global_step}")
        self._target_timer.enable()
        self._target_timer.reset()

      if should_stop:
        logging.info(f"request stop at step {global_step}")
        run_context.request_stop()


class SyncTrainingSaverControlHook(tf.estimator.SessionRunHook):

  def __init__(self, model_dir, target_timer, step_interval=100):
    self._model_dir = model_dir
    self._target_timer = target_timer
    self._step_interval = step_interval

  def begin(self):
    self._global_step_tensor = training_util._get_or_create_global_step_read()  # pylint: disable=protected-access

  def after_run(self, run_context, run_values):
    global_step = run_context.session.run(self._global_step_tensor)
    if global_step % self._step_interval == 0:
      check_path = os.path.join(self._model_dir, "ONLINE")
      if tf.io.gfile.exists(check_path):
        logging.info(f"{check_path} exists, enable timer")
        self._target_timer.enable()
      else:
        logging.info(f"{check_path} not exists, disable timer")
        self._target_timer.disable()




class SyncTrainingInfoHook(tf.estimator.SessionRunHook):

  def begin(self):
    self._last_timestamp = 0
    self._fetches = {}
    for table in ops.get_collection(hash_table_ops._HASH_TABLE_GRAPH_KEY):
      tensor_prefix = hash_table_ops._table_tensor_prefix(table)
      self._fetches[tensor_prefix] = table.size()

  def before_run(self, run_context):
    cur_time = int(time.time())
    if cur_time > self._last_timestamp + 600:
      self._last_timestamp = cur_time
      return tf.estimator.SessionRunArgs(self._fetches)
    else:
      return None

  def after_run(self, run_context, run_values):
    if run_values.results:
      logging.info("*** info: {}".format(run_values.results))


class ReqTimeControlDumpHook(tf.estimator.SessionRunHook):

  def __init__(self, model_dir, target_timer, step_interval=1000):
    self._model_dir = model_dir
    self._target_timer = target_timer
    self._step_interval = step_interval

  def begin(self):
    if hvd.rank() == 0:
      req_time_col = tf.compat.v1.get_collection("req_time")
      assert len(req_time_col) == 1
      self._req_time = tf.math.reduce_max(req_time_col[0])
    else:
      self._req_time = None

    self._global_step_tensor = training_util._get_or_create_global_step_read()  # pylint: disable=protected-access

    self._req_time_ph = tf.compat.v1.placeholder(tf.int64,
                                                 shape=[2],
                                                 name="hvd_req_time")
    self._req_time_bcast_op = mpi_ops.broadcast(self._req_time_ph, 0)

  def before_run(self, run_context):
    if hvd.rank() == 0:
      return session_run_hook.SessionRunArgs(
          fetches={'req_time': self._req_time})
    else:
      return None

  def after_run(self, run_context, run_values):
    global_step = run_context.session.run(self._global_step_tensor)
    if global_step % self._step_interval == 0:
      if hvd.rank() == 0:
        req_time = run_values.results['req_time']
        file_name = os.path.join(self._model_dir, "limit_req_time")
        if tf.io.gfile.exists(file_name):
          with tf.io.gfile.GFile(file_name) as f:
            limit_req_time = int(f.read())
        else:
          limit_req_time = -1
      else:
        req_time = 0
        limit_req_time = -1
      req_time0, limit_req_time0 = run_context.session.run(
          self._req_time_bcast_op,
          feed_dict={self._req_time_ph: [req_time, limit_req_time]})

      if req_time0 >= limit_req_time0 and limit_req_time0 > 0:
        self._target_timer.enable()
        self._target_timer.reset()
        run_context.request_stop()

    return super().after_run(run_context, run_values)
