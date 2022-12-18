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

import threading
from typing import Callable, Dict, NamedTuple

from absl import logging
from google.protobuf import text_format
import tensorflow as tf

from monolith.native_training import barrier_ops
from monolith.native_training import logging_ops
from monolith.native_training import utils
from monolith.native_training.runtime.ops import logging_ops_pb2


def get_ps_machine_info_shared_name(index: int):
  return f"ps_machine_info_{index}"


def _default_report(results: Dict[int, logging_ops_pb2.MachineHealthResult]):
  debugging_strs = []
  for idx, result in results.items():
    debugging_strs.append(
        f"PS {idx}: {text_format.MessageToString(result, as_one_line=True)}")
  logging.error("PS are not healthy:\n%s", "\n".join(debugging_strs))
  # TODO(leqi.zou): Give some alerts


class Config(NamedTuple):
  barrier_op: barrier_ops.BarrierOp
  num_ps: int
  ps_device_fn: Callable[[int], str] = utils.ps_device
  report_fn: Callable[[Dict[int, str]], None] = _default_report


class _PsHealthChecker:

  def __init__(self, config: Config):
    self._config = config
    # self._cancel = threading.Event()
    self._machine_status_tensors = []
    for i in range(config.num_ps):
      with tf.device(config.ps_device_fn(i)):
        handle = logging_ops.machine_info(
            shared_name=get_ps_machine_info_shared_name(i))
        self._machine_status_tensors.append(
            logging_ops.check_machine_health(handle))

  def create_threads(self, sess, coord: tf.train.Coordinator):
    # Daemon is important. It seems that if we have the error in the
    # after_create_session phase, the coordinator will never stop so
    # the process will be stuck forever.
    th = threading.Thread(target=self._run, args=(sess, coord), daemon=True)
    coord.register_thread(th)
    th.start()

  def _run(self, sess, coord: tf.train.Coordinator):
    while not coord.should_stop():
      status_list = sess.run(self._machine_status_tensors)
      results = {}
      should_stop = False
      for idx, status in enumerate(status_list):
        if len(status) > 0:
          should_stop = True
          result = logging_ops_pb2.MachineHealthResult()
          result.ParseFromString(status)
          results[idx] = result
      if should_stop:
        self._config.report_fn(results)
        self._config.barrier_op.place_barrier(sess)
        coord.wait_for_stop()
      coord.wait_for_stop(timeout=30.0)


class PsHealthCheckerHook(tf.estimator.SessionRunHook):

  def __init__(self, config: Config):
    self._config = config
    self._checker = None

  def begin(self):
    self._checker = _PsHealthChecker(self._config)

  def after_create_session(self, session, coord):
    self._checker.create_threads(session, coord)
