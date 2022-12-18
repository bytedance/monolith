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
from unittest import mock

import tensorflow as tf

from monolith.native_training import barrier_ops
from monolith.native_training import logging_ops
from monolith.native_training.hooks import ps_check_hooks
from monolith.native_training.runtime.ops import logging_ops_pb2


class PrepareMachineInfoHook(tf.estimator.SessionRunHook):
  """Used to create machine info after session creation"""

  def __init__(self, machine_info):
    self._machine_info = machine_info

  def after_create_session(self, session, coord):
    session.run(self._machine_info)


class RaiseErrorHook(tf.estimator.SessionRunHook):

  def __init__(self,
               raise_in_after_create_session=False,
               raise_in_before_run=False):
    self.raise_in_after_create_session = raise_in_after_create_session
    self.raise_in_before_run = raise_in_before_run
    self.exc = tf.errors.DeadlineExceededError(None, None, "Test exception")

  def after_create_session(self, session, coord):
    if self.raise_in_after_create_session:
      raise self.exc

  def before_run(self, run_context):
    if self.raise_in_before_run:
      print("RAISEd")
      raise self.exc


class PsCheckHooksTest(tf.test.TestCase):

  def _set_up_hook(self, report_fn=None, mem_limit=1 << 60):
    op = barrier_ops.BarrierOp(1)
    report_fn = report_fn or ps_check_hooks._default_report
    config = ps_check_hooks.Config(barrier_op=op,
                                   num_ps=1,
                                   ps_device_fn=lambda idx: None,
                                   report_fn=report_fn)
    machine_info = logging_ops.machine_info(
        mem_limit=mem_limit,
        shared_name=ps_check_hooks.get_ps_machine_info_shared_name(0))
    return [
        PrepareMachineInfoHook(machine_info),
        ps_check_hooks.PsHealthCheckerHook(config)
    ]

  def test_basic(self):
    hooks = self._set_up_hook()
    with tf.compat.v1.train.SingularMonitoredSession(hooks=hooks):
      time.sleep(1)

  def test_oom(self):
    report_fn = mock.MagicMock()
    hooks = self._set_up_hook(report_fn, mem_limit=0)
    with tf.compat.v1.train.SingularMonitoredSession(hooks=hooks):
      time.sleep(1)
    report_fn.assert_called_once()

  def test_raise_in_after_create_session(self):
    hooks = self._set_up_hook()

    def run():
      with tf.compat.v1.train.SingularMonitoredSession(
          hooks=hooks + [RaiseErrorHook(raise_in_after_create_session=True)]):
        pass

    self.assertRaises(tf.errors.DeadlineExceededError, run)

  def test_raise_in_before_run(self):
    hooks = self._set_up_hook()

    def run():
      t = tf.constant(1.0)
      with tf.compat.v1.train.SingularMonitoredSession(
          hooks=hooks + [RaiseErrorHook(raise_in_before_run=True)]) as sess:
        sess.run(t)

    self.assertRaises(tf.errors.DeadlineExceededError, run)

  def test_default_report(self):
    # This mainly for grammar check
    ps_check_hooks._default_report({1: logging_ops_pb2.MachineHealthResult()})


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
