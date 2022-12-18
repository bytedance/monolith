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

import os
import time
from unittest import mock

import tensorflow as tf

from monolith.native_training import barrier_ops
from monolith.native_training.hooks import controller_hooks


class ControllerHookTest(tf.test.TestCase):

  def testStop(self):
    helper = controller_hooks.StopHelper()
    op = barrier_ops.BarrierOp(
        1, barrier_callbacks=[helper.create_barrier_callback()])
    h1 = controller_hooks.ControllerHook(barrier_op=op)
    h2 = helper.create_stop_hook()
    dummy = tf.Variable(1)
    with tf.compat.v1.train.SingularMonitoredSession(hooks=[h1, h2]) as sess:
      sess.run(dummy)
      self.assertFalse(sess.should_stop())
      sess.run(h1.stop_op)
      # Session might be stopped. But request_stop might be fetched before
      # we run stop_op, so it is possible that it is not stopped yet.
      if not sess.should_stop():
        # Do a dummy run again. Session must be stopped after this.
        sess.run(dummy)
        self.assertTrue(sess.should_stop())

  def testSave(self):
    trigger_save = mock.MagicMock()
    h = controller_hooks.ControllerHook(trigger_save=trigger_save)
    dummy = tf.Variable(1)
    with tf.compat.v1.train.SingularMonitoredSession(hooks=[h]) as sess:
      sess.run(h.trigger_save_op)
      sess.run(dummy)
      sess.run(dummy)
      trigger_save.assert_called_once()


class QueryActionHookTest(tf.test.TestCase):

  @mock.patch("monolith.native_training.hooks.controller_hooks.QUERY_INTERVAL",
              0.1)
  def testStop(self):
    model_dir = os.path.join(os.environ["TEST_TMPDIR"],
                             "QueryActionHookTest_testStop")
    trigger_save = mock.MagicMock()
    h = controller_hooks.ControllerHook(trigger_save=trigger_save)
    qh = controller_hooks.QueryActionHook(model_dir, h)
    dummy = tf.constant(0)
    with tf.compat.v1.train.SingularMonitoredSession(hooks=[h, qh]) as sess:
      tf.io.gfile.makedirs(model_dir)
      query_path = os.path.join(model_dir, "monolith_action")
      with tf.io.gfile.GFile(query_path, "w") as f:
        f.write("action: TRIGGER_SAVE")
      now = time.time()
      while time.time() - now < 60 and tf.io.gfile.exists(query_path):
        time.sleep(0.1)
      sess.run(dummy)
      sess.run(dummy)
      trigger_save.assert_called_once()


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
