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
import time

import tensorflow as tf

from absl import logging
from freezegun import freeze_time
from tensorflow.python.platform import test

from monolith.native_training import session_run_hooks


class MockDateTime:

  def __init__(self, hour, minute):
    self.hour = hour
    self.minute = minute


class GlobalStepWaiterHookTest(tf.test.TestCase):

  def test_not_wait_for_step_zero(self):
    with tf.compat.v1.Graph().as_default():
      tf.compat.v1.train.get_or_create_global_step()
      hook = session_run_hooks.CustomGlobalStepWaiterHook(wait_until_step=0)
      hook.begin()
      with tf.compat.v1.Session() as sess:
        # Before run should return without waiting gstep increment.
        hook.before_run(
            tf.estimator.SessionRunContext(original_args=None, session=sess))

  @freeze_time("2012-01-14 10:00:00")
  def test_not_wait_if_tide_not_available(self):
    with tf.compat.v1.Graph().as_default():
      tf.compat.v1.train.get_or_create_global_step()
      hook = session_run_hooks.CustomGlobalStepWaiterHook(wait_until_step=0,
                                                          tide_start_hour=1,
                                                          tide_start_minute=0,
                                                          tide_end_hour=5,
                                                          tide_end_minute=0)
      hook.begin()
      with tf.compat.v1.Session() as sess:
        # Before run should return without waiting gstep increment.
        hook.before_run(
            tf.estimator.SessionRunContext(original_args=None, session=sess))

  @test.mock.patch.object(time, 'sleep')
  def test_wait_for_step(self, mock_sleep):
    with tf.compat.v1.Graph().as_default():
      gstep = tf.compat.v1.train.get_or_create_global_step()
      hook = session_run_hooks.CustomGlobalStepWaiterHook(wait_until_step=1000)
      hook.begin()

      with tf.compat.v1.Session() as sess:
        # Mock out calls to time.sleep() to update the global step.

        class Context(object):
          counter = 0

        def mock_sleep_side_effect(seconds):
          del seconds  # argument is ignored
          Context.counter += 1
          if Context.counter == 1:
            # The first time sleep() is called, we update the global_step from
            # 0 to 500.
            sess.run(tf.compat.v1.assign(gstep, 500))
          elif Context.counter == 2:
            # The second time sleep() is called, we update the global_step from
            # 500 to 1100.
            sess.run(tf.compat.v1.assign(gstep, 1100))
          else:
            raise AssertionError(
                'Expected before_run() to terminate after the second call to '
                'time.sleep()')

        mock_sleep.side_effect = mock_sleep_side_effect

        # Run the mocked-out interaction with the hook.
        self.evaluate(tf.compat.v1.global_variables_initializer())
        run_context = tf.estimator.SessionRunContext(original_args=None,
                                                     session=sess)
        hook.before_run(run_context)
        self.assertEqual(Context.counter, 2)


class MockSessionRunContext:

  def __init__(self):
    self.requested_stop = False

  def request_stop(self):
    logging.info("stop requested")
    self.requested_stop = True
    logging.info(self.requested_stop)


class TideStoppingHookTest(tf.test.TestCase):

  @freeze_time("2012-01-14 10:00:00")
  def test_stop_if_tide_not_available(self):
    with tf.compat.v1.Graph().as_default():
      hook = session_run_hooks.TideStoppingHook(tide_start_hour=1,
                                                tide_start_minute=0,
                                                tide_end_hour=5,
                                                tide_end_minute=0)
      hook.begin()
      with tf.compat.v1.Session() as _:
        # Before run should return without waiting gstep increment.
        context = MockSessionRunContext()
        hook.before_run(context)
        self.assertEqual(context.requested_stop, True)

  @freeze_time("2012-01-14 10:00:00")
  def test_do_not_stop_if_tide_available(self):
    with tf.compat.v1.Graph().as_default():
      hook = session_run_hooks.TideStoppingHook(tide_start_hour=1,
                                                tide_start_minute=0,
                                                tide_end_hour=12,
                                                tide_end_minute=0)
      hook.begin()
      with tf.compat.v1.Session() as _:
        # Before run should return without waiting gstep increment.
        context = MockSessionRunContext()
        hook.before_run(context)
        self.assertEqual(context.requested_stop, False)


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
