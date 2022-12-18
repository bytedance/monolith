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

import tensorflow as tf
import random
import time

from absl import logging
from datetime import datetime
from tensorflow.python.training import training_util


def before(hour1, minute1, hour2, minute2):
  if hour1 < hour2 or (hour1 == hour2 and minute1 < minute2):
    return True
  else:
    return False


def tide_available_now(tide_start_hour, tide_start_minute, tide_end_hour,
                       tide_end_minute):
  if before(tide_start_hour, tide_start_minute, tide_end_hour, tide_end_minute):
    if not before(datetime.utcnow().hour,
                  datetime.utcnow().minute,
                  tide_start_hour, tide_start_minute) and before(
                      datetime.utcnow().hour,
                      datetime.utcnow().minute, tide_end_hour, tide_end_minute):
      return True
    else:
      return False
  else:
    if before(datetime.utcnow().hour,
              datetime.utcnow().minute,
              tide_start_hour, tide_start_minute) or not before(
                  datetime.utcnow().hour,
                  datetime.utcnow().minute, tide_end_hour, tide_end_minute):
      return True
    else:
      return False


class CustomGlobalStepWaiterHook(tf.estimator.SessionRunHook):
  """Delays execution until global step reaches `wait_until_step`.

  This hook delays execution until global step reaches to `wait_until_step`. It
  is used to gradually start workers in distributed settings. One example usage
  would be setting `wait_until_step=int(K*log(task_id+1))` assuming that
  task_id=0 is the chief.
  """

  def __init__(self,
               wait_until_step,
               tide_start_hour=None,
               tide_start_minute=None,
               tide_end_hour=None,
               tide_end_minute=None,
               max_non_tide_wait_minute=10):
    """Initializes a `GlobalStepWaiterHook`.

    Args:
      wait_until_step: an `int` shows until which global step should we wait.
      tide_start_hour: the first hour in utc timezone when tide resources are available.
      tide_end_hour: the last hour in utc timezone when tide resources are available.
    """
    self._wait_until_step = wait_until_step
    self._tide_start_hour = tide_start_hour
    self._tide_start_minute = tide_start_minute
    self._tide_end_hour = tide_end_hour
    self._tide_end_minute = tide_end_minute
    self._hook_start_time = None
    self._non_tide_wait_second = random.randint(
        int(max_non_tide_wait_minute * 6), max_non_tide_wait_minute * 60)

  def begin(self):
    self._worker_is_started = False
    self._global_step_tensor = training_util._get_or_create_global_step_read()  # pylint: disable=protected-access
    if self._global_step_tensor is None:
      raise RuntimeError(
          "Global step should be created to use _GlobalStepWaiterHook.")
    if self._hook_start_time is None:
      self._hook_start_time = time.time()

  def before_run(self, run_context):
    if self._worker_is_started:
      return None

    if self._wait_until_step <= 0:
      self._worker_is_started = True
      return None

    logging.info("Waiting for global step %d before starting training.",
                 self._wait_until_step)
    while True:
      if self._tide_start_hour is not None and self._tide_end_hour is not None:
        if not tide_available_now(self._tide_start_hour,
                                  self._tide_start_minute, self._tide_end_hour,
                                  self._tide_end_minute):
          logging.info("Current UTC time: {} : {}".format(
              datetime.utcnow().hour,
              datetime.utcnow().minute))
          logging.info("Last hour in tide queue. Saving ckpt...")
          run_context.request_stop()
          return

      current_step = run_context.session.run(self._global_step_tensor)
      if current_step >= self._wait_until_step:
        self._worker_is_started = True
        return None

      if self._hook_start_time is not None and time.time(
      ) - self._hook_start_time > self._non_tide_wait_second:
        return None

      logging.log_every_n_seconds(
          logging.INFO, "Waiting for global step {} before starting training. "
          "Current step is {}.".format(self._wait_until_step, current_step), 60)
      time.sleep(0.5)


class TideStoppingHook(tf.estimator.SessionRunHook):

  def __init__(self,
               tide_start_hour=None,
               tide_start_minute=None,
               tide_end_hour=None,
               tide_end_minute=None):
    """Initializes a `GlobalStepWaiterHook`.

    Args:
      wait_until_step: an `int` shows until which global step should we wait.
      tide_start_hour: the first hour in utc timezone when tide resources are available.
      tide_end_hour: the last hour in utc timezone when tide resources are available.
    """
    self._tide_start_hour = tide_start_hour
    self._tide_start_minute = tide_start_minute
    self._tide_end_hour = tide_end_hour
    self._tide_end_minute = tide_end_minute

  def before_run(self, run_context):
    if self._tide_start_hour is not None and self._tide_end_hour is not None:
      if not tide_available_now(self._tide_start_hour, self._tide_start_minute,
                                self._tide_end_hour, self._tide_end_minute):
        logging.info("Current UTC time: {} : {}".format(
            datetime.utcnow().hour,
            datetime.utcnow().minute))
        logging.info("Last hour in tide queue. Saving ckpt...")
        run_context.request_stop()
