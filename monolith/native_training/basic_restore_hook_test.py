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
from tensorflow.python.training import session_run_hook

from monolith.native_training import basic_restore_hook


class CountCheckpointRestorerListener(
    basic_restore_hook.CheckpointRestorerListener):

  def __init__(self):
    self.begin_count = 0
    self.before_restore_count = 0
    self.after_restore_count = 0

  def begin(self):
    self.begin_count += 1

  def before_restore(self, session):
    self.before_restore_count += 1

  def after_restore(self, session):
    self.after_restore_count += 1

  def get_counts(self):
    return {
        'begin': self.begin_count,
        'before_restore': self.before_restore_count,
        'after_restore': self.after_restore_count
    }


class CountHook(session_run_hook.SessionRunHook):

  def __init__(self):
    self.after_create_session_count = 0
    self.before_run_count = 0
    self.after_run_count = 0
    self.end_count = 0

  def after_create_session(self, session, coord):
    self.after_create_session_count += 1

  def before_run(self, run_context):
    self.before_run_count += 1

  def after_run(self, run_context, run_values):
    self.after_run_count += 1

  def end(self, session):
    self.end_count += 1

  def get_counts(self):
    return {
        'after_create_session': self.after_create_session_count,
        'before_run': self.before_run_count,
        'after_run': self.after_run_count,
        'end': self.end_count,
    }


class CheckpointRestorerHookTest(tf.test.TestCase):

  def test_restore_only_in_after_create_session(self):
    with tf.compat.v1.Graph().as_default():
      global_step = tf.compat.v1.train.get_or_create_global_step()
      train_op = tf.compat.v1.assign_add(global_step, 1)
      listener = CountCheckpointRestorerListener()
      hook1 = basic_restore_hook.CheckpointRestorerHook(listeners=[listener])
      hook2 = CountHook()

      with tf.compat.v1.train.SingularMonitoredSession(
          hooks=[hook1, hook2]) as sess:
        # after_create_session
        self.assertEqual({
            'begin': 1,
            'before_restore': 1,
            'after_restore': 1,
        }, listener.get_counts())
        self.assertEqual(
            {
                'after_create_session': 1,
                'before_run': 0,
                'after_run': 0,
                'end': 0,
            }, hook2.get_counts())

        for _ in range(2):
          sess.run(train_op)

    self.assertEqual({
        'begin': 1,
        'before_restore': 1,
        'after_restore': 1,
    }, listener.get_counts())
    self.assertEqual(
        {
            'after_create_session': 1,
            'before_run': 2,
            'after_run': 2,
            'end': 1,
        }, hook2.get_counts())

  def test_two_listeners_with_restorer(self):
    with tf.compat.v1.Graph().as_default():
      global_step = tf.compat.v1.train.get_or_create_global_step()
      train_op = tf.compat.v1.assign_add(global_step, 1)
      listener1 = CountCheckpointRestorerListener()
      listener2 = CountCheckpointRestorerListener()
      hook = basic_restore_hook.CheckpointRestorerHook(
          listeners=[listener1, listener2])

      with tf.compat.v1.train.SingularMonitoredSession(hooks=[hook]) as sess:
        self.assertEqual({
            'begin': 1,
            'before_restore': 1,
            'after_restore': 1,
        }, listener1.get_counts())
        self.assertEqual(listener1.get_counts(), listener1.get_counts())


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
