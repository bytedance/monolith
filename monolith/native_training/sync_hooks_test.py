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

from monolith.native_training import sync_hooks


class CountHook(tf.estimator.SessionRunHook):

  def __init__(self):
    self.after_create_session_count = 0
    self.end_count = 0

  def after_create_session(self, session, coord):
    self.after_create_session_count += 1

  def end(self, session):
    self.end_count += 1

  def get_counts(self):
    return {
        'after_create_session': self.after_create_session_count,
        'end': self.end_count
    }


def get_local_helper(num_workers):
  return sync_hooks.SyncHelper(num_workers, is_chief=True, var_device=None)


class SyncHooksTest(tf.test.TestCase):

  def _after_create_session(self, sess, hooks):
    for hook in hooks:
      hook.after_create_session(sess, None)

  def _end(self, sess, hooks):
    for hook in hooks:
      hook.end(sess)

  def test_sync_process(self):
    with tf.compat.v1.Graph().as_default():
      helper = get_local_helper(2)
      chief_hook = sync_hooks.ChiefSyncHook(helper)
      worker_hook = sync_hooks.WorkerSyncHook(1, helper)
      worker_count_hook = CountHook()
      chief_count_hook = CountHook()

      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.local_variables_initializer())
        worker = threading.Thread(target=self._after_create_session,
                                  args=(sess, [worker_hook, worker_count_hook]))
        worker.daemon = True
        worker.start()

        time.sleep(1)
        # Worker hook is pending at 'after_create_session'.
        self.assertEqual({
            'after_create_session': 0,
            'end': 0,
        }, worker_count_hook.get_counts())

        chief_hook.after_create_session(sess, None)
        worker.join()
        self.assertEqual({
            'after_create_session': 1,
            'end': 0,
        }, worker_count_hook.get_counts())

        worker_hook.after_create_session(sess, None)
        chief = threading.Thread(target=self._end,
                                 args=(sess, [chief_hook, chief_count_hook]))
        chief.daemon = True
        chief.start()

        # Chief hook is pending at 'end'.
        self.assertEqual({
            'after_create_session': 0,
            'end': 0,
        }, chief_count_hook.get_counts())

        # Make sure logging is covered
        time.sleep(1)
        worker_hook.end(sess)
        chief.join()
        self.assertEqual({
            'after_create_session': 0,
            'end': 1,
        }, chief_count_hook.get_counts())

  def test_hook_helper(self):
    h = sync_hooks.TrainingHooksHelper(False, 0, 0)
    self.assertEqual(h.training_hooks, ())
    self.assertEqual(h.training_chief_hooks, ())

    h = sync_hooks.TrainingHooksHelper(True, 1, 0)
    # This only for grammar check
    h.training_hooks
    h.training_chief_hooks


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
