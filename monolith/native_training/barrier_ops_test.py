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

import tempfile
import threading
import time

import tensorflow as tf
from tensorflow.python.training import monitored_session

from monolith.native_training import barrier_ops


class BarrierOpsTest(tf.test.TestCase):

  def test_basic(self):
    barrier_op = barrier_ops.BarrierOp(2, False)
    with tf.compat.v1.Session() as sess:
      self.evaluate(tf.compat.v1.global_variables_initializer())
      self.evaluate(tf.compat.v1.local_variables_initializer())
      barrier_op.place_barrier(sess)
      self.assertTrue(barrier_op.is_barrier_placed(sess))
      barrier_op.remove_barrier(sess)
      self.assertTrue(barrier_op.is_barrier_removed(sess))

  def _run(self, train_op, sess, step=1):
    for i in range(step):
      sess.run(train_op)

  def test_barrier_hook_not_blocked(self):
    with tf.compat.v1.Graph().as_default():
      global_step = tf.compat.v1.train.get_or_create_global_step()
      train_op = tf.compat.v1.assign_add(global_step, 1)
      barrier_op = barrier_ops.BarrierOp(2, False)
      hook = barrier_ops.BarrierHook(1, barrier_op)

      with tf.compat.v1.Session() as sess:
        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.evaluate(tf.compat.v1.local_variables_initializer())
        mon_sess = monitored_session._HookedSession(sess, [hook])
        worker = threading.Thread(target=self._run,
                                  args=(train_op, mon_sess, 5))
        worker.daemon = True
        worker.start()
        worker.join()

        self.assertEqual(5, sess.run(global_step))

  def test_barrier_hook_blocked(self):
    with tf.compat.v1.Graph().as_default():
      global_step = tf.compat.v1.train.get_or_create_global_step()
      train_op = tf.compat.v1.assign_add(global_step, 1)

      called_variable = tf.Variable(False, trainable=False)
      barrier_action = "test_action"

      def action_callback(action, session):
        if action == barrier_action:
          session.run(called_variable.assign(True))

      barrier_op = barrier_ops.BarrierOp(2,
                                         False,
                                         barrier_callbacks=[action_callback])
      hook = barrier_ops.BarrierHook(1, barrier_op)

      with tf.compat.v1.Session() as sess:
        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.evaluate(tf.compat.v1.local_variables_initializer())
        mon_sess = monitored_session._HookedSession(sess, [hook])

        barrier_op.place_barrier(sess, action=barrier_action)
        worker = threading.Thread(target=self._run,
                                  args=(train_op, mon_sess, 5))
        worker.daemon = True
        worker.start()

        while not barrier_op.is_all_blocked(sess):
          time.sleep(0.1)
        # Hook is pending.
        self.assertEqual(1, sess.run(global_step))
        self.assertEqual(sess.run(called_variable), True)

        barrier_op.remove_barrier(sess)
        worker.join()
        self.assertTrue(barrier_op.is_none_blocked(sess))
        self.assertEqual(5, sess.run(global_step))


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
