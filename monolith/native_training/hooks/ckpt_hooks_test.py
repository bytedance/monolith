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
import threading

from absl import logging

import tensorflow as tf

from monolith.native_training.hooks import ckpt_hooks
from monolith.native_training.hooks import ckpt_hooks_pb2
from monolith.native_training import barrier_ops
from monolith.native_training import save_utils


class CountCheckpointSaverListener(tf.estimator.CheckpointSaverListener):

  def __init__(self):
    self.begin_count = 0
    self.before_save_count = 0
    self.after_save_count = 0

  def begin(self):
    self.begin_count += 1

  def before_save(self, session, global_step):
    self.before_save_count += 1

  def after_save(self, session, global_step):
    self.after_save_count += 1

  def get_counts(self):
    return {
        'begin': self.begin_count,
        'before_save': self.before_save_count,
        'after_save': self.after_save_count
    }


class FixedSessionCreator(tf.compat.v1.train.SessionCreator):

  def __init__(self, fixed_sess):
    self._sess = fixed_sess

  def create_session(self):
    return self._sess


class WorkerCkptHooksTest(tf.test.TestCase):

  def testIteratorSaveRestore(self):
    model_dir = os.path.join(os.environ["TEST_TMPDIR"], "iterator_save")
    tf.io.gfile.makedirs(model_dir)
    ds = tf.data.Dataset.from_tensor_slices([0, 1, 2, 3])
    it = tf.compat.v1.data.make_one_shot_iterator(ds)
    next_ele = it.get_next()
    helper = ckpt_hooks.WorkerCkptHelper(model_dir, 0)
    with self.session() as sess:
      sess.run(next_ele)
      ckpt_hooks.assign_ckpt_info(sess,
                                  ckpt_hooks_pb2.WorkerCkptInfo(global_step=10))
      save_callback = helper.create_save_iterator_callback()
      save_callback(ckpt_hooks.SAVE_ACTION, sess)

      self.assertAllEqual(sess.run(next_ele), 1)

    with tf.compat.v1.train.MonitoredSession(
        hooks=[helper.create_restorer_hook()]) as sess:
      # Restore happens
      self.assertAllEqual(sess.run(next_ele), 1)

  def testNoCkpt(self):
    model_dir = os.path.join(os.environ["TEST_TMPDIR"], "no_ckpt")
    helper = ckpt_hooks.WorkerCkptHelper(model_dir, 0)
    with tf.compat.v1.train.MonitoredSession(
        hooks=[helper.create_restorer_hook()]) as sess:
      pass

  def testNoSaveables(self):
    model_dir = os.path.join(os.environ["TEST_TMPDIR"], "no_saveables")
    tf.io.gfile.makedirs(model_dir)
    helper = ckpt_hooks.WorkerCkptHelper(model_dir, 0)
    with self.session() as sess:
      ckpt_hooks.assign_ckpt_info(sess,
                                  ckpt_hooks_pb2.WorkerCkptInfo(global_step=10))
      save_callback = helper.create_save_iterator_callback()
      save_callback(ckpt_hooks.SAVE_ACTION, sess)

  def testCkptDisabled(self):
    model_dir = os.path.join(os.environ["TEST_TMPDIR"], "ckpt_disabled")
    tf.io.gfile.makedirs(model_dir)
    ds = tf.data.Dataset.from_tensor_slices([0, 1, 2, 3])
    it = tf.compat.v1.data.make_one_shot_iterator(ds)
    next_ele = it.get_next()
    ckpt_hooks.disable_iterator_save_restore()
    helper = ckpt_hooks.WorkerCkptHelper(model_dir, 0)
    with self.session() as sess:
      sess.run(next_ele)
      ckpt_hooks.assign_ckpt_info(sess,
                                  ckpt_hooks_pb2.WorkerCkptInfo(global_step=10))
      save_callback = helper.create_save_iterator_callback()
      save_callback(ckpt_hooks.SAVE_ACTION, sess)

      self.assertAllEqual(sess.run(next_ele), 1)

    with tf.compat.v1.train.MonitoredSession(
        hooks=[helper.create_restorer_hook()]) as sess:
      # Restore should not happen
      self.assertAllEqual(sess.run(next_ele), 0)

  def test_saver_with_barrier(self):
    model_dir = os.path.join(os.environ["TEST_TMPDIR"], "saver_with_barrier")
    global_step = tf.compat.v1.train.get_or_create_global_step()
    train_op = tf.compat.v1.assign_add(global_step, 1)
    barrier_op = barrier_ops.BarrierOp(2, False)
    listener1 = ckpt_hooks.BarrierSaverListener(barrier_op)
    listener2 = CountCheckpointSaverListener()
    hook = save_utils.NoFirstSaveCheckpointSaverHook(
        model_dir,
        save_steps=1,
        listeners=[listener1, listener2],
        saver=tf.compat.v1.train.Saver())
    with tf.compat.v1.Session() as sess:

      g = tf.compat.v1.get_default_graph()
      sess.run(tf.compat.v1.global_variables_initializer())
      sess.run(tf.compat.v1.local_variables_initializer())

      def run():
        with g.as_default(), tf.compat.v1.train.MonitoredSession(
            session_creator=FixedSessionCreator(sess),
            hooks=[hook]) as mon_sess:
          mon_sess.run(train_op)

      worker = threading.Thread(target=run)
      worker.daemon = True
      worker.start()
      while not barrier_op.is_barrier_placed(sess):
        time.sleep(0.1)
      # Barrier is placed by save listener.
      self.assertEqual(1, sess.run(global_step))
      self.assertEqual({
          'begin': 1,
          'before_save': 0,
          'after_save': 0,
      }, listener2.get_counts())

      print("Start to wait")
      barrier_op.wait_until_barrier_removed(sess, 1)
      worker.join()
      self.assertEqual({
          'begin': 1,
          'before_save': 1,
          'after_save': 1,
      }, listener2.get_counts())


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
