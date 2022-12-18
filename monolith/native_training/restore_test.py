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

import tensorflow as tf

from monolith.native_training import basic_restore_hook
from monolith.native_training import hash_table_ops
from monolith.native_training import save_utils
from monolith.native_training import utils


def _generate_config(servers, job_name=utils.PS_JOB_NAME):
  """Generates a config based on servers"""
  cluster_def = tf.train.ClusterDef()
  job = cluster_def.job.add()
  job.name = job_name
  for i, server in enumerate(servers):
    job.tasks[i] = server.target[len('grpc://'):]
  session_config = tf.compat.v1.ConfigProto(cluster_def=cluster_def)
  session_config.experimental.share_session_state_in_clusterspec_propagation = True
  return session_config


def _get_id_tensor(x):
  return tf.constant(x, dtype=tf.int64)


class PartialRestoreTest(tf.test.TestCase):

  def build_graph(self):
    with tf.device(utils.ps_device(0)):
      global_step = tf.compat.v1.train.get_or_create_global_step()
      global_step_op = tf.compat.v1.assign_add(global_step, 1)
      v0 = tf.Variable(0, name="v0")
      op0 = tf.compat.v1.assign_add(v0, 1)
      hash_table0 = hash_table_ops.test_hash_table(1, name_suffix="0")
      add_op0 = hash_table0.assign_add(_get_id_tensor([0]),
                                       tf.constant([[1]],
                                                   dtype=tf.float32)).as_op()
      lookup0 = hash_table0.lookup(_get_id_tensor([0]))
    with tf.device(utils.ps_device(1)):
      v1 = tf.Variable(0, name="v1")
      op1 = tf.compat.v1.assign_add(v1, 1)
      hash_table1 = hash_table_ops.test_hash_table(1, name_suffix="1")
      add_op1 = hash_table1.assign_add(_get_id_tensor([1]),
                                       tf.constant([[1]],
                                                   dtype=tf.float32)).as_op()
      lookup1 = hash_table1.lookup(_get_id_tensor([1]))
    return tf.group(global_step_op, op0, op1, add_op0,
                    add_op1), v0, v1, lookup0, lookup1

  def test_restore_with_ps_monitor(self):
    basename = os.path.join(os.environ["TEST_TMPDIR"],
                            "test_restore_with_ps_monitor", "model.ckpt")

    with tf.compat.v1.Graph().as_default():
      train_op, v0, v1, lookup0, lookup1 = self.build_graph()
      ps_monitor = save_utils.PsMonitor(2)
      saver = save_utils.PartialRecoverySaver(sharded=True,
                                              max_to_keep=10,
                                              keep_checkpoint_every_n_hours=2,
                                              ps_monitor=ps_monitor)
      tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.SAVERS, saver)
      saver_listener = hash_table_ops.HashTableCheckpointSaverListener(basename)
      saver_hook = save_utils.NoFirstSaveCheckpointSaverHook(
          os.path.dirname(basename),
          save_steps=1,
          saver=saver,
          listeners=[saver_listener])
      restore_listener = hash_table_ops.HashTableCheckpointRestorerListener(
          basename, ps_monitor)
      restore_hook = basic_restore_hook.CheckpointRestorerHook(
          listeners=[restore_listener])

      server1 = tf.distribute.Server.create_local_server()
      server2 = tf.distribute.Server.create_local_server()
      config = _generate_config([server1, server2])

      # save checkpoint at first session
      with tf.compat.v1.train.SingularMonitoredSession(
          hooks=[restore_hook, saver_hook],
          master=server1.target,
          config=config,
          checkpoint_dir=os.path.dirname(basename)) as mon_sess:
        sess = mon_sess.raw_session()
        sess.run(train_op)
        v0_val = sess.run(v0)
        v1_val = sess.run(v1)
        embedding0 = sess.run(lookup0)
        embedding1 = sess.run(lookup1)

        self.assertAllEqual(v0_val, 1)
        self.assertAllEqual(v1_val, 1)
        self.assertAllEqual(embedding0, [[1]])
        self.assertAllEqual(embedding1, [[1]])

      # change variables at second session
      with tf.compat.v1.Session(server1.target, config=config) as sess:
        sess.run(train_op)
        v0_val = sess.run(v0)
        v1_val = sess.run(v1)
        embedding0 = sess.run(lookup0)
        embedding1 = sess.run(lookup1)

        self.assertAllEqual(v0_val, 2)
        self.assertAllEqual(v1_val, 2)
        self.assertAllEqual(embedding0, [[2]])
        self.assertAllEqual(embedding1, [[2]])

      server3 = tf.distribute.Server.create_local_server()
      server4 = tf.distribute.Server.create_local_server()
      config = _generate_config([server3, server4])

      # restore all variables at third session
      with tf.compat.v1.train.SingularMonitoredSession(
          hooks=[restore_hook, saver_hook],
          master=server3.target,
          config=config,
          checkpoint_dir=os.path.dirname(basename)) as mon_sess:
        sess = mon_sess.raw_session()
        v0_val = sess.run(v0)
        v1_val = sess.run(v1)
        embedding0 = sess.run(lookup0)
        embedding1 = sess.run(lookup1)

        self.assertAllEqual(v0_val, 1)
        self.assertAllEqual(v1_val, 1)
        self.assertAllEqual(embedding0, [[1]])
        self.assertAllEqual(embedding1, [[1]])

      server5 = tf.distribute.Server.create_local_server()
      config = _generate_config([server1, server5])

      # partial restore at fourth session
      with tf.compat.v1.train.SingularMonitoredSession(
          hooks=[restore_hook, saver_hook],
          master=server1.target,
          config=config,
          checkpoint_dir=os.path.dirname(basename)) as mon_sess:
        sess = mon_sess.raw_session()
        v0_val = sess.run(v0)
        v1_val = sess.run(v1)
        embedding0 = sess.run(lookup0)
        embedding1 = sess.run(lookup1)

        self.assertAllEqual(v0_val, 2)
        self.assertAllEqual(v1_val, 1)
        self.assertAllEqual(embedding0, [[2]])
        self.assertAllEqual(embedding1, [[1]])

  def test_restore_without_ps_monitor(self):
    basename = os.path.join(os.environ["TEST_TMPDIR"],
                            "test_restore_without_ps_monitor", "model.ckpt")

    with tf.compat.v1.Graph().as_default():
      train_op, v0, v1, lookup0, lookup1 = self.build_graph()
      saver = save_utils.PartialRecoverySaver(sharded=True,
                                              max_to_keep=10,
                                              keep_checkpoint_every_n_hours=2)
      tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.SAVERS, saver)
      saver_listener = hash_table_ops.HashTableCheckpointSaverListener(basename)
      saver_hook = save_utils.NoFirstSaveCheckpointSaverHook(
          os.path.dirname(basename),
          save_steps=1,
          saver=saver,
          listeners=[saver_listener])
      restore_listener = hash_table_ops.HashTableCheckpointRestorerListener(
          basename)
      restore_hook = basic_restore_hook.CheckpointRestorerHook(
          listeners=[restore_listener])

      server1 = tf.distribute.Server.create_local_server()
      server2 = tf.distribute.Server.create_local_server()
      config = _generate_config([server1, server2])

      # save checkpoint at first session
      with tf.compat.v1.train.SingularMonitoredSession(
          hooks=[restore_hook, saver_hook],
          master=server1.target,
          config=config,
          checkpoint_dir=os.path.dirname(basename)) as mon_sess:
        sess = mon_sess.raw_session()
        sess.run(train_op)
        v0_val = sess.run(v0)
        v1_val = sess.run(v1)
        embedding0 = sess.run(lookup0)
        embedding1 = sess.run(lookup1)

        self.assertAllEqual(v0_val, 1)
        self.assertAllEqual(v1_val, 1)
        self.assertAllEqual(embedding0, [[1]])
        self.assertAllEqual(embedding1, [[1]])

      # change variables at second session
      with tf.compat.v1.Session(server1.target, config=config) as sess:
        sess.run(train_op)
        v0_val = sess.run(v0)
        v1_val = sess.run(v1)
        embedding0 = sess.run(lookup0)
        embedding1 = sess.run(lookup1)

        self.assertAllEqual(v0_val, 2)
        self.assertAllEqual(v1_val, 2)
        self.assertAllEqual(embedding0, [[2]])
        self.assertAllEqual(embedding1, [[2]])

      # restore all variables at third session
      with tf.compat.v1.train.SingularMonitoredSession(
          hooks=[restore_hook, saver_hook],
          master=server1.target,
          config=config,
          checkpoint_dir=os.path.dirname(basename)) as mon_sess:
        sess = mon_sess.raw_session()
        v0_val = sess.run(v0)
        v1_val = sess.run(v1)
        embedding0 = sess.run(lookup0)
        embedding1 = sess.run(lookup1)

        self.assertAllEqual(v0_val, 1)
        self.assertAllEqual(v1_val, 1)
        self.assertAllEqual(embedding0, [[1]])
        self.assertAllEqual(embedding1, [[1]])


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
