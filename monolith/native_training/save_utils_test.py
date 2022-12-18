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

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

from freezegun import freeze_time
from unittest import mock
import numpy as np
import os
import six
import time

import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import saver_test_utils
from tensorflow.python.training import checkpoint_management

from monolith.native_training import save_utils


class SaveUtilsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._global_step = tf.compat.v1.train.get_or_create_global_step()
    self._saver = save_utils.PartialRecoverySaver(
        exempt_checkpoint_paths=['ckpt-10', 'ckpt-20'], max_to_keep=3)
    self._savepath = os.path.join(
        os.environ["TEST_TMPDIR"],
        type(self).__name__ + "_" + self._testMethodName, "ckpt")

  def create_test_ckpt(self, global_step_value: int):
    with self.session() as sess:
      self._global_step = tf.compat.v1.assign(self._global_step,
                                              global_step_value)
      sess.run(self._global_step)
      self._saver.save(sess, self._savepath, self._global_step)

  def test_get_ckpt_steps(self):
    helper = save_utils.SaveHelper(self._savepath)
    self.create_test_ckpt(10)
    self.create_test_ckpt(20)
    self.create_test_ckpt(300)
    ckpt_steps = helper.get_existing_checkpoint_steps()
    self.assertSetEqual(ckpt_steps, {10, 20, 300})

  def test_exempt_checkpoints(self):
    helper = save_utils.SaveHelper(self._savepath)
    self.create_test_ckpt(10)
    self.create_test_ckpt(20)
    self.create_test_ckpt(30)
    self.create_test_ckpt(40)
    self.create_test_ckpt(50)
    ckpt_steps = helper.get_existing_checkpoint_steps()
    self.assertSetEqual(ckpt_steps, {10, 20, 30, 40, 50})

    self.create_test_ckpt(60)
    ckpt_steps = helper.get_existing_checkpoint_steps()
    self.assertSetEqual(ckpt_steps, {10, 20, 40, 50, 60})


class SaverHookTest(tf.test.TestCase):

  def get_ckpt_dir(self):
    return os.path.join(os.environ["TEST_TMPDIR"],
                        type(self).__name__ + "_" + self._testMethodName)

  def test_basic(self):
    ckpt_dir = self.get_ckpt_dir()
    global_step = tf.compat.v1.train.get_or_create_global_step()
    global_step = tf.compat.v1.assign_add(global_step, 1)
    with tf.compat.v1.train.SingularMonitoredSession([
        save_utils.NoFirstSaveCheckpointSaverHook(checkpoint_dir=ckpt_dir,
                                                  save_steps=100)
    ]) as sess:
      sess.run(global_step)
    print(tf.io.gfile.glob(os.path.join(ckpt_dir, "*")))
    # Will not save at beginning.
    self.assertAllEqual(
        tf.io.gfile.glob(os.path.join(ckpt_dir, "model.ckpt-0\\.*")), [])
    # Will save after when session is closed.
    self.assertGreater(
        len(tf.io.gfile.glob(os.path.join(ckpt_dir, "model.ckpt-1\\.*"))), 0)

  def test_op_error(self):

    class AssertFailListener(tf.estimator.CheckpointSaverListener):

      def __init__(self):
        self._assert_op = tf.debugging.Assert(False, [False])

      def before_save(self, session, global_step_value):
        session.run(self._assert_op)

    class FinallyAfterSaveListener(tf.estimator.CheckpointSaverListener):

      def __init__(self):
        self._called = False

      @property
      def called(self):
        return self._called

      def after_save(self, session, global_step_value):
        self._called = True

    ckpt_dir = self.get_ckpt_dir()
    global_step = tf.compat.v1.train.get_or_create_global_step()
    l = FinallyAfterSaveListener()
    with tf.compat.v1.train.SingularMonitoredSession([
        save_utils.NoFirstSaveCheckpointSaverHook(
            checkpoint_dir=ckpt_dir,
            save_steps=100,
            listeners=[AssertFailListener()],
            finally_after_save_listeners=[l],
            ignore_save_errors=True,
            no_first_save=False)
    ]) as sess:
      pass
    self.assertTrue(l.called)


class SaverTest(tf.test.TestCase):

  def basicSaveRestore(self, variable_op):
    save_path = os.path.join(self.get_temp_dir(), "basic_save_restore")

    with self.session(graph=tf.Graph()) as sess:
      # Build a graph with 2 parameter nodes, and Save and
      # Restore nodes for them.
      v0 = variable_op(10.0, name="v0")
      v1 = variable_op(20.0, name="v1")
      v2 = saver_test_utils.CheckpointedOp(name="v2")
      v2_init = v2.insert("k1", 30.0)

      # Initialize all variables
      if not tf.executing_eagerly():
        self.evaluate([tf.compat.v1.global_variables_initializer(), v2_init])

        # Check that the parameter nodes have been initialized.
      self.assertEqual(10.0, self.evaluate(v0))
      self.assertEqual(20.0, self.evaluate(v1))
      self.assertEqual(b"k1", self.evaluate(v2.keys()))
      self.assertEqual(30.0, self.evaluate(v2.values()))

      # Save the initialized values in the file at "save_path"
      save = save_utils.PartialRecoverySaver(
          {
              "v0": v0,
              "v1": v1,
              "v2": v2.saveable
          }, restore_sequentially=True)
      val = save.save(sess, save_path)
      self.assertTrue(isinstance(val, six.string_types))
      self.assertEqual(save_path, val)

    # Start a second session.  In that session the parameter nodes
    # have not been initialized either.
    with self.session(graph=tf.Graph()) as sess:
      v0 = variable_op(-1.0, name="v0")
      v1 = variable_op(-1.0, name="v1")
      v2 = saver_test_utils.CheckpointedOp(name="v2")

      # Assert that the variables are not initialized.
      if not tf.executing_eagerly():
        self.assertEqual(
            len(tf.compat.v1.report_uninitialized_variables().eval()), 2)
        self.assertEqual(0, len(self.evaluate(v2.keys())))
        self.assertEqual(0, len(self.evaluate(v2.values())))
      # Restore the saved values in the parameter nodes.
      save = save_utils.PartialRecoverySaver({
          "v0": v0,
          "v1": v1,
          "v2": v2.saveable
      })
      save.restore(sess, save_path)
      # Check that the parameter nodes have been restored.
      self.assertEqual(10.0, self.evaluate(v0))
      self.assertEqual(20.0, self.evaluate(v1))
      self.assertEqual(b"k1", self.evaluate(v2.keys()))
      self.assertEqual(30.0, self.evaluate(v2.values()))

    # Build another graph with 2 nodes, initialized
    # differently, and a Restore node for them.
    with self.session(graph=tf.Graph()) as sess:
      v0_2 = variable_op(1000.0, name="v0")
      v1_2 = variable_op(2000.0, name="v1")
      v2_2 = saver_test_utils.CheckpointedOp(name="v2")
      v2_init = v2_2.insert("k1000", 3000.0)

      # Check that the parameter nodes have been initialized.
      if not tf.executing_eagerly():
        init_all_op = [tf.compat.v1.global_variables_initializer(), v2_init]
        self.evaluate(init_all_op)
        # TODO(xpan): Why _mutable_hash_table_v2 doesn't create empty
        # table as it claims in eager mode?
        self.assertEqual(b"k1000", self.evaluate(v2_2.keys()))
        self.assertEqual(3000.0, self.evaluate(v2_2.values()))
      self.assertEqual(1000.0, self.evaluate(v0_2))
      self.assertEqual(2000.0, self.evaluate(v1_2))

      # Restore the values saved earlier in the parameter nodes.
      save2 = save_utils.PartialRecoverySaver({
          "v0": v0_2,
          "v1": v1_2,
          "v2": v2_2.saveable
      })
      save2.restore(sess, save_path)
      # Check that the parameter nodes have been restored.
      self.assertEqual(10.0, self.evaluate(v0_2))
      self.assertEqual(20.0, self.evaluate(v1_2))
      self.assertEqual(b"k1", self.evaluate(v2_2.keys()))
      self.assertEqual(30.0, self.evaluate(v2_2.values()))

  def testBasic(self):
    self.basicSaveRestore(tf.Variable)

  def testSaveMaxToKeep(self):
    save_path = os.path.join(self.get_temp_dir(), "test_save_max_to_keep",
                             "model.ckpt")

    with self.session(graph=tf.Graph()) as sess:
      # Build a graph with 2 parameter nodes, and Save and
      # Restore nodes for them.
      v0 = tf.Variable(10.0, name="v0")
      v1 = tf.Variable(20.0, name="v1")
      v2 = saver_test_utils.CheckpointedOp(name="v2")
      v2_init = v2.insert("k1", 30.0)

      # Initialize all variables
      if not tf.executing_eagerly():
        self.evaluate([tf.compat.v1.global_variables_initializer(), v2_init])

        # Check that the parameter nodes have been initialized.
      self.assertEqual(10.0, self.evaluate(v0))
      self.assertEqual(20.0, self.evaluate(v1))
      self.assertEqual(b"k1", self.evaluate(v2.keys()))
      self.assertEqual(30.0, self.evaluate(v2.values()))

      # Save the initialized values in the file at "save_path"
      save = save_utils.PartialRecoverySaver(
          {
              "v0": v0,
              "v1": v1,
              "v2": v2.saveable
          },
          restore_sequentially=True,
          max_to_keep=2,
          exempt_checkpoint_paths=['model.ckpt-2'])
      val = save.save(sess, save_path, global_step=1)
      self.assertEqual(save_path + '-1', val)
      self.assertGreater(len(tf.io.gfile.glob(save_path + "-1\\.*")), 0)

      val = save.save(sess, save_path, global_step=2)
      self.assertEqual(save_path + '-2', val)
      self.assertGreater(len(tf.io.gfile.glob(save_path + "-1\\.*")), 0)
      self.assertGreater(len(tf.io.gfile.glob(save_path + "-2\\.*")), 0)

      val = save.save(sess, save_path, global_step=3)
      self.assertEqual(save_path + '-3', val)
      self.assertGreater(len(tf.io.gfile.glob(save_path + "-1\\.*")), 0)
      self.assertGreater(len(tf.io.gfile.glob(save_path + "-2\\.*")), 0)
      self.assertGreater(len(tf.io.gfile.glob(save_path + "-3\\.*")), 0)

      val = save.save(sess, save_path, global_step=4)
      self.assertEqual(save_path + '-4', val)
      self.assertEqual(len(tf.io.gfile.glob(save_path + "-1\\.*")), 0)
      self.assertGreater(len(tf.io.gfile.glob(save_path + "-2\\.*")), 0)
      self.assertGreater(len(tf.io.gfile.glob(save_path + "-3\\.*")), 0)
      self.assertGreater(len(tf.io.gfile.glob(save_path + "-4\\.*")), 0)

      save.restore(sess, val)
      # Check that the parameter nodes have been restored.
      self.assertEqual(10.0, self.evaluate(v0))
      self.assertEqual(20.0, self.evaluate(v1))
      self.assertEqual(b"k1", self.evaluate(v2.keys()))
      self.assertEqual(30.0, self.evaluate(v2.values()))

  @test_util.run_in_graph_and_eager_modes
  def testResourceBasic(self):
    self.basicSaveRestore(resource_variable_ops.ResourceVariable)

  def testResourceColocation(self):
    # train.Saver is V1 only API.
    with tf.Graph().as_default():
      partitioner = tf.compat.v1.fixed_size_partitioner(num_shards=2)
      with tf.device("/job:ps/device:GPU:0"):
        v = tf.compat.v1.get_variable("v0",
                                      shape=[10, 2],
                                      partitioner=partitioner,
                                      use_resource=True)
      save_utils.PartialRecoverySaver({"v0": v}).build()
      save_op = None
      for op in tf.compat.v1.get_default_graph().get_operations():
        if op.type == "SaveV2":
          save_op = op
          break
      assert save_op is not None
      for save_inp in save_op.inputs[3:]:
        # Input to SaveV2 op is placed on CPU of the same device as
        # the Variable.
        self.assertEqual("/job:ps/device:CPU:0", save_inp.device)

  def testResourceVariableReadOpsAddedDeterministically(self):
    graph_defs = []
    num_graphs = 10
    for _ in range(num_graphs):
      with tf.Graph().as_default() as g:
        for i in range(20):
          resource_variable_ops.ResourceVariable(i, name="var%s" % i)
        save_utils.PartialRecoverySaver()
        graph_defs.append(g.as_graph_def())
    for i in range(num_graphs - 1):
      self.assertEqual(graph_defs[i], graph_defs[i + 1])

  def testEagerBasic(self):
    with context.eager_mode():
      ckpt_prefix = os.path.join(self.get_temp_dir(), "ckpt")

      v1 = resource_variable_ops.ResourceVariable(3.14, name="v1")
      v2 = resource_variable_ops.ResourceVariable([1, 2], name="v2")
      save = save_utils.PartialRecoverySaver([v1, v2])
      save.save(None, ckpt_prefix)

      v1.assign(0.0)
      v2.assign([0, 0])
      self.assertNear(0.0, self.evaluate(v1), 1e-5)
      self.assertAllEqual([0, 0], self.evaluate(v2))

      save.restore(None, ckpt_prefix)
      self.assertNear(3.14, self.evaluate(v1), 1e-5)
      self.assertAllEqual([1, 2], self.evaluate(v2))

  def testEagerGraphCompatibility(self):
    # Save from graph mode and restore from eager mode.
    graph_ckpt_prefix = os.path.join(self.get_temp_dir(), "graph_ckpt")
    with context.graph_mode():
      with self.session(graph=tf.Graph()) as sess:
        # Create a graph model and save the checkpoint.
        w1 = resource_variable_ops.ResourceVariable(1.0, name="w1")
        w2 = resource_variable_ops.ResourceVariable(2.0, name="w2")
        graph_saver = save_utils.PartialRecoverySaver([w1, w2])
        self.evaluate(tf.compat.v1.global_variables_initializer())
        graph_saver.save(sess, graph_ckpt_prefix)

    with context.eager_mode():
      tf.compat.v1.reset_default_graph()

      w1 = resource_variable_ops.ResourceVariable(0.0, name="w1")
      w2 = resource_variable_ops.ResourceVariable(0.0, name="w2")

      graph_saver = save_utils.PartialRecoverySaver([w1, w2])
      graph_saver.restore(None, graph_ckpt_prefix)

      self.assertAllEqual(self.evaluate(w1), 1.0)
      self.assertAllEqual(self.evaluate(w2), 2.0)

    # Save from eager mode and restore from graph mode.
    eager_ckpt_prefix = os.path.join(self.get_temp_dir(), "eager_ckpt")
    with context.eager_mode():
      tf.compat.v1.reset_default_graph()

      w3 = resource_variable_ops.ResourceVariable(3.0, name="w3")
      w4 = resource_variable_ops.ResourceVariable(4.0, name="w4")

      graph_saver = save_utils.PartialRecoverySaver([w3, w4])
      graph_saver.save(None, eager_ckpt_prefix)

    with context.graph_mode():
      with self.session(graph=tf.Graph()) as sess:
        w3 = resource_variable_ops.ResourceVariable(0.0, name="w3")
        w4 = resource_variable_ops.ResourceVariable(0.0, name="w4")
        graph_saver = save_utils.PartialRecoverySaver([w3, w4])
        self.evaluate(tf.compat.v1.global_variables_initializer())
        graph_saver.restore(sess, eager_ckpt_prefix)
        self.assertAllEqual(w3, 3.0)
        self.assertAllEqual(w4, 4.0)

  @test_util.run_in_graph_and_eager_modes
  def testResourceSaveRestoreCachingDevice(self):
    save_path = os.path.join(self.get_temp_dir(), "resource_cache")
    with self.session(graph=tf.Graph()) as sess:
      v = resource_variable_ops.ResourceVariable([1],
                                                 caching_device="/cpu:0",
                                                 name="v")
      if tf.executing_eagerly():
        sess = None
      else:
        self.evaluate(tf.compat.v1.global_variables_initializer())
      save = save_utils.PartialRecoverySaver([v])
      save.save(sess, save_path)

      save2 = save_utils.PartialRecoverySaver([v])
      save2.restore(sess, save_path)
      self.assertEqual(self.evaluate(v), [1])

  def testNoAdditionalOpsAddedBySaverForResourceVariablesOutsideSaveScope(self):
    with tf.Graph().as_default() as g:
      v = resource_variable_ops.ResourceVariable(1.0, name="v")
      with tf.name_scope("saver1"):
        save_utils.PartialRecoverySaver()
      with tf.name_scope("saver2"):
        save_utils.PartialRecoverySaver({"name": v})
    ops_in_saver1_scope_but_not_save_scope = [
        op for op in g.get_operations()
        if (op.name.startswith("saver1/") and
            not op.name.startswith("saver1/save/"))
    ]
    self.assertEqual(ops_in_saver1_scope_but_not_save_scope, [])
    ops_in_saver2_scope_but_not_save_scope = [
        op for op in g.get_operations()
        if (op.name.startswith("saver2/") and
            not op.name.startswith("saver2/save/"))
    ]
    self.assertEqual(ops_in_saver2_scope_but_not_save_scope, [])

  def testSaveCopyRestoreWithSaveRelativePaths(self):
    """Save, copy checkpoint dir and restore from copied dir.

    This only works for save_relative_paths=True.
    """
    save_dir1 = os.path.join(self.get_temp_dir(), "save_dir1")
    os.mkdir(save_dir1)
    save_path1 = os.path.join(save_dir1, "save_copy_restore")

    # train.Saver is V1 only API.
    with tf.Graph().as_default():
      # Build a graph with 2 parameter nodes, and Save and
      # Restore nodes for them.
      v0 = tf.compat.v1.Variable(10.0, name="v0")
      v1 = tf.compat.v1.Variable(20.0, name="v1")
      v2 = saver_test_utils.CheckpointedOp(name="v2")
      v2_init = v2.insert("k1", 30.0)
      save = save_utils.PartialRecoverySaver(var_list={
          "v0": v0,
          "v1": v1,
          "v2": v2.saveable
      },
                                             restore_sequentially=True,
                                             save_relative_paths=True)
      init_all_op = [tf.compat.v1.global_variables_initializer(), v2_init]

      with self.cached_session() as sess:
        # Initialize all variables
        self.evaluate(init_all_op)

        # Check that the parameter nodes have been initialized.
        self.assertEqual(10.0, self.evaluate(v0))
        self.assertEqual(20.0, self.evaluate(v1))
        self.assertEqual(b"k1", self.evaluate(v2.keys()))
        self.assertEqual(30.0, self.evaluate(v2.values()))

        # Save the initialized values in the file at "save_path"
        val = save.save(sess, save_path1)
        self.assertTrue(isinstance(val, six.string_types))
        self.assertEqual(save_path1, val)

      self.assertEqual(tf.train.latest_checkpoint(save_dir1), save_path1)
      save_dir2 = os.path.join(self.get_temp_dir(), "save_dir2")
      os.renames(save_dir1, save_dir2)
      save_path2 = os.path.join(save_dir2, "save_copy_restore")
      self.assertEqual(tf.train.latest_checkpoint(save_dir2), save_path2)

      # Start a second session.  In that session the parameter nodes
      # have not been initialized either.
      with self.cached_session() as sess:
        v0 = tf.compat.v1.Variable(-1.0, name="v0")
        v1 = tf.compat.v1.Variable(-1.0, name="v1")
        v2 = saver_test_utils.CheckpointedOp(name="v2")
        save = save_utils.PartialRecoverySaver({
            "v0": v0,
            "v1": v1,
            "v2": v2.saveable
        })

        # Assert that the variables are not initialized.
        self.assertEqual(
            len(tf.compat.v1.report_uninitialized_variables().eval()), 2)
        self.assertEqual(0, len(self.evaluate(v2.keys())))
        self.assertEqual(0, len(self.evaluate(v2.values())))

        # Restore the saved values in the parameter nodes.
        save.restore(sess, save_path2)
        # Check that the parameter nodes have been restored.
        self.assertEqual(10.0, self.evaluate(v0))
        self.assertEqual(20.0, self.evaluate(v1))
        self.assertEqual(b"k1", self.evaluate(v2.keys()))
        self.assertEqual(30.0, self.evaluate(v2.values()))

  def testFilenameTensor(self):
    # train.Saver is V1 only API.
    with tf.Graph().as_default():
      v0 = tf.compat.v1.Variable(0, name="v0")
      filename = b"somerandomfilename"
      save = save_utils.PartialRecoverySaver({"v0": v0}, filename=filename)
      with self.cached_session() as sess:
        tensor = sess.graph.get_tensor_by_name(
            save.saver_def.filename_tensor_name)
        self.assertEqual(self.evaluate(tensor), filename)

  def testInvalidPath(self):
    v0 = tf.compat.v1.Variable(0, name="v0")
    with self.cached_session() as sess:
      save = save_utils.PartialRecoverySaver({"v0": v0})
      with self.assertRaisesRegex(
          ValueError, "The passed save_path is not a valid checkpoint:"):
        save.restore(sess, "invalid path")

  @test_util.run_v1_only("train.Saver is V1 only API.")
  def testInt64(self):
    save_path = os.path.join(self.get_temp_dir(), "int64")

    with self.cached_session() as sess:
      # Build a graph with 1 node, and save and restore for them.
      v = tf.compat.v1.Variable(np.int64(15), name="v")
      save = save_utils.PartialRecoverySaver({"v": v},
                                             restore_sequentially=True)
      self.evaluate(tf.compat.v1.global_variables_initializer())

      # Save the initialized values in the file at "save_path"
      val = save.save(sess, save_path)
      self.assertTrue(isinstance(val, six.string_types))
      self.assertEqual(save_path, val)

      with self.cached_session() as sess:
        v = tf.compat.v1.Variable(np.int64(-1), name="v")
        save = save_utils.PartialRecoverySaver({"v": v})

      with self.assertRaisesWithPredicateMatch(
          tf.errors.OpError, lambda e: "uninitialized value v" in e.message):
        self.evaluate(v)

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)
      # Check that the parameter nodes have been restored.
      self.assertEqual(np.int64(15), self.evaluate(v))

  def testSomeErrors(self):
    with tf.Graph().as_default():
      v0 = tf.compat.v1.Variable([10.0], name="v0")
      v1 = tf.compat.v1.Variable([20.0], name="v1")
      v2 = tf.compat.v1.Variable([20.0], name="v2")
      v2._set_save_slice_info(tf.Variable.SaveSliceInfo("v1", [1], [0], [1]))

      # By default the name used for "v2" will be "v1" and raise an error.
      with self.assertRaisesRegex(ValueError, "same name: v1"):
        save_utils.PartialRecoverySaver([v0, v1, v2])

      # The names are different and will work.
      save_utils.PartialRecoverySaver({"vee1": v1, "other": [v2]})

      # Partitioned variables also cause name conflicts.
      p_v1 = tf.compat.v1.get_variable(
          "p_v1",
          shape=[4, 5],
          partitioner=tf.compat.v1.fixed_size_partitioner(num_shards=2))
      p_v2 = tf.compat.v1.get_variable(
          "p_v2",
          shape=[4, 5],
          partitioner=tf.compat.v1.fixed_size_partitioner(num_shards=2))
      p_v2._name = "p_v1"
      with self.assertRaisesRegex(ValueError, "same name: p_v1"):
        save_utils.PartialRecoverySaver([p_v1, p_v2])

  def testSameName(self):
    with tf.Graph().as_default():
      v0 = tf.compat.v1.Variable([10.0], name="v0")
      v2 = saver_test_utils.CheckpointedOp(name="v2")

      # Saving one variable under two names raises an error.
      with self.assertRaisesRegex(
          ValueError, "The same saveable will be restored with two names: v0"):
        save_utils.PartialRecoverySaver({"v0": v0, "v0too": v0})

      # Ditto for custom saveables.
      with self.assertRaisesRegex(
          ValueError, "The same saveable will be restored with two names: v2"):
        save_utils.PartialRecoverySaver({
            "v2": v2.saveable,
            "v2too": v2.saveable
        })

      # Verify non-duplicate names work.
      save_utils.PartialRecoverySaver({"v0": v0, "v2": v2.saveable})

  @test_util.run_v1_only("train.Saver and VariableV1 are V1 only APIs.")
  def testBasicsWithListOfVariables(self):
    save_path = os.path.join(self.get_temp_dir(), "basics_with_list")

    with self.session(graph=tf.Graph()) as sess:
      # Build a graph with 2 parameter nodes, and Save and
      # Restore nodes for them.
      v0 = tf.compat.v1.Variable(10.0, name="v0")
      v1 = tf.compat.v1.Variable(20.0, name="v1")
      v2 = saver_test_utils.CheckpointedOp(name="v2")
      v2_init = v2.insert("k1", 30.0)
      save = save_utils.PartialRecoverySaver([v0, v1, v2.saveable])
      self.evaluate(tf.compat.v1.global_variables_initializer())
      v2_init.run()

      # Check that the parameter nodes have been initialized.
      self.assertEqual(10.0, self.evaluate(v0))
      self.assertEqual(20.0, self.evaluate(v1))
      self.assertEqual(b"k1", self.evaluate(v2.keys()))
      self.assertEqual(30.0, self.evaluate(v2.values()))

      # Save the initialized values in the file at "save_path"
      val = save.save(sess, save_path)
      self.assertTrue(isinstance(val, six.string_types))
      self.assertEqual(save_path, val)

    # Start a second session.  In that session the variables
    # have not been initialized either.
    with self.session(graph=tf.Graph()) as sess:
      v0 = tf.compat.v1.Variable(-1.0, name="v0")
      v1 = tf.compat.v1.Variable(-1.0, name="v1")
      v2 = saver_test_utils.CheckpointedOp(name="v2")
      save = save_utils.PartialRecoverySaver([v0, v1, v2.saveable])

      with self.assertRaisesWithPredicateMatch(
          tf.errors.OpError, lambda e: "uninitialized value v0" in e.message):
        self.evaluate(v0)
      with self.assertRaisesWithPredicateMatch(
          tf.errors.OpError, lambda e: "uninitialized value v1" in e.message):
        self.evaluate(v1)
      self.assertEqual(0, len(self.evaluate(v2.keys())))
      self.assertEqual(0, len(self.evaluate(v2.values())))

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)
      # Check that the parameter nodes have been restored.
      self.assertEqual(10.0, self.evaluate(v0))
      self.assertEqual(20.0, self.evaluate(v1))
      self.assertEqual(b"k1", self.evaluate(v2.keys()))
      self.assertEqual(30.0, self.evaluate(v2.values()))

    # Build another graph with 2 nodes, initialized
    # differently, and a Restore node for them.
    with self.session(graph=tf.Graph()) as sess:
      v0_2 = tf.compat.v1.Variable(1000.0, name="v0")
      v1_2 = tf.compat.v1.Variable(2000.0, name="v1")
      v2_2 = saver_test_utils.CheckpointedOp(name="v2")
      save2 = save_utils.PartialRecoverySaver([v0_2, v1_2, v2_2.saveable])
      v2_2.insert("k1000", 3000.0).run()
      self.evaluate(tf.compat.v1.global_variables_initializer())

      # Check that the parameter nodes have been initialized.
      self.assertEqual(1000.0, self.evaluate(v0_2))
      self.assertEqual(2000.0, self.evaluate(v1_2))
      self.assertEqual(b"k1000", self.evaluate(v2_2.keys()))
      self.assertEqual(3000.0, self.evaluate(v2_2.values()))
      # Restore the values saved earlier in the parameter nodes.
      save2.restore(sess, save_path)
      # Check that the parameter nodes have been restored.
      self.assertEqual(10.0, self.evaluate(v0_2))
      self.assertEqual(20.0, self.evaluate(v1_2))
      self.assertEqual(b"k1", self.evaluate(v2_2.keys()))
      self.assertEqual(30.0, self.evaluate(v2_2.values()))

  def _SaveAndLoad(self, var_name, var_value, other_value, save_path):
    with self.session(graph=tf.Graph()) as sess:
      var = resource_variable_ops.ResourceVariable(var_value, name=var_name)
      save = save_utils.PartialRecoverySaver({var_name: var})
      if not tf.executing_eagerly():
        self.evaluate(var.initializer)
      val = save.save(sess, save_path)
      self.assertEqual(save_path, val)
    with self.session(graph=tf.Graph()) as sess:
      var = resource_variable_ops.ResourceVariable(other_value, name=var_name)
      save = save_utils.PartialRecoverySaver({var_name: var})
      save.restore(sess, save_path)
      self.assertAllClose(var_value, self.evaluate(var))

  def testCacheRereadsFile(self):
    save_path = os.path.join(self.get_temp_dir(), "cache_rereads")
    # Save and reload one Variable named "var0".
    self._SaveAndLoad("var0", 0.0, 1.0, save_path)
    # Save and reload one Variable named "var1" in the same file.
    # The cached readers should know to re-read the file.
    self._SaveAndLoad("var1", 1.1, 2.2, save_path)

  def testAllowEmpty(self):
    save_path = os.path.join(self.get_temp_dir(), "allow_empty")
    # train.Saver is V1 only API.
    with tf.Graph().as_default(), self.cached_session() as sess:
      _ = tf.constant(1)
      save = save_utils.PartialRecoverySaver(allow_empty=True)
      val = save.save(sess, save_path)
      self.assertIsNone(val)
    with tf.Graph().as_default(), self.cached_session() as sess:
      save = save_utils.PartialRecoverySaver(allow_empty=True)
      save.restore(sess, save_path)

  def testGPU(self):
    if not tf.test.is_gpu_available():
      return
    save_path = os.path.join(self.get_temp_dir(), "gpu")
    with tf.compat.v1.Session("", graph=tf.Graph()) as sess:
      with sess.graph.device(tf.test.gpu_device_name()):
        v0_1 = tf.compat.v1.Variable(123.45)
      save = save_utils.PartialRecoverySaver({"v0": v0_1})
      self.evaluate(tf.compat.v1.global_variables_initializer())
      save.save(sess, save_path)

    with tf.compat.v1.Session("", graph=tf.Graph()) as sess:
      with sess.graph.device(tf.test.gpu_device_name()):
        v0_2 = tf.compat.v1.Variable(543.21)
      save = save_utils.PartialRecoverySaver({"v0": v0_2})
      self.evaluate(tf.compat.v1.global_variables_initializer())

  def testSharedServerOnGPU(self):
    if not tf.test.is_gpu_available():
      return
    save_path = os.path.join(self.get_temp_dir(), "gpu")
    with tf.compat.v1.Session("", graph=tf.Graph()) as sess:
      with sess.graph.device(tf.test.gpu_device_name()):
        v0_1 = tf.compat.v1.Variable(123.45)
      save = save_utils.PartialRecoverySaver({"v0": v0_1},
                                             sharded=True,
                                             allow_empty=True)
      self.evaluate(tf.compat.v1.global_variables_initializer())
      save.save(sess, save_path)

    with tf.compat.v1.Session("", graph=tf.Graph()) as sess:
      with sess.graph.device(tf.test.gpu_device_name()):
        v0_2 = tf.compat.v1.Variable(543.21)
      save = save_utils.PartialRecoverySaver({"v0": v0_2},
                                             sharded=True,
                                             allow_empty=True)
      self.evaluate(tf.compat.v1.global_variables_initializer())

  def testVariables(self):
    save_path = os.path.join(self.get_temp_dir(), "variables")
    with tf.compat.v1.Session("", graph=tf.Graph()) as sess:
      one = tf.compat.v1.Variable(1.0)
      twos = tf.compat.v1.Variable([2.0, 2.0, 2.0])
      v2 = saver_test_utils.CheckpointedOp(name="v2")
      init = tf.compat.v1.global_variables_initializer()
      save = save_utils.PartialRecoverySaver()
      init.run()
      v2.insert("k1", 3.0).run()
      save.save(sess, save_path)

    with tf.compat.v1.Session("", graph=tf.Graph()) as sess:
      one = tf.compat.v1.Variable(0.0)
      twos = tf.compat.v1.Variable([0.0, 0.0, 0.0])
      v2 = saver_test_utils.CheckpointedOp(name="v2")
      # Saver with no arg, defaults to 'all variables'.
      save = save_utils.PartialRecoverySaver()
      save.restore(sess, save_path)
      self.assertAllClose(1.0, self.evaluate(one))
      self.assertAllClose([2.0, 2.0, 2.0], self.evaluate(twos))
      self.assertEqual(b"k1", self.evaluate(v2.keys()))
      self.assertEqual(3.0, self.evaluate(v2.values()))

  def testVarListShouldBeEmptyInDeferredBuild(self):
    with tf.Graph().as_default():
      v = tf.compat.v1.Variable(1.0)
      with self.assertRaisesRegex(ValueError, "defer_build"):
        save_utils.PartialRecoverySaver([v], defer_build=True)

  def testBuildShouldBeCalledBeforeSaveInCaseOfDeferBuild(self):
    save_path = os.path.join(self.get_temp_dir(), "error_deferred_build")
    with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
      tf.compat.v1.Variable(1.0)
      saver = save_utils.PartialRecoverySaver(defer_build=True)
      with self.assertRaisesRegex(RuntimeError, "build"):
        saver.save(sess, save_path)

  def testDeferredBuild(self):
    save_path = os.path.join(self.get_temp_dir(), "deferred_build")
    with tf.compat.v1.Session("", graph=tf.Graph()) as sess:
      one = tf.compat.v1.Variable(1.0)
      save = save_utils.PartialRecoverySaver(defer_build=True)
      # if build is not deferred, saver cannot save the `twos`.
      twos = tf.compat.v1.Variable([2.0, 2.0, 2.0])
      init = tf.compat.v1.global_variables_initializer()
      save.build()
      init.run()
      save.save(sess, save_path)

    with tf.compat.v1.Session("", graph=tf.Graph()) as sess:
      one = tf.compat.v1.Variable(0.0)
      twos = tf.compat.v1.Variable([0.0, 0.0, 0.0])
      # Saver with no arg, defaults to 'all variables'.
      save = save_utils.PartialRecoverySaver()
      save.restore(sess, save_path)
      self.assertAllClose(1.0, self.evaluate(one))
      self.assertAllClose([2.0, 2.0, 2.0], self.evaluate(twos))

  @test_util.run_v1_only("train.Saver is V1 only API.")
  def testReshape(self):
    save_path = os.path.join(self.get_temp_dir(), "variables_reshape")
    with tf.compat.v1.Session("", graph=tf.Graph()) as sess:
      var = tf.compat.v1.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      init = tf.compat.v1.global_variables_initializer()
      save = save_utils.PartialRecoverySaver()
      init.run()
      save.save(sess, save_path)

    # Error when restoring with default reshape=False
    with tf.compat.v1.Session("", graph=tf.Graph()) as sess:
      var = tf.compat.v1.Variable([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
      save = save_utils.PartialRecoverySaver()
      with self.assertRaisesRegex(
          tf.errors.InvalidArgumentError,
          "Assign requires shapes of both tensors to match."):
        save.restore(sess, save_path)

    # Restored to new shape with reshape=True
    with tf.compat.v1.Session("", graph=tf.Graph()) as sess:
      var = tf.compat.v1.Variable([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
      save = save_utils.PartialRecoverySaver(reshape=True)
      save.restore(sess, save_path)
      self.assertAllClose([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                          self.evaluate(var))

  @test_util.run_in_graph_and_eager_modes
  def testSaveWithGlobalStep(self, pad_step_number=False):
    save_path = os.path.join(self.get_temp_dir(), "ckpt_with_global_step")
    global_step_int = 5
    # Save and reload one Variable named "var0".
    self._SaveAndLoad("var0", 0.0, 1.0, save_path)
    for use_tensor in [True, False]:
      with self.session(graph=tf.Graph()):
        var = resource_variable_ops.ResourceVariable(1.0, name="var0")
        save = save_utils.PartialRecoverySaver({var._shared_name: var},
                                               pad_step_number=pad_step_number)
        if tf.executing_eagerly():
          sess = None
        else:
          self.evaluate(var.initializer)
          sess = tf.compat.v1.get_default_session()
        if use_tensor:
          global_step = tf.constant(global_step_int)
          val = save.save(sess, save_path, global_step=global_step)
        else:
          val = save.save(sess, save_path, global_step=global_step_int)
        if pad_step_number:
          expected_save_path = "%s-%s" % (save_path,
                                          "{:08d}".format(global_step_int))
        else:
          expected_save_path = "%s-%d" % (save_path, global_step_int)
        self.assertEqual(expected_save_path, val)

  def testSaveWithGlobalStepWithPadding(self):
    self.testSaveWithGlobalStep(pad_step_number=True)

  def testSaveToNonexistingPath(self):
    file_io.write_string_to_file(
        os.path.join(self.get_temp_dir(), "actually_a_file"), "")
    paths = [
        os.path.join(self.get_temp_dir(), "nonexisting_dir/path"),
        os.path.join(self.get_temp_dir(), "other_nonexisting_dir/path1/path2"),
        os.path.join(self.get_temp_dir(), "actually_a_file/path"),
    ]

    for save_path in paths:
      # Build a graph with 2 parameter nodes, and Save and
      # Restore nodes for them.
      v0 = tf.compat.v1.Variable(10.0, name="v0")
      v1 = tf.compat.v1.Variable(20.0, name="v1")
      save = save_utils.PartialRecoverySaver({
          "v0": v0,
          "v1": v1
      },
                                             restore_sequentially=True)
      init_all_op = tf.compat.v1.global_variables_initializer()

      # In the case where the parent directory doesn't exist, whether or not the
      # save succeeds or fails is implementation dependent.  Therefore we allow
      # both cases.
      try:
        with self.cached_session() as sess:
          # Initialize all variables
          self.evaluate(init_all_op)

          # Check that the parameter nodes have been initialized.
          self.assertEqual(10.0, self.evaluate(v0))
          self.assertEqual(20.0, self.evaluate(v1))

          # Save the graph.
          save.save(sess, save_path)

        with self.cached_session() as sess:
          # Restore the saved values in the parameter nodes.
          save.restore(sess, save_path)
          # Check that the parameter nodes have been restored.
          self.assertEqual(10.0, self.evaluate(v0))
          self.assertEqual(20.0, self.evaluate(v1))
      except ValueError as exc:
        error_msg_template = "Parent directory of {} doesn't exist, can't save."
        self.assertEqual(error_msg_template.format(save_path), str(exc))

  def testSaveToURI(self):
    # ParseURI functions don't work on Windows yet.
    # TODO(jhseu): Remove this check when it works.
    if os.name == "nt":
      self.skipTest("Local URI support doesn't work on Windows")
    save_path = "file://" + os.path.join(self.get_temp_dir(), "uri")

    # Build a graph with 2 parameter nodes, and Save and
    # Restore nodes for them.
    v0 = tf.compat.v1.Variable(10.0, name="v0")
    v1 = tf.compat.v1.Variable(20.0, name="v1")
    save = save_utils.PartialRecoverySaver({
        "v0": v0,
        "v1": v1
    },
                                           restore_sequentially=True)
    init_all_op = tf.compat.v1.global_variables_initializer()

    with self.cached_session() as sess:
      # Initialize all variables
      self.evaluate(init_all_op)

      # Check that the parameter nodes have been initialized.
      self.assertEqual(10.0, self.evaluate(v0))
      self.assertEqual(20.0, self.evaluate(v1))
      save.save(sess, save_path)

  def testSaveRestoreAndValidateVariableDtype(self):
    for variable_op in [tf.Variable, resource_variable_ops.ResourceVariable]:
      save_path = os.path.join(self.get_temp_dir(), "basic_save_restore")

      # Build the first session.
      with self.session(graph=tf.Graph()) as sess:
        v0 = variable_op(10.0, name="v0", dtype=tf.dtypes.float32)

        if not tf.executing_eagerly():
          self.evaluate([tf.compat.v1.global_variables_initializer()])

        save = save_utils.PartialRecoverySaver({"v0": v0})
        save.save(sess, save_path)

      # Start a second session.
      with self.session(graph=tf.Graph()) as sess:
        v0_wrong_dtype = variable_op(1, name="v0", dtype=tf.dtypes.int32)
        # Restore the saved value with different dtype
        # in the parameter nodes.
        save = save_utils.PartialRecoverySaver({"v0": v0_wrong_dtype})
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    "original dtype"):
          save.restore(sess, save_path)

  # Test restoring large tensors (triggers a thread pool)
  def testRestoreLargeTensors(self):
    save_dir = self.get_temp_dir()

    def _model():
      small_v = [
          tf.compat.v1.get_variable("small%d" % i,
                                    shape=[10, 2],
                                    use_resource=True) for i in range(5)
      ]
      large_v = [
          tf.compat.v1.get_variable("large%d" % i,
                                    shape=[32000, 1000],
                                    use_resource=True) for i in range(3)
      ]
      return small_v + large_v

    save_graph = tf.Graph()
    with save_graph.as_default(), self.session(graph=save_graph) as sess:
      orig_vars = _model()
      self.evaluate(tf.compat.v1.global_variables_initializer())
      save = save_utils.PartialRecoverySaver(max_to_keep=1)
      self.evaluate(tf.compat.v1.global_variables_initializer())
      save.save(sess, save_dir)
      orig_vals = self.evaluate(orig_vars)

    restore_graph = tf.Graph()
    with restore_graph.as_default(), self.session(graph=restore_graph) as sess:
      restored_vars = _model()
      save = save_utils.PartialRecoverySaver(max_to_keep=1)
      save.restore(sess, save_dir)
      restored_vals = self.evaluate(restored_vars)

    for orig, restored in zip(orig_vals, restored_vals):
      self.assertAllEqual(orig, restored)


class SaveRestoreShardedTest(tf.test.TestCase):

  def testIterators(self):
    save_path = os.path.join(self.get_temp_dir(), "sharded_iterators")

    # Build a graph with 2 parameter nodes on different devices and save.
    with tf.compat.v1.Session(
        target="",
        config=config_pb2.ConfigProto(device_count={"CPU": 2})) as sess:
      with sess.graph.device("/cpu:0"):
        ds0 = tf.data.Dataset.range(10)
        it0 = tf.compat.v1.data.make_initializable_iterator(ds0)
        get_next0 = it0.get_next()
      saveable0 = iterator_ops._IteratorSaveable(it0._iterator_resource,
                                                 name="saveable_it0")

      with sess.graph.device("/cpu:1"):
        ds1 = tf.data.Dataset.range(20)
        it1 = tf.compat.v1.data.make_initializable_iterator(ds1)
        get_next1 = it1.get_next()
      saveable1 = iterator_ops._IteratorSaveable(it1._iterator_resource,
                                                 name="saveable_it1")
      saver = save_utils.PartialRecoverySaver(
          {
              "it0": saveable0,
              "it1": saveable1
          }, sharded=True)
      self.evaluate(it0.initializer)
      self.evaluate(it1.initializer)
      self.assertEqual(0, self.evaluate(get_next0))
      self.assertEqual(1, self.evaluate(get_next0))
      self.assertEqual(0, self.evaluate(get_next1))
      val = saver.save(sess, save_path)
      self.assertEqual(save_path, val)
      data_files = tf.io.gfile.glob(save_path + ".data*")
      self.assertEqual(2, len(data_files))

    # Restore
    with tf.compat.v1.Session(
        target="",
        config=config_pb2.ConfigProto(device_count={"CPU": 2})) as sess:
      with sess.graph.device("/cpu:0"):
        ds0 = tf.data.Dataset.range(10)
        it0 = tf.compat.v1.data.make_initializable_iterator(ds0)
        get_next0 = it0.get_next()
      saveable0 = iterator_ops._IteratorSaveable(it0._iterator_resource,
                                                 name="saveable_it0")

      with sess.graph.device("/cpu:1"):
        ds1 = tf.data.Dataset.range(20)
        it1 = tf.compat.v1.data.make_initializable_iterator(ds1)
        get_next1 = it1.get_next()
      saveable1 = iterator_ops._IteratorSaveable(it1._iterator_resource,
                                                 name="saveable_it1")
      saver = save_utils.PartialRecoverySaver(
          {
              "it0": saveable0,
              "it1": saveable1
          }, sharded=True)
      self.evaluate(it0.initializer)
      self.evaluate(it1.initializer)
      saver.restore(sess, save_path)
      self.assertEqual(2, self.evaluate(get_next0))
      self.assertEqual(1, self.evaluate(get_next1))

  def testIteratorsUnshardedRestore(self):
    save_path = os.path.join(self.get_temp_dir(), "restore_unsharded_iterators")

    # Build a graph with 2 parameter nodes on different devices and save.
    with tf.compat.v1.Session(
        target="",
        config=config_pb2.ConfigProto(device_count={"CPU": 2})) as sess:
      with sess.graph.device("/cpu:0"):
        ds0 = tf.data.Dataset.range(10)
        it0 = tf.compat.v1.data.make_initializable_iterator(ds0)
        get_next0 = it0.get_next()
      saveable0 = iterator_ops._IteratorSaveable(it0._iterator_resource,
                                                 name="saveable_it0")

      with sess.graph.device("/cpu:1"):
        ds1 = tf.data.Dataset.range(20)
        it1 = tf.compat.v1.data.make_initializable_iterator(ds1)
        get_next1 = it1.get_next()
      saveable1 = iterator_ops._IteratorSaveable(it1._iterator_resource,
                                                 name="saveable_it1")
      saver = save_utils.PartialRecoverySaver(
          {
              "it0": saveable0,
              "it1": saveable1
          }, sharded=True)
      self.evaluate(it0.initializer)
      self.evaluate(it1.initializer)
      self.assertEqual(0, self.evaluate(get_next0))
      self.assertEqual(1, self.evaluate(get_next0))
      self.assertEqual(0, self.evaluate(get_next1))
      val = saver.save(sess, save_path)
      self.assertEqual(save_path, val)
      data_files = tf.io.gfile.glob(save_path + ".data*")
      self.assertEqual(2, len(data_files))

    # Restore
    with tf.compat.v1.Session(
        target="",
        config=config_pb2.ConfigProto(device_count={"CPU": 2})) as sess:
      with sess.graph.device("/cpu:0"):
        ds0 = tf.data.Dataset.range(10)
        it0 = tf.compat.v1.data.make_initializable_iterator(ds0)
        get_next0 = it0.get_next()
      saveable0 = iterator_ops._IteratorSaveable(it0._iterator_resource,
                                                 name="saveable_it0")

      with sess.graph.device("/cpu:1"):
        ds1 = tf.data.Dataset.range(20)
        it1 = tf.compat.v1.data.make_initializable_iterator(ds1)
        get_next1 = it1.get_next()
      saveable1 = iterator_ops._IteratorSaveable(it1._iterator_resource,
                                                 name="saveable_it1")
      saver = save_utils.PartialRecoverySaver(
          {
              "it0": saveable0,
              "it1": saveable1
          }, sharded=False)
      self.evaluate(it0.initializer)
      self.evaluate(it1.initializer)
      saver.restore(sess, save_path)
      self.assertEqual(2, self.evaluate(get_next0))
      self.assertEqual(1, self.evaluate(get_next1))


class MaxToKeepTest(tf.test.TestCase):

  def _get_test_dir(self, dirname):
    test_dir = os.path.join(self.get_temp_dir(), dirname)
    tf.io.gfile.makedirs(test_dir)
    return test_dir

  def assertCheckpointState(self, model_checkpoint_path,
                            all_model_checkpoint_paths, save_dir):
    checkpoint_state = tf.train.get_checkpoint_state(save_dir)
    self.assertEqual(checkpoint_state.model_checkpoint_path,
                     model_checkpoint_path)
    self.assertEqual(checkpoint_state.all_model_checkpoint_paths,
                     all_model_checkpoint_paths)

  def testMaxToKeepEager(self):
    with context.eager_mode():
      save_dir = self._get_test_dir("max_to_keep_eager")

      v = tf.compat.v1.Variable(10.0, name="v")
      save = save_utils.PartialRecoverySaver({"v": v}, max_to_keep=2)
      self.evaluate(tf.compat.v1.global_variables_initializer())
      if not tf.executing_eagerly():
        self.assertEqual([], save.last_checkpoints)

      s1 = save.save(None, os.path.join(save_dir, "s1"))
      self.assertEqual([s1], save.last_checkpoints)
      self.assertTrue(tf.compat.v1.train.checkpoint_exists(s1))
      self.assertCheckpointState(model_checkpoint_path=s1,
                                 all_model_checkpoint_paths=[s1],
                                 save_dir=save_dir)

      s2 = save.save(None, os.path.join(save_dir, "s2"))
      self.assertEqual([s1, s2], save.last_checkpoints)
      self.assertTrue(tf.compat.v1.train.checkpoint_exists(s1))
      self.assertTrue(tf.compat.v1.train.checkpoint_exists(s2))
      self.assertCheckpointState(model_checkpoint_path=s2,
                                 all_model_checkpoint_paths=[s1, s2],
                                 save_dir=save_dir)

      s3 = save.save(None, os.path.join(save_dir, "s3"))
      self.assertEqual([s2, s3], save.last_checkpoints)
      self.assertFalse(tf.compat.v1.train.checkpoint_exists(s1))
      self.assertTrue(tf.compat.v1.train.checkpoint_exists(s2))
      self.assertTrue(tf.compat.v1.train.checkpoint_exists(s3))
      self.assertCheckpointState(model_checkpoint_path=s3,
                                 all_model_checkpoint_paths=[s2, s3],
                                 save_dir=save_dir)

      # Create a second helper, identical to the first.
      save2 = save_utils.PartialRecoverySaver({"v": v}, max_to_keep=2)
      save2.set_last_checkpoints_with_time([
          (s, np.inf) for s in save.last_checkpoints
      ])

      # Exercise the first helper.

      # Adding s2 again (old s2 is removed first, then new s2 appended)
      s2 = save.save(None, os.path.join(save_dir, "s2"))
      self.assertEqual([s3, s2], save.last_checkpoints)
      self.assertFalse(tf.compat.v1.train.checkpoint_exists(s1))
      self.assertTrue(tf.compat.v1.train.checkpoint_exists(s3))
      self.assertTrue(tf.compat.v1.train.checkpoint_exists(s2))
      self.assertCheckpointState(model_checkpoint_path=s2,
                                 all_model_checkpoint_paths=[s3, s2],
                                 save_dir=save_dir)

      # Adding s1 (s3 should now be deleted as oldest in list)
      s1 = save.save(None, os.path.join(save_dir, "s1"))
      self.assertEqual([s2, s1], save.last_checkpoints)
      self.assertFalse(tf.compat.v1.train.checkpoint_exists(s3))
      self.assertTrue(tf.compat.v1.train.checkpoint_exists(s2))
      self.assertCheckpointState(model_checkpoint_path=s1,
                                 all_model_checkpoint_paths=[s2, s1],
                                 save_dir=save_dir)

      s2 = save2.save(None, os.path.join(save_dir, "s2"))
      self.assertEqual([s3, s2], save2.last_checkpoints)
      # Created by the first helper.
      self.assertTrue(tf.compat.v1.train.checkpoint_exists(s1))
      # Deleted by the first helper.
      self.assertFalse(tf.compat.v1.train.checkpoint_exists(s3))

  def testNonSharded(self):
    save_dir = self._get_test_dir("max_to_keep_non_sharded")

    # train.Saver is V1 only API.
    with tf.Graph().as_default(), self.cached_session() as sess:
      v = tf.compat.v1.Variable(10.0, name="v")
      save = save_utils.PartialRecoverySaver({"v": v}, max_to_keep=2)
      self.evaluate(tf.compat.v1.global_variables_initializer())
      self.assertEqual([], save.last_checkpoints)

      s1 = save.save(sess, os.path.join(save_dir, "s1"))
      self.assertEqual([s1], save.last_checkpoints)
      self.assertTrue(tf.compat.v1.train.checkpoint_exists(s1))
      self.assertCheckpointState(model_checkpoint_path=s1,
                                 all_model_checkpoint_paths=[s1],
                                 save_dir=save_dir)

      s2 = save.save(sess, os.path.join(save_dir, "s2"))
      self.assertEqual([s1, s2], save.last_checkpoints)
      self.assertTrue(tf.compat.v1.train.checkpoint_exists(s1))
      self.assertTrue(tf.compat.v1.train.checkpoint_exists(s2))
      self.assertCheckpointState(model_checkpoint_path=s2,
                                 all_model_checkpoint_paths=[s1, s2],
                                 save_dir=save_dir)

      s3 = save.save(sess, os.path.join(save_dir, "s3"))
      self.assertEqual([s2, s3], save.last_checkpoints)
      self.assertFalse(tf.compat.v1.train.checkpoint_exists(s1))
      self.assertTrue(tf.compat.v1.train.checkpoint_exists(s2))
      self.assertTrue(tf.compat.v1.train.checkpoint_exists(s3))
      self.assertCheckpointState(model_checkpoint_path=s3,
                                 all_model_checkpoint_paths=[s2, s3],
                                 save_dir=save_dir)

      # Create a second helper, identical to the first.
      save2 = save_utils.PartialRecoverySaver(saver_def=save.as_saver_def())
      save2.set_last_checkpoints_with_time([
          (s, np.inf) for s in save.last_checkpoints
      ])

      # Create a third helper, with the same configuration but no knowledge of
      # previous checkpoints.
      save3 = save_utils.PartialRecoverySaver(saver_def=save.as_saver_def())

      # Exercise the first helper.

      # Adding s2 again (old s2 is removed first, then new s2 appended)
      s2 = save.save(sess, os.path.join(save_dir, "s2"))
      self.assertEqual([s3, s2], save.last_checkpoints)
      self.assertFalse(tf.compat.v1.train.checkpoint_exists(s1))
      self.assertFalse(
          tf.compat.v1.train.checkpoint_exists(
              checkpoint_management.meta_graph_filename(s1)))
      self.assertTrue(tf.compat.v1.train.checkpoint_exists(s3))
      self.assertTrue(
          tf.compat.v1.train.checkpoint_exists(
              checkpoint_management.meta_graph_filename(s3)))
      self.assertTrue(tf.compat.v1.train.checkpoint_exists(s2))
      self.assertTrue(
          tf.compat.v1.train.checkpoint_exists(
              checkpoint_management.meta_graph_filename(s2)))
      self.assertCheckpointState(model_checkpoint_path=s2,
                                 all_model_checkpoint_paths=[s3, s2],
                                 save_dir=save_dir)

      # Adding s1 (s3 should now be deleted as oldest in list)
      s1 = save.save(sess, os.path.join(save_dir, "s1"))
      self.assertEqual([s2, s1], save.last_checkpoints)
      self.assertFalse(tf.compat.v1.train.checkpoint_exists(s3))
      self.assertFalse(
          tf.compat.v1.train.checkpoint_exists(
              checkpoint_management.meta_graph_filename(s3)))
      self.assertTrue(tf.compat.v1.train.checkpoint_exists(s2))
      self.assertTrue(
          tf.compat.v1.train.checkpoint_exists(
              checkpoint_management.meta_graph_filename(s2)))
      self.assertTrue(tf.compat.v1.train.checkpoint_exists(s1))
      self.assertTrue(
          tf.compat.v1.train.checkpoint_exists(
              checkpoint_management.meta_graph_filename(s1)))
      self.assertCheckpointState(model_checkpoint_path=s1,
                                 all_model_checkpoint_paths=[s2, s1],
                                 save_dir=save_dir)

      # Exercise the second helper.

      # Adding s2 again (old s2 is removed first, then new s2 appended)
      s2 = save2.save(sess, os.path.join(save_dir, "s2"))
      self.assertEqual([s3, s2], save2.last_checkpoints)
      # Created by the first helper.
      self.assertTrue(tf.compat.v1.train.checkpoint_exists(s1))
      self.assertTrue(
          tf.compat.v1.train.checkpoint_exists(
              checkpoint_management.meta_graph_filename(s1)))
      # Deleted by the first helper.
      self.assertFalse(tf.compat.v1.train.checkpoint_exists(s3))
      self.assertFalse(
          tf.compat.v1.train.checkpoint_exists(
              checkpoint_management.meta_graph_filename(s3)))
      self.assertTrue(tf.compat.v1.train.checkpoint_exists(s2))
      self.assertTrue(
          tf.compat.v1.train.checkpoint_exists(
              checkpoint_management.meta_graph_filename(s2)))
      self.assertCheckpointState(model_checkpoint_path=s2,
                                 all_model_checkpoint_paths=[s3, s2],
                                 save_dir=save_dir)

      # Adding s1 (s3 should now be deleted as oldest in list)
      s1 = save2.save(sess, os.path.join(save_dir, "s1"))
      self.assertEqual([s2, s1], save2.last_checkpoints)
      self.assertFalse(tf.compat.v1.train.checkpoint_exists(s3))
      self.assertFalse(
          tf.compat.v1.train.checkpoint_exists(
              checkpoint_management.meta_graph_filename(s3)))
      self.assertTrue(tf.compat.v1.train.checkpoint_exists(s2))
      self.assertTrue(
          tf.compat.v1.train.checkpoint_exists(
              checkpoint_management.meta_graph_filename(s2)))
      self.assertTrue(tf.compat.v1.train.checkpoint_exists(s1))
      self.assertTrue(
          tf.compat.v1.train.checkpoint_exists(
              checkpoint_management.meta_graph_filename(s1)))
      self.assertCheckpointState(model_checkpoint_path=s1,
                                 all_model_checkpoint_paths=[s2, s1],
                                 save_dir=save_dir)

      # Exercise the third helper.

      # Adding s2 again (but helper is unaware of previous s2)
      s2 = save3.save(sess, os.path.join(save_dir, "s2"))
      self.assertEqual([s2], save3.last_checkpoints)
      # Created by the first helper.
      self.assertTrue(tf.compat.v1.train.checkpoint_exists(s1))
      self.assertTrue(
          tf.compat.v1.train.checkpoint_exists(
              checkpoint_management.meta_graph_filename(s1)))
      # Deleted by the first helper.
      self.assertFalse(tf.compat.v1.train.checkpoint_exists(s3))
      self.assertFalse(
          tf.compat.v1.train.checkpoint_exists(
              checkpoint_management.meta_graph_filename(s3)))
      self.assertTrue(tf.compat.v1.train.checkpoint_exists(s2))
      self.assertTrue(
          tf.compat.v1.train.checkpoint_exists(
              checkpoint_management.meta_graph_filename(s2)))
      # Even though the file for s1 exists, this saver isn't aware of it, which
      # is why it doesn't end up in the checkpoint state.
      self.assertCheckpointState(model_checkpoint_path=s2,
                                 all_model_checkpoint_paths=[s2],
                                 save_dir=save_dir)

      # Adding s1 (s3 should not be deleted because helper is unaware of it)
      s1 = save3.save(sess, os.path.join(save_dir, "s1"))
      self.assertEqual([s2, s1], save3.last_checkpoints)
      self.assertFalse(tf.compat.v1.train.checkpoint_exists(s3))
      self.assertFalse(
          tf.compat.v1.train.checkpoint_exists(
              checkpoint_management.meta_graph_filename(s3)))
      self.assertTrue(tf.compat.v1.train.checkpoint_exists(s2))
      self.assertTrue(
          tf.compat.v1.train.checkpoint_exists(
              checkpoint_management.meta_graph_filename(s2)))
      self.assertTrue(tf.compat.v1.train.checkpoint_exists(s1))
      self.assertTrue(
          tf.compat.v1.train.checkpoint_exists(
              checkpoint_management.meta_graph_filename(s1)))
      self.assertCheckpointState(model_checkpoint_path=s1,
                                 all_model_checkpoint_paths=[s2, s1],
                                 save_dir=save_dir)

  def testSharded(self):
    save_dir = self._get_test_dir("max_to_keep_sharded")

    with tf.compat.v1.Session(
        target="",
        config=config_pb2.ConfigProto(device_count={"CPU": 2})) as sess:
      with sess.graph.device("/cpu:0"):
        v0 = tf.compat.v1.Variable(111, name="v0")
      with sess.graph.device("/cpu:1"):
        v1 = tf.compat.v1.Variable(222, name="v1")
      save = save_utils.PartialRecoverySaver({
          "v0": v0,
          "v1": v1
      },
                                             sharded=True,
                                             max_to_keep=2)
      self.evaluate(tf.compat.v1.global_variables_initializer())
      self.assertEqual([], save.last_checkpoints)

      s1 = save.save(sess, os.path.join(save_dir, "s1"))
      self.assertEqual([s1], save.last_checkpoints)
      self.assertEqual(4, len(tf.io.gfile.glob(s1 + "*")))

      self.assertTrue(
          tf.io.gfile.exists(checkpoint_management.meta_graph_filename(s1)))

      s2 = save.save(sess, os.path.join(save_dir, "s2"))
      self.assertEqual([s1, s2], save.last_checkpoints)
      self.assertEqual(4, len(tf.io.gfile.glob(s1 + "*")))
      self.assertTrue(
          tf.io.gfile.exists(checkpoint_management.meta_graph_filename(s1)))
      self.assertEqual(4, len(tf.io.gfile.glob(s2 + "*")))
      self.assertTrue(
          tf.io.gfile.exists(checkpoint_management.meta_graph_filename(s2)))

      s3 = save.save(sess, os.path.join(save_dir, "s3"))
      self.assertEqual([s2, s3], save.last_checkpoints)
      self.assertEqual(0, len(tf.io.gfile.glob(s1 + "*")))
      self.assertFalse(
          tf.io.gfile.exists(checkpoint_management.meta_graph_filename(s1)))
      self.assertEqual(4, len(tf.io.gfile.glob(s2 + "*")))
      self.assertTrue(
          tf.io.gfile.exists(checkpoint_management.meta_graph_filename(s2)))
      self.assertEqual(4, len(tf.io.gfile.glob(s3 + "*")))
      self.assertTrue(
          tf.io.gfile.exists(checkpoint_management.meta_graph_filename(s3)))

  def testNoMaxToKeep(self):
    save_dir = self._get_test_dir("no_max_to_keep")
    save_dir2 = self._get_test_dir("max_to_keep_0")

    with self.cached_session() as sess:
      v = tf.compat.v1.Variable(10.0, name="v")
      self.evaluate(tf.compat.v1.global_variables_initializer())

      # Test max_to_keep being None.
      save = save_utils.PartialRecoverySaver({"v": v}, max_to_keep=None)
      self.assertEqual([], save.last_checkpoints)
      s1 = save.save(sess, os.path.join(save_dir, "s1"))
      self.assertEqual([], save.last_checkpoints)
      self.assertTrue(tf.compat.v1.train.checkpoint_exists(s1))
      s2 = save.save(sess, os.path.join(save_dir, "s2"))
      self.assertEqual([], save.last_checkpoints)
      self.assertTrue(tf.compat.v1.train.checkpoint_exists(s2))

      # Test max_to_keep being 0.
      save2 = save_utils.PartialRecoverySaver({"v": v}, max_to_keep=0)
      self.assertEqual([], save2.last_checkpoints)
      s1 = save2.save(sess, os.path.join(save_dir2, "s1"))
      self.assertEqual([], save2.last_checkpoints)
      self.assertTrue(tf.compat.v1.train.checkpoint_exists(s1))
      s2 = save2.save(sess, os.path.join(save_dir2, "s2"))
      self.assertEqual([], save2.last_checkpoints)
      self.assertTrue(tf.compat.v1.train.checkpoint_exists(s2))

  def testNoMetaGraph(self):
    save_dir = self._get_test_dir("no_meta_graph")

    with self.cached_session() as sess:
      v = tf.compat.v1.Variable(10.0, name="v")
      save = save_utils.PartialRecoverySaver({"v": v})
      self.evaluate(tf.compat.v1.global_variables_initializer())

      s1 = save.save(sess, os.path.join(save_dir, "s1"), write_meta_graph=False)
      self.assertTrue(tf.compat.v1.train.checkpoint_exists(s1))
      self.assertFalse(
          tf.io.gfile.exists(checkpoint_management.meta_graph_filename(s1)))


class RecoverLastCheckpointsTest(tf.test.TestCase):

  def _get_test_dir(self, dirname):
    test_dir = os.path.join(self.get_temp_dir(), dirname)
    tf.io.gfile.makedirs(test_dir)
    return test_dir

  def assertCheckpointState(self, model_checkpoint_path,
                            all_model_checkpoint_paths, save_dir):
    checkpoint_state = tf.train.get_checkpoint_state(save_dir)
    self.assertEqual(checkpoint_state.model_checkpoint_path,
                     model_checkpoint_path)
    self.assertEqual(checkpoint_state.all_model_checkpoint_paths,
                     all_model_checkpoint_paths)

  def test_recover_last_checkpoints(self):
    with context.eager_mode():
      save_dir = self._get_test_dir("recover_last_checkpoints")

      v = tf.compat.v1.Variable(10.0, name="v")
      save = save_utils.PartialRecoverySaver({"v": v}, max_to_keep=10)
      self.evaluate(tf.compat.v1.global_variables_initializer())
      self.assertEqual([], save.last_checkpoints)

      s1 = save.save(None, os.path.join(save_dir, "ckpt-1"))
      s2 = save.save(None, os.path.join(save_dir, "ckpt-2"))
      s3 = save.save(None, os.path.join(save_dir, "ckpt-3"))
      self.assertEqual([s1, s2, s3], save.last_checkpoints)
      self.assertTrue(tf.compat.v1.train.checkpoint_exists(s1))
      self.assertTrue(tf.compat.v1.train.checkpoint_exists(s2))
      self.assertTrue(tf.compat.v1.train.checkpoint_exists(s3))
      self.assertCheckpointState(model_checkpoint_path=s3,
                                 all_model_checkpoint_paths=[s1, s2, s3],
                                 save_dir=save_dir)

      # Create another saver and recover last checkpoints.
      save2 = save_utils.PartialRecoverySaver({"v": v}, max_to_keep=10)
      self.assertEqual([], save2.last_checkpoints)
      save2.recover_last_checkpoints([s1, s2, s3])
      self.assertEqual([s1, s2, s3], save2.last_checkpoints)

      # Remove a checkpoint and check that last checkpoints are
      # restored correctly.
      for fname in tf.io.gfile.glob("{}*".format(s1)):
        tf.io.gfile.remove(fname)
      self.assertFalse(tf.compat.v1.train.checkpoint_exists(s1))

      # Create another saver and recover last checkpoints. The removed
      # checkpoint would be correctly omitted.
      save3 = save_utils.PartialRecoverySaver({"v": v}, max_to_keep=10)
      self.assertEqual([], save3.last_checkpoints)
      save3.recover_last_checkpoints([s1, s2, s3])
      self.assertEqual([s2, s3], save3.last_checkpoints)
      s4 = save3.save(None, os.path.join(save_dir, "ckpt-4"))
      self.assertCheckpointState(model_checkpoint_path=s4,
                                 all_model_checkpoint_paths=[s2, s3, s4],
                                 save_dir=save_dir)


class KeepCheckpointEveryNHoursTest(tf.test.TestCase):

  def _get_test_dir(self, dirname):
    test_dir = os.path.join(self.get_temp_dir(), dirname)
    tf.io.gfile.makedirs(test_dir)
    return test_dir

  @test_util.run_in_graph_and_eager_modes
  @mock.patch.object(save_utils, "time")
  def testNonSharded(self, mock_time):
    save_dir = self._get_test_dir("keep_checkpoint_every_n_hours")

    with self.cached_session() as sess:
      v = tf.compat.v1.Variable([10.0], name="v")
      # Run the initializer NOW to avoid the 0.5s overhead of the first Run()
      # call, which throws the test timing off in fastbuild mode.
      self.evaluate(tf.compat.v1.global_variables_initializer())
      # Create a saver that will keep the last 2 checkpoints plus one every 0.7
      # seconds.
      start_time = time.time()
      mock_time.time.return_value = start_time
      save = save_utils.PartialRecoverySaver({"v": v},
                                             max_to_keep=2,
                                             keep_checkpoint_every_n_hours=0.7 /
                                             3600)
      self.assertEqual([], save.last_checkpoints)

      # Wait till 1 seconds have elapsed so s1 will be old enough to keep.
      # sleep may return early, don't trust it.
      mock_time.time.return_value = start_time + 1.0
      s1 = save.save(sess, os.path.join(save_dir, "s1"))
      self.assertEqual([s1], save.last_checkpoints)

      s2 = save.save(sess, os.path.join(save_dir, "s2"))
      self.assertEqual([s1, s2], save.last_checkpoints)

      # We now have 2 'last_checkpoints': [s1, s2].  The next call to Save(),
      # would normally delete s1, because max_to_keep is 2.  However, s1 is
      # older than 0.7s so we must keep it.
      s3 = save.save(sess, os.path.join(save_dir, "s3"))
      self.assertEqual([s2, s3], save.last_checkpoints)

      # s1 should still be here, we are Not checking now to reduce time
      # variance in the test.

      # We now have 2 'last_checkpoints': [s2, s3], and s1 on disk.  The next
      # call to Save(), will delete s2, because max_to_keep is 2, and because
      # we already kept the old s1. s2 is very close in time to s1 so it gets
      # deleted.
      s4 = save.save(sess, os.path.join(save_dir, "s4"))
      self.assertEqual([s3, s4], save.last_checkpoints)

      # Check that s1 is still here, but s2 is gone.
      self.assertTrue(tf.compat.v1.train.checkpoint_exists(s1))
      self.assertFalse(tf.compat.v1.train.checkpoint_exists(s2))
      self.assertTrue(tf.compat.v1.train.checkpoint_exists(s3))
      self.assertTrue(tf.compat.v1.train.checkpoint_exists(s4))


class SaveRestoreWithVariableNameMap(tf.test.TestCase):

  def _testNonReshape(self, variable_op):
    save_path = os.path.join(self.get_temp_dir(), "non_reshape")

    with self.session(graph=tf.Graph()) as sess:
      # Build a graph with 2 parameter nodes, and Save and
      # Restore nodes for them.
      v0 = variable_op(10.0, name="v0")
      v1 = variable_op(20.0, name="v1")
      save = save_utils.PartialRecoverySaver({
          "save_prefix/v0": v0,
          "save_prefix/v1": v1
      })
      self.evaluate(tf.compat.v1.global_variables_initializer())

      # Check that the parameter nodes have been initialized.
      self.assertEqual(10.0, self.evaluate(v0))
      self.assertEqual(20.0, self.evaluate(v1))

      # Save the initialized values in the file at "save_path"
      # Use a variable name map to set the saved tensor names
      val = save.save(sess, save_path)
      self.assertTrue(isinstance(val, six.string_types))
      self.assertEqual(save_path, val)

      # Verify that the original names are not in the Saved file
      save = save_utils.PartialRecoverySaver({"v0": v0, "v1": v1})
      with self.assertRaisesOpError("not found in checkpoint"):
        save.restore(sess, save_path)

    # Verify that the mapped names are present in the Saved file and can be
    # Restored using remapped names.
    with self.session(graph=tf.Graph()) as sess:
      v0 = variable_op(-1.0, name="v0")
      v1 = variable_op(-1.0, name="v1")

      if not tf.executing_eagerly():
        with self.assertRaisesOpError("uninitialized"):
          self.evaluate(v0)
        with self.assertRaisesOpError("uninitialized"):
          self.evaluate(v1)

      save = save_utils.PartialRecoverySaver({
          "save_prefix/v0": v0,
          "save_prefix/v1": v1
      })
      save.restore(sess, save_path)

      # Check that the parameter nodes have been restored.
      if not tf.executing_eagerly():
        self.assertEqual(10.0, self.evaluate(v0))
        self.assertEqual(20.0, self.evaluate(v1))

    # Add a prefix to the node names in the current graph and Restore using
    # remapped names.
    with self.session(graph=tf.Graph()) as sess:
      v0 = variable_op(-1.0, name="restore_prefix/v0")
      v1 = variable_op(-1.0, name="restore_prefix/v1")

      if not tf.executing_eagerly():
        with self.assertRaisesOpError("uninitialized"):
          self.evaluate(v0)
        with self.assertRaisesOpError("uninitialized"):
          self.evaluate(v1)

      # Restore the saved values in the parameter nodes.
      save = save_utils.PartialRecoverySaver({
          "save_prefix/v0": v0,
          "save_prefix/v1": v1
      })
      save.restore(sess, save_path)

      # Check that the parameter nodes have been restored.
      self.assertEqual(10.0, self.evaluate(v0))
      self.assertEqual(20.0, self.evaluate(v1))

  @test_util.run_in_graph_and_eager_modes
  def testNonReshapeResourceVariable(self):
    self._testNonReshape(resource_variable_ops.ResourceVariable)

  def testNonReshapeVariable(self):
    self._testNonReshape(tf.Variable)


class SecondOrStepTimerWithTideSettingTest(tf.test.TestCase):

  def testNoTideSetting(self):
    timer = save_utils.SecondOrStepTimerWithTideSetting(every_secs=10)
    with freeze_time("2012-01-14 02:00:00") as freezer:
      timer.update_last_triggered_step(5)
      freezer.tick(5.0)
      self.assertEqual(False, timer.should_trigger_for_step(10))
      freezer.tick(10.0)
      self.assertEqual(True, timer.should_trigger_for_step(10))

  def testTideAvailable(self):
    timer = save_utils.SecondOrStepTimerWithTideSetting(every_secs=10,
                                                        tide_start_hour=1,
                                                        tide_start_minute=0,
                                                        tide_end_hour=3,
                                                        tide_end_minute=0,
                                                        tide_every_secs=5)
    with freeze_time("2012-01-14 02:00:00") as freezer:
      timer.update_last_triggered_step(5)
      freezer.tick(5.0)
      self.assertEqual(False, timer.should_trigger_for_step(10))
      freezer.tick(10.0)
      self.assertEqual(True, timer.should_trigger_for_step(10))

  def testTideNotAvailable(self):
    timer = save_utils.SecondOrStepTimerWithTideSetting(every_secs=10,
                                                        tide_start_hour=1,
                                                        tide_start_minute=0,
                                                        tide_end_hour=3,
                                                        tide_end_minute=0,
                                                        tide_every_secs=5)
    with freeze_time("2012-01-14 04:00:00") as freezer:
      timer.update_last_triggered_step(5)
      freezer.tick(2.0)
      self.assertEqual(False, timer.should_trigger_for_step(10))
      freezer.tick(7.0)
      self.assertEqual(True, timer.should_trigger_for_step(10))


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
