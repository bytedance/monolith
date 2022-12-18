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
from tensorflow.data.experimental import CheckpointInputPipelineHook

from monolith.native_training.distribute import distributed_dataset
from monolith.native_training import native_task_context
from monolith.native_training.hooks import session_hooks


def gen_test_files(files_dir):
  """
  Generates following files under the folder.
  a_0.txt
  a_1.txt
  ...
  e_1.txt
  In each file, it will be some like (this is coming from a_0.txt)
  a.0.0
  a.0.1
  """
  for c in range(97, 102):
    for i in range(2):
      with tf.io.gfile.GFile(
          os.path.join(files_dir, '{}_{}.txt'.format(chr(c), i)), 'w+') as f:
        f.write('\n'.join(['{}.{}.{}'.format(chr(c), i, j) for j in range(2)]))


class DynamicShardingDatasetTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = os.environ["TEST_TMPDIR"]
    self.data_dir = os.path.join(self.test_dir, 'test_data')
    if not tf.io.gfile.exists(self.data_dir):
      tf.io.gfile.makedirs(self.data_dir)
      gen_test_files(self.data_dir)
    self.glob_patterns = [
        os.path.join(self.data_dir, basename)
        for basename in ['a_*.txt', 'b_*.txt', 'c_*.txt', 'd_*.txt', 'e_*.txt']
    ]

  def get_test_session(self):
    return tf.compat.v1.train.SingularMonitoredSession(
        hooks=[session_hooks.SetCurrentSessionHook()])

  def testBasic(self):
    ds = distributed_dataset.create_dynamic_sharding_dataset(self.glob_patterns)
    it = tf.compat.v1.data.make_one_shot_iterator(ds)
    element = it.get_next()
    with self.get_test_session() as sess:
      names = []
      for i in range(10):
        names.append(sess.run(element))

      expected = []
      for i in range(97, 102):
        for j in range(2):
          expected.append(
              os.path.join(self.data_dir, "{}_{}.txt".format(chr(i), j)))
      self.assertAllEqual(names, expected)

  def testEof(self):
    ds = distributed_dataset.create_dynamic_sharding_dataset([])
    it = tf.compat.v1.data.make_one_shot_iterator(ds)
    v = tf.Variable(0)
    element = it.get_next()
    with tf.control_dependencies([element]):
      add_op = v.assign_add(1)
    with self.get_test_session() as sess:
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(add_op)
      # Make sure v is not changed
      self.assertAllEqual(sess.run(v), 0)

  def testWithOtherDataset(self):
    filename_dataset = distributed_dataset.create_dynamic_sharding_dataset(
        self.glob_patterns)
    dataset = filename_dataset.flat_map(tf.data.TextLineDataset)
    it = tf.compat.v1.data.make_one_shot_iterator(dataset)
    element = it.get_next()
    with self.get_test_session() as sess:
      lines = []
      for i in range(3):
        lines.append(sess.run(element).decode())
      self.assertAllEqual(lines, ["a.0.0", "a.0.1", "a.1.0"])

  def testSaveRestore(self):
    filename_dataset = distributed_dataset.create_dynamic_sharding_dataset(
        self.glob_patterns)
    dataset = filename_dataset.flat_map(tf.data.TextLineDataset)
    it = tf.compat.v1.data.make_one_shot_iterator(dataset)
    element = it.get_next()
    saveable_obj = tf.data.experimental.make_saveable_from_iterator(
        it, external_state_policy="ignore")
    saver = tf.compat.v1.train.Saver(var_list=[saveable_obj] +
                                     tf.compat.v1.global_variables())
    with self.get_test_session() as sess:
      real_sess = session_hooks.get_current_session()
      self.assertAllEqual(sess.run(element).decode(), "a.0.0")
      save_path = saver.save(
          real_sess, os.path.join(os.environ["TEST_TMPDIR"], "save_restore"))
      self.assertAllEqual(sess.run(element).decode(), "a.0.1")
      saver.restore(real_sess, save_path)
      self.assertAllEqual(sess.run(element).decode(), "a.0.1")


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
