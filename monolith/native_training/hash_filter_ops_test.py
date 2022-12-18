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
from typing import List

import tensorflow as tf
from tensorflow.python.lib.io import tf_record

import monolith.native_training.hash_filter_ops as ops
from monolith.native_training.runtime.hash_table import \
    embedding_hash_table_pb2


def get_config_str(occurrence_threshold=0):
  config = embedding_hash_table_pb2.SlotOccurrenceThresholdConfig()
  config.default_occurrence_threshold = occurrence_threshold
  return config.SerializeToString()


class HashFilterOpsTest(tf.test.TestCase):

  def _count_files(self, basename: str):
    return len(tf.io.gfile.glob(basename + "*"))

  def _GetHashFilterSplitMetaDump(self, ckpt_file: str):
    for record in tf_record.tf_record_iterator(ckpt_file):
      return embedding_hash_table_pb2.HashFilterSplitMetaDump.FromString(record)
    return None

  def test_hash_filter_basic(self):
    config = get_config_str(3)
    hash_filter = ops.create_hash_filter(100, 7, config)
    # we choose a key that is unique enough so they won't collide with each other.
    ids = tf.constant([1, 3 << 17, 1], dtype=tf.int64)
    embedding = tf.zeros([3, 2])
    loss = ops.intercept_gradient(hash_filter, ids, embedding)
    grad = tf.gradients(loss, embedding)[0]
    with self.session() as sess:
      grad_value = sess.run(grad)
      self.assertAllEqual(grad_value, [[0, 0], [0, 0], [0, 0]])
      grad_value = sess.run(grad)
      self.assertAllEqual(grad_value, [[0, 0], [0, 0], [1, 1]])
      grad_value = sess.run(grad)
      self.assertAllEqual(grad_value, [[1, 1], [0, 0], [1, 1]])
      grad_value = sess.run(grad)
      self.assertAllEqual(grad_value, [[1, 1], [1, 1], [1, 1]])

  def test_hash_filter_save_restore(self):
    config = get_config_str(3)
    hash_filter = ops.create_hash_filter(100, 7, config)
    # we choose a key that is unique enough so they won't collide with each other.
    ids = tf.constant([1, 3 << 17, 1], dtype=tf.int64)
    embedding = tf.zeros([3, 2])
    loss = ops.intercept_gradient(hash_filter, ids, embedding)
    grad = tf.gradients(loss, embedding)[0]
    base_folder = os.path.join(os.environ["TEST_TMPDIR"],
                               "test_hash_filter_save_restore")
    with self.session() as sess:
      # save checkpoint 0
      ckpt_basename_0 = os.path.join(base_folder, "hash_filter_test_0")
      hash_filter_save_op = ops.save_hash_filter(hash_filter, ckpt_basename_0,
                                                 True)
      sess.run(hash_filter_save_op)
      self.assertEqual(self._count_files(ckpt_basename_0), 7)

      # restore checkpoint 0
      hash_filter_restore_op = ops.restore_hash_filter(hash_filter,
                                                       ckpt_basename_0, True)
      sess.run(hash_filter_restore_op)

      grad_value = sess.run(grad)
      self.assertAllEqual(grad_value, [[0, 0], [0, 0], [0, 0]])
      grad_value = sess.run(grad)
      self.assertAllEqual(grad_value, [[0, 0], [0, 0], [1, 1]])

      # save checkpoint 1
      ckpt_basename_1 = os.path.join(base_folder, "hash_filter_test_1")
      hash_filter_save_op = ops.save_hash_filter(hash_filter, ckpt_basename_1,
                                                 True)
      sess.run(hash_filter_save_op)
      files = sorted(tf.io.gfile.glob(ckpt_basename_1 + "*"))
      self.assertEqual(self._count_files(ckpt_basename_1), 7)
      # restore checkpoint 1
      hash_filter_restore_op = ops.restore_hash_filter(hash_filter,
                                                       ckpt_basename_1, True)
      sess.run(hash_filter_restore_op)
      grad_value = sess.run(grad)
      self.assertAllEqual(grad_value, [[1, 1], [0, 0], [1, 1]])
      grad_value = sess.run(grad)
      self.assertAllEqual(grad_value, [[1, 1], [1, 1], [1, 1]])
      ckpt_basename = os.path.join(base_folder, "hash_filter_test")

  def test_hash_filter_save_restore_across_multiple_filters(self):
    config = get_config_str(2)

    # Each hash filter contains up to 2 elements.
    hash_filter = ops.create_hash_filter(300, 100, config)
    # we choose a key that is unique enough so they won't collide with each other.
    ids = tf.constant([1, 1 << 17, 2 << 17, 1], dtype=tf.int64)
    embedding = tf.zeros([4, 2])
    loss = ops.intercept_gradient(hash_filter, ids, embedding)
    grad = tf.gradients(loss, embedding)[0]
    base_folder = os.path.join(
        os.environ["TEST_TMPDIR"],
        "test_hash_filter_save_restore_across_multiple_filters")
    with self.session() as sess:
      # save checkpoint 0
      ckpt_basename_0 = os.path.join(base_folder, "hash_filter_test_0")
      hash_filter_save_op = ops.save_hash_filter(hash_filter, ckpt_basename_0,
                                                 True)
      sess.run(hash_filter_save_op)

      # Verify checkpoint content
      ckpt_0_files = sorted(tf.io.gfile.glob(ckpt_basename_0 + "*"))
      self.assertEqual(len(ckpt_0_files), 100)
      for file in ckpt_0_files:
        dump = self._GetHashFilterSplitMetaDump(file)
        self.assertEqual(dump.total_size, 3)
        self.assertEqual(dump.num_elements, 0)
        self.assertEqual(dump.sliding_hash_filter_meta.split_num, 100)
        self.assertEqual(dump.sliding_hash_filter_meta.head, 0)
        self.assertEqual(dump.sliding_hash_filter_meta.head_increment, 0)

      # restore checkpoint 0
      hash_filter_restore_op = ops.restore_hash_filter(hash_filter,
                                                       ckpt_basename_0, True)
      sess.run(hash_filter_restore_op)

      grad_value = sess.run(grad)
      self.assertAllEqual(grad_value, [[0, 0], [0, 0], [0, 0], [0, 0]])
      grad_value = sess.run(grad)
      self.assertAllEqual(grad_value, [[1, 1], [0, 0], [0, 0], [1, 1]])

      # save checkpoint 1
      ckpt_basename_1 = os.path.join(base_folder, "hash_filter_test_1")
      hash_filter_save_op = ops.save_hash_filter(hash_filter, ckpt_basename_1,
                                                 True)
      sess.run(hash_filter_save_op)

      # verify checkpoint 1
      ckpt_1_files = sorted(tf.io.gfile.glob(ckpt_basename_1 + "*"))
      self.assertEqual(len(ckpt_1_files), 100)
      for file in ckpt_1_files[:4]:
        dump = self._GetHashFilterSplitMetaDump(file)
        self.assertEqual(dump.total_size, 3)
        self.assertEqual(dump.num_elements, 2)
        self.assertEqual(dump.sliding_hash_filter_meta.split_num, 100)
        self.assertEqual(dump.sliding_hash_filter_meta.head, 4)
        self.assertEqual(dump.sliding_hash_filter_meta.head_increment, 4)
      for file in ckpt_1_files[4:]:
        dump = self._GetHashFilterSplitMetaDump(file)
        self.assertEqual(dump.total_size, 3)
        self.assertEqual(dump.num_elements, 0)
        self.assertEqual(dump.sliding_hash_filter_meta.split_num, 100)
        self.assertEqual(dump.sliding_hash_filter_meta.head, 4)
        self.assertEqual(dump.sliding_hash_filter_meta.head_increment, 4)

      # restore checkpoint 1
      hash_filter_restore_op = ops.restore_hash_filter(hash_filter,
                                                       ckpt_basename_1, True)
      sess.run(hash_filter_restore_op)
      grad_value = sess.run(grad)
      self.assertAllEqual(grad_value, [[1, 1], [1, 1], [1, 1], [1, 1]])
      grad_value = sess.run(grad)
      self.assertAllEqual(grad_value, [[1, 1], [1, 1], [1, 1], [1, 1]])

  def test_dummy_hash_filter_basic(self):
    hash_filter = ops.create_dummy_hash_filter()
    # we choose a key that is unique enough so they won't collide with each other.
    ids = tf.constant([1, 3 << 17, 1], dtype=tf.int64)
    embedding = tf.zeros([3, 2])
    loss = ops.intercept_gradient(hash_filter, ids, embedding)
    grad = tf.gradients(loss, embedding)[0]
    with self.session() as sess:
      grad_value = sess.run(grad)
      self.assertAllEqual(grad_value, [[1, 1], [1, 1], [1, 1]])

  def test_dummy_hash_filter_save_restore(self):
    basename = "dummy_hash_filter"
    hash_filter = ops.create_dummy_hash_filter()
    # we choose a key that is unique enough so they won't collide with each other.
    ids = tf.constant([1, 3 << 17, 1], dtype=tf.int64)
    embedding = tf.zeros([3, 2])
    loss = ops.intercept_gradient(hash_filter, ids, embedding)
    grad = tf.gradients(loss, embedding)[0]
    with self.session() as sess:
      grad_value = sess.run(grad)
      self.assertAllEqual(grad_value, [[1, 1], [1, 1], [1, 1]])
      hash_filter_save_op = ops.save_hash_filter(hash_filter, basename, False)
      sess.run(hash_filter_save_op)
      self.assertEqual(self._count_files(basename), 0)
      grad_value = sess.run(grad)
      self.assertAllEqual(grad_value, [[1, 1], [1, 1], [1, 1]])
      hash_filter_restore_op = ops.restore_hash_filter(hash_filter, basename,
                                                       False)
      hash_filter_restore_op = ops.restore_hash_filter(hash_filter, basename,
                                                       False)
      self.assertAllEqual(grad_value, [[1, 1], [1, 1], [1, 1]])
      self.assertEqual(self._count_files(basename), 0)

  def test_restore_not_found(self):
    with self.session() as sess:
      non_existent_files = os.path.join(os.environ["TEST_TMPDIR"],
                                        "test_restore_not_found",
                                        "hash_filters")
      config = get_config_str(2)
      hash_filter = ops.create_hash_filter(300, 7, config)
      restore_op = ops.restore_hash_filter(hash_filter, non_existent_files,
                                           True)
      with self.assertRaises(Exception):
        sess.run(restore_op)


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
