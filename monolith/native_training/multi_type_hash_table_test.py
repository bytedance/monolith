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

import hashlib
from typing import Dict, List
from unittest import mock

import tensorflow as tf

from monolith.native_training import entry
from monolith.native_training import hash_table_ops
from monolith.native_training import learning_rate_functions
from monolith.native_training import multi_type_hash_table
from monolith.native_training import test_utils


def factory(idx: int, config):
  return hash_table_ops.hash_table_from_config(config=config,
                                               name_suffix=str(idx))


def _id(x):
  return tf.constant(x, dtype=tf.int64)


def _value(x):
  return tf.constant(x, dtype=tf.float32)


class MultiTypeHashTableTest(tf.test.TestCase):

  def test_basic(self):
    with self.session() as sess:
      hash_table = multi_type_hash_table.MultiTypeHashTable(
          {
              "slot0": test_utils.generate_test_hash_table_config(1),
              "slot1": test_utils.generate_test_hash_table_config(2),
              "slot2": test_utils.generate_test_hash_table_config(2),
          }, factory)
      hash_table = hash_table.assign_add({
          "slot0": (_id([0]), _value([[1]])),
          "slot1": (_id([1]), _value([[2, 2]])),
          "slot2": (_id([2, 3]), _value([[4, 4], [8, 8]]))
      })
      values_dict = hash_table.lookup({
          "slot0": _id([0]),
          "slot1": _id([1]),
          "slot2": _id([2, 3]),
      })
      values_dict = sess.run(values_dict)
    expected_values_dict = {
        "slot0": [[1]],
        "slot1": [[2, 2]],
        "slot2": [[4, 4], [8, 8]]
    }
    for slot, values in values_dict.items():
      self.assertAllEqual(values, expected_values_dict[slot])

  def test_apply_gradients(self):
    with self.session() as sess:
      hash_table = multi_type_hash_table.MultiTypeHashTable(
          {
              "slot0": test_utils.generate_test_hash_table_config(1),
              "slot1": test_utils.generate_test_hash_table_config(2),
          }, factory)
      values_dict = hash_table.lookup({
          "slot0": _id([0]),
          "slot1": _id([1, 2]),
      })
      grads = [tf.constant(2.0), tf.constant([[1.0, 3.0], [2.0, 4.0]])]
      global_step = tf.constant(0, dtype=tf.int64)
      hash_table = hash_table.apply_gradients(
          {
              "slot0": (_id([0]), grads[0]),
              "slot1": (_id([1, 2]), grads[1]),
          }, global_step)
      values_dict = hash_table.lookup({
          "slot0": _id([0]),
          "slot1": _id([1, 2]),
      })
      values_dict = sess.run(values_dict)
      expected_dict = {"slot0": [[-2]], "slot1": [[-1, -3], [-2, -4]]}
      for key in expected_dict:
        self.assertAllEqual(values_dict[key], expected_dict[key])

  def test_apply_gradients_with_learning_rate_decay(self):
    with self.session() as sess:
      global_step = tf.compat.v1.train.get_or_create_global_step()
      self.evaluate(tf.compat.v1.global_variables_initializer())
      hash_table = multi_type_hash_table.MultiTypeHashTable(
          {
              "slot0":
                  test_utils.generate_test_hash_table_config(
                      1,
                      learning_rate=learning_rate_functions.PolynomialDecay(
                          initial_learning_rate=0.1,
                          decay_steps=10,
                          end_learning_rate=1.1)),
              "slot1":
                  test_utils.generate_test_hash_table_config(
                      2,
                      learning_rate=learning_rate_functions.PolynomialDecay(
                          initial_learning_rate=0.1,
                          decay_steps=10,
                          end_learning_rate=1.1)),
          }, factory)
      values_dict = hash_table.lookup({
          "slot0": _id([0]),
          "slot1": _id([1, 2]),
      })
      grads = [tf.constant(2.0), tf.constant([[1.0, 3.0], [2.0, 4.0]])]
      hash_table = hash_table.apply_gradients(
          {
              "slot0": (_id([0]), grads[0]),
              "slot1": (_id([1, 2]), grads[1]),
          }, global_step)
      values_dict = hash_table.lookup({
          "slot0": _id([0]),
          "slot1": _id([1, 2]),
      })
      values_dict = sess.run(values_dict)
      expected_dict = {"slot0": [[-0.2]], "slot1": [[-0.1, -0.3], [-0.2, -0.4]]}
      for key in expected_dict:
        self.assertAllClose(values_dict[key], expected_dict[key])

  def test_apply_gradients_without_lookup(self):
    with self.session() as sess:
      hash_table = multi_type_hash_table.MultiTypeHashTable(
          {
              "slot0": test_utils.generate_test_hash_table_config(1),
              "slot1": test_utils.generate_test_hash_table_config(2)
          }, factory)
      global_step = tf.constant(0, dtype=tf.int64)
      hash_table = hash_table.apply_gradients(
          {
              "slot0": (_id([1]), tf.constant(3.0)),
              "slot1": (_id([2, 2]), tf.constant([[1.1, 3.1], [2.2, 4.2]])),
          }, global_step)
      values_dict = hash_table.lookup({"slot0": _id([1]), "slot1": _id([1, 2])})
      values_dict = sess.run(values_dict)
      expected_dict = {"slot0": [[-3]], "slot1": [[0, 0], [-3.3, -7.3]]}
      for key in expected_dict:
        self.assertAllClose(values_dict[key], expected_dict[key])

  def test_fused_lookup(self):
    with self.session() as sess:
      hash_table = multi_type_hash_table.MultiTypeHashTable(
          {
              "slot0": test_utils.generate_test_hash_table_config(1),
              "slot1": test_utils.generate_test_hash_table_config(2),
              "slot2": test_utils.generate_test_hash_table_config(2),
          }, factory)
      hash_table = hash_table.assign_add({
          "slot0": (_id([0]), _value([[1]])),
          "slot1": (_id([1]), _value([[2, 2]])),
          "slot2": (_id([2, 3]), _value([[4, 4], [8, 8]]))
      })
      values_dict = hash_table.fused_lookup([0, 1, 2, 3], [1, 1, 2], 1)
      embeddings, recv_splits, id_offsets, emb_offsets, emb_sizes = sess.run(
          values_dict)
    self.assertAllEqual(embeddings, [1, 2, 2, 4, 4, 8, 8])
    self.assertAllEqual(recv_splits, [7])
    self.assertAllEqual(id_offsets, [0, 1, 2])
    self.assertAllEqual(emb_offsets, [0, 1, 3])
    self.assertAllEqual(emb_sizes, [1, 2, 4])

  def test_fused_lookup_multi_shards(self):
    with self.session() as sess:
      hash_table = multi_type_hash_table.MultiTypeHashTable(
          {
              "slot0": test_utils.generate_test_hash_table_config(1),
              "slot1": test_utils.generate_test_hash_table_config(2),
              "slot2": test_utils.generate_test_hash_table_config(2),
          }, factory)
      hash_table = hash_table.assign_add({
          "slot0": (_id([0]), _value([[1]])),
          "slot1": (_id([1]), _value([[2, 2]])),
          "slot2": (_id([2, 3]), _value([[4, 4], [8, 8]]))
      })
      values_dict = hash_table.fused_lookup([0, 2, 1, 3], [1, 0, 1, 0, 1, 1], 2)
      embeddings, recv_splits, id_offsets, emb_offsets, emb_sizes = sess.run(
          values_dict)
    self.assertAllEqual(embeddings, [1, 4, 4, 2, 2, 8, 8])
    self.assertAllEqual(recv_splits, [3, 4])
    self.assertAllEqual(id_offsets, [0, 1, 1, 2, 2, 3])
    self.assertAllEqual(emb_offsets, [0, 1, 1, 3, 3, 5])
    self.assertAllEqual(emb_sizes, [1, 0, 2, 0, 2, 2])

  def test_fused_apply_gradients(self):
    with self.session() as sess:
      hash_table = multi_type_hash_table.MultiTypeHashTable(
          {
              "slot0": test_utils.generate_test_hash_table_config(1),
              "slot1": test_utils.generate_test_hash_table_config(2)
          }, factory)
      ids = tf.constant([0, 1, 2], dtype=tf.int64)
      fused_slot_size = tf.constant([1, 2])
      embeddings, _, id_offsets, emb_offsets, _ = hash_table.fused_lookup(
          ids, fused_slot_size, 1)
      grads = tf.constant([2.0, 1.0, 3.0, 2.0, 4.0])
      hash_table = hash_table.fused_apply_gradient(
          ids, fused_slot_size, grads, id_offsets, emb_offsets,
          tf.constant(0, dtype=tf.int64), tf.constant(0, dtype=tf.int64), 1)
      lookup_op = hash_table.fused_lookup(ids, fused_slot_size, 1)
      embeddings, recv_splits, id_offsets, emb_offsets, emb_sizes = sess.run(
          lookup_op)
    self.assertAllEqual(embeddings, [-2, -1, -3, -2, -4])
    self.assertAllEqual(recv_splits, [5])
    self.assertAllEqual(id_offsets, [0, 1])
    self.assertAllEqual(emb_offsets, [0, 1])
    self.assertAllEqual(emb_sizes, [1, 4])

  def test_fused_apply_gradients_missing_tables(self):
    with self.session() as sess:
      hash_table = multi_type_hash_table.MultiTypeHashTable(
          {
              "slot0": test_utils.generate_test_hash_table_config(1),
              "slot1": test_utils.generate_test_hash_table_config(2)
          }, factory)
      ids = tf.constant([1, 1], dtype=tf.int64)
      fused_slot_size = tf.constant([1, 0, 1, 0])
      embeddings, _, id_offsets, emb_offsets, _ = hash_table.fused_lookup(
          ids, fused_slot_size, 2)
      grads = tf.constant([1.0, 2.0])
      hash_table = hash_table.fused_apply_gradient(
          ids, fused_slot_size, grads, id_offsets, emb_offsets,
          tf.constant(0, dtype=tf.int64), tf.constant(0, dtype=tf.int64), 2)
      lookup_op = hash_table.fused_lookup(ids, fused_slot_size, 2)
      embeddings, recv_splits, id_offsets, emb_offsets, emb_sizes = sess.run(
          lookup_op)
    self.assertAllEqual(embeddings, [-3, -3])
    self.assertAllEqual(recv_splits, [1, 1])
    self.assertAllEqual(id_offsets, [0, 1, 1, 2])
    self.assertAllEqual(emb_offsets, [0, 1, 1, 2])
    self.assertAllEqual(emb_sizes, [1, 0, 1, 0])


def _multi_type_factory(slot_to_config):
  return multi_type_hash_table.MultiTypeHashTable(slot_to_config, factory)


class MergedMultiTypeHashTable(tf.test.TestCase):

  def testBasic(self):
    with self.session() as sess:
      hash_table = multi_type_hash_table.MergedMultiTypeHashTable(
          {
              "slot0": test_utils.generate_test_hash_table_config(1),
              "slot1": test_utils.generate_test_hash_table_config(2),
              "slot2": test_utils.generate_test_hash_table_config(2),
          }, _multi_type_factory)
      # slot 1 & 2 should be merged.
      updated_hash_table = hash_table.assign_add({
          "slot0": (_id([0]), _value([[1]])),
          "slot1": (_id([1]), _value([[2, 2]])),
          "slot2": (_id([1]), _value([[4, 4]]))
      })
      values_dict = updated_hash_table.lookup({
          "slot0": _id([0]),
          "slot1": _id([1]),
          "slot2": _id([1]),
      })
      values_dict = sess.run(values_dict)
      expected_values_dict = {
          "slot0": [[1]],
          "slot1": [[6, 6]],
          "slot2": [[6, 6]]
      }
      for slot, values in values_dict.items():
        self.assertAllEqual(values, expected_values_dict[slot])
      global_step = tf.constant(0, dtype=tf.int64)
      updated_hash_table = hash_table.apply_gradients(
          {
              "slot0": (_id([0]), _value([[-1]])),
              "slot1": (_id([1]), _value([[1, 1]])),
              "slot2": (_id([1]), _value([[1, 1]]))
          }, global_step)
      values_dict = updated_hash_table.lookup({
          "slot0": _id([0]),
          "slot1": _id([1]),
      })
      values_dict = sess.run(values_dict)
      expected_values_dict = {"slot0": [[2]], "slot1": [[4, 4]]}
      for slot, values in values_dict.items():
        self.assertAllEqual(values, expected_values_dict[slot])

  def testNameStability(self):
    factory = mock.MagicMock()

    def call(slot_to_config: Dict[str, entry.HashTableConfigInstance]):
      self.assertListEqual(list(slot_to_config.keys()),
                           ["e21904dd414d1780e5fc904866dc69c2"])
      return _multi_type_factory(slot_to_config)

    factory.side_effect = call
    hash_table = multi_type_hash_table.MergedMultiTypeHashTable(
        {
            "slot0": test_utils.generate_test_hash_table_config(1),
            "slot1": test_utils.generate_test_hash_table_config(1),
        }, factory)

  def testRestoreName(self):
    factory = mock.MagicMock()

    def call(slot_to_config: Dict[str, entry.HashTableConfigInstance]):
      config = next(iter(slot_to_config.values()))
      expected_name = hashlib.md5("slot_0".encode()).hexdigest()
      self.assertListEqual(config.extra_restore_names, [expected_name])
      return _multi_type_factory(slot_to_config)

    hash_table = multi_type_hash_table.MergedMultiTypeHashTable(
        {
            "fc_slot_0": test_utils.generate_test_hash_table_config(1),
        }, factory)


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
