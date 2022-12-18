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

import datetime
import os
import random
from typing import Dict, List

import numpy as np
import tensorflow as tf
from tensorflow.python.training import monitored_session

from monolith.native_training import basic_restore_hook
from monolith.native_training import entry
from monolith.native_training import hash_filter_ops
from monolith.native_training import learning_rate_functions
from monolith.native_training import save_utils
import monolith.native_training.hash_table_ops as ops
from monolith.native_training.runtime.hash_table import \
    embedding_hash_table_pb2


def _get_id_tensor(x):
  return tf.constant(x, dtype=tf.int64)


def test_hash_table_with_hash_filters(dim_size,
                                      hash_filters,
                                      name_suffix="0",
                                      learning_rate=1.0) -> ops.HashTable:
  """
  Returns a hash table which essentially is a |dim_size| float 
  table with sgd optimizer.
  """
  table_config = embedding_hash_table_pb2.EmbeddingHashTableConfig()
  table_config.cuckoo.SetInParent()
  segment = table_config.entry_config.segments.add()
  segment.dim_size = dim_size
  segment.opt_config.sgd.SetInParent()
  segment.init_config.zeros.SetInParent()
  config = entry.HashTableConfigInstance(table_config, [learning_rate])
  return ops.hash_table_from_config(config=config,
                                    hash_filter=hash_filters[0],
                                    name_suffix=name_suffix)


def test_hash_table(*args, **kwargs):
  """Serialize and deserialize hash table to make sure this process works fine"""
  with tf.name_scope("scope") as scope:
    h = ops.test_hash_table(*args, **kwargs)
    proto = h.to_proto(export_scope=scope)
    return ops.HashTable.from_proto(proto, import_scope=scope)


class HashTableOpsTest(tf.test.TestCase):

  def test_basic(self):
    with tf.compat.v1.Session() as sess:
      dim_size = 1
      hash_table = ops.vocab_hash_table(3, dim_size)
      hash_table = hash_table.assign_add(_get_id_tensor([0, 1]),
                                         tf.ones([2, dim_size]))
      embeddings = hash_table.lookup(_get_id_tensor([0, 1, 2]))
      size = hash_table.size()
      embeddings, size = sess.run([embeddings, size])
    self.assertAllEqual(embeddings, [[1], [1], [0]])
    self.assertAllEqual(size, 2)
    self.assertNotEqual(hash_table.name, "MonolithHashTable")

  def test_assign(self):
    with tf.compat.v1.Session() as sess:
      dim_size = 1
      hash_table = ops.vocab_hash_table(3, dim_size)
      hash_table = hash_table.assign(_get_id_tensor([0, 1]),
                                     tf.ones([2, dim_size]))
      embeddings1 = hash_table.lookup(_get_id_tensor([0, 1, 2]))

      # Ensure the second assign happens after the first lookup
      with tf.control_dependencies([embeddings1]):
        hash_table = hash_table.assign(
            _get_id_tensor([1]),
            tf.constant([5 for _ in range(dim_size)], dtype=tf.float32))
        embeddings2 = hash_table.lookup(_get_id_tensor([0, 1, 2]))
      embeddings1, embeddings2 = sess.run([embeddings1, embeddings2])
    self.assertAllEqual(embeddings1, [[1], [1], [0]])
    self.assertAllEqual(embeddings2, [[1], [5], [0]])
    self.assertNotEqual(hash_table.name, "MonolithHashTable")

  def test_lookup_entry(self):
    table = test_hash_table(1)
    updated_table = table.assign(_get_id_tensor([0, 1, 2]),
                                 [[0.1], [0.2], [0.3]])
    self.evaluate(updated_table.as_op())
    entry_strs = table.lookup_entry(_get_id_tensor([0, 1, 2, 3]))
    entry_strs = self.evaluate(entry_strs)
    nums = list()
    for i in range(3):
      # OK to parse
      dump = embedding_hash_table_pb2.EntryDump()
      dump.ParseFromString(entry_strs[i])
      nums.append(dump.num)
    self.assertAllClose(nums, [[0.1], [0.2], [0.3]])
    self.assertEqual(entry_strs[3], b"")

  def test_save_as_tensor(self):
    table = test_hash_table(1)
    updated_table = table.assign(_get_id_tensor([0, 1, 2]),
                                 [[0.1], [0.2], [0.3]])
    self.evaluate(updated_table.as_op())
    _, dump_str = table.save_as_tensor(0, 1, 1000, 0)
    dump_str = self.evaluate(dump_str)
    for i in range(len(dump_str)):
      # OK to parse
      dump = embedding_hash_table_pb2.EntryDump()
      dump.ParseFromString(dump_str[i])

  def testNameConflict(self):
    with self.session() as sess:
      hash_table = test_hash_table(1, name_suffix="same_suffix")
      with self.assertRaises(ValueError):
        test_hash_table(1, name_suffix="same_suffix")

  def test_gradients(self):
    with tf.compat.v1.Session() as sess:
      hash_table = test_hash_table(1, learning_rate=0.1)
      id_tensor = _get_id_tensor([0, 0, 1])
      embeddings = hash_table.lookup(id_tensor)
      loss = -embeddings
      grads = tf.gradients(loss, embeddings)
      global_step = _get_id_tensor(0)
      hash_table = hash_table.apply_gradients(id_tensor,
                                              grads[0],
                                              global_step=global_step)

      new_embeddings = hash_table.lookup(_get_id_tensor([0, 1]))
      new_embeddings = sess.run(new_embeddings)
    self.assertAllClose(new_embeddings, [[0.2], [0.1]])

  def test_gradients_with_learning_rate_fn(self):
    with tf.compat.v1.Session() as sess:
      hash_table = test_hash_table(1, learning_rate=lambda: 0.1)
      id_tensor = _get_id_tensor([0, 0, 1])
      embeddings = hash_table.lookup(id_tensor)
      loss = -embeddings
      grads = tf.gradients(loss, embeddings)
      global_step = _get_id_tensor(0)
      hash_table = hash_table.apply_gradients(id_tensor,
                                              grads[0],
                                              global_step=global_step)
      new_embeddings = hash_table.lookup(_get_id_tensor([0, 1]))
      new_embeddings = sess.run(new_embeddings)
    self.assertAllClose(new_embeddings, [[0.2], [0.1]])

  def test_gradients_with_learning_rate_decay(self):
    with tf.compat.v1.Session() as sess:
      global_step = tf.compat.v1.train.get_or_create_global_step()
      self.evaluate(tf.compat.v1.global_variables_initializer())
      self.evaluate(tf.compat.v1.assign_add(global_step, 1))
      hash_table = test_hash_table(
          1,
          learning_rate=learning_rate_functions.PolynomialDecay(
              initial_learning_rate=0.01,
              decay_steps=10,
              end_learning_rate=0.11))
      id_tensor = _get_id_tensor([0, 0, 1])
      embeddings = hash_table.lookup(id_tensor)
      loss = -embeddings
      grads = tf.gradients(loss, embeddings)
      hash_table = hash_table.apply_gradients(id_tensor,
                                              grads[0],
                                              global_step=global_step)

      new_embeddings = hash_table.lookup(_get_id_tensor([0, 1]))
      new_embeddings = sess.run(new_embeddings)
    self.assertAllClose(new_embeddings, [[0.04], [0.02]])

  def test_gradients_with_dedup(self):
    vec_dim = 10
    with tf.compat.v1.Session() as sess:
      hash_table = test_hash_table(vec_dim, learning_rate=0.1)
      id_tensor = _get_id_tensor([0, 1, 0, 1, 0])
      embeddings = hash_table.lookup(id_tensor)
      loss = -embeddings
      grads = tf.gradients(loss, embeddings)
      global_step = _get_id_tensor(0)
      hash_table = hash_table.apply_gradients(id_tensor,
                                              grads[0],
                                              global_step=global_step,
                                              enable_dedup=True)

      new_embeddings = hash_table.lookup(_get_id_tensor([0, 1]))
      new_embeddings = sess.run(new_embeddings)
    expected_output = [[0.3 for _ in range(vec_dim)],
                       [0.2 for _ in range(vec_dim)]]
    self.assertAllClose(new_embeddings, expected_output)

  def test_gradients_with_different_ids(self):
    with tf.compat.v1.Session() as sess:
      hash_table = test_hash_table(1, learning_rate=0.1)
      embeddings = hash_table.lookup(_get_id_tensor([0, 0, 1]))
      loss = -embeddings
      grads = tf.gradients(loss, embeddings)
      global_step = _get_id_tensor(0)
      hash_table = hash_table.apply_gradients(_get_id_tensor([1, 0, 1]),
                                              grads[0],
                                              global_step=global_step)

      new_embeddings = hash_table.lookup(_get_id_tensor([0, 1]))
      new_embeddings = sess.run(new_embeddings)
    self.assertAllClose(new_embeddings, [[0.1], [0.2]])

  def test_gradients_with_hash_filter(self):
    with tf.compat.v1.Session() as sess:
      hash_table = test_hash_table(1,
                                   enable_hash_filter=True,
                                   learning_rate=0.1,
                                   occurrence_threshold=3)
      id_tensor = _get_id_tensor([0, 0, 1])
      embeddings = hash_table.lookup(id_tensor)
      loss = -embeddings
      grads = tf.gradients(loss, embeddings)
      global_step = _get_id_tensor(0)
      hash_table = hash_table.apply_gradients(id_tensor,
                                              grads[0],
                                              global_step=global_step)

      expected_results = [
          # occurrence_threshold=3
          # id 0, first apply gradient changes count=1, first apply gradient changes count=2
          # both <=3, no real update.
          # id 1, first apply gradient changes count=1 <= 3, no real update
          [[0], [0]],
          # id 0, first apply gradient changes count=3, first apply gradient changes count=4
          # first update <= 3, second update > 3, update once
          # id 1, first apply gradient changes count=2 <= 3, no real update
          [[0.1], [0]],
          # id 0, first apply gradient changes count=5, first apply gradient changes count=6
          # both update count > 3, update twice
          # id 1, first apply gradient changes count=3 <= 3, no real update
          [[0.3], [0.0]],
          # id 0, first apply gradient changes count=7, first apply gradient changes count=8
          # both update count > 3, update twice
          # id 1, first apply gradient changes count=4 > 3, update once
          [[0.5], [0.1]]
      ]
      for i in range(0, 4):
        new_embeddings = hash_table.lookup(_get_id_tensor([0, 1]))
        new_embeddings = sess.run(new_embeddings)
        self.assertAllClose(new_embeddings, expected_results[i])

  def test_save_restore(self):
    with self.session() as sess:
      hash_table = test_hash_table(1)
      hash_table = hash_table.assign_add(
          _get_id_tensor([-1, 1]), tf.constant([[1], [2]], dtype=tf.float32))
      base_name = os.path.join(os.environ["TEST_TMPDIR"], "test_save_restore",
                               "table")
      hash_table = hash_table.save(base_name)
      sess.run(hash_table.as_op())

    with self.session() as sess:
      hash_table2 = test_hash_table(1, False)
      hash_table2 = hash_table2.restore(base_name)
      embedding = hash_table2.lookup(_get_id_tensor([-1, 1]))
      embedding = sess.run(embedding)
    self.assertAllEqual(embedding, [[1], [2]])

  def test_restore_from_another_table(self):
    with self.session() as sess:
      hash_table1 = test_hash_table(1)
      hash_table1 = hash_table1.assign(_get_id_tensor([1]),
                                       tf.constant([[1]], dtype=tf.float32))
      base_name = os.path.join(os.environ["TEST_TMPDIR"],
                               "test_restore_from_another_table", "table")
      hash_table1 = hash_table1.save(base_name)
      sess.run(hash_table1.as_op())
      hash_table2 = test_hash_table(1, extra_restore_names=[hash_table1.name])
      hash_table2 = hash_table2.restore(base_name)
      embedding = hash_table2.lookup(_get_id_tensor([1]))
      embedding = sess.run(embedding)
    self.assertAllEqual(embedding, [[1]])

  def test_save_restore_with_feature_eviction_assign_add(self):
    with self.session() as sess:
      # Default feature eviction time is expire_time.
      # Feature with ts older than expire_time will be evicted.
      expire_time = 1
      hash_table = test_hash_table(dim_size=1, expire_time=expire_time)
      max_ts = 10000000
      expire_time_in_sec = expire_time * 24 * 3600
      evict_ts = max_ts - expire_time_in_sec - 1
      hash_table = hash_table.assign_add(_get_id_tensor([1]),
                                         tf.constant([[1]], dtype=tf.float32),
                                         tf.constant(evict_ts, dtype=tf.int64))

      # Feature with keep_ts which is newer than expire_time.
      # It will not be evicted after save.
      keep_ts = max_ts - expire_time_in_sec + 1
      hash_table = hash_table.assign_add(_get_id_tensor([2]),
                                         tf.constant([[2]], dtype=tf.float32),
                                         tf.constant(keep_ts, dtype=tf.int64))

      # Feature with max_ts date will be kept and also it will update the internal max_req_time.
      hash_table = hash_table.assign_add(_get_id_tensor([3]),
                                         tf.constant([[3]], dtype=tf.float32),
                                         tf.constant(max_ts, dtype=tf.int64))
      base_name = os.path.join(
          os.environ["TEST_TMPDIR"],
          "test_save_restore_with_feature_eviction_assign_add", "table")
      hash_table = hash_table.save(base_name)
      sess.run(hash_table.as_op())

    with self.session() as sess:
      hash_table2 = test_hash_table(1, False)
      hash_table2 = hash_table2.restore(base_name)
      embedding = hash_table2.lookup(_get_id_tensor([1, 2, 3]))
      embedding = sess.run(embedding)
    self.assertAllEqual(embedding, [[0], [2], [3]])

  def test_save_restore_with_feature_eviction_apply_gradients(self):
    with self.session() as sess:
      # Default feature eviction time is expire_time.
      # Feature with evic_ts older than expire_time will be evicted.
      expire_time = 1
      hash_table = test_hash_table(dim_size=1, expire_time=expire_time)
      max_ts = 10000000
      expire_time_in_sec = expire_time * 24 * 3600
      evict_ts = max_ts - expire_time_in_sec - 1
      global_step = _get_id_tensor(0)
      hash_table = hash_table.apply_gradients(_get_id_tensor([1]),
                                              tf.constant([[1]],
                                                          dtype=tf.float32),
                                              global_step,
                                              req_time=tf.constant(
                                                  evict_ts, dtype=tf.int64))

      # Feature with keep_ts which is newer than expire_time.
      # It will not be evicted after save.
      keep_ts = max_ts - expire_time_in_sec + 1
      global_step = _get_id_tensor(0)
      hash_table = hash_table.apply_gradients(_get_id_tensor([2]),
                                              tf.constant([[2]],
                                                          dtype=tf.float32),
                                              global_step,
                                              req_time=tf.constant(
                                                  keep_ts, dtype=tf.int64))

      # Feature with max_ts will be kept and also it will update the internal max_req_time.
      global_step = _get_id_tensor(0)
      hash_table = hash_table.apply_gradients(_get_id_tensor([3]),
                                              tf.constant([[3]],
                                                          dtype=tf.float32),
                                              global_step,
                                              req_time=tf.constant(
                                                  max_ts, dtype=tf.int64))

      base_name = os.path.join(
          os.environ["TEST_TMPDIR"],
          "test_save_restore_with_feature_eviction_apply_gradients", "table")
      hash_table = hash_table.save(base_name)
      sess.run(hash_table.as_op())

    with self.session() as sess:
      hash_table2 = test_hash_table(1, False)
      hash_table2 = hash_table2.restore(base_name)
      embedding = hash_table2.lookup(_get_id_tensor([1, 2, 3]))
      embedding = sess.run(embedding)
    self.assertAllEqual(embedding, [[0], [-2], [-3]])

  def test_entry_ttl_zero(self):
    basename = os.path.join(os.environ["TEST_TMPDIR"], "test_entry_ttl",
                            "table")
    with self.session() as sess:
      hash_table = test_hash_table(1, expire_time=0)
      hash_table = hash_table.assign_add(
          _get_id_tensor([-1, 1]), tf.constant([[1], [2]], dtype=tf.float32))
      hash_table = hash_table.save(basename)
      sess.run(hash_table.as_op())
    with self.session() as sess:
      hash_table2 = test_hash_table(1)
      hash_table2 = hash_table2.restore(basename)
      embedding = hash_table2.lookup(_get_id_tensor([-1, 1]))
      embedding = sess.run(embedding)
    self.assertAllEqual(embedding, [[0], [0]])

  def test_entry_ttl_not_zero(self):
    basename = os.path.join(os.environ["TEST_TMPDIR"],
                            "test_entry_ttl_not_zero", "table")
    with self.session() as sess:
      hash_table = test_hash_table(1, expire_time=60 * 60)
      hash_table = hash_table.assign_add(
          _get_id_tensor([-1, 1]), tf.constant([[1], [2]], dtype=tf.float32))
      hash_table = hash_table.save(basename)
      sess.run(hash_table.as_op())
    with self.session() as sess:
      hash_table2 = test_hash_table(1)
      hash_table2 = hash_table2.restore(basename)
      embedding = hash_table2.lookup(_get_id_tensor([-1, 1]))
      embedding = sess.run(embedding)
    self.assertAllEqual(embedding, [[1], [2]])

  def test_entry_ttl_by_slots(self):
    basename = os.path.join(os.environ["TEST_TMPDIR"],
                            "test_entry_ttl_by_slots", "table")
    table_config = embedding_hash_table_pb2.EmbeddingHashTableConfig()
    table_config.cuckoo.SetInParent()
    segment = table_config.entry_config.segments.add()
    segment.dim_size = 1
    segment.opt_config.sgd.SetInParent()
    segment.init_config.zeros.SetInParent()
    table_config.slot_expire_time_config.default_expire_time = 60 * 60
    slot_expire_time_1 = table_config.slot_expire_time_config.slot_expire_times.add(
    )
    slot_expire_time_1.slot = 1
    slot_expire_time_1.expire_time = 0
    slot_expire_time_2 = table_config.slot_expire_time_config.slot_expire_times.add(
    )
    slot_expire_time_2.slot = 2
    slot_expire_time_2.expire_time = 1
    hash_filters = hash_filter_ops.create_hash_filters(0, False)
    config = entry.HashTableConfigInstance(table_config, [1.0])

    with self.session() as sess:
      id_1 = (1 << 48)
      id_2 = (2 << 48)
      name_suffix = tf.compat.v1.get_default_graph().unique_name("")
      hash_table = ops.hash_table_from_config(config,
                                              hash_filter=hash_filters[0],
                                              name_suffix=name_suffix)
      hash_table = hash_table.assign_add(
          _get_id_tensor([id_1, id_2]), tf.constant([[1], [2]],
                                                    dtype=tf.float32),
          tf.constant(100, dtype=tf.int64))
      hash_table = hash_table.save(basename)
      sess.run(hash_table.as_op())

    basename_new = os.path.join(os.environ["TEST_TMPDIR"],
                                "test_entry_ttl_by_slots", "table_new")
    with self.session() as sess:
      name_suffix = tf.compat.v1.get_default_graph().unique_name("")
      hash_table2 = ops.hash_table_from_config(config,
                                               hash_filter=hash_filters[0],
                                               name_suffix=name_suffix)
      hash_table2 = hash_table2.restore(basename)
      embedding_2 = hash_table2.lookup(_get_id_tensor([id_1, id_2]))
      embedding_2 = sess.run(embedding_2)
      hash_table2 = hash_table2.save(basename_new)
      sess.run(hash_table2.as_op())
    self.assertAllEqual(embedding_2, [[0], [2]])

    with self.session() as sess:
      hash_table3 = test_hash_table(1)
      hash_table3 = hash_table3.restore(basename_new)
      embedding_3 = hash_table3.lookup(_get_id_tensor([id_1, id_2]))
    self.assertAllEqual(embedding_3, [[0], [2]])

  def test_restore_not_found(self):
    with self.session() as sess:
      non_existent_files = os.path.join(os.environ["TEST_TMPDIR"],
                                        "test_restore_not_found", "table")
      hash_table2 = test_hash_table(1)
      hash_table2 = hash_table2.restore(non_existent_files)
      with self.assertRaises(Exception):
        sess.run(hash_table2.as_op())

  def test_save_restore_hook(self):
    basename = os.path.join(os.environ["TEST_TMPDIR"], "test_save_restore_hook",
                            "model.ckpt")
    hash_filter = hash_filter_ops.create_dummy_hash_filter()
    hash_table = test_hash_table(1)
    add_op = hash_table.assign_add(_get_id_tensor([0]),
                                   tf.constant([[1]],
                                               dtype=tf.float32)).as_op()
    sub_op = hash_table.assign_add(_get_id_tensor([0]),
                                   tf.constant([[-1]],
                                               dtype=tf.float32)).as_op()
    embedding = hash_table.lookup(_get_id_tensor([0]))
    saver_listener = ops.HashTableCheckpointSaverListener(basename)
    # We need to create some variables to make saver happy.
    tf.compat.v1.train.create_global_step()
    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(),
                                     sharded=True,
                                     max_to_keep=10,
                                     keep_checkpoint_every_n_hours=2)
    saver_hook = tf.estimator.CheckpointSaverHook(os.path.dirname(basename),
                                                  save_steps=1000,
                                                  saver=saver,
                                                  listeners=[saver_listener])
    restorer_listener = ops.HashTableCheckpointRestorerListener(basename)
    restore_hook = basic_restore_hook.CheckpointRestorerHook(
        listeners=[restorer_listener])

    with self.session() as sess:
      saver_hook.begin()
      sess.run(tf.compat.v1.global_variables_initializer())
      # In the estimator API, graph will be finalized before calling hook
      g = tf.compat.v1.get_default_graph()
      g.finalize()
      sess.run(add_op)
      saver_hook.after_create_session(sess, None)
      sess.run(sub_op)
      # restore will override sub_op
      restore_hook.after_create_session(sess, None)
      embedding = sess.run(embedding)
    self.assertAllEqual(embedding, [[1]])

  def test_restore_after_save(self):
    ckpt_prefix = os.path.join(os.environ["TEST_TMPDIR"],
                               "test_restore_after_save", "model.ckpt")
    hash_table = test_hash_table(1)
    assign_1_op = hash_table.assign(_get_id_tensor([0]),
                                    tf.constant([[1]],
                                                dtype=tf.float32)).as_op()
    assign_2_op = hash_table.assign(_get_id_tensor([0]),
                                    tf.constant([[2]],
                                                dtype=tf.float32)).as_op()
    emb = hash_table.lookup(_get_id_tensor([0]))

    class AssignSaverListener(tf.estimator.CheckpointSaverListener):

      def after_save(self, session, global_step_value):
        session.run(assign_2_op)

    # We need to create some variables to make saver happy.
    tf.compat.v1.train.create_global_step()
    saver = tf.compat.v1.train.Saver()
    saver_hook = tf.estimator.CheckpointSaverHook(
        os.path.dirname(ckpt_prefix),
        save_steps=100,
        saver=saver,
        listeners=[
            ops.HashTableCheckpointSaverListener(ckpt_prefix),
            AssignSaverListener(),
            ops.HashTableRestorerSaverLitsener(ckpt_prefix)
        ])

    with self.session() as sess:
      saver_hook.begin()
      sess.run(tf.compat.v1.global_variables_initializer())
      sess.run(assign_1_op)
      saver_hook.after_create_session(sess, None)
      self.assertAllEqual([[1]], sess.run(emb))

  def test_save_restore_hook_with_feature_eviction_assign_add(self):
    basename = os.path.join(
        os.environ["TEST_TMPDIR"],
        "test_save_restore_hook_with_feature_eviction_assign_add", "model.ckpt")
    hash_filter = hash_filter_ops.create_dummy_hash_filter()
    # Default feature eviction time is expire_time.
    # Feature with ts older than expire_time will be evicted.
    expire_time = 1
    hash_table = test_hash_table(dim_size=1, expire_time=expire_time)
    max_ts = 10000000
    expire_time_in_sec = expire_time * 24 * 3600
    evict_ts = max_ts - expire_time_in_sec - 1
    assign_op_1 = hash_table.assign_add(_get_id_tensor([1]),
                                        tf.constant([[1]], dtype=tf.float32),
                                        tf.constant(evict_ts,
                                                    dtype=tf.int64)).as_op()

    keep_ts = max_ts - expire_time_in_sec + 1
    assign_op_2 = hash_table.assign_add(_get_id_tensor([2]),
                                        tf.constant([[2]], dtype=tf.float32),
                                        tf.constant(keep_ts,
                                                    dtype=tf.int64)).as_op()

    assign_op_3 = hash_table.assign_add(_get_id_tensor([3]),
                                        tf.constant([[3]], dtype=tf.float32),
                                        tf.constant(max_ts,
                                                    dtype=tf.int64)).as_op()

    embedding = hash_table.lookup(_get_id_tensor([1, 2, 3]))
    saver_listener = ops.HashTableCheckpointSaverListener(basename)
    # We need to create some variables to make saver happy.
    tf.compat.v1.train.create_global_step()
    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(),
                                     sharded=True,
                                     max_to_keep=10,
                                     keep_checkpoint_every_n_hours=2)
    saver_hook = tf.estimator.CheckpointSaverHook(os.path.dirname(basename),
                                                  save_steps=1000,
                                                  saver=saver,
                                                  listeners=[saver_listener])
    restorer_listener = ops.HashTableCheckpointRestorerListener(basename)
    restore_hook = basic_restore_hook.CheckpointRestorerHook(
        listeners=[restorer_listener])

    with self.session() as sess:
      saver_hook.begin()
      sess.run(tf.compat.v1.global_variables_initializer())
      # In the estimator API, graph will be finalized before calling hook
      g = tf.compat.v1.get_default_graph()
      g.finalize()
      sess.run(assign_op_1)
      sess.run(assign_op_2)
      sess.run(assign_op_3)
      embedding_values = sess.run(embedding)
      self.assertAllEqual(embedding_values, [[1], [2], [3]])

      saver_hook.after_create_session(sess, None)
      restore_hook.after_create_session(sess, None)
      embedding_values = sess.run(embedding)
      self.assertAllEqual(embedding, [[0], [2], [3]])

  def test_save_restore_hook_with_feature_eviction_apply_gradients(self):
    basename = os.path.join(
        os.environ["TEST_TMPDIR"],
        "test_save_restore_hook_with_feature_eviction_apply_gradients",
        "model.ckpt")
    hash_filter = hash_filter_ops.create_dummy_hash_filter()
    # Default feature eviction time is expire_time.
    # Feature with ts older than expire_time will be evicted.
    expire_time = 1
    hash_table = test_hash_table(dim_size=1, expire_time=expire_time)
    max_ts = 10000000
    expire_time_in_sec = expire_time * 24 * 3600
    evict_ts = max_ts - expire_time_in_sec - 1
    global_step = _get_id_tensor(0)
    assign_op_1 = hash_table.apply_gradients(
        _get_id_tensor([1]),
        tf.constant([[1]], dtype=tf.float32),
        global_step,
        req_time=tf.constant(evict_ts, dtype=tf.int64)).as_op()

    ts_to_keep = max_ts - expire_time_in_sec + 1
    global_step = _get_id_tensor(0)
    assign_op_2 = hash_table.apply_gradients(
        _get_id_tensor([2]),
        tf.constant([[2]], dtype=tf.float32),
        global_step,
        req_time=tf.constant(ts_to_keep, dtype=tf.int64)).as_op()

    global_step = _get_id_tensor(0)
    assign_op_3 = hash_table.apply_gradients(
        _get_id_tensor([3]),
        tf.constant([[3]], dtype=tf.float32),
        global_step,
        req_time=tf.constant(max_ts, dtype=tf.int64)).as_op()

    embedding = hash_table.lookup(_get_id_tensor([1, 2, 3]))
    saver_listener = ops.HashTableCheckpointSaverListener(basename)
    # We need to create some variables to make saver happy.
    tf.compat.v1.train.create_global_step()
    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(),
                                     sharded=True,
                                     max_to_keep=10,
                                     keep_checkpoint_every_n_hours=2)
    saver_hook = tf.estimator.CheckpointSaverHook(os.path.dirname(basename),
                                                  save_steps=1000,
                                                  saver=saver,
                                                  listeners=[saver_listener])
    restorer_listener = ops.HashTableCheckpointRestorerListener(basename)
    restore_hook = basic_restore_hook.CheckpointRestorerHook(
        listeners=[restorer_listener])

    with self.session() as sess:
      saver_hook.begin()
      sess.run(tf.compat.v1.global_variables_initializer())
      # In the estimator API, graph will be finalized before calling hook
      g = tf.compat.v1.get_default_graph()
      g.finalize()
      sess.run(assign_op_1)
      sess.run(assign_op_2)
      sess.run(assign_op_3)
      embedding_values = sess.run(embedding)
      self.assertAllEqual(embedding_values, [[-1], [-2], [-3]])

      saver_hook.after_create_session(sess, None)
      restore_hook.after_create_session(sess, None)
      embedding_values = sess.run(embedding)
      self.assertAllEqual(embedding, [[0], [-2], [-3]])

  def test_save_restore_hook_with_no_req_time_feature_eviction_apply_gradients(
      self):
    basename = os.path.join(
        os.environ["TEST_TMPDIR"],
        "test_save_restore_hook_with_no_req_time_feature_eviction_apply_gradients",
        "model.ckpt")
    hash_filter = hash_filter_ops.create_dummy_hash_filter()
    hash_table = test_hash_table(dim_size=1, expire_time=1)
    global_step = _get_id_tensor(0)
    assign_op_1 = hash_table.apply_gradients(
        _get_id_tensor([1]), tf.constant([[1]], dtype=tf.float32),
        global_step).as_op()

    assign_op_2 = hash_table.apply_gradients(
        _get_id_tensor([2]), tf.constant([[2]], dtype=tf.float32),
        global_step).as_op()

    embedding = hash_table.lookup(_get_id_tensor([1, 2]))
    saver_listener = ops.HashTableCheckpointSaverListener(basename)
    # We need to create some variables to make saver happy.
    tf.compat.v1.train.create_global_step()
    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(),
                                     sharded=True,
                                     max_to_keep=10,
                                     keep_checkpoint_every_n_hours=2)
    saver_hook = tf.estimator.CheckpointSaverHook(os.path.dirname(basename),
                                                  save_steps=1000,
                                                  saver=saver,
                                                  listeners=[saver_listener])
    restorer_listener = ops.HashTableCheckpointRestorerListener(basename)
    restore_hook = basic_restore_hook.CheckpointRestorerHook(
        listeners=[restorer_listener])

    with self.session() as sess:
      saver_hook.begin()
      sess.run(tf.compat.v1.global_variables_initializer())
      # In the estimator API, graph will be finalized before calling hook
      g = tf.compat.v1.get_default_graph()
      g.finalize()
      sess.run(assign_op_1)
      sess.run(assign_op_2)
      embedding_values = sess.run(embedding)
      self.assertAllEqual(embedding_values, [[-1], [-2]])

      saver_hook.after_create_session(sess, None)
      restore_hook.after_create_session(sess, None)
      embedding_values = sess.run(embedding)
      self.assertAllEqual(embedding, [[-1], [-2]])

  def test_save_restore_hook_with_zero_req_time_feature_eviction_apply_gradients(
      self):
    basename = os.path.join(
        os.environ["TEST_TMPDIR"],
        "test_save_restore_hook_with_zero_req_time_feature_eviction_apply_gradients",
        "model.ckpt")
    hash_filter = hash_filter_ops.create_dummy_hash_filter()
    hash_table = test_hash_table(dim_size=1, expire_time=1)
    global_step = _get_id_tensor(0)
    assign_op_1 = hash_table.apply_gradients(_get_id_tensor([1]),
                                             tf.constant([[1]],
                                                         dtype=tf.float32),
                                             global_step,
                                             req_time=tf.constant(
                                                 0, dtype=tf.int64)).as_op()

    global_step = _get_id_tensor(0)
    assign_op_2 = hash_table.apply_gradients(_get_id_tensor([2]),
                                             tf.constant([[2]],
                                                         dtype=tf.float32),
                                             global_step,
                                             req_time=tf.constant(
                                                 0, dtype=tf.int64)).as_op()

    embedding = hash_table.lookup(_get_id_tensor([1, 2]))
    saver_listener = ops.HashTableCheckpointSaverListener(basename)
    # We need to create some variables to make saver happy.
    tf.compat.v1.train.create_global_step()
    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(),
                                     sharded=True,
                                     max_to_keep=10,
                                     keep_checkpoint_every_n_hours=2)
    saver_hook = tf.estimator.CheckpointSaverHook(os.path.dirname(basename),
                                                  save_steps=1000,
                                                  saver=saver,
                                                  listeners=[saver_listener])
    restorer_listener = ops.HashTableCheckpointRestorerListener(basename)
    restore_hook = basic_restore_hook.CheckpointRestorerHook(
        listeners=[restorer_listener])

    with self.session() as sess:
      saver_hook.begin()
      sess.run(tf.compat.v1.global_variables_initializer())
      # In the estimator API, graph will be finalized before calling hook
      g = tf.compat.v1.get_default_graph()
      g.finalize()
      sess.run(assign_op_1)
      sess.run(assign_op_2)
      embedding_values = sess.run(embedding)
      self.assertAllEqual(embedding_values, [[-1], [-2]])

      saver_hook.after_create_session(sess, None)
      restore_hook.after_create_session(sess, None)
      embedding_values = sess.run(embedding)
      self.assertAllEqual(embedding, [[-1], [-2]])

  def test_save_restore_hook_with_same_req_time_feature_eviction_apply_gradients(
      self):
    basename = os.path.join(
        os.environ["TEST_TMPDIR"],
        "test_save_restore_hook_with_same_req_time_feature_eviction_apply_gradients",
        "model.ckpt")
    hash_filter = hash_filter_ops.create_dummy_hash_filter()
    hash_table = test_hash_table(dim_size=1, expire_time=1)
    global_step = _get_id_tensor(0)
    assign_op_1 = hash_table.apply_gradients(_get_id_tensor([1]),
                                             tf.constant([[1]],
                                                         dtype=tf.float32),
                                             global_step,
                                             req_time=tf.constant(
                                                 100, dtype=tf.int64)).as_op()

    global_step = _get_id_tensor(0)
    assign_op_2 = hash_table.apply_gradients(_get_id_tensor([2]),
                                             tf.constant([[2]],
                                                         dtype=tf.float32),
                                             global_step,
                                             req_time=tf.constant(
                                                 100, dtype=tf.int64)).as_op()

    embedding = hash_table.lookup(_get_id_tensor([1, 2]))
    saver_listener = ops.HashTableCheckpointSaverListener(basename)
    # We need to create some variables to make saver happy.
    tf.compat.v1.train.create_global_step()
    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(),
                                     sharded=True,
                                     max_to_keep=10,
                                     keep_checkpoint_every_n_hours=2)
    saver_hook = tf.estimator.CheckpointSaverHook(os.path.dirname(basename),
                                                  save_steps=1000,
                                                  saver=saver,
                                                  listeners=[saver_listener])
    restorer_listener = ops.HashTableCheckpointRestorerListener(basename)
    restore_hook = basic_restore_hook.CheckpointRestorerHook(
        listeners=[restorer_listener])

    with self.session() as sess:
      saver_hook.begin()
      sess.run(tf.compat.v1.global_variables_initializer())
      # In the estimator API, graph will be finalized before calling hook
      g = tf.compat.v1.get_default_graph()
      g.finalize()
      sess.run(assign_op_1)
      sess.run(assign_op_2)
      embedding_values = sess.run(embedding)
      self.assertAllEqual(embedding_values, [[-1], [-2]])

      saver_hook.after_create_session(sess, None)
      restore_hook.after_create_session(sess, None)
      embedding_values = sess.run(embedding)
      self.assertAllEqual(embedding, [[-1], [-2]])

  def test_delete_save_path(self):
    basename = os.path.join(os.environ["TEST_TMPDIR"], "test_delete_save_path",
                            "model.ckpt")
    helper = save_utils.SaveHelper(basename)

    class HashTableCheckpointRestore(ops.HashTableCheckpointRestorerListener):

      def restore_checkpoint(self, sess, global_step_value):
        path_prefix = helper.get_ckpt_asset_dir(
            helper.get_ckpt_prefix(global_step_value))
        self._restore_from_path_prefix(sess, path_prefix)

    class HashFilterCheckpointRestore(
        hash_filter_ops.HashFilterCheckpointRestorerListener):

      def restore_checkpoint(self, sess, global_step_value):
        path_prefix = helper.get_ckpt_asset_dir(
            helper.get_ckpt_prefix(global_step_value))
        self._restore_from_path_prefix(sess, path_prefix)

    config = embedding_hash_table_pb2.SlotOccurrenceThresholdConfig()
    config.default_occurrence_threshold = 0
    enable_hash_filter = True
    hash_filters = hash_filter_ops.create_hash_filters(
        0, enable_hash_filter, config.SerializeToString())
    hash_table = test_hash_table_with_hash_filters(dim_size=1,
                                                   hash_filters=hash_filters)
    add_op = hash_table.assign_add(_get_id_tensor([0]),
                                   tf.constant([[1]],
                                               dtype=tf.float32)).as_op()
    sub_op = hash_table.assign_add(_get_id_tensor([0]),
                                   tf.constant([[-1]],
                                               dtype=tf.float32)).as_op()
    lookup_op = hash_table.lookup(_get_id_tensor([0]))
    global_step = tf.compat.v1.train.get_or_create_global_step()
    train_op = tf.compat.v1.assign_add(global_step, 1)

    hash_table_saver_listener = ops.HashTableCheckpointSaverListener(basename)
    hash_filter_saver_listener = hash_filter_ops.HashFilterCheckpointSaverListener(
        basename, hash_filters, True)
    saver = save_utils.PartialRecoverySaver(tf.compat.v1.global_variables(),
                                            sharded=True,
                                            max_to_keep=1,
                                            keep_checkpoint_every_n_hours=2)
    saver_hook = save_utils.NoFirstSaveCheckpointSaverHook(
        os.path.dirname(basename),
        save_steps=1,
        saver=saver,
        listeners=[hash_table_saver_listener, hash_filter_saver_listener])

    hash_table_restorer_listener = HashTableCheckpointRestore(basename)
    hash_filter_restorer_listener = HashFilterCheckpointRestore(
        basename, hash_filters, True)

    with tf.compat.v1.train.SingularMonitoredSession(
        hooks=[saver_hook],
        checkpoint_dir=os.path.dirname(basename)) as mon_sess:
      sess = mon_sess.raw_session()
      sess.run(add_op)
      # let saving happen in step 1 and step 10.
      mon_sess.run(train_op)
      for _ in range(8):
        sess.run(train_op)
      mon_sess.run(train_op)

      # hash table checkpoint 1 is deleted.
      with self.assertRaises(Exception):
        hash_table_restorer_listener.restore_checkpoint(sess, 1)
      # hash filter checkpoint 1 is deleted.
      with self.assertRaises(Exception):
        hash_filter_restorer_listener.restore_checkpoint(sess, 1)
      sess.run(sub_op)
      # checkpoint 10 is OK.
      hash_table_restorer_listener.restore_checkpoint(sess, 10)
      hash_filter_restorer_listener.restore_checkpoint(sess, 10)
      embedding = sess.run(lookup_op)
      self.assertAllEqual(embedding, [[1]])

  def test_save_restore_with_hash_table_clear_logic(self):
    basename = os.path.join(os.environ["TEST_TMPDIR"],
                            "test_save_restore_with_hash_table_clear_logic",
                            "model.ckpt")
    hash_filter = hash_filter_ops.create_dummy_hash_filter()
    hash_table = test_hash_table(1)
    add_op_0 = hash_table.assign_add(_get_id_tensor([0]),
                                     tf.constant([[1]],
                                                 dtype=tf.float32)).as_op()
    add_op_1 = hash_table.assign_add(_get_id_tensor([1]),
                                     tf.constant([[1]],
                                                 dtype=tf.float32)).as_op()
    embedding_0 = hash_table.lookup(_get_id_tensor([0]))
    embedding_1 = hash_table.lookup(_get_id_tensor([1]))
    saver_listener = ops.HashTableCheckpointSaverListener(basename)
    # We need to create some variables to make saver happy.
    tf.compat.v1.train.create_global_step()
    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(),
                                     sharded=True,
                                     max_to_keep=10,
                                     keep_checkpoint_every_n_hours=2)
    saver_hook = tf.estimator.CheckpointSaverHook(os.path.dirname(basename),
                                                  save_steps=1000,
                                                  saver=saver,
                                                  listeners=[saver_listener])
    restorer_listener = ops.HashTableCheckpointRestorerListener(basename)
    restore_hook = basic_restore_hook.CheckpointRestorerHook(
        listeners=[restorer_listener])

    with self.session() as sess:
      saver_hook.begin()
      sess.run(tf.compat.v1.global_variables_initializer())
      # In the estimator API, graph will be finalized before calling hook
      g = tf.compat.v1.get_default_graph()
      g.finalize()
      sess.run(add_op_0)
      saver_hook.after_create_session(sess, None)

      sess.run(add_op_1)

      embedding_value = sess.run(embedding_1)
      self.assertAllEqual(embedding_value, [[1]])

      restore_hook.after_create_session(sess, None)

      # update before save will be restored from checkpoint.
      embedding_value = sess.run(embedding_0)
      self.assertAllEqual(embedding_value, [[1]])

      # update after save will not be restored from checkpoint.
      embedding_value = sess.run(embedding_1)
      self.assertAllEqual(embedding_value, [[0]])

  def test_hash_table_and_hash_filter_save_restore_hook_together(self):
    basename = os.path.join(
        os.environ["TEST_TMPDIR"],
        "test_hash_table_and_hash_filter_save_restore_hook_together",
        "model.ckpt")

    config = embedding_hash_table_pb2.SlotOccurrenceThresholdConfig()
    config.default_occurrence_threshold = 2
    enable_hash_filter = True
    hash_filters = hash_filter_ops.create_hash_filters(
        0, enable_hash_filter, config.SerializeToString())
    hash_table = test_hash_table_with_hash_filters(dim_size=1,
                                                   hash_filters=hash_filters)
    add_op = hash_table.assign_add(_get_id_tensor([0]),
                                   tf.constant([[1]],
                                               dtype=tf.float32)).as_op()
    embedding = hash_table.lookup(_get_id_tensor([0]))
    hash_table_saver_listener = ops.HashTableCheckpointSaverListener(basename)
    hash_filter_saver_listener = hash_filter_ops.HashFilterCheckpointSaverListener(
        basename, hash_filters, enable_hash_filter)

    # We need to create some variables to make saver happy.
    tf.compat.v1.train.create_global_step()
    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(),
                                     sharded=True,
                                     max_to_keep=10,
                                     keep_checkpoint_every_n_hours=2)
    saver_hook = tf.estimator.CheckpointSaverHook(
        os.path.dirname(basename),
        save_steps=1000,
        saver=saver,
        listeners=[hash_table_saver_listener, hash_filter_saver_listener])

    hash_table_restorer_listener = ops.HashTableCheckpointRestorerListener(
        basename)
    hash_filter_restorer_listener = hash_filter_ops.HashFilterCheckpointRestorerListener(
        basename, hash_filters, True)
    restore_hook = basic_restore_hook.CheckpointRestorerHook(
        listeners=[hash_table_restorer_listener, hash_filter_restorer_listener])

    with self.session() as sess:
      saver_hook.begin()
      sess.run(tf.compat.v1.global_variables_initializer())
      # In the estimator API, graph will be finalized before calling hook
      g = tf.compat.v1.get_default_graph()
      g.finalize()
      # add_op not actually works as count after adding in hash filter is 1.
      sess.run(add_op)
      embedding_value = sess.run(embedding)
      self.assertAllEqual(embedding_value, [[0]])
      # save hash filter ckpt with count is 1.
      saver_hook.after_create_session(sess, None)

      embedding_value = sess.run(embedding)

      # add_op not actually works as count after adding in hash filter is 2.
      sess.run(add_op)
      embedding_value = sess.run(embedding)
      self.assertAllEqual(embedding_value, [[0]])

      # add_op works as count after adding in hash filter is 3.
      sess.run(add_op)
      embedding_value = sess.run(embedding)
      self.assertAllEqual(embedding_value, [[1]])

      # restore hash table ckpt (embedding value is 0)
      # and hash filter ckpt (count is 1)
      restore_hook.after_create_session(sess, None)
      embedding_value = sess.run(embedding)
      self.assertAllEqual(embedding_value, [[0]])
      #add_op not works as count in hash filter is 2 after it restored from ckpt.
      sess.run(add_op)
      embedding_value = sess.run(embedding)
      self.assertAllEqual(embedding_value, [[0]])

      # add_op works as count after adding in hash filter is 3.
      sess.run(add_op)
      embedding_value = sess.run(embedding)
      self.assertAllEqual(embedding_value, [[1]])

      # restore again to test everything is good.
      # restore hash table ckpt (embedding value is 0)
      # and hash filter ckpt (count is 1)
      restore_hook.after_create_session(sess, None)
      embedding_value = sess.run(embedding)
      self.assertAllEqual(embedding_value, [[0]])
      #add_op not works as count in hash filter is 2 after it restored from ckpt.
      sess.run(add_op)
      embedding_value = sess.run(embedding)
      self.assertAllEqual(embedding_value, [[0]])

      # add_op works as count after adding in hash filter is 3.
      sess.run(add_op)
      embedding_value = sess.run(embedding)
      self.assertAllEqual(embedding_value, [[1]])

  def test_two_hash_table_whose_name_is_prefix(self):
    with tf.compat.v1.Session() as sess:
      dim_size = 1
      hash_table1 = test_hash_table(dim_size)
      hash_table2 = test_hash_table(dim_size)
      basename = os.path.join(os.environ["TEST_TMPDIR"],
                              "test_two_hash_table_whose_name_is_prefix")
      hash_table1 = hash_table1.save(basename + "/table1")
      hash_table2 = hash_table2.save(basename + "/table10")
      sess.run([hash_table1.as_op(), hash_table2.as_op()])
      hash_table1 = hash_table1.restore(basename + "/table1")
      hash_table2 = hash_table2.restore(basename + "/table10")
      sess.run([hash_table1.as_op(), hash_table2.as_op()])

  def test_fused_lookup(self):
    with tf.compat.v1.Session() as sess:
      hash_tables = []
      dim_sizes = [1, 1, 2]
      for x in range(len(dim_sizes)):
        dim_size = dim_sizes[x]
        hash_table = ops.vocab_hash_table(9, dim_size)
        hash_table = hash_table.assign(
            _get_id_tensor([0 + 3 * x, 1 + 3 * x]),
            tf.ones([2, dim_size]) if x % 2 == 0 else tf.zeros([2, dim_size]))
        hash_tables.append(hash_table)
      embeddings = ops.fused_lookup(
          [hash_table.table for hash_table in hash_tables],
          _get_id_tensor([0, 4, 6, 1, 3, 7]),
          fused_slot_size=tf.constant([1, 1, 1, 1, 1, 1]),
          num_of_shards=2)
      embeddings, recv_splits, id_offsets, emb_offsets, emb_dims = sess.run(
          embeddings)
    self.assertAllEqual(embeddings, [1, 0, 1, 1, 1, 0, 1, 1])
    self.assertAllEqual(recv_splits, [4, 4])
    self.assertAllEqual(id_offsets, [0, 1, 2, 3, 4, 5])
    self.assertAllEqual(emb_offsets, [0, 1, 2, 4, 5, 6])
    self.assertAllEqual(emb_dims, [1, 1, 2, 1, 1, 2])

  def test_fused_optimize(self):
    with tf.compat.v1.Session() as sess:
      hash_tables = []
      dim_sizes = [1, 2]
      fused_slot_size = tf.constant([1, 1, 1, 1])
      ids = _get_id_tensor([0, 4, 1, 3])
      for x in range(len(dim_sizes)):
        dim_size = dim_sizes[x]
        hash_table = ops.vocab_hash_table(6, dim_size)
        hash_table = hash_table.assign(
            _get_id_tensor([0 + 3 * x, 1 + 3 * x]),
            tf.ones([2, dim_size]) if x == 0 else tf.zeros([2, dim_size]))
        hash_tables.append(hash_table)
      hash_table_resource = [hash_table.table for hash_table in hash_tables]
      #embeddings=[1, 0, 0, 1, 0, 0]
      embeddings, recv_splits, id_offsets, emb_offsets, emb_dims = ops.fused_lookup(
          hash_table_resource, ids, fused_slot_size, num_of_shards=2)
      new_tables = ops.fused_apply_gradient(hash_table_resource,
                                            ids,
                                            fused_slot_size,
                                            tf.constant(
                                                [-1, -2, -2, -1, -2, -2],
                                                dtype=tf.float32),
                                            id_offsets,
                                            emb_offsets,
                                            tf.constant([0.1, 0.1],
                                                        dtype=tf.float32),
                                            tf.constant([1, 1], dtype=tf.int32),
                                            tf.constant(0, dtype=tf.int64),
                                            tf.constant(0, dtype=tf.int64),
                                            num_of_shards=2)
      with tf.control_dependencies(new_tables):
        lookup_op = ops.fused_lookup(hash_table_resource,
                                     ids,
                                     fused_slot_size,
                                     num_of_shards=2)
      embeddings, recv_splits, id_offsets, emb_offsets, emb_dims = sess.run(
          lookup_op)
    self.assertAllClose(embeddings, [1.1, 0.2, 0.2, 1.1, 0.2, 0.2])
    self.assertAllEqual(recv_splits, [3, 3])
    self.assertAllEqual(id_offsets, [0, 1, 2, 3])
    self.assertAllEqual(emb_offsets, [0, 1, 3, 4])
    self.assertAllEqual(emb_dims, [1, 2, 1, 2])

  def test_batch_softmax_optimizer(self):
    table_config = embedding_hash_table_pb2.EmbeddingHashTableConfig()
    table_config.cuckoo.SetInParent()
    segment = table_config.entry_config.segments.add()
    segment.dim_size = 1
    segment.opt_config.batch_softmax.SetInParent()
    segment.init_config.zeros.SetInParent()
    segment.comp_config.fp32.SetInParent()
    learning_rate = 0.1
    config = entry.HashTableConfigInstance(table_config, [learning_rate])
    with self.session() as sess:
      hash_table = ops.hash_table_from_config(config=config,
                                              name_suffix='batch_softmax')
      for global_step in range(1000):
        fids = list()
        if global_step % 5 == 0:
          fids.append(0)
        if global_step % 10 == 0:
          fids.append(1)
        if len(fids) == 0:
          continue
        id_tensor = _get_id_tensor(fids)
        global_step = _get_id_tensor(global_step)
        hash_table = hash_table.apply_gradients(id_tensor,
                                                tf.constant([0.1 for _ in fids],
                                                            dtype=tf.float32),
                                                global_step=global_step)
      item_step_interval = hash_table.lookup(_get_id_tensor([0, 1]))
      item_step_interval = tf.math.maximum(item_step_interval,
                                           tf.constant([1.0], dtype=tf.float32))
      item_step_interval = sess.run(item_step_interval)
    self.assertAllClose([1 / val for val in item_step_interval], [[0.2], [0.1]],
                        atol=0.01)

  def test_extract_fid(self):
    entry = embedding_hash_table_pb2.EntryDump()
    entry.id = 1 << 48
    slot_tensor = ops.extract_slot_from_entry([entry.SerializeToString()])
    self.assertAllEqual(self.evaluate(slot_tensor), [1])

  def test_meta_graph_export(self):
    table = test_hash_table(2)
    meta = tf.compat.v1.train.export_meta_graph()
    self.assertIn(ops._HASH_TABLE_GRAPH_KEY, meta.collection_def)


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
