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

from absl.testing import parameterized
import hashlib
import os
from typing import Dict, List
from unittest import mock

import tensorflow as tf

from monolith.native_training import basic_restore_hook
from monolith.native_training import entry
from monolith.native_training import learning_rate_functions
from monolith.native_training import multi_hash_table_ops
from monolith.native_training import test_utils


def _id(x):
  return tf.constant(x, dtype=tf.int64)


def _value(x):
  return tf.constant(x, dtype=tf.float32)


def from_configs(configs, *args, **kwargs):
  """We do a serialization/deserialization to make sure it worked in all cases."""
  with tf.name_scope("scope") as scope:
    table = multi_hash_table_ops.MultiHashTable.from_configs(
        configs, *args, **kwargs)
    proto = table.to_proto(export_scope=scope)
    table = multi_hash_table_ops.MultiHashTable.from_proto(proto,
                                                           import_scope=scope)
  return table


class MultiTypeHashTableTest(tf.test.TestCase, parameterized.TestCase):

  def test_lookup_assign_add(self):

    multi_table = multi_hash_table_ops.MultiHashTable.from_configs(
        configs={
            "slot0": test_utils.generate_test_hash_table_config(1),
            "not_used": test_utils.generate_test_hash_table_config(2),
            "slot1": test_utils.generate_test_hash_table_config(2),
            "slot2": test_utils.generate_test_hash_table_config(2),
        })
    multi_table = multi_table.assign_add(
        slot_to_id_and_value={
            "slot0": (_id([0]), _value([[1]])),
            "slot1": (_id([1]), _value([[2, 2]])),
            "slot2": (_id([2, 3]), _value([[4, 4], [8, 8]]))
        })
    values_dict = multi_table.lookup(slot_to_id={
        "slot0": _id([0]),
        "slot1": _id([1]),
        "slot2": _id([2, 3]),
    })
    with tf.compat.v1.train.SingularMonitoredSession() as sess:
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
      multi_table = from_configs(
          configs={
              "slot0": test_utils.generate_test_hash_table_config(1),
              "slot1": test_utils.generate_test_hash_table_config(2),
          })
      sess.run(multi_table.initializer)
      values_dict = multi_table.lookup(slot_to_id={
          "slot0": _id([0]),
          "slot1": _id([1, 2]),
      })
      grads = [tf.constant(2.0), tf.constant([[1.0, 3.0], [2.0, 4.0]])]
      global_step = tf.constant(0, dtype=tf.int64)
      multi_table = multi_table.apply_gradients(slot_to_id_and_grad={
          "slot0": (_id([0]), grads[0]),
          "slot1": (_id([1, 2]), grads[1]),
      },
                                                global_step=global_step)
      values_dict = multi_table.lookup(slot_to_id={
          "slot0": _id([0]),
          "slot1": _id([1, 2]),
      })
      values_dict = sess.run(values_dict)
      expected_dict = {"slot0": [[-2]], "slot1": [[-1, -3], [-2, -4]]}
      for key in expected_dict:
        self.assertAllEqual(values_dict[key], expected_dict[key])

  def test_save_restore(self):
    with tf.Graph().as_default(), self.session() as sess:
      table_0 = from_configs(
          configs={
              "slot0": test_utils.generate_test_hash_table_config(1),
              "slot1": test_utils.generate_test_hash_table_config(2),
              "slot2": test_utils.generate_test_hash_table_config(2),
          })
      table_0 = table_0.assign_add(
          slot_to_id_and_value={
              "slot0": (_id([0, 1]), _value([[1], [2]])),
              "slot1": (_id([2, 3, 4, 5]),
                        _value([[1, 2], [2, 3], [3, 4], [4, 5]])),
              "slot2": (_id([6, 7, 8, 9, 10]),
                        _value([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]))
          })
      basename = os.path.join(os.environ["TEST_TMPDIR"], "test_save_restore",
                              table_0.shared_name)
      table_0 = table_0.save(basename)
      sess.run(table_0.initializer)
      sess.run(table_0.as_op())

    with tf.Graph().as_default(), self.session() as sess:
      table_1 = from_configs(
          configs={
              "slot0": test_utils.generate_test_hash_table_config(1),
              "slot2": test_utils.generate_test_hash_table_config(2),
              "slot3": test_utils.generate_test_hash_table_config(3),
          })
      table_1 = table_1.restore(basename)
      values_dict = table_1.lookup(slot_to_id={
          "slot0": _id([0, 1]),
          "slot2": _id([6, 7, 8, 9, 10]),
      })
      sess.run(table_1.initializer)
      values_dict = sess.run(values_dict)
      expected_values_dict = {
          "slot0": [[1], [2]],
          "slot2": [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
      }
      for slot, values in values_dict.items():
        self.assertAllEqual(values, expected_values_dict[slot])

  def test_save_restore_hook(self):
    basename = os.path.join(os.environ["TEST_TMPDIR"], "test_save_restore_hook",
                            "model.ckpt")
    table = from_configs(
        configs={
            "slot0": test_utils.generate_test_hash_table_config(1),
            "slot1": test_utils.generate_test_hash_table_config(2),
            "slot2": test_utils.generate_test_hash_table_config(2),
        })
    add_op = table.assign_add(
        slot_to_id_and_value={
            "slot0": (_id([0]), _value([[1]])),
            "slot1": (_id([1]), _value([[2, 2]])),
            "slot2": (_id([2, 3]), _value([[4, 4], [8, 8]]))
        }).as_op()
    sub_op = table.assign_add(
        slot_to_id_and_value={
            "slot0": (_id([0]), _value([[-1]])),
            "slot1": (_id([1]), _value([[-2, -3]])),
            "slot2": (_id([2, 3]), _value([[-4, -5], [-6, -7]]))
        }).as_op()
    values_dict = table.lookup(slot_to_id={
        "slot0": _id([0]),
        "slot1": _id([1]),
        "slot2": _id([2, 3]),
    })
    saver_listener = multi_hash_table_ops.MultiHashTableCheckpointSaverListener(
        basename)
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
    restorer_listener = multi_hash_table_ops.MultiHashTableCheckpointRestorerListener(
        basename)
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
      values_dict = sess.run(values_dict)
      expected_values_dict = {
          "slot0": [[1]],
          "slot1": [[2, 2]],
          "slot2": [[4, 4], [8, 8]]
      }
      for slot, values in values_dict.items():
        self.assertAllEqual(values, expected_values_dict[slot])

  def test_meta_graph_export(self):
    multi_table = from_configs(
        configs={
            "slot0": test_utils.generate_test_hash_table_config(1),
            "slot1": test_utils.generate_test_hash_table_config(2),
        })
    meta = tf.compat.v1.train.export_meta_graph()
    self.assertIn(multi_hash_table_ops._MULTI_HASH_TABLE_GRAPH_KEY,
                  meta.collection_def)


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
