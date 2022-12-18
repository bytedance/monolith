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
from tensorflow.python.framework import test_util
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


class GpuHashTableOpsTest(tf.test.TestCase):
  '''
  def test_lookup(self):
    with tf.device("/GPU:0"):
      with tf.compat.v1.Session() as sess:
        dim_size = 2
        hash_table = ops.test_hash_table(dim_size, use_adagrad=True, use_gpu = True)
        embeddings = hash_table.lookup(_get_id_tensor([0, 1, 2]))
        size = hash_table.size()
        embeddings, size = sess.run([embeddings, size])
        print("test_lookup")
        print(np.array(embeddings).shape)

    self.assertAllEqual(embeddings, [[1, 1], [1, 1], [1, 1]])
    self.assertAllEqual(size, 3)
    self.assertNotEqual(hash_table.name, "MonolithHashTable")

  def test_assign_lookup(self):
    with tf.device("/GPU:0"):
      with tf.compat.v1.Session() as sess:
        dim_size = 1
        hash_table = ops.test_hash_table(dim_size, use_adagrad=True, use_gpu = True)
        hash_table = hash_table.assign(_get_id_tensor([1, 2]),
                                      tf.constant([[10], [10]], dtype=tf.float32))
        embeddings1 = hash_table.lookup(_get_id_tensor([0, 1, 2]))
        # Ensure the second assign happens after the first lookup
        with tf.control_dependencies([embeddings1]):
          hash_table = hash_table.assign(
              _get_id_tensor([2]),
              tf.constant([5], dtype=tf.float32))
          embeddings2 = hash_table.lookup(_get_id_tensor([0, 1, 2]))
        embeddings1, embeddings2 = sess.run([embeddings1, embeddings2])
        print("test_assign_lookup")
        print(np.array(embeddings1).shape)
        print(np.array(embeddings2).shape)

    self.assertAllEqual(embeddings1, [[1], [10], [10]])
    self.assertAllEqual(embeddings2, [[1], [10], [5]])
    self.assertNotEqual(hash_table.name, "MonolithHashTable")

  def test_assign_lookup_wrong(self):
    with tf.device("/GPU:0"):
      with tf.compat.v1.Session() as sess:
        dim_size = 1
        hash_table = ops.test_hash_table(dim_size, use_adagrad=True, use_gpu = True)
        hash_table = hash_table.assign(_get_id_tensor([1, 2]),
                                      tf.constant([[10], [10]], dtype=tf.float32))
        embeddings1 = hash_table.lookup(_get_id_tensor([0, 1, 2]))

        embeddings1 = sess.run([embeddings1])
        print("test_assign_lookup_wrong")
        print(np.array(embeddings1).shape)


    self.assertNotEqual(hash_table.name, "MonolithHashTable")

  def test_optimize(self):
    with tf.device("/GPU:0"):
      with tf.compat.v1.Session() as sess:
        dim_size = 1
        hash_table = ops.test_hash_table(dim_size, learning_rate=0.1, use_adagrad=True, use_gpu = True)
        id_tensor = _get_id_tensor([0, 1])
        embeddings = hash_table.lookup(id_tensor)

        # Ensure the second assign happens after the first lookup
        with tf.control_dependencies([embeddings]):
          grads = tf.constant([-1, -1], dtype=tf.float32)
          global_step = _get_id_tensor(0)
          hash_table = hash_table.apply_gradients(id_tensor,
                                                  grads,
                                                  global_step=global_step)

          new_embeddings = hash_table.lookup(_get_id_tensor([0, 1]))
        embeddings, new_embeddings = sess.run([embeddings, new_embeddings])
      self.assertAllClose(new_embeddings, [[1.0953462],[1.0953462]])    
      self.assertNotEqual(hash_table.name, "MonolithHashTable")
  '''

  @test_util.run_gpu_only
  def test_fused_lookup(self):

    with tf.compat.v1.Session() as sess:
      with tf.device("/GPU:0"):
        hash_tables = []
        dim_sizes = [1, 2, 3]
        for x in range(len(dim_sizes)):
          dim_size = dim_sizes[x]
          hash_table = ops.test_hash_table(dim_size,
                                           learning_rate=0.1,
                                           use_adagrad=True,
                                           use_gpu=True)

          hash_table = hash_table.assign(
              _get_id_tensor([0 + 3 * x, 1 + 3 * x]),
              tf.ones([2, dim_size]) if x % 2 == 0 else tf.zeros([2, dim_size]))

          hash_tables.append(hash_table.table)

        embeddings = ops.fused_lookup(
            #hash_table_tesor,
            hash_tables,
            _get_id_tensor([0, 3, 3, 6, 1, 4, 7]),
            fused_slot_size=tf.constant([1, 2, 1, 1, 1, 1]),
            num_of_shards=2)

        embeddings, recv_splits, id_offsets, emb_offsets, emb_dims = sess.run(
            embeddings)
    self.assertAllEqual(embeddings, [1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1])
    self.assertAllEqual(recv_splits, [8, 6])
    self.assertAllEqual(id_offsets, [0, 1, 3, 4, 5, 6])
    self.assertAllEqual(emb_offsets, [0, 1, 5, 8, 9, 11])
    self.assertAllEqual(emb_dims, [1, 4, 3, 1, 2, 3])

  @test_util.run_gpu_only
  def test_fused_optimize(self):
    with tf.compat.v1.Session() as sess:
      with tf.device("/GPU:0"):
        hash_tables = []
        dim_sizes = [1, 2]
        fused_slot_size = tf.constant([1, 1, 1, 1])
        ids = _get_id_tensor([0, 4, 1, 3])
        for x in range(len(dim_sizes)):
          dim_size = dim_sizes[x]
          hash_table = ops.test_hash_table(dim_size,
                                           learning_rate=0.1,
                                           use_adagrad=True,
                                           use_gpu=True)
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
                                              tf.constant([1, 1],
                                                          dtype=tf.int32),
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
    self.assertAllClose(
        embeddings,
        [1.0953462, 0.09877297, 0.09877297, 1.0953462, 0.09877297, 0.09877297])
    self.assertAllEqual(recv_splits, [3, 3])
    self.assertAllEqual(id_offsets, [0, 1, 2, 3])
    self.assertAllEqual(emb_offsets, [0, 1, 3, 4])
    self.assertAllEqual(emb_dims, [1, 2, 1, 2])


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
