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
import numpy as np
import tensorflow as tf
import sys

from monolith.native_training import hash_filter_ops
import monolith.native_training.hash_table_ops as ops


def _get_id_tensor(x):
  return tf.constant(x, dtype=tf.int64)


# TODO: use tf.test.Benchmark
class HashTableOpsBenchmark(tf.test.TestCase):

  def test_lookup(self):
    with tf.compat.v1.Session() as sess:
      len, dim_size = (10000, 32)
      id_tensor = _get_id_tensor([x for x in range(len)])
      hash_table = ops.test_hash_table(dim_size)
      hash_table = hash_table.assign_add(id_tensor[:-5],
                                         tf.ones([len, dim_size]))
      hash_table = hash_table.assign_add(id_tensor[-5:],
                                         tf.zeros([len, dim_size]))
      iters = 100
      embedding_one = [float(1) * iters for _ in range(32)]
      embedding_zero = [0 for _ in range(32)]
      start = time.time()
      _embeddings = hash_table.lookup(id_tensor)
      for _ in range(iters):
        embeddings = sess.run(_embeddings)
      total_wall_time = time.time() - start
      print('wall time: {}'.format(total_wall_time / iters))
    self.assertAllClose(embeddings[:-5],
                        [embedding_one for _ in range(len - 5)])
    self.assertAllClose(embeddings[-5:], [embedding_zero for _ in range(5)])

  def test_lookup_multi_thread(self):
    with tf.compat.v1.Session() as sess:
      len, dim_size = (10000, 32)
      id_tensor = _get_id_tensor([x for x in range(len)])
      hash_table = ops.test_hash_table(dim_size)
      hash_table = hash_table.assign_add(id_tensor[:-5],
                                         tf.ones([len, dim_size]))
      hash_table = hash_table.assign_add(id_tensor[-5:],
                                         tf.zeros([len, dim_size]))
      iters = 100
      embedding_one = [float(1) * iters for _ in range(32)]
      embedding_zero = [0 for _ in range(32)]
      start = time.time()
      _embeddings = hash_table.lookup(id_tensor, use_multi_threads=True)
      for _ in range(iters):
        embeddings = sess.run(_embeddings)
      total_wall_time = time.time() - start
      print('wall time(MT): {}'.format(total_wall_time / iters))
    self.assertAllClose(embeddings[:-5],
                        [embedding_one for _ in range(len - 5)])
    self.assertAllClose(embeddings[-5:], [embedding_zero for _ in range(5)])

  def test_basic_optimize(self):
    with tf.compat.v1.Session() as sess:
      len, dim_size = (1000000, 32)
      # We assume each ID is appeared 4 times.
      id_tensor = _get_id_tensor([x // 4 for x in range(len)])
      hash_table = ops.test_hash_table(dim_size,
                                       learning_rate=0.001,
                                       use_adagrad=True)
      hash_table = hash_table.assign_add(id_tensor[:-5],
                                         tf.ones([len, dim_size]))
      hash_table = hash_table.assign_add(id_tensor[-5:],
                                         tf.zeros([len, dim_size]))
      start = time.time()
      embeddings = hash_table.lookup(id_tensor)
      loss = -embeddings
      grads = tf.gradients(loss, embeddings)
      hash_table = hash_table.apply_gradients(zip(grads, [embeddings]))
      embeddings = hash_table.lookup(id_tensor)
      embeddings = sess.run(embeddings)
      total_wall_time = time.time() - start
      print('wall time: {}'.format(total_wall_time))

  def test_multi_threads_optimize(self):
    with tf.compat.v1.Session() as sess:
      len, dim_size = (1000000, 32)
      # We assume each ID is appeared 4 times.
      id_tensor = _get_id_tensor([x // 4 for x in range(len)])
      hash_table = ops.test_hash_table(dim_size,
                                       learning_rate=0.001,
                                       use_adagrad=True)
      hash_table = hash_table.assign_add(id_tensor[:-5],
                                         tf.ones([len, dim_size]))
      hash_table = hash_table.assign_add(id_tensor[-5:],
                                         tf.zeros([len, dim_size]))
      start = time.time()
      embeddings = hash_table.lookup(id_tensor)
      loss = -embeddings
      grads = tf.gradients(loss, embeddings)
      hash_table = hash_table.apply_gradients(zip(grads, [embeddings]),
                                              use_multi_threads=True)
      embeddings = hash_table.lookup(id_tensor)
      embeddings = sess.run(embeddings)
      total_wall_time = time.time() - start
      print('wall time: {}'.format(total_wall_time))

  def test_multi_threads_optimize_with_dedup(self):
    with tf.compat.v1.Session() as sess:
      len, dim_size = (1000000, 32)
      # We assume each ID is appeared 4 times.
      id_tensor = _get_id_tensor([x // 4 for x in range(len)])
      hash_table = ops.test_hash_table(dim_size,
                                       learning_rate=0.001,
                                       use_adagrad=True)
      hash_table = hash_table.assign_add(id_tensor[:-5],
                                         tf.ones([len, dim_size]))
      hash_table = hash_table.assign_add(id_tensor[-5:],
                                         tf.zeros([len, dim_size]))
      start = time.time()
      embeddings = hash_table.lookup(id_tensor)
      loss = -embeddings
      grads = tf.gradients(loss, embeddings)
      hash_table = hash_table.apply_gradients(zip(grads, [embeddings]),
                                              use_multi_threads=True,
                                              enable_dedup=True)
      embeddings = hash_table.lookup(id_tensor)
      embeddings = sess.run(embeddings)
      total_wall_time = time.time() - start
      print('wall time: {}'.format(total_wall_time))


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
