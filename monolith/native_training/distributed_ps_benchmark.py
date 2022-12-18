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

import time
import os
import shutil
from absl import flags
import tensorflow as tf
from tensorflow.core.protobuf import cluster_pb2, config_pb2

from monolith.native_training import (distributed_ps, hash_filter_ops, \
                                      hash_table_ops, utils)
from monolith.native_training.runtime.hash_table import \
    embedding_hash_table_pb2

PROFILE = False


def _generate_config(servers, job_name=utils.PS_JOB_NAME):
  """Generates a config based on servers"""
  cluster_def = cluster_pb2.ClusterDef()
  job = cluster_def.job.add()
  job.name = job_name
  for i, server in enumerate(servers):
    job.tasks[i] = server.target[len('grpc://'):]
  return config_pb2.ConfigProto(cluster_def=cluster_def)


def _get_vocab_hash_table_factory(dim: int):

  def factory(name_suffix: str, hash_filter: tf.Tensor):
    config = embedding_hash_table_pb2.EmbeddingHashTableConfig()
    config.cuckoo.SetInParent()
    segment = config.entry_config.segments.add()
    segment.dim_size = dim
    segment.opt_config.sgd.learning_rate = 1.0
    segment.init_config.zeros.SetInParent()
    return hash_table_ops.hash_table_from_config(config=config,
                                                 hash_filter=hash_filter,
                                                 name_suffix=name_suffix)

  return factory


class DistributedHashTableTest(tf.test.TestCase):

  def lookup(self, enable_dedup, real_run=True):
    ps_num = 10
    servers = [
        tf.distribute.Server.create_local_server() for _ in range(ps_num)
    ]
    server0 = servers[0]
    num_elements, dim = 1000000, 16
    config = _generate_config(servers)
    if PROFILE and real_run:
      log_dir = "/tmp/distributed_ps_benchmark/lookup{}".format(
          "_dedup" if enable_dedup else "")
      if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
      tf.profiler.experimental.start(log_dir)
    with tf.compat.v1.Session(server0.target, config=config) as sess:
      hash_filters = hash_filter_ops.create_hash_filters(ps_num, False)
      hash_table = distributed_ps.DistributedHashTable(
          ps_num, hash_filters, _get_vocab_hash_table_factory(dim))
      hash_table = hash_table.assign_add(
          tf.constant([x for x in range(num_elements)], dtype=tf.int64),
          tf.constant([[x for _ in range(dim)] for x in range(num_elements)],
                      dtype=tf.float32))
      start = time.time()
      if real_run:
        values = hash_table.lookup(tf.constant(
            [x // 2 for x in range(num_elements)], dtype=tf.int64),
                                   use_multi_threads=True,
                                   enable_dedup=enable_dedup)
        values = sess.run(values)
        print("wall time(MT) enable_dedup={}: cost {}".format(
            str(enable_dedup),
            time.time() - start))
        self.assertAllEqual(
            values, [[x // 2 for _ in range(dim)] for x in range(num_elements)])
      else:
        sess.run(hash_table.as_op())
        print("wall time(overhead): cost {}".format(time.time() - start))
    if PROFILE and real_run:
      tf.profiler.experimental.stop()

  def apply_gradients(self, real_run=True):
    ps_num = 10
    servers = [
        tf.distribute.Server.create_local_server() for _ in range(ps_num)
    ]
    server0 = servers[0]
    num_elements, dim = 1000000, 16
    config = _generate_config(servers)
    if PROFILE and real_run:
      log_dir = "/tmp/distributed_ps_benchmark/apply_gradients"
      if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
      tf.profiler.experimental.start(log_dir)
    with tf.compat.v1.Session(server0.target, config=config) as sess:
      hash_filters = hash_filter_ops.create_hash_filters(ps_num, False)
      hash_table = distributed_ps.DistributedHashTable(
          ps_num, hash_filters, _get_vocab_hash_table_factory(dim))
      hash_table = hash_table.assign_add(
          tf.constant([x for x in range(num_elements)], dtype=tf.int64),
          tf.constant([[1 for _ in range(dim)] for _ in range(num_elements)],
                      dtype=tf.float32))
      embeddings = hash_table.lookup(tf.constant(
          [x // 2 for x in range(num_elements)], dtype=tf.int64),
                                     use_multi_threads=True,
                                     enable_dedup=True)
      loss = tf.multiply(0.3, embeddings)
      grads = tf.gradients(loss, embeddings)
      start = time.time()
      if real_run:
        hash_table = hash_table.apply_gradients(
            tf.constant([x // 2 for x in range(num_elements)], dtype=tf.int64),
            grads[0])
        if PROFILE:
          sess.run(hash_table.as_op())
        else:
          values = hash_table.lookup(tf.constant(
              [x // 2 for x in range(num_elements)], dtype=tf.int64),
                                     use_multi_threads=True,
                                     enable_dedup=True)
          values = sess.run(values)
          self.assertAllClose(
              values, [[0.4 for _ in range(dim)] for x in range(num_elements)])
        print("wall time(MT): cost {}".format(time.time() - start))
      else:
        grads = sess.run(grads[0])
        if not PROFILE:
          self.assertAllClose(
              grads, [[0.3 for _ in range(dim)] for x in range(num_elements)])
        print("wall time(overhead): cost {}".format(time.time() - start))
    if PROFILE and real_run:
      tf.profiler.experimental.stop()

  def test_lookup_overhead(self):
    self.lookup(False, False)

  def test_lookup(self):
    self.lookup(False)

  def test_lookup_dedup(self):
    self.lookup(True)

  def test_apply_gradients_overhead(self):
    self.apply_gradients(False)

  def test_apply_gradients(self):
    self.apply_gradients(True)


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
