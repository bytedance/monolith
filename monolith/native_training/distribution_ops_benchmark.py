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
import shutil
import time
import tensorflow as tf

from monolith.native_training import distribution_ops


class DistributionOpsBenchmarkTest(tf.test.TestCase):

  def map_id_to_embedding(self, use_multi_threads):
    log_dir = "/tmp/distribution_ops_benchmark/map_id_to_embedding{}".format(
        "_multi_threads" if use_multi_threads else "")
    if os.path.exists(log_dir):
      shutil.rmtree(log_dir)
    options = tf.profiler.experimental.ProfilerOptions(host_tracer_level=3,
                                                       python_tracer_level=0,
                                                       device_tracer_level=0)
    tf.profiler.experimental.start(log_dir, options=options)

    num_elements, dim, ps_num = 1000000, 16, 10
    ids = tf.constant([x for x in range(num_elements)], dtype=tf.int64)
    embeddings = tf.constant(
        [[x for x in range(dim)] for _ in range(num_elements)],
        dtype=tf.float32)

    indices = tf.math.floormod(ids, ps_num)
    split_ids = distribution_ops.split_by_indices(indices, ids, ps_num)
    split_embeddings = distribution_ops.split_by_indices(
        indices, embeddings, ps_num)
    embeddings_mapped = distribution_ops.map_id_to_embedding(
        split_ids, split_embeddings, ids, use_multi_threads=use_multi_threads)

    self.assertAllEqual(embeddings, embeddings_mapped)
    tf.profiler.experimental.stop()

  def test_gather_embeddings_by_ids_basic(self):
    num_features = 100000
    with tf.compat.v1.Session() as sess:
      embeddings = tf.ones([num_features, 32])
      id_tensor = tf.constant([x for x in range(num_features)], dtype=tf.int64)
      input = tf.constant([[y % num_features
                            for y in range(x, x + 4)]
                           for x in range(num_features)],
                          dtype=tf.int64)
      output = distribution_ops.gather_embeddings_by_input(
          id_tensor, embeddings, input)
      start = time.time()
      output = sess.run(output)
      total_wall_time = time.time() - start
      print('wall time: {}'.format(total_wall_time))
    with tf.compat.v1.Session() as sess:
      embeddings = tf.ones([num_features, 256])
      id_tensor = tf.constant([x for x in range(num_features)], dtype=tf.int64)
      input = tf.constant([[y % num_features
                            for y in range(x, x + 2)]
                           for x in range(num_features)],
                          dtype=tf.int64)
      output = distribution_ops.gather_embeddings_by_input(
          id_tensor, embeddings, input)
      start = time.time()
      output = sess.run(output)
      total_wall_time = time.time() - start
      print('wall time: {}'.format(total_wall_time))

  def test_gather_embeddings_by_ids_multi_threads(self):
    num_features = 100000
    with tf.compat.v1.Session() as sess:
      embeddings = tf.ones([num_features, 32])
      id_tensor = tf.constant([x for x in range(num_features)], dtype=tf.int64)
      input = tf.constant([[y % num_features
                            for y in range(x, x + 4)]
                           for x in range(num_features)],
                          dtype=tf.int64)
      output = distribution_ops.gather_embeddings_by_input(
          id_tensor, embeddings, input, use_multi_threads=True)
      start = time.time()
      output = sess.run(output)
      total_wall_time = time.time() - start
      print('wall time: {}'.format(total_wall_time))
    with tf.compat.v1.Session() as sess:
      embeddings = tf.ones([num_features, 256])
      id_tensor = tf.constant([x for x in range(num_features)], dtype=tf.int64)
      input = tf.constant([[y % num_features
                            for y in range(x, x + 2)]
                           for x in range(num_features)],
                          dtype=tf.int64)
      output = distribution_ops.gather_embeddings_by_input(
          id_tensor, embeddings, input, use_multi_threads=True)
      start = time.time()
      output = sess.run(output)
      total_wall_time = time.time() - start
      print('wall time: {}'.format(total_wall_time))

  def test_map_id_to_embedding(self):
    self.map_id_to_embedding(False)

  def test_map_id_to_embedding_multi_threads(self):
    self.map_id_to_embedding(True)


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
