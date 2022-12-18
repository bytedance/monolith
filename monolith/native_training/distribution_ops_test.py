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

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import test_util
from monolith.native_training import distribution_ops
import random


class DistributionOpsTest(tf.test.TestCase):

  def test_split_by_indices(self):
    with tf.compat.v1.Session() as sess:
      ids = tf.constant([0, 1, 2, 2, 3], dtype=tf.int64)
      indices = tf.math.floormod(ids, 3)
      splits = distribution_ops.split_by_indices(indices, ids, num_splits=3)
      splits = sess.run(splits)

    expected_splits = [[0, 3], [1], [2, 2]]
    for split, expected_split in zip(splits, expected_splits):
      self.assertAllEqual(split, expected_split)

  def test_reorder_by_indices(self):
    with tf.compat.v1.Session() as sess:
      ids = tf.constant([0, 1, 2, 2, 3, 5], dtype=tf.int64)
      indices = tf.cast(tf.math.floormod(ids, 3), dtype=tf.int32)
      reorder_op = distribution_ops.reorder_by_indices(ids,
                                                       indices,
                                                       num_of_shards=3)
      output, split_sizes = sess.run(reorder_op)

    expected_output = [3, 0, 1, 5, 2]
    expected_split_sizes = [2, 1, 2]
    self.assertAllEqual(output, expected_output)
    self.assertAllEqual(split_sizes, expected_split_sizes)

  def test_split_by_indices_gradient(self):
    with self.session() as sess:
      indices = tf.constant([0, 1, 0], dtype=tf.int64)
      tensor = tf.constant([[0, 0], [1, 1], [2, 2]], dtype=tf.float32)
      splits = distribution_ops.split_by_indices(indices, tensor, num_splits=3)
      grad = tf.gradients(splits, tensor)[0]
      grad = sess.run(grad)
    self.assertAllEqual(grad, [[1, 1], [1, 1], [1, 1]])

  def test_split_by_indices_empty_gradient(self):
    with self.session() as sess:
      indices = tf.constant([], dtype=tf.int64)
      tensor = tf.constant([], dtype=tf.float32)
      splits = distribution_ops.split_by_indices(indices, tensor, num_splits=3)
      grad, = tf.gradients(splits, tensor)
      grad = sess.run(grad)
    self.assertAllEqual(grad, [])

  def test_ragged_split_by_indices(self):
    with self.session() as sess:
      indices = tf.constant([0, 1, 0, 1], dtype=tf.int64)
      num = tf.ragged.constant([[], [], [4, 3, 2], [1], [], []], dtype=tf.int64)
      splits, pos = distribution_ops.ragged_split_by_indices(indices,
                                                             num,
                                                             num_splits=2)
      splits, pos = sess.run([splits, pos])
    expected_splits = (
        [[], [], [4, 2], [], [], []],
        [[], [], [3], [1], [], []],
    )
    for split, expected_split in zip(splits, expected_splits):
      self.assertAllEqual(split, expected_split)

    expected_pos = (
        [[], [], [0, 2], [], [], []],
        [[], [], [1], [3], [], []],
    )
    for p1, p2 in zip(pos, expected_pos):
      self.assertAllEqual(p1, p2)

  def test_unique_key_with_value_and_offset_and_fill_with_offset_map(self):
    key = tf.ragged.constant([[], [0, 1, 2, 1, 0], [0, 1, 0], []],
                             dtype=tf.int64)
    dims = [1, 2, 3, 4]
    result = distribution_ops.unique_key_with_value_and_offset(key, dims)
    self.assertAllEqual(result.unique_key, [[], [0, 1, 2], [0, 1], []])
    self.assertAllEqual(result.value_offset,
                        [[], [[0, 8], [2, 6], [4]], [[10, 16], [13]], []])
    value = tf.range(12, dtype=tf.float32)
    filled_tensor = distribution_ops.fill_with_offset_map(
        tf.ragged.constant([[], [0, 1, 2], [3, 4], []], dtype=tf.int64), value,
        result.value_offset, result.value_buffer, dims)

    buffer = distribution_ops.finalize_shared_tensor([filled_tensor],
                                                     dtype=tf.float32,
                                                     shape=[None])
    self.assertAllEqual(
        buffer, [0, 1, 2, 3, 4, 5, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 6, 7, 8])
    grad, = tf.gradients([buffer], [value], [tf.range(19, dtype=tf.float32)])
    self.assertAllEqual(grad, [8, 10, 8, 10, 4, 5, 26, 28, 30, 13, 14, 15])

  def test_fill_with_offset_map_error_case(self):
    key = tf.ragged.constant([[], [0, 1, 2, 1, 0], [0, 1, 0], []],
                             dtype=tf.int64)
    dims = [1, 2, 3, 4]
    result = distribution_ops.unique_key_with_value_and_offset(key, dims)
    value = tf.range(10, dtype=tf.float32)  # expected size: 12
    filled_tensor = distribution_ops.fill_with_offset_map(
        tf.ragged.constant([[], [0, 1, 2], [3, 4], []], dtype=tf.int64), value,
        result.value_offset, result.value_buffer, dims)
    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(filled_tensor)

  def test_unique_key_with_value_and_offset_empty(self):
    key = tf.ragged.constant([[], [], []], dtype=tf.int64)
    result = distribution_ops.unique_key_with_value_and_offset(key, [1, 2, 3])
    self.assertAllEqual(result.unique_key, [[], [], []])
    self.assertAllEqual(result.value_offset, [[], [], []])

  def test_map_id_to_embedding(self):
    with tf.compat.v1.Session() as sess:
      ids1 = tf.constant([1], dtype=tf.int64)
      embeddings1 = tf.constant([[1, 1]], dtype=tf.float32)
      ids2 = tf.constant([2], dtype=tf.int64)
      embeddings2 = tf.constant([[2, 2]], dtype=tf.float32)
      input = tf.constant([[1], [2]], dtype=tf.int64)
      output = distribution_ops.map_id_to_embedding([ids1, ids2],
                                                    [embeddings1, embeddings2],
                                                    input,
                                                    use_multi_threads=False)
      output = sess.run(output)
    self.assertAllEqual(output, [[[1, 1]], [[2, 2]]])

  def test_map_id_to_embedding_multi_threads(self):
    with tf.compat.v1.Session() as sess:
      num_elements, dim, ps_num = 1000, 16, 10
      ids = tf.constant([x for x in range(num_elements)], dtype=tf.int64)
      embeddings = tf.constant(
          [[x for x in range(dim)] for _ in range(num_elements)],
          dtype=tf.float32)

      indices = tf.math.floormod(ids, ps_num)
      split_ids = distribution_ops.split_by_indices(indices, ids, ps_num)
      split_embeddings = distribution_ops.split_by_indices(
          indices, embeddings, ps_num)
      embeddings_mapped = distribution_ops.map_id_to_embedding(
          split_ids, split_embeddings, ids, use_multi_threads=True)

      embeddings = sess.run(embeddings)
      embeddings_mapped = sess.run(embeddings_mapped)

    self.assertAllEqual(embeddings, embeddings_mapped)

  def test_map_id_to_embedding_gradient(self):
    with self.session() as sess:
      ids1 = tf.constant([1], dtype=tf.int64)
      embeddings1 = tf.constant([[0, 0]], dtype=tf.float32)
      ids2 = tf.constant([2], dtype=tf.int64)
      embeddings2 = tf.constant([[0, 0]], dtype=tf.float32)
      input = tf.constant([1, 1, 2], dtype=tf.int64)
      output = distribution_ops.map_id_to_embedding([ids1, ids2],
                                                    [embeddings1, embeddings2],
                                                    input,
                                                    use_multi_threads=False)
      target_output = tf.constant([[2, 2], [2, 2], [2, 2]], dtype=tf.float32)
      loss = target_output - output
      grads = tf.gradients(loss, [embeddings1, embeddings2])
      grads = sess.run(grads)

    expected_grads = [[[-2, -2]], [[-1, -1]]]
    for grads_part, expexted_grads_part in zip(grads, expected_grads):
      self.assertAllEqual(grads_part, expexted_grads_part)

  def test_gather_embeddings_by_ids(self):
    with tf.compat.v1.Session() as sess:
      ids = tf.constant([1, 2, 3], dtype=tf.int64)
      embeddings = tf.constant([[1, 1], [2, 2], [3, 3]], dtype=tf.float32)
      input = tf.constant([[2], [1], [2]], dtype=tf.int64)
      output = distribution_ops.gather_embeddings_by_input(
          ids, embeddings, input)
      output, index_mapping = sess.run(output)
    self.assertAllEqual(output, [[[2, 2]], [[1, 1]], [[2, 2]]])
    self.assertAllEqual(index_mapping, [[1], [0], [1]])

  def test_gather_embeddings_by_ids_gradient(self):
    with self.session() as sess:
      ids = tf.constant([1, 2, 3], dtype=tf.int64)
      embeddings = tf.constant([[1, 1], [2, 2], [3, 3]], dtype=tf.float32)
      input = tf.constant([[1], [2], [1]], dtype=tf.int64)
      output, index_mapping = distribution_ops.gather_embeddings_by_input(
          ids, embeddings, input)

      target_output = tf.constant([[[2, 2]], [[2, 2]], [[2, 2]]],
                                  dtype=tf.float32)
      loss = target_output - output
      grads = tf.gradients(loss, embeddings)
      grads = sess.run(grads)

    expected_grads = [[-2, -2], [-1, -1], [0, 0]]
    self.assertAllEqual(grads[0], expected_grads)

  def test_gather_embeddings_by_ids_gradient_back_prop(self):
    with self.session() as sess:
      ids = tf.constant([2, 3, 1], dtype=tf.int64)
      grads = tf.constant([[1, 1], [2, 2], [4, 4], [8, 8]], dtype=tf.float32)
      # implies the input tensor with id value [3, 2, 3, 1]
      index_mapping = tf.constant([1, 0, 1, 2], dtype=tf.int64)
      emb_grads = distribution_ops.gather_embeddings_by_ids_gradient_back_prop(
          ids, grads, index_mapping)
    self.assertAllEqual(emb_grads, [[2, 2], [5, 5], [8, 8]])

  @test_util.run_gpu_only
  def test_fused_gather_embeddings_by_input(self):
    with tf.compat.v1.Session() as sess, test_util.use_gpu():
      # inputs = [
      #   tf.constant([2, 3, 1, 2, 7, 2], dtype=tf.int64),
      #   tf.constant([5, 8, 4, 4, 5, 11, 6], dtype=tf.int64)
      # ]
      # shard_indices: [[2, 0, 1, 2, 1, 2], [2, 2, 1, 1, 2, 2, 0]]
      # fused_ids: [3, 6, 1, 7, 4, 2, 5, 8, 11]
      # fused_slot_sizes: [1, 1, 2, 1, 1, 3]
      embedding_dims = [3, 2]
      fused_embeddings = tf.constant([
          1.1, 1.2, 1.3, 2.1, 2.2, 3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 5.1, 5.2, 6.1,
          6.2, 6.3, 7.1, 7.2, 8.1, 8.2, 9.1, 9.2
      ],
                                     dtype=tf.float32)
      SCALE = (12345, 11777
              )  # To test the number of elements larger than GPU grid
      fused_embedding_offsets = [
          tf.constant([13, 0, 5, 13, 8, 13] * SCALE[0], dtype=tf.int32),
          tf.constant([16, 18, 11, 11, 16, 20, 3] * SCALE[1], dtype=tf.int32)
      ]
      output = distribution_ops.fused_gather_embeddings_by_input(
          fused_embeddings, fused_embedding_offsets, embedding_dims)
      outputs = sess.run(output)

    expected_outputs = [[[6.1, 6.2, 6.3], [1.1, 1.2, 1.3], [3.1, 3.2, 3.3],
                         [6.1, 6.2, 6.3], [4.1, 4.2, 4.3], [6.1, 6.2, 6.3]] *
                        SCALE[0],
                        [[7.1, 7.2], [8.1, 8.2], [5.1, 5.2], [5.1, 5.2],
                         [7.1, 7.2], [9.1, 9.2], [2.1, 2.2]] * SCALE[1]]
    self.assertAllClose(outputs, expected_outputs)

  def test_fused_gather_embeddings_by_input_gradient(self):
    with tf.compat.v1.Session() as sess, test_util.use_gpu():
      # The size of one-dimensional fused_embeddings.
      with tf.device("CPU:0"):
        fused_embeddings_size = tf.constant(22, dtype=tf.int32)
      embedding_dims = [3, 2]
      SCALE = 888  # To test float sum precision loss on CPU and GPU
      grads = [
          tf.constant([[1.1, 1.2, 1.3], [2.1, 2.2, 2.3], [3.1, 3.2, 3.3],
                       [4.1, 4.2, 4.3], [5.1, 5.2, 5.3], [6.1, 6.2, 6.3]] *
                      SCALE,
                      dtype=tf.float32),
          tf.constant([[1.4, 1.5], [2.4, 2.5], [3.4, 3.5], [4.4, 4.5],
                       [5.4, 5.5], [6.4, 6.5], [7.4, 7.5]] * SCALE,
                      dtype=tf.float32)
      ]
      embedding_offsets = [
          tf.constant([13, 0, 5, 13, 8, 13] * SCALE, dtype=tf.int32),
          tf.constant([16, 18, 11, 11, 16, 20, 3] * SCALE, dtype=tf.int32)
      ]
      output_t = distribution_ops.fused_gather_embeddings_by_input_gradient(
          fused_embeddings_size, grads, embedding_offsets, embedding_dims)
      self.assertAllEqual(output_t.shape[0],
                          22)  # shape inference when applicable
      output = sess.run(output_t)
    expected_output = [
        2.1,
        2.2,
        2.3,  # id 3   offset 0
        7.4,
        7.5,  # id 6   offset 3
        3.1,
        3.2,
        3.3,  # id 1   offset 5
        5.1,
        5.2,
        5.3,  # id 7   offset 8
        7.8,
        8.0,  # id 4   offset 11
        11.3,
        11.6,
        11.9,  # id 2   offset 13
        6.8,
        7.0,  # id 5   offset 16
        2.4,
        2.5,  # id 8   offset 18
        6.4,
        6.5,  # id 11  offset 20
    ]
    self.assertAllClose(output,
                        np.asarray(expected_output) * SCALE,
                        rtol=1e-7 * SCALE)

  def test_reduce_mean(self):
    with tf.compat.v1.Session() as sess:
      id_indices = tf.constant([[0], [0], [1]], dtype=tf.int64)
      id_values = tf.constant([[4, 4], [2, 2], [1, 1]], dtype=tf.float32)
      reduced = distribution_ops.reduce_mean(id_indices, id_values, [2])
      reduced = sess.run(reduced)
    self.assertAllEqual(reduced, [[3, 3], [1, 1]])

  def test_reduce_mean_gradient(self):
    with self.session() as sess:
      id_indices = tf.constant([[0], [0]], dtype=tf.int64)
      id_values = tf.constant([[0, 0], [0, 0]], dtype=tf.float32)
      reduced = distribution_ops.reduce_mean(id_indices, id_values, [1])
      target = tf.constant([[-2, -4]], dtype=tf.float32)
      loss = target - 2 * reduced
      grads = tf.gradients(loss, id_values)[0]
      grads = sess.run(grads)
    self.assertAllEqual(grads, [[-1, -1], [-1, -1]])

  def test_reduce_sum(self):
    with tf.compat.v1.Session() as sess:
      id_indices = tf.constant([[0], [0], [1]], dtype=tf.int64)
      id_values = tf.constant([[1, 1], [2, 2], [4, 4]], dtype=tf.float32)
      reduced = distribution_ops.reduce_sum(id_indices, id_values, [2])
      reduced = sess.run(reduced)
    self.assertAllEqual(reduced, [[3, 3], [4, 4]])

  def test_reduce_sum_gradient(self):
    with self.session() as sess:
      id_indices = tf.constant([[0], [0]], dtype=tf.int64)
      id_values = tf.constant([[0, 0], [0, 0]], dtype=tf.float32)
      reduced = distribution_ops.reduce_sum(id_indices, id_values, [1])
      target = tf.constant([[10, 99]], dtype=tf.float32)
      loss = target - reduced
      grads = tf.gradients(loss, id_values)[0]
      grads = sess.run(grads)
    self.assertAllEqual(grads, [[-1, -1], [-1, -1]])

  def test_reduce_sqrtn(self):
    with tf.compat.v1.Session() as sess:
      id_indices = tf.constant([[0], [0], [1]], dtype=tf.int64)
      id_values = tf.constant([[3, 3], [4, 4], [4, 4]], dtype=tf.float32)
      reduced = distribution_ops.reduce_sqrtn(id_indices, id_values, [2])
      reduced = sess.run(reduced)
    self.assertAllClose(reduced, [[5, 5], [4, 4]])

  def test_reduce_sqrtn_gradient(self):
    with self.session() as sess:
      id_indices = tf.constant([[0], [0]], dtype=tf.int64)
      id_values = tf.constant([[3, 4], [4, 3]], dtype=tf.float32)
      reduced = distribution_ops.reduce_sqrtn(id_indices, id_values, [1])
      target = tf.constant([[10, 15]], dtype=tf.float32)
      loss = target - reduced
      grads = tf.gradients(loss, id_values)[0]
      grads = sess.run(grads)
    self.assertAllClose(grads, [[-0.6, -0.8], [-0.8, -0.6]])

  def test_reduce_sqrtn_gradient_zero(self):
    with self.session() as sess:
      id_indices = tf.constant([[0], [0]], dtype=tf.int64)
      id_values = tf.constant([[0, 0], [0, 0]], dtype=tf.float32)
      reduced = distribution_ops.reduce_sqrtn(id_indices, id_values, [1])
      target = tf.constant([[10, 15]], dtype=tf.float32)
      loss = target - reduced
      grads = tf.gradients(loss, id_values)[0]
      grads = sess.run(grads)
    self.assertAllClose(grads, [[0, 0], [0, 0]])

  def test_fused_reduce_sum_and_split(self):
    # Test split.
    with tf.compat.v1.Session() as sess, sess.graph.device(lambda op: '/CPU:0'):
      id_indices = tf.constant([0, 0, 1], dtype=tf.int64)
      id_values = tf.constant([[1, 1, 1], [2, 2, 1], [4, 4, 2]],
                              dtype=tf.float32)
      reduced = distribution_ops.fused_reduce_sum_and_split(
          id_indices, id_values, 2, [2, 1])
      reduced = sess.run(reduced)
    self.assertAllEqual(reduced[0], [[3, 3], [4, 4]])
    self.assertAllEqual(reduced[1], [[2], [2]])
    # Test a different split type.
    with tf.compat.v1.Session() as sess, sess.graph.device(lambda op: '/CPU:0'):
      id_indices = tf.constant([0, 0, 1], dtype=tf.int64)
      id_values = tf.constant([[1, 1, 1], [2, 2, 1], [4, 4, 2]],
                              dtype=tf.float32)
      reduced = distribution_ops.fused_reduce_sum_and_split(
          id_indices, id_values, 2, [1, 2])
      reduced = sess.run(reduced)
    self.assertAllEqual(reduced[0], [[3], [4]])
    self.assertAllEqual(reduced[1], [[3, 2], [4, 2]])
    # Test non-consecutive indicies
    with tf.compat.v1.Session() as sess, sess.graph.device(lambda op: '/CPU:0'):
      id_indices = tf.constant([0, 0, 2], dtype=tf.int64)
      id_values = tf.constant([[1, 1, 1], [2, 2, 1], [4, 4, 2]],
                              dtype=tf.float32)
      reduced = distribution_ops.fused_reduce_sum_and_split(
          id_indices, id_values, 4, [1, 2])
      reduced = sess.run(reduced)
    self.assertAllEqual(reduced[0], [[3], [0], [4], [0]])
    self.assertAllEqual(reduced[1], [[3, 2], [0, 0], [4, 2], [0, 0]])

  def test_fused_reduce_sum_and_split_grad(self):
    # Test split.
    with tf.compat.v1.Session() as sess, sess.graph.device(lambda op: '/CPU:0'):
      id_indices = tf.constant([0, 0, 1], dtype=tf.int64)
      id_values = tf.constant([[1, 1, 1], [2, 2, 1], [4, 4, 2]],
                              dtype=tf.float32)
      reduced_result = distribution_ops.fused_reduce_sum_and_split(
          id_indices, id_values, 2, [2, 1])
      grads = tf.gradients(reduced_result, id_values)[0]
      grads = sess.run(grads)
    self.assertAllEqual(grads, [[1, 1, 1], [1, 1, 1], [1, 1, 1]])

  @test_util.run_gpu_only
  def test_fused_reduce_scatter(self):
    with tf.compat.v1.Session() as sess, test_util.use_gpu():
      id_indices = [
          tf.constant([0, 0, 1], dtype=tf.int32),
          tf.constant([0, 0, 1], dtype=tf.int32),
          tf.constant([], dtype=tf.int32, shape=[0]),
          tf.constant([0, 0, 2, 2], dtype=tf.int32),
      ]
      id_values = [
          tf.constant([[1, 1, 1], [2, 2, 1], [4, 4, 2]], dtype=tf.float32),
          tf.constant([[1, 1, 1], [2, 2, 1], [4, 4, 2]], dtype=tf.float32),
          tf.constant([], dtype=tf.float32, shape=[0, 3]),
          tf.constant([[1, 1, 1, 1, 1], [2, 2, 1, 1, 1], [4, 4, 2, 2, 2],
                       [4, 4, 2, 2, 2]],
                      dtype=tf.float32)
      ]
      shapes = [(2, 3), (4, 3), (2, 3), (4, 5)]
      reduced_tensors = distribution_ops.fused_sorted_segment_sum(
          id_indices, id_values, shapes)
      truth_tensors = [
          tf.scatter_nd(tf.expand_dims(i, -1), v, s)
          for i, v, s in zip(id_indices, id_values, shapes)
      ]

      reduced = sess.run(reduced_tensors)
      truth = sess.run(truth_tensors)
      expected = [[[3, 3, 2], [4, 4, 2]],
                  [[3, 3, 2], [4, 4, 2], [0, 0, 0], [0, 0, 0]],
                  [[0, 0, 0], [0, 0, 0]],
                  [[3, 3, 2, 2, 2], [0, 0, 0, 0, 0], [8, 8, 4, 4, 4],
                   [0, 0, 0, 0, 0]]]
      for r, e, t in zip(reduced, expected, truth):
        self.assertAllClose(r, e)
        self.assertAllClose(e, t)
      # Gradient Check
      gs_expected = sess.run(tf.gradients(truth_tensors, id_values))
      gs = sess.run(tf.gradients(reduced_tensors, id_values))
      self.assertAllClose(gs, gs_expected)

  @test_util.run_gpu_only
  def test_fused_reduce_and_split_gpu(self):
    num_rows = 102
    batch_size = 256
    emb_lens = [i * 2 - 1 for i in range(1, num_rows + 1)]
    slice_dims = []
    for l in emb_lens:
      if l < 4:
        slices = [1 for i in range(l)]
      else:
        slices = [l // 4] * 4
        slices[-1] += l % 4
      slice_dims.append(slices)

    row_lens = [i for i in range(0, batch_size)]
    random.shuffle(row_lens)
    rows_before_reduction = sum(row_lens)
    shapes = [
        tf.convert_to_tensor([batch_size, emb_lens[i]], dtype=tf.int64)
        for i in range(num_rows)
    ]
    ragged_tensors = [tf.ragged.range(row_lens) for j in range(num_rows)]
    value_rowids = [t.value_rowids() for t in ragged_tensors]
    splits = [t.row_splits for t in ragged_tensors]

    with tf.compat.v1.Session() as sess, test_util.use_gpu():
      embeddings = [
          tf.ones((rows_before_reduction, emb_lens[i])) for i in range(num_rows)
      ]

      outputs = distribution_ops.fused_reduce_and_split_gpu(
          splits, embeddings, slice_dims)
      outputs2 = []
      for i in range(num_rows):
        temp1 = tf.scatter_nd(tf.expand_dims(value_rowids[i], -1),
                              embeddings[i], shapes[i])
        outputs2.extend(tf.split(temp1, slice_dims[i], axis=1))
      self.assertEqual(len(outputs), len(outputs2))
      for i in range(len(outputs)):
        rand = tf.random.uniform(outputs[i].shape)
        outputs[i] *= rand
        outputs2[i] *= rand

      grads = tf.gradients(outputs, embeddings)
      grads2 = tf.gradients(outputs2, embeddings)

      val_flags = []
      for i in range(len(outputs)):
        val_flags.append(tf.reduce_all(tf.equal(outputs[i], outputs2[i])))
      val_flag = tf.reduce_all(val_flags)

      self.assertEqual(len(grads), len(grads2))
      grad_flags = []
      for i in range(len(grads)):
        grad_flags.append(tf.reduce_all(tf.equal(grads[i], grads2[i])))
      grad_flag = tf.reduce_all(grad_flags)

      f1, f2 = sess.run([val_flag, grad_flag])
      self.assertTrue(f1)
      self.assertTrue(f2)


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
