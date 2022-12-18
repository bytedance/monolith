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

from monolith.native_training import distribution_ops


class DistributionOpsTest(tf.test.TestCase):

  def _test_fused_reorder_by_indices(self,
                                     ids_list,
                                     shard_num,
                                     expected_output,
                                     expected_split_sizes,
                                     expected_sharded_slot_sizes,
                                     dim_sizes=None,
                                     expected_embedding_offsets=None):
    if dim_sizes is None:
      # Fake dim_sizes for testing
      dim_sizes = [2 for _ in range(len(ids_list))]
    with tf.compat.v1.Session() as sess:
      ids_list = [tf.convert_to_tensor(ids, dtype=tf.int64) for ids in ids_list]
      reorder_op = distribution_ops.fused_reorder_by_indices(
          ids_list, num_of_shards=shard_num, dim_sizes=dim_sizes)
      print('>>>', reorder_op[3])
      output, split_sizes, sharded_slot_sizes, embedding_offsets = sess.run(
          reorder_op)
      # print(output, split_sizes, sharded_slot_sizes)

      self.assertAllEqual(output, expected_output)
      self.assertAllEqual(split_sizes, expected_split_sizes)
      self.assertAllEqual(sharded_slot_sizes, expected_sharded_slot_sizes)
      # print(embedding_offsets)
    if expected_embedding_offsets:
      for r, e in zip(embedding_offsets, expected_embedding_offsets):
        self.assertAllEqual(r, e)

  def test_fused_reorder_by_indices(self):

    # ids_list, shard_num
    # expected_output, expected_split_sizes, expected_sharded_slot_sizes
    self._test_fused_reorder_by_indices(
        # Fallback to original reorder_by_indices,
        #  but keeping the inner-merged-slot order
        [[0, 1, 2, 2, 3, 5]],
        3,
        [0, 3, 1, 2, 5],
        [2, 1, 2],
        [2, 1, 2])

    self._test_fused_reorder_by_indices(
        # Extra slot
        [[0, 1, 2, 2, 3, 5], []],
        3,
        [0, 3, 1, 2, 5],
        [2, 1, 2],
        [2, 0, 1, 0, 2, 0])

    self._test_fused_reorder_by_indices(
        # plus 2*shard_num
        [[0, 1, 2, 2, 3, 5], [6, 7, 8, 8, 9, 11]],
        3,
        [0, 3, 6, 9, 1, 7, 2, 5, 8, 11],
        [4, 2, 4],
        [2, 2, 1, 1, 2, 2])

    self._test_fused_reorder_by_indices(
        # Empty slots
        [[], []],
        2,
        [],
        [0, 0],
        [0, 0, 0, 0])

    self._test_fused_reorder_by_indices([[0, 1, 4, 5], [2, 3, 6, 7]], 2,
                                        [0, 4, 2, 6, 1, 5, 3, 7], [4, 4],
                                        [2, 2, 2, 2])

    self._test_fused_reorder_by_indices([[0, 1, 0], [3, 2, 3], [5, 6, 7]],
                                        2, [0, 2, 6, 1, 3, 5, 7], [3, 4],
                                        [1, 1, 1, 1, 1, 2],
                                        dim_sizes=[1, 2, 3],
                                        expected_embedding_offsets=[[0, 6, 0],
                                                                    [7, 1, 7],
                                                                    [9, 3, 12]])

    self._test_fused_reorder_by_indices(
        # Imagine the expected fused_embeddings as follows:
        #   [1.1, 1.2, 1.3,     # 3   # slot 0, dim 3, offset 0
        #    2.1, 2.2,          # 6   # slot 1, dim 2, offset 3
        #    3.1, 3.2, 3.3,     # 1   # slot 0, dim 3, offset 5
        #    4.1, 4.2, 4.3,     # 7   # slot 0, dim 3, offset 8
        #    5.1, 5.2,          # 4   # slot 1, dim 2, offset 11
        #    6.1, 6.2, 6.3,     # 2   # slot 0, dim 3, offset 13
        #    7.1, 7.2,          # 5   # slot 1, dim 2, offset 16
        #    8.1, 8.2,          # 8   # slot 1, dim 2, offset 18
        #    9.1, 9.2],         # 11  # slot 1, dim 2, offset 20
        [[2, 3, 1, 2, 7, 2], [5, 8, 4, 4, 5, 11, 6]],
        3,
        [3, 6, 1, 7, 4, 2, 5, 8, 11],
        [2, 3, 4],
        [1, 1, 2, 1, 1, 3],
        dim_sizes=[3, 2],
        expected_embedding_offsets=[[13, 0, 5, 13, 8, 13],
                                    [16, 18, 11, 11, 16, 20, 3]])

  def test_ragged_tensor_workflow(self):
    with tf.Graph().as_default():
      a = tf.RaggedTensor.from_tensor(tf.constant([[0], [1]], dtype=tf.int64))
      b = tf.RaggedTensor.from_tensor(tf.constant([[2], [3]], dtype=tf.int64))
      c = tf.RaggedTensor.from_tensor(tf.constant([[4], [5]], dtype=tf.int64))
      d = tf.RaggedTensor.from_tensor(tf.constant([[6], [7]], dtype=tf.int64))
      # Currently for merged slots A, B
      # the order ['A', 'B'] is based on merged_slot_to_config;
      # the mapping is based on MergedMultiTypeHashTable._slot_mapping: {'a': 'A', 'b': 'B', 'c': 'A', 'd', 'B'}
      merged_slot_values = [
          tf.concat([a.values, c.values], 0),
          tf.concat([b.values, d.values], 0)
      ]
      self._test_fused_reorder_by_indices(merged_slot_values, 2,
                                          [0, 4, 2, 6, 1, 5, 3, 7], [4, 4],
                                          [2, 2, 2, 2])


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
