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

import abc

import tensorflow as tf

from monolith.native_training import device_utils
from monolith.native_training import distribution_ops
from monolith.native_training import ragged_utils


class Combiner(abc.ABC):

  def __init__(self, max_seq_length: int):
    self._max_seq_length = max_seq_length

  @property
  def max_seq_length(self):
    return self._max_seq_length

  @abc.abstractmethod
  def combine(self,
              key: tf.RaggedTensor,
              embedding: tf.Tensor,
              name: str = None):
    pass


class ReduceSum(Combiner):

  def __init__(self):
    super().__init__(0)

  def combine(self,
              key: tf.RaggedTensor,
              embedding: tf.Tensor,
              name: str = None):
    return distribution_ops.reduce_sum(tf.expand_dims(
        ragged_utils.fused_value_rowids(key), -1),
                                       embedding,
                                       tf.expand_dims(key.nrows(), 0),
                                       name=name)


class ReduceMean(Combiner):

  def __init__(self):
    super().__init__(0)

  def combine(self,
              key: tf.RaggedTensor,
              embedding: tf.Tensor,
              name: str = None):
    return distribution_ops.reduce_mean(tf.expand_dims(
        ragged_utils.fused_value_rowids(key), -1),
                                        embedding,
                                        tf.expand_dims(key.nrows(), 0),
                                        name=name)


class FirstN(Combiner):

  def __init__(self, seq_length: int):
    assert seq_length > 0, "seq_length must be greater than 0"
    super().__init__(seq_length)

  def combine(self,
              key: tf.RaggedTensor,
              embedding: tf.Tensor,
              name: str = None):
    """For rows with smaller number of embeddings than seq_length,
    automatically append embedding elements which are all zero (default to scatter_nd).
    Tensor's shape is (batch, seq_length, dim) """
    name = name or "FirstNCombiner"
    with tf.name_scope(name):
      if not isinstance(embedding, tf.Tensor):
        embedding = tf.convert_to_tensor(embedding)
      batch_size_tensor = key.nrows()
      key_sparse = key.to_sparse()
      indices = key_sparse.indices

      shape = tf.stack([
          batch_size_tensor,
          tf.math.reduce_max([self.max_seq_length, key_sparse.dense_shape[1]]),
          embedding.shape.as_list()[1]
      ])
      with device_utils.maybe_device_if_allowed('/device:GPU:0'):
        scattered = tf.scatter_nd(indices, embedding, shape)
        # We use slice here instead of array composition because of the shape problem.
        return tf.slice(scattered, [0, 0, 0], [-1, self.max_seq_length, -1])
