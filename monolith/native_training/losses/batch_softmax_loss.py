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

import tensorflow as tf


def batch_softmax_loss(query: tf.Tensor,
                       item: tf.Tensor,
                       item_step_interval: tf.Tensor,
                       r: tf.Tensor,
                       normalize: bool = True,
                       temperature: float = 1.0) -> tf.Tensor:
  """
  Batch Softmax Loss

  Args:
    query (:obj:`tf.Tensor`): query 向量, shape=(batch_size, k)
    item (:obj:`tf.Tensor`): item 向量, shape=(batch_size, k)
    item_step_interval (:obj:`tf.Tensor`): item 出现的平均 global step 间隔, shape=(batch_size,)
    r (:obj:`tf.Tensor`): query 对 item 感兴趣程度权重
    normalize (:obj:`bool`): 是否对 query/item 向量归一化
    temperature (:obj:`float`): hyper-parameter tuned to maximize retrieval metrics such as recall or precision
  """

  if temperature <= 0:
    raise ValueError(
        "temperature should be positive, while got {}".format(temperature))
  if normalize:
    query = tf.linalg.l2_normalize(query, axis=1)
    item = tf.linalg.l2_normalize(item, axis=1)

  # (batch_size, batch_size)
  similarity = tf.matmul(query, item, transpose_b=True) / temperature
  # The first looked-up item_step_interval is filled by zeros
  item_step_interval = tf.math.maximum(item_step_interval,
                                       tf.constant([1.0], dtype=tf.float32))
  item_frequency = 1 / item_step_interval
  similarity = tf.math.exp(similarity - tf.math.log(item_frequency))
  loss = -tf.reduce_sum(
      tf.multiply(
          r,
          tf.math.log(
              tf.linalg.tensor_diag_part(similarity) /
              tf.reduce_sum(similarity, axis=1))))

  return loss
