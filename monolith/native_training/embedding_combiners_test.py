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

from monolith.native_training import embedding_combiners


class CombinerTest(tf.test.TestCase):

  def testReduceSum(self):
    key = tf.RaggedTensor.from_row_lengths([1, 2, 3], [1, 2])
    emb = [[1.0], [2.0], [3.0]]
    comb = embedding_combiners.ReduceSum()
    result = self.evaluate(comb.combine(key, emb))
    self.assertAllClose(result, [[1.0], [5.0]])

  def testFirstN(self):
    key = tf.RaggedTensor.from_row_lengths([1, 2, 3, 4, 5, 6], [1, 2, 3])
    emb = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
    comb = embedding_combiners.FirstN(2)
    result = self.evaluate(comb.combine(key, emb))
    self.assertAllClose(result,
                        [[[1.0], [0.0]], [[2.0], [3.0]], [[4.0], [5.0]]])

  def testFirstNUnknownShape(self):
    key = tf.compat.v1.ragged.placeholder(tf.int64, 1, [])
    emb = tf.compat.v1.placeholder(tf.float32, shape=[None, 6])
    comb = embedding_combiners.FirstN(2)
    result = comb.combine(key, emb)
    self.assertAllEqual(result.shape, [None, 2, 6])


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
