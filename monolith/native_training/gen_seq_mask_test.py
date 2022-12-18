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

import tensorflow as tf

from monolith.native_training.gen_seq_mask import gen_seq_mask


class GenSeqMaskTest(tf.test.TestCase):

  def test_gen_seq_mask_int32(self):
    split = tf.constant([0, 5, 7, 9, 13], dtype=tf.int32)
    mask = gen_seq_mask(split, 6)
    result = tf.constant([[1, 1, 1, 1, 1, 0], [1, 1, 0, 0, 0, 0],
                          [1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0]],
                         dtype=tf.int32)
    self.assertAllEqual(mask, result)

  def test_gen_seq_mask_int64(self):
    split = tf.constant([0, 5, 7, 9, 13], dtype=tf.int64)
    mask = gen_seq_mask(split, 6)
    result = tf.constant([[1, 1, 1, 1, 1, 0], [1, 1, 0, 0, 0, 0],
                          [1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0]],
                         dtype=tf.int64)
    self.assertAllEqual(mask, result)


if __name__ == "__main__":
  tf.test.main()
