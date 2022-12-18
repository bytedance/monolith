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

from monolith.native_training import tensor_utils


class TensorUtilsTest(tf.test.TestCase):

  def test_maybe_squeeze_3d_tensor(self):
    x = tf.ragged.constant([[0, 1], [2]])
    y = tf.RaggedTensor.from_uniform_row_length(x, 1)
    sx = tensor_utils.maybe_squeeze_3d_tensor(x)
    sy = tensor_utils.maybe_squeeze_3d_tensor(y)
    with self.session() as sess:
      sx_value, sy_value = sess.run([sx, sy])
      for squeezed in (sx_value, sy_value):
        self.assertAllEqual(squeezed, [[0, 1], [2]])

  def test_pack_tensors(self):
    x = tf.constant([1, 2], dtype=tf.int64)
    y = tf.constant([[4, 5], [6, 7]], dtype=tf.int64)
    d = {"x": x, "y": y}
    packed_d = tensor_utils.pack_tensors(d)
    unpacked_d = tensor_utils.unpack_tensors(tensor_utils.get_keyed_shape(d),
                                             packed_d)
    with self.session() as sess:
      packed_d_value = sess.run(packed_d)
      self.assertAllEqual(packed_d_value[0], [1, 2, 4, 5, 6, 7])
      self.assertAllEqual(packed_d_value[1], [2, 4])
      unpacked_d_value = sess.run(unpacked_d)
      original_d = sess.run(d)
      for key in sorted(unpacked_d_value):
        self.assertAllEqual(unpacked_d_value[key], original_d[key])

  def test_pack_typed_keyed_tensors(self):
    t1 = tf.constant([[0, 0, 1, 0], [0, 2, 0, 3]], dtype=tf.int64)
    t2 = tf.constant([0, 3, 1, 2], dtype=tf.int64)
    t3 = tf.constant([9.1, 2.2], dtype=tf.float32)
    t4 = tf.constant([[1.1, 2.2], [3.3, 4.4]], dtype=tf.float32)
    t5 = tf.constant([3, 4, 5, 6, 7, 8, 9], dtype=tf.float64)
    d1 = {"t1": t1, "t2": t2}
    d2 = {"t4": t4, "t3": t3}
    d3 = {"t5": t5}
    l = [d1, d2, d3]
    packed_l = tensor_utils.pack_typed_keyed_tensors(l)
    unpacked_l = tensor_utils.unpack_packed_tensors(
        tensor_utils.get_typed_keyed_shape(l), packed_l)

    packed_d1 = tensor_utils.pack_tensors(d1)
    packed_d2 = tensor_utils.pack_tensors(d2)
    packed_d3 = tensor_utils.pack_tensors(d3)
    with self.session() as sess:
      packed_l_value = sess.run(packed_l)
      packed_d1_value = sess.run(packed_d1)
      packed_d2_value = sess.run(packed_d2)
      packed_d3_value = sess.run(packed_d3)
      self.assertAllEqual(packed_l_value[0], packed_d1_value[0])
      self.assertAllEqual(packed_l_value[1], packed_d2_value[0])
      self.assertAllEqual(packed_l_value[2], packed_d3_value[0])
      self.assertAllEqual(packed_l_value[3], [2, 2, 1, 8, 4, 2, 4, 7])

      unpacked_l_value = sess.run(unpacked_l)
      l_value = sess.run(l)
      for i, d in enumerate(unpacked_l_value):
        for key in sorted(d):
          self.assertAllEqual(d[key], l_value[i][key])

  def test_pack_typed_keyed_tensors_with_placeholder(self):
    t1 = tf.compat.v1.placeholder(tf.int32,
                                  shape=(
                                      None,
                                      3,
                                      4,
                                  ),
                                  name="t1_placeholder")
    t2 = tf.compat.v1.placeholder(tf.int32,
                                  shape=(None, 2),
                                  name="t2_place_holder")
    t3 = tf.compat.v1.placeholder(tf.float32,
                                  shape=(None,),
                                  name="t3_placeholder")
    t4 = tf.compat.v1.placeholder(tf.float32,
                                  shape=(None, 2, 2),
                                  name="t4_placeholder")
    d1 = {"t1": t1, "t2": t2}
    d2 = {"t3": t3, "t4": t4}
    l = [d1, d2]

    t5 = tf.constant([
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24
    ],
                     dtype=tf.int32)
    t5 = tf.reshape(t5, [-1, 3, 4])
    t6 = tf.constant([9, 8, 7, 6])
    t6 = tf.reshape(t6, [-1, 2])

    t7 = tf.constant([9, 12, 15], dtype=tf.float32)
    t8 = tf.constant([1, 2, 3, 4, 5, 6, 7, 8], dtype=tf.float32)
    t8 = tf.reshape(t8, [-1, 2, 2])

    packed_l = tensor_utils.pack_typed_keyed_tensors(l)
    unpacked_l = tensor_utils.unpack_packed_tensors(
        tensor_utils.get_typed_keyed_shape(l), packed_l)

    with self.session() as sess:
      t5_value = sess.run(t5)
      t6_value = sess.run(t6)
      t7_value = sess.run(t7)
      t8_value = sess.run(t8)

      unpacked_l_value = sess.run(unpacked_l,
                                  feed_dict={
                                      t1: t5_value,
                                      t2: t6_value,
                                      t3: t7_value,
                                      t4: t8_value
                                  })

      original_l_value = sess.run(l,
                                  feed_dict={
                                      t1: t5_value,
                                      t2: t6_value,
                                      t3: t7_value,
                                      t4: t8_value
                                  })

      for d1, d2 in zip(unpacked_l_value, original_l_value):
        for key in sorted(d1):
          self.assertAllEqual(d1[key], d2[key])

  def test_split_tensors_with_type_and_merge_dicts(self):
    t1 = tf.constant([[0, 0, 1, 0], [0, 2, 0, 3]], dtype=tf.int64)
    t2 = tf.constant([0, 3, 1, 2], dtype=tf.int64)
    t3 = tf.constant([9.1, 2.2], dtype=tf.float32)
    t4 = tf.constant([[1.1, 2.2], [3.3, 4.4]], dtype=tf.float32)
    t5 = tf.constant([3, 4, 5, 6, 7, 8, 9], dtype=tf.float64)
    d1 = {"t1": t1, "t2": t2}
    d2 = {"t4": t4, "t3": t3}
    d3 = {"t5": t5}
    l = [d2, d3, d1]
    total_d = {"t1": t1, "t2": t2, "t4": t4, "t3": t3, "t5": t5}

    split_total_d_l = tensor_utils.split_tensors_with_type(total_d)
    total_d_assemble = tensor_utils.merge_dicts(split_total_d_l)

    with self.session() as sess:
      split_total_d_l_value = sess.run(split_total_d_l)
      l_value = sess.run(l)
      for i, d in enumerate(split_total_d_l_value):
        for key in sorted(d):
          self.assertAllEqual(d[key], l_value[i][key])

      total_d_assemble_value = sess.run(total_d_assemble)
      total_d_value = sess.run(total_d)
      for key in sorted(total_d_assemble_value):
        self.assertAllEqual(total_d_value[key], total_d_assemble_value[key])


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
