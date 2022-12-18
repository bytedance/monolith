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

import monolith.core.hyperparams as _params
from monolith.core.feature import FeatureSlot, FeatureColumnV1, Env


class FeatureSlotTest(tf.test.TestCase):

  def test_has_bias(self):
    params = _params.Params()
    params.define('qr_multi_hashing', False, '')
    params.define('qr_hashing_threshold', 100000000, '')
    params.define('qr_collision_rate', 4, '')
    params.define('use_random_init_embedding_for_oov', False, '')
    params.define('merge_vector', False, '')
    env = Env({}, params)
    fs_1 = FeatureSlot(env=env, slot_id=1, has_bias=True)

    self.assertEqual(len(fs_1.feature_slices), 1)
    self.assertEqual(fs_1.feature_slices[0].dim, 1)
    self.assertEqual(fs_1.feature_slices[0].slice_index, 0)

  def test_add_feature_slice(self):
    params = _params.Params()
    params.define('qr_multi_hashing', False, '')
    params.define('qr_hashing_threshold', 100000000, '')
    params.define('qr_collision_rate', 4, '')
    params.define('use_random_init_embedding_for_oov', False, '')
    params.define('merge_vector', False, '')
    env = Env({}, params)
    fs_1 = FeatureSlot(env=env, slot_id=1, has_bias=True)

    fs_1.add_feature_slice(dim=10)

    self.assertEqual(len(fs_1.feature_slices), 2)
    self.assertEqual(fs_1.feature_slices[0].dim, 1)
    self.assertEqual(fs_1.feature_slices[0].slice_index, 0)
    self.assertEqual(fs_1.feature_slices[1].dim, 10)
    self.assertEqual(fs_1.feature_slices[1].slice_index, 1)


class FeatureColumnV1Test(tf.test.TestCase):

  def test_add_feature_column(self):
    params = _params.Params()
    params.define('qr_multi_hashing', False, '')
    params.define('qr_hashing_threshold', 100000000, '')
    params.define('qr_collision_rate', 4, '')
    params.define('use_random_init_embedding_for_oov', False, '')
    params.define('merge_vector', False, '')

    env = Env({}, params)
    fs_1 = FeatureSlot(env=env, slot_id=1, has_bias=True)
    fs_1.add_feature_slice(dim=10)
    fc_1 = FeatureColumnV1(fs_1, 'fc_name_1')

    self.assertEqual(len(fs_1._feature_columns), 1)

  def test_merge_split_vector_in_same_slot(self):
    params = _params.Params()
    params.define('qr_multi_hashing', False, '')
    params.define('qr_hashing_threshold', 100000000, '')
    params.define('qr_collision_rate', 4, '')
    params.define('use_random_init_embedding_for_oov', False, '')
    params.define('merge_vector', True, '')

    env = Env({}, params)

    # Test merge logic.
    fs_1 = FeatureSlot(env=env, slot_id=1, has_bias=True)
    slice_1_1 = fs_1.add_feature_slice(dim=2)
    fs_2 = FeatureSlot(env=env, slot_id=2, has_bias=True)
    fs_3 = FeatureSlot(env=env, slot_id=3, has_bias=False)
    slice_3_0 = fs_3.add_feature_slice(dim=2)
    slice_3_1 = fs_3.add_feature_slice(dim=3)
    fs_4 = FeatureSlot(env=env, slot_id=4, has_bias=True)
    slice_4_1 = fs_4.add_feature_slice(dim=2)
    slice_4_2 = fs_4.add_feature_slice(dim=3)
    slice_4_3 = fs_4.add_feature_slice(dim=4)

    fc_1 = FeatureColumnV1(fs_1, 'fc_name_1')
    fc_1.embedding_lookup(slice_1_1)
    fc_2 = FeatureColumnV1(fs_2, 'fc_name_2')
    fc_3 = FeatureColumnV1(fs_3, 'fc_name_3')
    fc_3.embedding_lookup(slice_3_0)
    fc_3.embedding_lookup(slice_3_1)
    fc_4 = FeatureColumnV1(fs_4, 'fc_name_4')
    fc_4.embedding_lookup(slice_4_1)
    fc_4.embedding_lookup(slice_4_2)
    fc_4.embedding_lookup(slice_4_3)
    fc_5 = FeatureColumnV1(fs_4, 'fc_name_5')
    fc_5.embedding_lookup(slice_4_1)
    fc_5.embedding_lookup(slice_4_2)
    fc_5.embedding_lookup(slice_4_3)

    env._merge_vector_in_same_slot()
    # Check the length of merged feature slices in FeatureSlot
    self.assertEqual(len(fs_1._merged_feature_slices), 2)
    self.assertEqual(len(fs_2._merged_feature_slices), 1)
    self.assertEqual(len(fs_3._merged_feature_slices), 1)
    self.assertEqual(len(fs_4._merged_feature_slices), 2)
    # Check the dim of each merged feature slice in FeatureSlot
    self.assertEqual(fs_1._merged_feature_slices[0].dim, 1)
    self.assertEqual(fs_1._merged_feature_slices[1].dim, 2)
    self.assertEqual(fs_2._merged_feature_slices[0].dim, 1)
    self.assertEqual(fs_3._merged_feature_slices[0].dim, 5)
    self.assertEqual(fs_4._merged_feature_slices[0].dim, 1)
    self.assertEqual(fs_4._merged_feature_slices[1].dim, 9)
    # Check the dim of each merged feature slice in FeatureColumn
    self.assertTrue(fs_1._merged_feature_slices[0] in
                    fc_1._merged_feature_slice_to_tf_placeholder)
    self.assertTrue(fs_1._merged_feature_slices[1] in
                    fc_1._merged_feature_slice_to_tf_placeholder)
    self.assertTrue(fs_2._merged_feature_slices[0] in
                    fc_2._merged_feature_slice_to_tf_placeholder)
    self.assertTrue(fs_3._merged_feature_slices[0] in
                    fc_3._merged_feature_slice_to_tf_placeholder)
    self.assertTrue(fs_4._merged_feature_slices[0] in
                    fc_4._merged_feature_slice_to_tf_placeholder)
    self.assertTrue(fs_4._merged_feature_slices[1] in
                    fc_4._merged_feature_slice_to_tf_placeholder)
    self.assertTrue(fs_4._merged_feature_slices[0] in
                    fc_5._merged_feature_slice_to_tf_placeholder)
    self.assertTrue(fs_4._merged_feature_slices[1] in
                    fc_5._merged_feature_slice_to_tf_placeholder)

    # Test split logic
    env._tpu_features = {}
    env._tpu_features["fc_name_1_0"] = tf.constant([[1]])
    env._tpu_features["fc_name_1_1"] = tf.constant([[2, 3]])
    env._tpu_features["fc_name_2_0"] = tf.constant([[4]])
    env._tpu_features["fc_name_3_0"] = tf.constant([[7, 8, 9, 10, 11]])
    env._tpu_features["fc_name_4_0"] = tf.constant([[12]])
    env._tpu_features["fc_name_4_1"] = tf.constant(
        [[13, 14, 15, 16, 17, 18, 19, 20, 21]])
    env._tpu_features["fc_name_5_0"] = tf.constant([[12]])
    env._tpu_features["fc_name_5_1"] = tf.constant(
        [[13, 14, 15, 16, 17, 18, 19, 20, 21]])

    with tf.compat.v1.Session() as sess:
      env._split_merged_embedding(fs_1)
      env._split_merged_embedding(fs_2)
      env._split_merged_embedding(fs_3)
      env._split_merged_embedding(fs_4)

      features = sess.run(env._tpu_features)
      self.assertAllEqual(features["fc_name_1_0"], [[1]])
      self.assertAllEqual(features["fc_name_1_1"], [[2, 3]])
      self.assertAllEqual(features["fc_name_2_0"], [[4]])
      self.assertAllEqual(features["fc_name_3_0"], [[7, 8]])
      self.assertAllEqual(features["fc_name_3_1"], [[9, 10, 11]])
      self.assertAllEqual(features["fc_name_4_0"], [[12]])
      self.assertAllEqual(features["fc_name_4_1"], [[13, 14]])
      self.assertAllEqual(features["fc_name_4_2"], [[15, 16, 17]])
      self.assertAllEqual(features["fc_name_4_3"], [[18, 19, 20, 21]])
      self.assertAllEqual(features["fc_name_5_0"], [[12]])
      self.assertAllEqual(features["fc_name_5_1"], [[13, 14]])
      self.assertAllEqual(features["fc_name_5_2"], [[15, 16, 17]])
      self.assertAllEqual(features["fc_name_5_3"], [[18, 19, 20, 21]])


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
