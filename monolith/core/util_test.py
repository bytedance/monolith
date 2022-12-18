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

import tensorflow.compat.v1 as tf

import monolith.core.util as util


class UtilTest(tf.test.TestCase):
  """Base class for tpu test."""

  root_path = "gs://test_folder/unzipped_tf_records_corrected_repartitioned/"

  def test_range_dataset_single(self):
    expected_results = [
        "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200502/00/part",
    ]
    with self.session() as sess:
      input_dataset = tf.data.Dataset.from_tensor_slices([
          "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200501/00/part",
          "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200502/00/part",
          "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200503/00/part",
      ])
      dataset = util.range_dateset(input_dataset, self.root_path, "20200502",
                                   "20200502")
      iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
      next_element = iterator.get_next()
      i = 0
      try:
        while True:
          self.assertEqual(sess.run(next_element).decode(), expected_results[i])
          i += 1
      except tf.errors.OutOfRangeError:
        pass
      self.assertEqual(i, len(expected_results))

  def test_range_dataset_multiple(self):
    expected_results = [
        "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200502/00/part",
        "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200503/00/part",
        "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200503/01/part",
    ]
    with self.session() as sess:
      input_dataset = tf.data.Dataset.from_tensor_slices([
          "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200501/00/part",
          "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200502/00/part",
          "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200503/00/part",
          "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200503/01/part",
          "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200504/01/part",
      ])
      dataset = util.range_dateset(input_dataset, self.root_path, "20200502",
                                   "20200503")
      iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
      next_element = iterator.get_next()
      i = 0
      try:
        while True:
          self.assertEqual(sess.run(next_element).decode(), expected_results[i])
          i += 1
      except tf.errors.OutOfRangeError:
        pass
      self.assertEqual(i, len(expected_results))

  def test_range_dataset_out_of_boundary(self):
    expected_results = [
        "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200501/00/part",
        "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200502/00/part",
    ]
    with self.session() as sess:
      input_dataset = tf.data.Dataset.from_tensor_slices([
          "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200501/00/part",
          "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200502/00/part",
      ])
      dataset = util.range_dateset(input_dataset, self.root_path, "20200401",
                                   "20200505")
      iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
      next_element = iterator.get_next()
      i = 0
      try:
        while True:
          self.assertEqual(sess.run(next_element).decode(), expected_results[i])
          i += 1
      except tf.errors.OutOfRangeError:
        pass
      self.assertEqual(i, len(expected_results))

  def test_range_dataset_no_start_date(self):
    expected_results = [
        "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200501/00/part",
        "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200502/00/part",
    ]
    with self.session() as sess:
      input_dataset = tf.data.Dataset.from_tensor_slices([
          "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200501/00/part",
          "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200502/00/part",
      ])
      dataset = util.range_dateset(input_dataset,
                                   self.root_path,
                                   start_date=None,
                                   end_date="20200505")
      iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
      next_element = iterator.get_next()
      i = 0
      try:
        while True:
          self.assertEqual(sess.run(next_element).decode(), expected_results[i])
          i += 1
      except tf.errors.OutOfRangeError:
        pass
      self.assertEqual(i, len(expected_results))

  def test_range_dataset_no_end_date(self):
    expected_results = [
        "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200502/00/part",
    ]
    with self.session() as sess:
      input_dataset = tf.data.Dataset.from_tensor_slices([
          "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200501/00/part",
          "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200502/00/part",
      ])
      dataset = util.range_dateset(input_dataset,
                                   self.root_path,
                                   start_date="20200502",
                                   end_date=None)
      iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
      next_element = iterator.get_next()
      i = 0
      try:
        while True:
          self.assertEqual(sess.run(next_element).decode(), expected_results[i])
          i += 1
      except tf.errors.OutOfRangeError:
        pass
      self.assertEqual(i, len(expected_results))


if __name__ == "__main__":
  tf.disable_eager_execution()
  tf.test.main()