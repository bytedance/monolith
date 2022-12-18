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

from absl import logging
import numpy as np
import random
import tensorflow as tf

from idl.matrix.proto import proto_parser_pb2
from monolith.native_training.data.training_instance.python import parse_instance_ops as ops
from monolith.native_training.data.training_instance.python import parser_utils


def make_fid_v1(slot_id, fid):
  return (slot_id << 54) | fid


def make_fid_v2(slot_id, fid):
  return (slot_id << 48) | fid


def get_test_fidv2():
  return [make_fid_v2(100, i) for i in range(10)]


def generate_instance():
  instance = proto_parser_pb2.Instance()
  v1_fids = [make_fid_v1(i, i) for i in range(10)]
  v2_fids = get_test_fidv2()
  instance.fid.extend(v1_fids)
  fid_feature = instance.feature.add()
  fid_feature.name = "fidv2"
  fid_feature.fid.extend(v2_fids)
  float_feature = instance.feature.add()
  float_feature.name = "ue"
  float_feature.float_value.extend([float(i * 1e-5) for i in range(16)])
  int64_feature = instance.feature.add()
  int64_feature.name = "int64_feature"
  int64_feature.int64_value.append(100)
  string_feature = instance.feature.add()
  string_feature.name = "string_feature"
  string_feature.bytes_value.append(b"test_string")
  instance.label.extend([1.1, 2.2, 3.3])
  instance.line_id.uid = 110
  instance.line_id.sample_rate = 0.5
  instance.line_id.req_time = 64
  instance.line_id.actions.extend([0, 100])
  instance.line_id.user_id = "123"
  return instance


class RaggedEncodingHelperTest(tf.test.TestCase):

  def testExpandContract(self):
    with tf.compat.v1.Session() as sess:
      rt = tf.RaggedTensor.from_row_splits(values=[3, 1, 4, 1, 5, 9, 2, 6],
                                           row_splits=[0, 4, 4, 7, 8, 8])
      rt_copy = tf.RaggedTensor.from_row_splits(values=[3, 1, 4, 1, 5, 9, 2, 6],
                                                row_splits=[0, 4, 4, 7, 8, 8])
      d = {"slot_2": rt}
      assert rt._row_partition._value_rowids is None
      d = parser_utils.RaggedEncodingHelper.expand(
          d, with_precomputed_value_rowids=True)
      print(d)
      assert len(d) == 1
      self.assertAllEqual(sess.run(d["slot_2"]["value_rowids"]),
                          sess.run(rt_copy.value_rowids()))
      d = parser_utils.RaggedEncodingHelper.contract(d)
      assert len(d) == 1
      self.assertAllEqual(sess.run(d["slot_2"]), sess.run(rt))
      self.assertAllEqual(sess.run(d["slot_2"]._row_partition._value_rowids),
                          sess.run(rt_copy.value_rowids()))


class ParseInstancesTest(tf.test.TestCase):

  def testParseInstance(self):
    instance = generate_instance()
    body = instance.SerializeToString()
    with tf.compat.v1.Session() as sess:
      features = ops.parse_instances2(
          [body, body],
          fidv1_features=list(range(10)),
          fidv2_features=["fidv2"],
          float_features=["ue"],
          float_feature_dims=[16],
          int64_features=["int64_feature"],
          int64_feature_dims=[1],
          string_features=["string_feature"],
          string_feature_dims=[1],
          misc_float_features=["sample_rate", "label"],
          misc_float_dims=[1, 3],
          misc_int64_features=["uid", "actions"],
          misc_int64_dims=[1, 2],
          misc_string_features=["user_id"],
          misc_string_dims=[1])
      features = sess.run(features)
      self.assertEqual(
          len([fidv1_key for fidv1_key in features if "slot" in fidv1_key]), 10)
      self.assertAllEqual(
          features["slot_1"],
          tf.compat.v1.ragged.constant_value([[make_fid_v2(1, 1)]] * 2))
      self.assertAllEqual(
          features["fidv2"],
          tf.compat.v1.ragged.constant_value([get_test_fidv2()] * 2))
      self.assertAllClose(features["int64_feature"], [[100]] * 2)
      self.assertAllEqual(features["string_feature"], [[b"test_string"]] * 2)
      self.assertAllClose(features["ue"],
                          [[float(i * 1e-5) for i in range(16)]] * 2)
      self.assertAllClose(features["sample_rate"], [[0.5]] * 2)
      self.assertAllClose(features["label"], [[1.1, 2.2, 3.3]] * 2)
      self.assertAllEqual(features["uid"], [[110]] * 2)
      self.assertAllEqual(features["actions"], [[0, 100]] * 2)
      self.assertAllEqual(features["user_id"], [["123"]] * 2)

  def testParseInstanceV1Only(self):
    instance = generate_instance()
    body = instance.SerializeToString()
    with tf.compat.v1.Session() as sess:
      features = ops.parse_instances2([body], fidv1_features=[1])
      features = sess.run(features)
      self.assertAllEqual(
          features["slot_1"],
          tf.compat.v1.ragged.constant_value([[make_fid_v1(1, 1)]]))

  def testParseInstanceWithMissingFields(self):
    instance = generate_instance()
    body = instance.SerializeToString()
    with tf.compat.v1.Session() as sess:
      features = ops.parse_instances2(
          [body],
          fidv1_features=list(range(11)),
          fidv2_features=["fidv2", "fidv2_2"],
          float_features=["ue", "ue2"],
          float_feature_dims=[16, 8],
          int64_features=["int64_feature", "missing_int64_feature"],
          int64_feature_dims=[1, 10],
          string_features=["string_feature", "missing_string_feature"],
          string_feature_dims=[1, 10])
      features = sess.run(features)
      # It should be an empty tensor for the last FID element
      self.assertAllEqual(features["slot_10"],
                          tf.compat.v1.ragged.constant_value([[]]))
      self.assertAllEqual(features["fidv2_2"],
                          tf.compat.v1.ragged.constant_value([[]]))
      # It should be an zero tensor for the second UE element
      self.assertAllEqual(features["ue2"], [[0 for i in range(8)]])
      # It should be an zero tensor for the second int64 element
      self.assertAllEqual(features["missing_int64_feature"],
                          [[0 for i in range(10)]])
      self.assertAllEqual(features["missing_string_feature"],
                          [["" for i in range(10)]])


class RawParseInstanceTest(tf.test.TestCase):

  def test_concat(self):
    serialized = [generate_instance().SerializeToString()]
    tensors = ops.monolith_raw_parse_instance(T=[tf.int64, tf.int64],
                                              serialized=serialized,
                                              fidv1_features=[0, 1],
                                              fidv2_features=["fidv2"],
                                              fid_output_type="CONCAT")

    with self.session() as sess:
      tensors = sess.run(tensors)
      self.assertAllEqual(tensors[0], [0, 1, 2, len(get_test_fidv2()) + 2])
      self.assertAllEqual(
          tensors[1], [make_fid_v2(0, 0), make_fid_v2(1, 1)] + get_test_fidv2())


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
