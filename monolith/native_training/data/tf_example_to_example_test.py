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

import math

from absl import app, logging
import tensorflow as tf
import numpy as np

from monolith.native_training.data.feature_utils import tf_example_to_example
from monolith.native_training.data.parsers import parse_examples

# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy(
    )  # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# @tf.py_function(Tout=tf.string)
def serialize_example(feature0, feature1, feature2, feature3, feature4):
  """
  Creates a tf.train.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.train.Example-compatible
  # data type.
  feature = {
      'feature0': _int64_feature(feature0),
      'feature1': _int64_feature(feature1),
      'feature2': _bytes_feature(feature2),
      'feature3': _float_feature(feature3),
      'feature4': _float_feature(feature4),
  }

  # Create a Features message using tf.train.Example.

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()


def get_fid_v2(slot: int, signature: int):
  fid_v2_mask = (1 << 48) - 1
  return (slot << 48) | (signature & fid_v2_mask)


def calc_hash_value(val: float):
  return int(math.log2(abs(val) + 1))


class TFExampleToExampleTest(tf.test.TestCase):

  def test_tf_example_to_example(self):
    tf.compat.v1.disable_v2_behavior()
    logging.set_verbosity(logging.INFO)

    # The number of observations in the dataset.
    n_observations = int(1e4)

    # Boolean feature, encoded as False or True.
    feature0 = np.random.choice([False, True], n_observations)

    # Integer feature, random from 0 to 4.
    feature1 = np.random.randint(0, 5, n_observations)

    # String feature.
    strings = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
    feature2 = strings[feature1]

    # Float feature, from a standard normal distribution.
    feature3 = np.random.randn(n_observations)

    feature4 = np.random.randn(n_observations)

    filename = '/tmp/test.tfrecord'
    # Write the `tf.train.Example` observations to the file.
    with tf.io.TFRecordWriter(filename) as writer:
      for i in range(n_observations):
        example = serialize_example(feature0[i], feature1[i], feature2[i],
                                    feature3[i], feature4[i])
        writer.write(example)

    filenames = [filename]
    dataset = tf.data.TFRecordDataset(filenames)

    # logging.info(dataset)
    # for raw_record in dataset.take(1):
    #   example = tf.train.Example()
    #   example.ParseFromString(raw_record.numpy())
    #   print(example)

    def map_fn(tensor: tf.Tensor):
      return tf_example_to_example(tensor,
                                   sparse_features={
                                       "feature0": 1,
                                       "feature1": 2,
                                       "feature4": 3
                                   },
                                   dense_features=["feature2"],
                                   label="feature3",
                                   instance_weight=None)

    def parse_fn(variant: tf.Tensor):
      return parse_examples(
          variant,
          sparse_features=["not_existed1", "feature0", "feature1", "feature4"],
          dense_features=[
              "label", "feature2", "feature3", "not_existed2", "instance_weight"
          ],
          dense_feature_types=[
              tf.float32, tf.string, tf.float32, tf.float32, tf.float32
          ],
          dense_feature_shapes=[1, 1, 1, 1, 1],
      )

    batch_size = 2
    dataset = dataset.map(map_fn)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(parse_fn)
    session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    session_config.graph_options.rewrite_options.disable_meta_optimizer = True
    with tf.compat.v1.Session(config=session_config) as sess:
      it = tf.compat.v1.data.make_one_shot_iterator(dataset)
      element = it.get_next()
      element["label"] = tf.reshape(element['label'], shape=(-1,))
      element["feature2"] = tf.reshape(element['feature2'], shape=(-1,))
      element["feature3"] = tf.reshape(element['feature3'], shape=(-1,))
      element["not_existed2"] = tf.reshape(element['not_existed2'], shape=(-1,))
      element["instance_weight"] = tf.reshape(element['instance_weight'],
                                              shape=(-1,))
      for i in range(n_observations // batch_size):
        features = sess.run(fetches=element)
        self.assertAllEqual(features['not_existed1'].values.shape, (0,))
        feature0_fids = [
            get_fid_v2(1, val)
            for val in feature0[i * batch_size:(i + 1) * batch_size]
        ]
        feature1_fids = [
            get_fid_v2(2, val)
            for val in feature1[i * batch_size:(i + 1) * batch_size]
        ]
        feature4_fids = [
            get_fid_v2(3, calc_hash_value(val))
            for val in feature4[i * batch_size:(i + 1) * batch_size]
        ]
        self.assertAllEqual(features['feature0'].values, feature0_fids)
        self.assertAllEqual(features['feature1'].values, feature1_fids)
        self.assertAllEqual(features['feature4'].values, feature4_fids)
        self.assertAllClose(features['label'],
                            feature3[i * batch_size:(i + 1) * batch_size])
        self.assertAllClose(features['feature3'], [0, 0])
        self.assertAllClose(features['not_existed2'], [0, 0])
        self.assertAllClose(features['instance_weight'], [1.0, 1.0])


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
