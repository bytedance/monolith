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

from absl import logging
from monolith.native_training.data.training_instance.python.instance_dataset_ops import PBInstanceDataset
from monolith.native_training.data.training_instance.python.parse_instance_ops import parse_instances
from tensorflow.python.framework import sparse_tensor

FIDV1_FEATURES = [i for i in range(1, 10)]
FIDV2_FEATURES = ["fc_360d_ml_convert_cid", "fc_360d_ml_convert_advertiser_id"]
FLOAT_FEATURES = ["fc_muse_finish_rough_10168_uid_d128"]
FLOAT_FEATURES_DIM = [128]
INT64_FEATURES = ["fc_dense_external_action"]
INT64_FEATURE_DIM = [1]


def parse(serialized):
  return parse_instances(serialized, FIDV1_FEATURES, FIDV2_FEATURES,
                         FLOAT_FEATURES, FLOAT_FEATURES_DIM, INT64_FEATURES,
                         INT64_FEATURE_DIM)


def testInstanceDataset():

  # with self.session() as sess:
  with tf.compat.v1.Session() as sess:
    logging.warning("PBInstanceDatasetV2 process is Starting")
    dataset = PBInstanceDataset(
        file_name="",
        has_sort_id=True,
        kafka_dump_prefix=True,
    )
    dataset = dataset.batch(32).map(parse)
    it = tf.compat.v1.data.make_one_shot_iterator(dataset)
    element = it.get_next()
    logging.warning("PBInstanceDatasetV2 next process is Finished")
    elements = sess.run(element)
    logging.warning(element)
    logging.warning(elements["sample_rate"])


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  testInstanceDataset()
