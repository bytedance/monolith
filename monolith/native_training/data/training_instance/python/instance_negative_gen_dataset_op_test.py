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
import numpy as np
from collections import defaultdict

from monolith.native_training.data.training_instance.python.instance_dataset_op import PBDataset, PbType, PBInstanceDataset, InstanceNegativeGenDataset
from monolith.native_training.data.training_instance.python.parse_instance_ops import parse_variant_instances, parse_instances
from monolith.native_training.data.training_instance.python.pb_datasource_ops import variant_dummy
from tensorflow.python.framework import sparse_tensor

FILE_NAME = 'monolith/native_training/data/training_instance/instance.pb'
CHANNEL_SLOT = 357
GROUP_SLOTS = [200,201,202,203,204,205,206,210,211,212,213,214,215,\
        216,217,218,219,220,221,222,223,224,225,230,231,232,233,234,235,236,237,238,239,240,241,242]
LABEL_FIELD = 'actions'
LABEL_INDEX = 0
NEGATIVE_LABEL = -2
NEGATIVE_LABEL2 = -1
CHANNEL_FEATURE_NAME = ""
GROUP_FEATURES_NAME = []
GID = 'gid'

CHANNEL_SLOT_NAME = 'slot_' + str(CHANNEL_SLOT)
GROUP_SLOT_NAME = 'slot_200'

CHANNEL = 6435440280980561277


def parse1(pb_varient: tf.Tensor):
  FIDV1_FEATURES = [
    1, 3, 4, 5, 7, 8, 9, 31, 32, 33, 35, 36, 37, 38, 42, 44, 60, 61, 62, 63, 65, 66, 67, 68, 72, 74, 90, 91, 92, 93, 95, 120, \
    121, 122, 123, 125, 126, 128, 150, 151, 152, 153, 155, 156, 158, 180, 181, 182, 183, 185, 186, 188, 192, 193, 194, 200, 201, \
    202, 204, 206, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 230, 231, 232, 233, 234, 235, \
    236, 237, 238, 239, 240, 242, 357, 358, 359, 360, 361, 410, 411, 412, 413, 415, 416, 418, 422, 423, 424, 446, 472, 475, 515, 516
  ]
  return parse_variant_instances(pb_varient,
                                 FIDV1_FEATURES,
                                 misc_int64_features=[GID])


class InsNegativeDatasetTest(tf.test.TestCase):

  def testNegativeGen(self):
    with self.session() as sess:
      dataset = PBDataset(file_name=FILE_NAME,
                          has_sort_id=True,
                          kafka_dump=True,
                          kafka_dump_prefix=False,
                          input_pb_type=PbType.Instance,
                          output_pb_type=PbType.Instance)

      dataset = dataset.negative_gen(neg_num=7,
                                     channel_slot=CHANNEL_SLOT,
                                     group_slots=GROUP_SLOTS,
                                     per_channel_sample=True,
                                     start_num=0,
                                     max_group_num_per_channel=10000,
                                     label_field=LABEL_FIELD,
                                     label_index=0,
                                     negative_label=NEGATIVE_LABEL,
                                     use_neg_ins=True)

      dataset = dataset.batch(8, drop_remainder=True).map(parse1)
      it = tf.compat.v1.data.make_one_shot_iterator(dataset)
      element = it.get_next()
      count = 0

      channel_res = []
      group_res = []
      label_res = []

      while True:
        try:
          ret = sess.run(element)
          channel_res.append(ret[CHANNEL_SLOT_NAME].flat_values)
          group_res.append(ret[GROUP_SLOT_NAME].flat_values)
          label_res.append(ret[LABEL_FIELD])
          count += 8
          if count > 16:
            break
        except tf.errors.OutOfRangeError:
          logging.info("got eof")
          break

      for i in range(1, 8):
        self.assertEqual(channel_res[0][0], channel_res[0][i])
      self.assertEqual(label_res[0][1], NEGATIVE_LABEL)

  def testRingBufferCache(self):
    with self.session() as sess:
      dataset = PBDataset(file_name=FILE_NAME,
                          has_sort_id=True,
                          kafka_dump=True,
                          kafka_dump_prefix=False,
                          input_pb_type=PbType.Instance,
                          output_pb_type=PbType.Instance)

      max_group_num_per_channel = 2
      dataset = dataset.negative_gen(
          neg_num=7,
          channel_slot=CHANNEL_SLOT,
          group_slots=GROUP_SLOTS,
          per_channel_sample=True,
          start_num=0,
          max_group_num_per_channel=max_group_num_per_channel,
          label_field=LABEL_FIELD,
          label_index=0,
          negative_label=NEGATIVE_LABEL,
          use_neg_ins=True)
      dataset = dataset.batch(8, drop_remainder=True).map(parse1)
      it = tf.compat.v1.data.make_one_shot_iterator(dataset)
      element = it.get_next()
      count = 0

      channel_res = []
      group_res = []
      label_res = []
      gid_res = []

      while True:
        try:
          ret = sess.run(element)
          channel_res.append(ret[CHANNEL_SLOT_NAME].flat_values)
          group_res.append(ret[GROUP_SLOT_NAME].flat_values)
          label_res.append(ret[LABEL_FIELD])
          gid_res.append(ret[GID])

          count += 8
          if count > 1024:
            break
        except tf.errors.OutOfRangeError:
          logging.info("got eof")
          break

      res_by_channel = defaultdict(list)
      for i in range(100):
        channel = channel_res[i][0]
        res_by_channel[channel].append(i)

      valid_count = 0
      for channel in res_by_channel:
        one_channel_res = res_by_channel[channel]
        if len(one_channel_res) <= max_group_num_per_channel:
          continue
        idx0 = one_channel_res[0]
        idx1 = one_channel_res[1]
        idx2 = one_channel_res[2]


        if gid_res[idx0][0] != gid_res[idx1][0] and gid_res[idx0][0] != gid_res[idx2][0] \
                and gid_res[idx1][0] != gid_res[idx2][0]:
          for fid in group_res[idx2]:
            self.assertNotIn(fid, group_res[idx0])
            valid_count += 1
      logging.info('checkout count ' + str(valid_count))

  def testIgnoreReaNegInstance(self):
    with self.session() as sess:
      dataset = PBDataset(file_name=FILE_NAME,
                          has_sort_id=True,
                          kafka_dump=True,
                          kafka_dump_prefix=False,
                          input_pb_type=PbType.Instance,
                          output_pb_type=PbType.Instance)

      dataset = dataset.negative_gen(neg_num=7,
                                     channel_slot=CHANNEL_SLOT,
                                     group_slots=GROUP_SLOTS,
                                     per_channel_sample=True,
                                     start_num=0,
                                     max_group_num_per_channel=10000,
                                     label_field=LABEL_FIELD,
                                     label_index=0,
                                     negative_label=NEGATIVE_LABEL,
                                     use_neg_ins=True)
      dataset = InstanceNegativeGenDataset(input_dataset=dataset,
                                           neg_num=2,
                                           channel_slot=CHANNEL_SLOT,
                                           group_slots=GROUP_SLOTS,
                                           per_channel_sample=True,
                                           start_num=0,
                                           max_group_num_per_channel=10000,
                                           label_field=LABEL_FIELD,
                                           label_index=0,
                                           negative_label=NEGATIVE_LABEL2,
                                           use_neg_ins=False)
      dataset = dataset.batch(8, drop_remainder=True).map(parse1)
      it = tf.compat.v1.data.make_one_shot_iterator(dataset)
      element = it.get_next()
      count = 0

      channel_res = []
      group_res = []
      label_res = []

      while True:
        try:
          ret = sess.run(element)
          label_res.append(ret[LABEL_FIELD])
          count += 8
          if count > 16:
            break
        except tf.errors.OutOfRangeError:
          logging.info("got eof")
          break

      self.assertEqual(label_res[0][1], NEGATIVE_LABEL2)

  def testUseNegInstance(self):
    with self.session() as sess:
      dataset = PBDataset(file_name=FILE_NAME,
                          has_sort_id=True,
                          kafka_dump=True,
                          kafka_dump_prefix=False,
                          input_pb_type=PbType.Instance,
                          output_pb_type=PbType.Instance)

      dataset = dataset.negative_gen(neg_num=2,
                                     channel_slot=CHANNEL_SLOT,
                                     group_slots=GROUP_SLOTS,
                                     per_channel_sample=True,
                                     start_num=0,
                                     max_group_num_per_channel=10000,
                                     label_field=LABEL_FIELD,
                                     label_index=0,
                                     negative_label=NEGATIVE_LABEL,
                                     use_neg_ins=True)
      dataset = InstanceNegativeGenDataset(input_dataset=dataset,
                                           neg_num=2,
                                           channel_slot=CHANNEL_SLOT,
                                           group_slots=GROUP_SLOTS,
                                           per_channel_sample=True,
                                           start_num=0,
                                           max_group_num_per_channel=10000,
                                           label_field=LABEL_FIELD,
                                           label_index=0,
                                           negative_label=NEGATIVE_LABEL2,
                                           use_neg_ins=True)
      dataset = dataset.batch(8, drop_remainder=True).map(parse1)
      it = tf.compat.v1.data.make_one_shot_iterator(dataset)
      element = it.get_next()
      count = 0

      channel_res = []
      group_res = []
      label_res = []

      while True:
        try:
          ret = sess.run(element)
          label_res.append(ret[LABEL_FIELD])
          count += 8
          if count > 16:
            break
        except tf.errors.OutOfRangeError:
          logging.info("got eof")
          break

      self.assertEqual(label_res[0][1], NEGATIVE_LABEL2)
      self.assertEqual(label_res[0][2], NEGATIVE_LABEL2)
      self.assertEqual(label_res[0][3], NEGATIVE_LABEL)
      self.assertEqual(label_res[0][4], NEGATIVE_LABEL)


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
