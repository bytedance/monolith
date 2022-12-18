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

from monolith.native_training.data.datasets import PBDataset, PbType, DynamicMatchingFilesDataset
from monolith.native_training.data.parsers import parse_instances, parse_examples, parse_example_batch
from monolith.native_training.data.feature_utils import switch_slot, feature_combine

features = {
    'f_spm_1': 301,
    'f_spm_3': 303,
    'f_spm_2': 302,
    'f_spm_4': 304,
    'f_user_id': 1,
    'f_user_ctx_network': 61,
    'f_user_id-f_page': 504,
    'f_scm': 306,
    'f_goods_id': 200,
    'f_goods_sale_number_1000': 225,
    'f_goods_praise_cnt': 229,
    'f_spm': 300,
    'f_page': 305,
    'f_is_dup': 310,
    'f_user_ctx_platform': 52,
    'f_goods_title_terms': 209,
    'f_goods_tags_terms': 211,
    'f_user_test09_array_int32': 554,
    'f_user_test15_array_float': 540,
    'f_user_test14_array_bool': 543,
    'f_user_test12_array_uint64': 551,
    'f_user_test10_array_int64': 549
}


group_slots = [200,201,202,203,204,205,206,210,211,212,213,214,215,\
               216,217,218,219,220,221,222,223,224,225,230,231,232,233,234,235,236,237,238,239,240,241,242]


def parse_inst_exam(tensor: tf.Tensor, out_type):
  fidv1_features = [
      1, 2, 32, 33, 36, 38, 42, 50, 54, 56, 60, 66, 120, 150, 180, 182, 192,
      220, 333, 410, 412, 422, 446
  ]
  if out_type == PbType.INSTANCE:
    return parse_instances(tensor,
                           fidv1_features,
                           dense_features=['label'],
                           dense_feature_shapes=[2],
                           dense_feature_types=[tf.float32],
                           extra_features=['uid', 'req_time', 'item_id'],
                           extra_feature_shapes=[1, 1, 1])
  else:
    return parse_examples(
        tensor,
        sparse_features=[f'fc_slot_{slot}' for slot in fidv1_features],
        dense_features=['label'],
        dense_feature_shapes=[2],
        dense_feature_types=[tf.float32],
        extra_features=['uid', 'req_time', 'item_id'],
        extra_feature_shapes=[1, 1, 1])


def parse_eb(tensor: tf.Tensor, out_type):
  if out_type == PbType.INSTANCE:
    feature_dict = parse_instances(
        tensor,
        fidv1_features=list(features.values()),
        dense_features=['label'],
        dense_feature_shapes=[2],
        dense_feature_types=[tf.float32],
        extra_features=['uid', 'req_time', 'item_id'],
        extra_feature_shapes=[1, 1, 1])
  else:
    feature_dict = parse_examples(tensor,
                                  sparse_features=list(features.keys()),
                                  dense_features=['label'],
                                  dense_feature_shapes=[2],
                                  dense_feature_types=[tf.float32],
                                  extra_features=['uid', 'req_time', 'item_id'],
                                  extra_feature_shapes=[1, 1, 1])
    feature_dict['f_page'] = switch_slot(feature_dict['f_page'], slot=306)
    feature_dict['f_user_id-f_goods_tags_terms'] = feature_combine(
        feature_dict['f_user_id'], feature_dict['f_goods_tags_terms'], slot=505)
  return feature_dict


class DataOpsTest(tf.test.TestCase):

  def target(self, input_pb_type, output_pb_type):
    filter_fn = None

    if input_pb_type == PbType.INSTANCE:
      lagrangex_header = False
      has_sort_id, kafka_dump, kafka_dump_prefix = True, True, False
      file_name = "monolith/native_training/data/training_instance/instance.pb"
    elif input_pb_type == PbType.EXAMPLE:
      lagrangex_header = False
      has_sort_id, kafka_dump, kafka_dump_prefix = True, True, False
      file_name = "monolith/native_training/data/training_instance/example.pb"
    else:
      lagrangex_header = True
      has_sort_id, kafka_dump, kafka_dump_prefix = False, False, False
      file_name = "monolith/native_training/data/training_instance/examplebatch.data"

    def parser(tensor: tf.Tensor):
      if output_pb_type == PbType.PLAINTEXT:
        return parse_inst_exam(tensor, input_pb_type)
      elif input_pb_type != PbType.EXAMPLEBATCH:
        return parse_inst_exam(tensor, output_pb_type)
      else:
        return parse_eb(tensor, output_pb_type)

    dataset = PBDataset(file_name=file_name,
                        lagrangex_header=lagrangex_header,
                        has_sort_id=has_sort_id,
                        kafka_dump=kafka_dump,
                        kafka_dump_prefix=kafka_dump_prefix,
                        input_pb_type=input_pb_type,
                        output_pb_type=output_pb_type)
    if input_pb_type == PbType.EXAMPLEBATCH:
      variant_type = 'instance' if output_pb_type == PbType.INSTANCE else 'example'
      dataset = dataset.instance_reweight(
          action_priority="2,7,0,1,3,4,5,6,8,9,10,11",
          reweight=
          "0:0:1,1:0:1,2:3:-1,3:0:1,4:0:1,5:0:1,6:0:1,7:6:1,8:0:1,9:0:1,10:0:1,11:0:-1",
          variant_type=variant_type)
    if filter_fn is not None:
      dataset = dataset.filter(filter_fn)
    dataset = dataset.batch(8, drop_remainder=True).map(parser)

    for feature in dataset.take(5):
      self.assertIn(len(feature), {26, 27})

  def testExampleBatch2Instance(self):
    self.target(PbType.EXAMPLEBATCH, PbType.INSTANCE)

  def testExample2Instance(self):
    self.target(PbType.EXAMPLE, PbType.INSTANCE)

  def testInstance2Instance(self):
    self.target(PbType.INSTANCE, PbType.INSTANCE)

  def testExampleBatch(self):
    lagrangex_header = True
    has_sort_id, kafka_dump, kafka_dump_prefix = False, False, False
    file_name = "monolith/native_training/data/training_instance/examplebatch.data"
    input_pb_type, output_pb_type = PbType.EXAMPLEBATCH, PbType.EXAMPLEBATCH
    dataset = PBDataset(file_name=file_name,
                        lagrangex_header=lagrangex_header,
                        has_sort_id=has_sort_id,
                        kafka_dump=kafka_dump,
                        kafka_dump_prefix=kafka_dump_prefix,
                        input_pb_type=input_pb_type,
                        output_pb_type=output_pb_type)

    def parser(tensor):
      freatues = parse_example_batch(
          tensor,
          sparse_features=list(features.keys()),
          dense_features=['label'],
          dense_feature_shapes=[2],
          dense_feature_types=[tf.float32],
          extra_features=['uid', 'req_time', 'item_id'],
          extra_feature_shapes=[1, 1, 1])
      return freatues

    dataset = dataset.map(parser)
    for feature in dataset.take(5):
      self.assertIn(len(feature), {26, 27})


if __name__ == '__main__':
  tf.test.main()
