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

import io
from absl import logging
import os
import uuid
import struct
import tensorflow as tf
import tempfile
from typing import List, BinaryIO
from idl.matrix.proto import proto_parser_pb2, example_pb2
from monolith.native_training.data.datasets import PBDataset, PbType
from monolith.native_training.data.parsers import parse_instances, \
  parse_examples
from monolith.native_training.model_export.data_gen_utils import lg_header, sort_header
from monolith.native_training.data.feature_utils import (
    add_action, add_label, feature_combine, filter_by_fids, filter_by_label,
    filter_by_value, scatter_label, switch_slot, map_id, use_field_as_label,
    label_upper_bound, label_normalization, multi_label_gen, string_to_variant,
    variant_to_zeros, has_variant)

fid_v1_mask = (1 << 54) - 1
fid_v2_mask = (1 << 48) - 1


def get_fid_v1(slot: int, signautre: int):
  return (slot << 54) | (signautre & fid_v1_mask)


def get_fid_v2(slot: int, signature: int):
  return (slot << 48) | (signature & fid_v2_mask)


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

group_slots = [
    200, 201, 202, 203, 204, 205, 206, 210, 211, 212, 213, 214, 215, 216, 217,
    218, 219, 220, 221, 222, 223, 224, 225, 230, 231, 232, 233, 234, 235, 236,
    237, 238, 239, 240, 241, 242
]


def parse_instance_or_example(tensor: tf.Tensor, out_type):
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
                           extra_features=[
                               'uid', 'req_time', 'item_id', 'actions',
                               'video_finish_percent'
                           ],
                           extra_feature_shapes=[1, 1, 1, 2, 1])
  else:
    return parse_examples(
        tensor,
        sparse_features=[f'fc_slot_{slot}' for slot in fidv1_features],
        dense_features=['label'],
        dense_feature_shapes=[2],
        dense_feature_types=[tf.float32],
        extra_features=[
            'uid', 'req_time', 'item_id', 'actions', 'video_finish_percent'
        ],
        extra_feature_shapes=[1, 1, 1, 3, 1])


def parse_example_batch(tensor: tf.Tensor, out_type):
  if out_type == PbType.INSTANCE:
    feature_dict = parse_instances(tensor,
                                   fidv1_features=list(features.values()),
                                   dense_features=['label'],
                                   dense_feature_shapes=[2],
                                   dense_feature_types=[tf.float32],
                                   extra_features=[
                                       'uid', 'req_time', 'item_id', 'actions',
                                       'video_finish_percent'
                                   ],
                                   extra_feature_shapes=[1, 1, 1, 3, 1])
  else:
    feature_dict = parse_examples(tensor,
                                  sparse_features=list(features.keys()),
                                  dense_features=['label'],
                                  dense_feature_shapes=[2],
                                  dense_feature_types=[tf.float32],
                                  extra_features=[
                                      'uid', 'req_time', 'item_id', 'actions',
                                      'video_finish_percent'
                                  ],
                                  extra_feature_shapes=[1, 1, 1, 3, 1])
    feature_dict['f_page'] = switch_slot(feature_dict['f_page'], slot=306)
    feature_dict['f_user_id-f_goods_tags_terms'] = feature_combine(
        feature_dict['f_user_id'], feature_dict['f_goods_tags_terms'], slot=505)
  return feature_dict


def generate_instance(labels: List[int],
                      actions: List[int],
                      chnid: int = None,
                      did: str = None,
                      fid_v1_list: List[int] = None,
                      device_type: str = None):
  instance = proto_parser_pb2.Instance()
  instance.fid.extend(fid_v1_list if fid_v1_list else [])
  instance.label.extend(labels)
  instance.line_id.user_id = "test_{}".format(uuid.uuid4())
  instance.line_id.uid = 100
  instance.line_id.sample_rate = 0.5
  instance.line_id.actions.extend(actions)
  if chnid is not None:
    instance.line_id.chnid = chnid
  if did is not None:
    instance.line_id.did = did
  if device_type is not None:
    instance.line_id.device_type = device_type
  return instance


def write_instance_into_file(file: BinaryIO, instance):
  sort_id = str(instance.line_id.user_id)
  file.write(struct.pack('<Q', len(sort_id)))
  file.write(sort_id.encode())
  instance_serialized = instance.SerializeToString()
  file.write(struct.pack('<Q', len(instance_serialized)))
  file.write(instance_serialized)


class DataOpsTest(tf.test.TestCase):
  '''
  def pb_dataset_target(self,
                        input_pb_type,
                        output_pb_type,
                        filter_fn=None,
                        add_action_fn=None,
                        return_result_key='actions',
                        num_return_items=2):
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
        return parse_instance_or_example(tensor, input_pb_type)
      elif input_pb_type != PbType.EXAMPLEBATCH:
        return parse_instance_or_example(tensor, output_pb_type)
      else:
        return parse_example_batch(tensor, output_pb_type)

    with tf.Graph().as_default():
      config = tf.compat.v1.ConfigProto()
      config.graph_options.rewrite_options.disable_meta_optimizer = True
      with self.session(config=config) as sess:
        dataset = PBDataset(file_name=file_name,
                            lagrangex_header=lagrangex_header,
                            has_sort_id=has_sort_id,
                            kafka_dump=kafka_dump,
                            kafka_dump_prefix=kafka_dump_prefix,
                            input_pb_type=input_pb_type,
                            output_pb_type=output_pb_type)
        if add_action_fn is not None:
          dataset = dataset.map(add_action_fn)

        if input_pb_type == PbType.EXAMPLEBATCH:
          variant_type = 'instance' if output_pb_type == PbType.INSTANCE else 'example'
          dataset = dataset.instance_reweight(
              action_priority="2,7,0,1,3,4,5,6,8,9,10,11",
              reweight=
              "0:0:1,1:0:1,2:3:-1,3:0:1,4:0:1,5:0:1,6:0:1,7:6:1,8:0:1,9:0:1,10:0:1,11:0:-1",
              variant_type=variant_type)
        if filter_fn is not None:
          dataset = dataset.filter(filter_fn)

        batch_size = 4
        dataset = dataset.batch(batch_size, drop_remainder=True).map(parser)
        it = tf.compat.v1.data.make_one_shot_iterator(dataset)
        element = it.get_next()

        results = list()
        for _ in range(num_return_items):
          try:
            element_result = sess.run(element)
            results.append(element_result[return_result_key])
          except tf.errors.OutOfRangeError:
            break
        return results

  def test_input_instance_output_instance(self):
    actions = self.pb_dataset_target(input_pb_type=PbType.INSTANCE,
                                     output_pb_type=PbType.INSTANCE)
    self.assertAllEqual(actions[0], [[1, 0], [1, 0], [1, 0], [1, 0]])
    self.assertAllEqual(actions[1], [[1, 0], [1, 0], [1, 0], [1, 0]])

  def test_input_instance_output_instance_add_action(self):
    actions = self.pb_dataset_target(
        input_pb_type=PbType.INSTANCE,
        output_pb_type=PbType.INSTANCE,
        add_action_fn=lambda variant: add_action(
            variant, 'sample_rate', 'ge', 0, 2, variant_type='instance'))
    self.assertAllEqual(actions[0], [[1, 2], [1, 2], [1, 2], [1, 2]])
    self.assertAllEqual(actions[1], [[1, 2], [1, 2], [1, 2], [1, 2]])

  def test_input_instance_output_example(self):
    actions = self.pb_dataset_target(input_pb_type=PbType.INSTANCE,
                                     output_pb_type=PbType.EXAMPLE)
    self.assertAllEqual(actions[0],
                        [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]])
    self.assertAllEqual(actions[1],
                        [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]])

  def test_input_instance_output_example_add_action(self):
    actions = self.pb_dataset_target(input_pb_type=PbType.INSTANCE,
                                     output_pb_type=PbType.EXAMPLE,
                                     add_action_fn=lambda variant: add_action(
                                         variant,
                                         'req_time',
                                         'between', [1622667900, 1622667911],
                                         2,
                                         variant_type='example'))
    self.assertAllEqual(actions[0],
                        [[1, 2, 0], [1, 2, 0], [1, 2, 0], [1, 2, 0]])
    self.assertAllEqual(actions[1],
                        [[1, 2, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]])

  def test_input_example_output_instance(self):
    actions = self.pb_dataset_target(input_pb_type=PbType.EXAMPLE,
                                     output_pb_type=PbType.INSTANCE)
    self.assertAllEqual(actions[0], [[1, 0], [1, 0], [1, 0], [1, 0]])
    self.assertAllEqual(actions[1], [[1, 0], [1, 0], [1, 0], [1, 0]])

  def test_input_example_output_instance_add_action(self):
    actions = self.pb_dataset_target(
        input_pb_type=PbType.EXAMPLE,
        output_pb_type=PbType.INSTANCE,
        add_action_fn=lambda variant: add_action(variant,
                                                 'req_time',
                                                 'in', [1622667900, 1622667911],
                                                 2,
                                                 variant_type='instance'))
    self.assertAllEqual(actions[0], [[1, 2], [1, 2], [1, 0], [1, 2]])
    self.assertAllEqual(actions[1], [[1, 2], [1, 2], [1, 2], [1, 0]])

  def test_input_example_output_example(self):
    actions = self.pb_dataset_target(input_pb_type=PbType.EXAMPLE,
                                     output_pb_type=PbType.EXAMPLE)
    self.assertAllEqual(actions[0],
                        [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]])
    self.assertAllEqual(actions[1],
                        [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]])

  def test_input_example_output_example_add_action(self):
    actions = self.pb_dataset_target(
        input_pb_type=PbType.EXAMPLE,
        output_pb_type=PbType.EXAMPLE,
        add_action_fn=lambda variant: add_action(
            variant, 'uid', 'eq', 62975225690081677, 2, variant_type='example'))
    self.assertAllEqual(actions[0],
                        [[1, 0, 0], [1, 0, 0], [1, 2, 0], [1, 0, 0]])
    self.assertAllEqual(actions[1],
                        [[1, 0, 0], [1, 0, 0], [1, 2, 0], [1, 0, 0]])

  def test_input_example_batch_output_instance(self):
    actions = self.pb_dataset_target(input_pb_type=PbType.EXAMPLEBATCH,
                                     output_pb_type=PbType.INSTANCE)
    self.assertAllEqual(actions[0],
                        [[2, 0, 0], [2, 0, 0], [2, 0, 0], [2, 0, 0]])
    self.assertAllEqual(actions[1],
                        [[2, 0, 0], [2, 0, 0], [2, 0, 0], [2, 0, 0]])

  def test_input_example_batch_output_instance_add_action(self):
    actions = self.pb_dataset_target(
        input_pb_type=PbType.EXAMPLEBATCH,
        output_pb_type=PbType.INSTANCE,
        add_action_fn=lambda variant: add_action(variant,
                                                 'video_finish_percent',
                                                 'ge',
                                                 0,
                                                 3,
                                                 variant_type='instance'))
    self.assertAllEqual(actions[0],
                        [[2, 3, 0], [2, 3, 0], [2, 3, 0], [2, 3, 0]])
    self.assertAllEqual(actions[1],
                        [[2, 3, 0], [2, 3, 0], [2, 3, 0], [2, 3, 0]])

  def test_input_example_batch_output_example(self):
    actions = self.pb_dataset_target(input_pb_type=PbType.EXAMPLEBATCH,
                                     output_pb_type=PbType.EXAMPLE)
    self.assertAllEqual(actions[0],
                        [[2, 0, 0], [2, 0, 0], [2, 0, 0], [2, 0, 0]])
    self.assertAllEqual(actions[1],
                        [[2, 0, 0], [2, 0, 0], [2, 0, 0], [2, 0, 0]])

  def test_input_example_batch_output_example_add_action(self):
    actions = self.pb_dataset_target(
        input_pb_type=PbType.EXAMPLEBATCH,
        output_pb_type=PbType.EXAMPLE,
        add_action_fn=lambda variant: add_action(
            variant, 'video_finish_percent', 'le', 0, 3, variant_type='example')
    )
    self.assertAllEqual(actions[0],
                        [[2, 3, 0], [2, 3, 0], [2, 3, 0], [2, 3, 0]])
    self.assertAllEqual(actions[1],
                        [[2, 3, 0], [2, 3, 0], [2, 3, 0], [2, 3, 0]])

  def test_input_instance_output_instance_add_label(self):
    mock_batch_num = 100
    add_label_config = '1,2:3:1.0;4::0.5'

    def mock_instance_for_add_label(batch_num: int = 200):
      tmpfile = tempfile.mkstemp()[1]
      labels = [[], [], [], []]

      # for task1: 1,2 -> positive, 3     -> negative
      # for task2: 4   -> positive, other -> negative/invalid(depends on sampling)

      # instance1: task1 -> positive, task2 -> positive
      # instance2: task1 -> positive, task2 -> negative/invalid(depends on sampling)
      # instance3: task1 -> negative, task2 -> positive
      # instance4: task1 -> invalid,  task2 -> negative/invalid(depends on sampling)
      actions = [[1, 2, 4], [1], [3, 4], [5]]
      with io.open(tmpfile, 'wb') as writer:
        for _ in range(batch_num):
          for label, action in zip(labels, actions):
            instance = generate_instance(label, action)
            write_instance_into_file(writer, instance)
      return tmpfile

    file_name = mock_instance_for_add_label(mock_batch_num)
    logging.info('file_name: %s', file_name)

    def parser(tensor: tf.Tensor):
      return parse_instances(tensor,
                             fidv1_features=list(features.values()),
                             dense_features=['label'],
                             dense_feature_shapes=[2],
                             dense_feature_types=[tf.float32],
                             extra_features=[
                                 'uid', 'req_time', 'item_id', 'actions',
                                 'video_finish_percent'
                             ],
                             extra_feature_shapes=[1, 1, 1, 3, 1])

    with tf.Graph().as_default():
      config = tf.compat.v1.ConfigProto()
      config.graph_options.rewrite_options.disable_meta_optimizer = True
      with self.session(config=config) as sess:
        dataset = PBDataset(file_name=file_name,
                            lagrangex_header=False,
                            has_sort_id=True,
                            kafka_dump=False,
                            kafka_dump_prefix=False,
                            input_pb_type=PbType.INSTANCE,
                            output_pb_type=PbType.INSTANCE)
        dataset = dataset.map(
            lambda variant: add_label(variant,
                                      config=add_label_config,
                                      negative_value=-1.0,
                                      new_sample_rate=1.0,
                                      variant_type='instance'))
        dataset = dataset.filter(lambda variant: filter_by_label(
            variant, label_threshold=[-100, -100], variant_type='instance'))

        batch_size = 4
        dataset = dataset.batch(batch_size, drop_remainder=False).map(parser)
        it = tf.compat.v1.data.make_one_shot_iterator(dataset)

        valid_instance_num = 0
        for _ in range(mock_batch_num):
          try:
            element = it.get_next()
            element_result = sess.run(element)
            valid_instance_num += len(element_result['label'])
          except tf.errors.OutOfRangeError:
            break
    logging.info('Valid instance number: %d', valid_instance_num)
    self.assertAllInRange(valid_instance_num, 340, 360)
    os.remove(file_name)

  def test_input_instance_output_instance_label_upper_bound(self):
    labels = self.pb_dataset_target(
        input_pb_type=PbType.INSTANCE,
        output_pb_type=PbType.INSTANCE,
        add_action_fn=lambda variant: label_upper_bound(
            variant, label_upper_bounds=[0.5, 0.5], variant_type='instance'),
        return_result_key='label')
    self.assertAllEqual(labels[0], [[0, 0.5], [0, 0.5], [0, 0.5], [0, 0.5]])
    self.assertAllEqual(labels[1], [[0, 0.5], [0, 0.5], [0, 0.5], [0, 0.5]])

  def test_input_instance_output_instance_label_normalization(self):
    labels = self.pb_dataset_target(
        input_pb_type=PbType.INSTANCE,
        output_pb_type=PbType.INSTANCE,
        add_action_fn=lambda variant: label_normalization(
            variant,
            norm_methods=['scale', 'repow'],
            norm_values=[0.5, 3],
            variant_type='instance'),
        return_result_key='label')
    self.assertAllEqual(labels[0], [[0, 8], [0, 8], [0, 8], [0, 8]])
    self.assertAllEqual(labels[1], [[0, 8], [0, 8], [0, 8], [0, 8]])

  def test_input_examplebatch_output_instance_use_field_as_label(self):
    labels = self.pb_dataset_target(
        input_pb_type=PbType.INSTANCE,
        output_pb_type=PbType.INSTANCE,
        add_action_fn=lambda variant: use_field_as_label(
            variant, 'sample_rate', False, 0, variant_type='instance'),
        return_result_key='label')
    self.assertAllEqual(labels[0], [[1, 1], [1, 1], [1, 1], [1, 1]])
    self.assertAllEqual(labels[1], [[1, 1], [1, 1], [1, 1], [1, 1]])

    labels = self.pb_dataset_target(
        input_pb_type=PbType.INSTANCE,
        output_pb_type=PbType.INSTANCE,
        add_action_fn=lambda variant: use_field_as_label(label_upper_bound(
            variant, label_upper_bounds=[0.5, 0.5], variant_type='instance'),
                                                         'sample_rate',
                                                         True,
                                                         1.1,
                                                         variant_type='instance'
                                                        ),
        return_result_key='label')
    # Original label is [0, 0.5], new label = max(original_label, [1, 1])
    self.assertAllEqual(labels[0], [[1, 1], [1, 1], [1, 1], [1, 1]])
    self.assertAllEqual(labels[1], [[1, 1], [1, 1], [1, 1], [1, 1]])

    labels = self.pb_dataset_target(
        input_pb_type=PbType.INSTANCE,
        output_pb_type=PbType.INSTANCE,
        add_action_fn=lambda variant: use_field_as_label(label_upper_bound(
            variant, label_upper_bounds=[0.5, 0.5], variant_type='instance'),
                                                         'sample_rate',
                                                         True,
                                                         0.9,
                                                         variant_type='instance'
                                                        ),
        return_result_key='label')
    # Original label is [0, 0.5], new label = max(original_label, [0, 0])
    self.assertAllEqual(labels[0], [[0, 0.5], [0, 0.5], [0, 0.5], [0, 0.5]])
    self.assertAllEqual(labels[1], [[0, 0.5], [0, 0.5], [0, 0.5], [0, 0.5]])

  def test_input_instance_output_instance_filter_by_label_equals(self):
    labels = self.pb_dataset_target(
        input_pb_type=PbType.INSTANCE,
        output_pb_type=PbType.INSTANCE,
        filter_fn=lambda variant: filter_by_label(variant,
                                                  label_threshold=[0, 1],
                                                  filter_equal=False,
                                                  variant_type='instance'),
        return_result_key='label',
        num_return_items=100)
    self.assertEqual(len(labels), 100)
    self.assertAllEqual(labels[0], [[0, 1], [0, 1], [0, 1], [0, 1]])
    self.assertAllEqual(labels[1], [[0, 1], [0, 1], [0, 1], [0, 1]])

    labels = self.pb_dataset_target(
        input_pb_type=PbType.INSTANCE,
        output_pb_type=PbType.INSTANCE,
        filter_fn=lambda variant: filter_by_label(variant,
                                                  label_threshold=[0, 1],
                                                  filter_equal=True,
                                                  variant_type='instance'),
        return_result_key='label',
        num_return_items=100)
    self.assertEqual(len(labels), 49)
    self.assertAllEqual(labels[0], [[0, 2], [0, 2], [0, 2], [0, 2]])
    self.assertAllEqual(labels[1], [[0, 2], [0, 2], [0, 2], [0, 2]])

  def test_input_instance_output_instance_scatter_label(self):
    mock_batch_num = 1
    scatter_label_config = '100:3,200:1,300:4'

    def mock_instance_for_scatter_label(batch_num: int = 200):
      tmpfile = tempfile.mkstemp()[1]
      labels = [[1], [2], [3], []]
      actions = [[], [], [], []]
      chnids = [0, 100, 200, 300]
      with io.open(tmpfile, 'wb') as writer:
        for _ in range(batch_num):
          for label, action, chnid in zip(labels, actions, chnids):
            instance = generate_instance(label, action, chnid)
            write_instance_into_file(writer, instance)
      return tmpfile

    file_name = mock_instance_for_scatter_label(mock_batch_num)
    logging.info('file_name: %s', file_name)

    def parser(tensor: tf.Tensor):
      return parse_instances(tensor,
                             fidv1_features=list(features.values()),
                             dense_features=['label'],
                             dense_feature_shapes=[5],
                             dense_feature_types=[tf.float32],
                             extra_features=[
                                 'uid', 'req_time', 'item_id', 'actions',
                                 'video_finish_percent'
                             ],
                             extra_feature_shapes=[1, 1, 1, 3, 1])

    with tf.Graph().as_default():
      config = tf.compat.v1.ConfigProto()
      config.graph_options.rewrite_options.disable_meta_optimizer = True
      with self.session(config=config) as sess:
        dataset = PBDataset(file_name=file_name,
                            lagrangex_header=False,
                            has_sort_id=True,
                            kafka_dump=False,
                            kafka_dump_prefix=False,
                            input_pb_type=PbType.INSTANCE,
                            output_pb_type=PbType.INSTANCE)
        dataset = dataset.map(lambda variant: scatter_label(
            variant, config=scatter_label_config, variant_type='instance'))
        dataset = dataset.filter(lambda variant: filter_by_label(
            variant,
            label_threshold=[-100, -100, -100, -100, -100],
            variant_type='instance'))

        batch_size = 4
        dataset = dataset.batch(batch_size, drop_remainder=False).map(parser)
        it = tf.compat.v1.data.make_one_shot_iterator(dataset)

        try:
          element = it.get_next()
          element_result = sess.run(element)
          self.assertAllEqual(len(element_result['label']), 2)
          self.assertAllClose(
              element_result['label'],
              [[
                  -3.4028235e+38, -3.4028235e+38, -3.4028235e+38, 2.0000000e+00,
                  -3.4028235e+38
              ],
               [
                   -3.4028235e+38, 3.0000000e+00, -3.4028235e+38,
                   -3.4028235e+38, -3.4028235e+38
               ]])
        except tf.errors.OutOfRangeError:
          self.assertTrue(False)
    os.remove(file_name)

  def test_filter_by_value(self):
    file_name = "monolith/native_training/data/training_instance/instance.pb"
    config = tf.compat.v1.ConfigProto()
    config.graph_options.rewrite_options.disable_meta_optimizer = True

    def filter_fn(ts):
      return filter_by_value(ts,
                             field_name='req_id',
                             op='endswith',
                             operand=['kjhfjh', 'huggfyfi'])

    with self.session(config=config) as sess:
      dataset = PBDataset(file_name=file_name,
                          lagrangex_header=False,
                          has_sort_id=True,
                          kafka_dump=True,
                          kafka_dump_prefix=False,
                          input_pb_type=PbType.INSTANCE,
                          output_pb_type=PbType.INSTANCE)
      dataset = dataset.filter(filter_fn)
      it = tf.compat.v1.data.make_one_shot_iterator(dataset)
      element = it.get_next()
      result = None
      try:
        result = sess.run(element)
      except tf.errors.OutOfRangeError:
        self.assertTrue(result is None)

  def test_filter_by_value_not_in(self):
    mock_batch_num = 1

    def mock_instance_for_filter_by_value(batch_num: int = 200):
      tmpfile = tempfile.mkstemp()[1]
      labels = [[1], [2], [3], []]
      actions = [[], [], [], []]
      chnids = [10, 20, 30, 40]
      dids = ['hello', 'world', '300', '400']
      with io.open(tmpfile, 'wb') as writer:
        for _ in range(batch_num):
          for label, action, chnid, did in zip(labels, actions, chnids, dids):
            instance = generate_instance(label, action, chnid, did)
            write_instance_into_file(writer, instance)
      return tmpfile

    file_name = mock_instance_for_filter_by_value(mock_batch_num)
    logging.info('file_name: %s', file_name)

    # generate FilterValues serialized files
    tmp_filter_values_file_string = tempfile.mkstemp()[1]
    with tf.io.gfile.GFile(tmp_filter_values_file_string, 'w') as f:
      filter_values = example_pb2.FilterValues()
      filter_values.bytes_list.value.extend([b'hello', b'world', b'excluded'])
      f.write(filter_values.SerializeToString())
    tmp_filter_values_file_int64 = tempfile.mkstemp()[1]
    with tf.io.gfile.GFile(tmp_filter_values_file_int64, 'w') as f:
      filter_values = example_pb2.FilterValues()
      filter_values.int64_list.value.extend([20, 30, 666])
      f.write(filter_values.SerializeToString())

    def parser(tensor: tf.Tensor):
      return parse_instances(tensor,
                             fidv1_features=list(features.values()),
                             dense_features=['label'],
                             dense_feature_shapes=[5],
                             dense_feature_types=[tf.float32],
                             extra_features=['uid', 'req_time', 'did'],
                             extra_feature_shapes=[1, 1, 1])

    with tf.Graph().as_default():
      config = tf.compat.v1.ConfigProto()
      config.graph_options.rewrite_options.disable_meta_optimizer = True
      with self.session(config=config) as sess:
        dataset_base = PBDataset(file_name=file_name,
                                 lagrangex_header=False,
                                 has_sort_id=True,
                                 kafka_dump=False,
                                 kafka_dump_prefix=False,
                                 input_pb_type=PbType.INSTANCE,
                                 output_pb_type=PbType.INSTANCE)
        dataset_filter_by_list = dataset_base.filter(
            lambda variant: filter_by_value(variant,
                                            field_name='did',
                                            op='not-in',
                                            operand=['hello', 'world']))
        dataset_filter_by_file_string = dataset_base.filter(
            lambda variant: filter_by_value(variant,
                                            field_name='did',
                                            op='not-in',
                                            operand=None,
                                            operand_filepath=
                                            tmp_filter_values_file_string))
        dataset_filter_by_file_int64 = dataset_base.filter(
            lambda variant: filter_by_value(variant,
                                            field_name='chnid',
                                            op='in',
                                            operand=None,
                                            operand_filepath=
                                            tmp_filter_values_file_int64))

        batch_size = 4
        dataset_filter_by_list = dataset_filter_by_list.batch(
            batch_size, drop_remainder=False).map(parser)
        dataset_filter_by_file_string = dataset_filter_by_file_string.batch(
            batch_size, drop_remainder=False).map(parser)
        dataset_filter_by_file_int64 = dataset_filter_by_file_int64.batch(
            batch_size, drop_remainder=False).map(parser)

        try:
          # test for filter by not-in list
          it = tf.compat.v1.data.make_one_shot_iterator(dataset_filter_by_list)
          element = it.get_next()
          element_result = sess.run(element)
          self.assertAllEqual(len(element_result['did']), 2)
          self.assertAllEqual(element_result['did'], [[b'300'], [b'400']])
          # test for filter by not-in file
          it = tf.compat.v1.data.make_one_shot_iterator(
              dataset_filter_by_file_string)
          element = it.get_next()
          element_result = sess.run(element)
          self.assertAllEqual(len(element_result['did']), 2)
          self.assertAllEqual(element_result['did'], [[b'300'], [b'400']])
          # test for filter by in file
          it = tf.compat.v1.data.make_one_shot_iterator(
              dataset_filter_by_file_int64)
          element = it.get_next()
          element_result = sess.run(element)
          self.assertAllEqual(len(element_result['did']), 2)
          self.assertAllEqual(element_result['did'], [[b'world'], [b'300']])
        except tf.errors.OutOfRangeError:
          self.assertTrue(False)

    os.remove(file_name)
    os.remove(tmp_filter_values_file_string)
    os.remove(tmp_filter_values_file_int64)

  def test_map_id(self):
    inputs = tf.constant([123, 456, 789, 912], dtype=tf.int32)
    map_dict = {123: 0, 456: 1, 789: 2}
    config = tf.compat.v1.ConfigProto()
    config.graph_options.rewrite_options.disable_meta_optimizer = True
    with self.session(config=config) as sess:
      out_ts = map_id(tensor=inputs, map_dict=map_dict)
      out = sess.run(out_ts)
      self.assertListEqual(list(out), [0, 1, 2, -1])

  def test_filter_by_fids(self):
    mock_batch_num = 1
    batch_size = 4

    def mock_instance(batch_num: int = 200):
      tmpfile = tempfile.mkstemp()[1]
      with io.open(tmpfile, 'wb') as writer:
        for _ in range(batch_num):
          for i in range(batch_size + 1):
            instance = generate_instance(
                [], [],
                fid_v1_list=[get_fid_v1(2, i),
                             get_fid_v1(3, i)] if i > 0 else [get_fid_v1(2, i)])
            write_instance_into_file(writer, instance)
      return tmpfile

    file_name = mock_instance(mock_batch_num)
    logging.info('file_name: %s', file_name)

    def parser(tensor: tf.Tensor):
      return parse_instances(tensor, fidv1_features=[2, 3])

    with tf.Graph().as_default():
      config = tf.compat.v1.ConfigProto()
      config.graph_options.rewrite_options.disable_meta_optimizer = True
      with self.session(config=config) as sess:
        dataset = PBDataset(file_name=file_name,
                            lagrangex_header=False,
                            has_sort_id=True,
                            kafka_dump=False,
                            kafka_dump_prefix=False,
                            input_pb_type=PbType.INSTANCE,
                            output_pb_type=PbType.INSTANCE)
        dataset = dataset.filter(lambda variant: filter_by_fids(
            variant, select_slots=[2, 3], variant_type='instance'))

        dataset = dataset.batch(batch_size, drop_remainder=False).map(parser)
        it = tf.compat.v1.data.make_one_shot_iterator(dataset)

        try:
          element = it.get_next()
          element_result = sess.run(element)
          self.assertAllEqual(element_result['slot_2'].values,
                              [get_fid_v1(2, i + 1) for i in range(batch_size)])
          self.assertAllEqual(element_result['slot_3'].values,
                              [get_fid_v1(3, i + 1) for i in range(batch_size)])
        except tf.errors.OutOfRangeError:
          self.assertTrue(False)
    os.remove(file_name)

  def test_multi_label_gen(self):
    mock_batch_num = 1
    head_to_idx = {'ios': 3, 'wp': 1, 'android': 4, 'other': 0}

    def mock_instance_for_multi_label_gen(batch_num: int = 10):
      tmpfile = tempfile.mkstemp()[1]
      labels = [[1], [2], [3], [1]]
      actions = [[1, 2], [3], [2], [1]]
      chnids = [0, 100, 200, 300]
      device_types = ['ios', 'wp', 'android', 'ios']
      with io.open(tmpfile, 'wb') as writer:
        for _ in range(batch_num):
          for label, action, chnid, device_type in zip(labels, actions, chnids,
                                                       device_types):
            instance = generate_instance(label,
                                         action,
                                         chnid,
                                         device_type=device_type)
            write_instance_into_file(writer, instance)
      return tmpfile

    file_name = mock_instance_for_multi_label_gen(mock_batch_num)
    logging.info('file_name: %s', file_name)

    def parser(tensor: tf.Tensor):
      return parse_instances(tensor,
                             dense_features=['label'],
                             dense_feature_shapes=[5],
                             dense_feature_types=[tf.float32],
                             extra_features=[
                                 'uid', 'req_time', 'item_id', 'actions',
                                 'device_type'
                             ],
                             extra_feature_shapes=[1, 1, 1, 3, 1])

    with tf.Graph().as_default():
      config = tf.compat.v1.ConfigProto()
      config.graph_options.rewrite_options.disable_meta_optimizer = True
      with self.session(config=config) as sess:
        dataset = PBDataset(file_name=file_name,
                            lagrangex_header=False,
                            has_sort_id=True,
                            kafka_dump=False,
                            kafka_dump_prefix=False,
                            input_pb_type=PbType.INSTANCE,
                            output_pb_type=PbType.INSTANCE)
        dataset = dataset.map(
            lambda variant: multi_label_gen(variant,
                                            head_to_index=head_to_idx,
                                            head_field='device_type',
                                            use_origin_label=False,
                                            pos_actions=[3, 2],
                                            neg_actions=[1],
                                            action_priority='4,3,2,1,0',
                                            variant_type='instance'))

        batch_size = 4
        dataset = dataset.batch(batch_size, drop_remainder=False).map(parser)
        it = tf.compat.v1.data.make_one_shot_iterator(dataset)
        try:
          element = it.get_next()
          element_result = sess.run(element)
          self.assertAllClose(
              element_result['label'],
              [[
                  -3.4028235e+38, -3.4028235e+38, -3.4028235e+38, 1.0000000e+00,
                  -3.4028235e+38
              ],
               [
                   -3.4028235e+38, 1.0000000e+00, -3.4028235e+38,
                   -3.4028235e+38, -3.4028235e+38
               ],
               [
                   -3.4028235e+38, -3.4028235e+38, -3.4028235e+38,
                   -3.4028235e+38, 1.0000000e+00
               ],
               [
                   -3.4028235e+38, -3.4028235e+38, -3.4028235e+38,
                   0.0000000e+00, -3.4028235e+38
               ]])
        except tf.errors.OutOfRangeError:
          self.assertTrue(False)
    os.remove(file_name)
  '''

  def test_string_to_variant(self):
    insts = []
    has_header, lg_header_flag = True, False
    sort_id, kafka_dump, kafka_dump_prefix = True, False, True
    for i in range(10):
      inst = proto_parser_pb2.Instance()
      inst.fid.extend([i for i in range(1, 20)])
      inst.line_id.chnid = 1
      inst_str = inst.SerializeToString()
      if lg_header_flag:
        header = lg_header(None)
      else:
        header = sort_header(sort_id, kafka_dump, kafka_dump_prefix)
      if i == 3:
        inst_str = b''
      if has_header:
        data = struct.pack(f'<{len(header)}sQ{len(inst_str)}s', header,
                           len(inst_str), inst_str)
      else:
        data = inst_str
      insts.append(data)
    with tf.Graph().as_default():
      config = tf.compat.v1.ConfigProto()
      config.graph_options.rewrite_options.disable_meta_optimizer = True
      ips = tf.constant(value=insts, dtype=tf.string, shape=(10,), name='insts')
      ops = string_to_variant(ips,
                              variant_type='instance',
                              has_header=has_header,
                              lagrangex_header=lg_header_flag,
                              has_sort_id=sort_id,
                              kafka_dump=kafka_dump,
                              kafka_dump_prefix=kafka_dump_prefix,
                              chnids=[1, 2],
                              datasources=["1", "2"],
                              default_datasource='3')
      zeros = variant_to_zeros(ops)
      with self.session(config=config) as sess:
        element_result = sess.run(zeros)
    self.assertAllEqual(ips.shape, ops.shape)

  def test_has_variant(self):
    inst = proto_parser_pb2.Instance()
    inst.fid.extend([i for i in range(1, 20)])
    inst_str = inst.SerializeToString()
    data = struct.pack(f'<Q{len(inst_str)}s', len(inst_str), inst_str)

    with tf.Graph().as_default():
      config = tf.compat.v1.ConfigProto()
      config.graph_options.rewrite_options.disable_meta_optimizer = True
      ips = tf.constant(value=[data], dtype=tf.string, shape=tuple())
      ops = string_to_variant(ips,
                              variant_type='instance',
                              has_header=True,
                              lagrangex_header=False,
                              has_sort_id=False,
                              kafka_dump=False,
                              kafka_dump_prefix=False)
      out = has_variant(ops, variant_type='instance')
      with self.session(config=config) as sess:
        element_result = sess.run(out)
        self.assertTrue(element_result)


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
