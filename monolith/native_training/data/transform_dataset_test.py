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
import time

from absl import logging
import os
import uuid
import struct
import tensorflow as tf
import tempfile
from typing import List, BinaryIO
from idl.matrix.proto import proto_parser_pb2, example_pb2
from monolith.native_training.data.datasets import PBDataset, PbType
from monolith.native_training.data.parsers import (parse_instances,
                                                   parse_examples)
from monolith.native_training.data.transform import transforms

fid_v1_mask = (1 << 54) - 1
fid_v2_mask = (1 << 48) - 1


def get_fid_v1(slot: int, signautre: int):
  return (slot << 54) | (signautre & fid_v1_mask)


def get_fid_v2(slot: int, signature: int):
  return (slot << 48) | (signature & fid_v2_mask)


def mock_instance_line_id(index: int, instance, actions: List[int]):
  instance.line_id.user_id = "test_{}".format(uuid.uuid4())
  instance.line_id.uid = 100
  instance.line_id.read_count = 0 if 20 <= index < 40 else 1
  instance.line_id.video_play_time = 0.0 if 20 <= index < 30 else 1.0
  instance.line_id.req_time = int(time.time())
  instance.line_id.sample_rate = 0.5
  instance.line_id.actions.extend(actions)


def generate_instance_or_example(variant_type: str,
                                 index: int,
                                 labels: List[int],
                                 actions: List[int],
                                 fid_v1_list: List[int] = None):
  assert variant_type in {"instance", "example"}
  if variant_type == "instance":
    instance = proto_parser_pb2.Instance()
    instance.fid.extend(fid_v1_list if fid_v1_list else [])
  else:
    instance = example_pb2.Example()
    named_feature = instance.named_feature.add()
    named_feature.name = "fc_slot_1"
    named_feature.feature.fid_v1_list.value.extend(fid_v1_list)

  instance.label.extend(labels)
  mock_instance_line_id(index, instance, actions)
  return instance


def write_instance_into_file(file: BinaryIO, instance):
  sort_id = str(instance.line_id.user_id)
  file.write(struct.pack('<Q', len(sort_id)))
  file.write(sort_id.encode())
  instance_serialized = instance.SerializeToString()
  file.write(struct.pack('<Q', len(instance_serialized)))
  file.write(instance_serialized)


class DataOpsTest(tf.test.TestCase):

  def mock_instance_or_example(self, variant_type: str, batch_num: int,
                               batch_size: int):
    tmpfile = tempfile.mkstemp()[1]
    with io.open(tmpfile, 'wb') as writer:
      for i in range(batch_num * batch_size):
        instance = generate_instance_or_example(
            variant_type=variant_type,
            index=i,
            labels=[i, 1 if i <= 2 else 0],
            actions=[2 if 30 <= i < 35 else 0],
            fid_v1_list=[get_fid_v1(2, i), get_fid_v1(3, i)]
            if i < 10 else [290956192322012601])
        write_instance_into_file(writer, instance)
    return tmpfile

  def instance_or_example_test(self, variant_type: str):
    mock_batch_num = 10
    batch_size = 4
    file_name = self.mock_instance_or_example(variant_type, mock_batch_num,
                                              batch_size)
    logging.info('file_name: %s', file_name)

    def parser(tensor: tf.Tensor):
      if variant_type == "instance":
        return parse_instances(tensor,
                               fidv1_features=[2, 3],
                               dense_features=['label'],
                               dense_feature_shapes=[2])
      else:
        return parse_examples(tensor,
                              sparse_features=["fc_slot_1"],
                              dense_features=['label'],
                              dense_feature_shapes=[2])

    with tf.Graph().as_default():
      config = tf.compat.v1.ConfigProto()
      config.graph_options.rewrite_options.disable_meta_optimizer = True
      with self.session(config=config) as sess:
        dataset = PBDataset(file_name=file_name,
                            lagrangex_header=False,
                            has_sort_id=True,
                            kafka_dump=False,
                            kafka_dump_prefix=False,
                            input_pb_type=PbType.INSTANCE
                            if variant_type == 'instance' else PbType.EXAMPLE,
                            output_pb_type=PbType.INSTANCE
                            if variant_type == 'instance' else PbType.EXAMPLE)

        dataset = dataset.transform(t=transforms.Compose([
            transforms.FilterByFid(select_fids=[290956192322012601]),
            transforms.FilterByValue(field_name='read_count',
                                     op='in',
                                     operand=[0]),
            transforms.LogicalOr(
                transforms.FilterByValue(field_name='video_play_time',
                                         op='eq',
                                         operand=0.0),
                transforms.FilterByAction(has_actions=[2]))
        ]),
                                    variant_type=variant_type)
        dataset = dataset.batch(batch_size, drop_remainder=False).map(parser)
        it = tf.compat.v1.data.make_one_shot_iterator(dataset)

        total_count = 0
        while True:
          try:
            element = it.get_next()
            element_result = sess.run(element)
            logging.info('element: %s', element_result)
            total_count += element_result["label"].shape[0]
          except tf.errors.OutOfRangeError:
            break
        self.assertEqual(total_count, 15)
    os.remove(file_name)

  def test_instance(self):
    self.instance_or_example_test(variant_type='instance')

  def test_example(self):
    self.instance_or_example_test(variant_type='example')


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
