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
import time
import tensorflow as tf

from absl import logging, flags
import numpy as np
from struct import unpack

from monolith.native_training.data.datasets import PBDataset, InstanceReweightDataset, PbType, \
  FilePBDataset, KafkaDataset
from monolith.native_training.data.parsers import parse_instances, parse_examples, parse_example_batch
from monolith.native_training.data.feature_utils import filter_by_fids, filter_by_value, negative_sample, \
  switch_slot, feature_combine, special_strategy
from idl.matrix.proto.example_pb2 import Example, ExampleBatch
from monolith.native_training.model_export.data_gen_utils import gen_random_data_file, ParserArgs
from tensorflow.python.framework import sparse_tensor
from monolith.native_training.estimator import RunConfig
from monolith.native_training.hooks import session_hooks

FLAGS = flags.FLAGS
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
    feature_dict = parse_examples(
        tensor,
        sparse_features=list(features.keys()),
        dense_features=['label'],
        dense_feature_shapes=[2],
        dense_feature_types=[tf.float32],
        extra_features=['uid', 'req_time', 'item_id', 'actions'],
        extra_feature_shapes=[1, 1, 1, 1])
    feature_dict['f_page'] = switch_slot(feature_dict['f_page'], slot=306)
    feature_dict['f_user_id-f_goods_tags_terms'] = feature_combine(
        feature_dict['f_user_id'], feature_dict['f_goods_tags_terms'], slot=505)
  return feature_dict


class DataOpsTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    cwd = os.getcwd()
    cls.patterns = [os.path.join(cwd, "tmp_data", "part-*")]
    cls._files = []
    args = ParserArgs(fidv1_features=[i for i in range(1, 10)],
                      extra_features=[
                          'uid', 'sample_rate', 'req_time', 'actions',
                          'stay_time'
                      ],
                      extra_feature_shapes=[1, 1, 1, 1, 1],
                      batch_size=16,
                      variant_type='instance')
    for i in range(3):
      tf.io.gfile.makedirs(os.path.join(cwd, "tmp_data"))
      file_name = os.path.join(cwd, "tmp_data", f"part-{i}")
      gen_random_data_file(file_name,
                           args,
                           num_batch=10,
                           sort_id=True,
                           kafka_dump_prefix=False)
      cls._files.append(file_name)

  @classmethod
  def tearDownClass(cls):
    for file_name in cls._files:
      tf.io.gfile.remove(file_name)

  def pb_dataset_target(self, input_pb_type, output_pb_type, filter_fn=None):
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
        it = tf.compat.v1.data.make_initializable_iterator(dataset)
        element = it.get_next()
        sess.run(it.initializer)
        count = 0
        while True:
          try:
            element_num = sess.run(element)
            # print(element_num)
            count += 8
          except tf.errors.OutOfRangeError:
            break
        logging.info("The number of batch is: {}".format(count))

  def testInstance2Instance(self):
    self.pb_dataset_target(input_pb_type=PbType.INSTANCE,
                           output_pb_type=PbType.INSTANCE)

  def testInstance2Example(self):
    self.pb_dataset_target(input_pb_type=PbType.INSTANCE,
                           output_pb_type=PbType.EXAMPLE)

  def testExample2Example(self):
    self.pb_dataset_target(input_pb_type=PbType.EXAMPLE,
                           output_pb_type=PbType.EXAMPLE)

  def testExample2Instance(self):
    self.pb_dataset_target(input_pb_type=PbType.EXAMPLE,
                           output_pb_type=PbType.INSTANCE)

  def testExampleBatch2Example(self):
    self.pb_dataset_target(input_pb_type=PbType.EXAMPLEBATCH,
                           output_pb_type=PbType.EXAMPLE)

  def testExampleBatch2Instance(self):
    self.pb_dataset_target(input_pb_type=PbType.EXAMPLEBATCH,
                           output_pb_type=PbType.INSTANCE)

  def testInstanceWithPBInstanceDataset(self):
    self.pb_dataset_target(input_pb_type=PbType.INSTANCE,
                           output_pb_type=PbType.PLAINTEXT)

  def testExampleWithPBInstanceDataset(self):
    self.pb_dataset_target(input_pb_type=PbType.EXAMPLE,
                           output_pb_type=PbType.PLAINTEXT)

  def testSetFilterInstance(self):
    self.pb_dataset_target(
        input_pb_type=PbType.EXAMPLEBATCH,
        output_pb_type=PbType.INSTANCE,
        filter_fn=lambda variant: filter_by_fids(variant, has_actions=[1, 2]))

  def testSetFilterExample(self):
    self.pb_dataset_target(
        input_pb_type=PbType.EXAMPLEBATCH,
        output_pb_type=PbType.EXAMPLE,
        filter_fn=lambda variant: filter_by_fids(
            variant, has_actions=[1, 2], variant_type='example'))

  def testValueFilterInstance(self):
    self.pb_dataset_target(input_pb_type=PbType.EXAMPLEBATCH,
                           output_pb_type=PbType.INSTANCE,
                           filter_fn=lambda variant: filter_by_value(
                               variant, "sample_rate", "ge", 0.8))

  def testValueFilterInInstance(self):
    self.pb_dataset_target(input_pb_type=PbType.EXAMPLEBATCH,
                           output_pb_type=PbType.INSTANCE,
                           filter_fn=lambda variant: filter_by_value(
                               variant, "chnid", "in", [0, 2, 5]))

  def testValueFilterEqInstance(self):
    self.pb_dataset_target(
        input_pb_type=PbType.EXAMPLEBATCH,
        output_pb_type=PbType.INSTANCE,
        filter_fn=lambda variant: filter_by_value(variant, "chnid", "eq", 0))

  def testValueFilterBewteenInstance(self):
    self.pb_dataset_target(input_pb_type=PbType.EXAMPLEBATCH,
                           output_pb_type=PbType.INSTANCE,
                           filter_fn=lambda variant: filter_by_value(
                               variant, "sample_rate", "between", [0.1, 0.9]))

  def testValueFilterStrInstance(self):
    self.pb_dataset_target(
        input_pb_type=PbType.EXAMPLEBATCH,
        output_pb_type=PbType.INSTANCE,
        filter_fn=lambda variant: filter_by_value(variant, "vid", "eq", 'scm'))

  def testValueFilterAnyInstance(self):
    self.pb_dataset_target(input_pb_type=PbType.EXAMPLEBATCH,
                           output_pb_type=PbType.INSTANCE,
                           filter_fn=lambda variant: filter_by_value(
                               variant, "actions", "any", [2, 5, 7]))

  def testValueFilterAllInstance(self):
    self.pb_dataset_target(input_pb_type=PbType.EXAMPLEBATCH,
                           output_pb_type=PbType.INSTANCE,
                           filter_fn=lambda variant: filter_by_value(
                               variant, "actions", "all", [2, 5, 7]))

  def testValueFilterDiffInstance(self):
    self.pb_dataset_target(input_pb_type=PbType.EXAMPLEBATCH,
                           output_pb_type=PbType.INSTANCE,
                           filter_fn=lambda variant: filter_by_value(
                               variant, "actions", "diff", [2, 5, 7]))

  def testSpecialStrategyInstance(self):
    self.pb_dataset_target(
        input_pb_type=PbType.EXAMPLEBATCH,
        output_pb_type=PbType.INSTANCE,
        filter_fn=lambda variant: special_strategy(
            variant, [2, 5, 7], "2:0.7:-1,5:0.9:1,4:0.2:0,7:1.0:1"))

  def testValueFilterExample(self):
    self.pb_dataset_target(
        input_pb_type=PbType.EXAMPLEBATCH,
        output_pb_type=PbType.EXAMPLE,
        filter_fn=lambda variant: filter_by_value(
            variant, "sample_rate", "ge", 0.8, variant_type='example'))

  def testExampleBatchPredScalar(self):
    eb = ExampleBatch()
    file_name = "monolith/native_training/data/training_instance/examplebatch.data"

    with tf.Graph().as_default():
      config = tf.compat.v1.ConfigProto()
      config.graph_options.rewrite_options.disable_meta_optimizer = True
      examples_placeholder = tf.compat.v1.placeholder(dtype=tf.string,
                                                      shape=(None,))
      parsed_results = parse_example_batch(
          examples_placeholder,
          sparse_features=list(features.keys()),
          dense_features=['label'],
          dense_feature_shapes=[2],
          dense_feature_types=[tf.float32],
          extra_features=['uid', 'req_time', 'item_id'],
          extra_feature_shapes=[1, 1, 1])

      with self.session(config=config) as sess:
        with open(file_name, 'rb') as stream:
          stream.read(8)  # strip lagrangex_header
          size = unpack("<Q", stream.read(8))[0]
          eb_str = stream.read(size)
          results = sess.run(fetches=[parsed_results],
                             feed_dict={examples_placeholder: [eb_str]})

  def testExampleBatchPredBatch(self):
    eb = ExampleBatch()
    file_name = "monolith/native_training/data/training_instance/examplebatch.data"

    with tf.Graph().as_default():
      config = tf.compat.v1.ConfigProto()
      config.graph_options.rewrite_options.disable_meta_optimizer = True
      examples_placeholder = tf.compat.v1.placeholder(dtype=tf.string)
      parsed_results = parse_example_batch(
          examples_placeholder,
          sparse_features=list(features.keys()),
          dense_features=['label'],
          dense_feature_shapes=[2],
          dense_feature_types=[tf.float32],
          extra_features=['uid', 'req_time', 'item_id'],
          extra_feature_shapes=[1, 1, 1])

      with self.session(config=config) as sess:
        with open(file_name, 'rb') as stream:
          stream.read(8)  # strip lagrangex_header
          size = unpack("<Q", stream.read(8))[0]
          eb_str = stream.read(size)
          results = sess.run(fetches=[parsed_results],
                             feed_dict={examples_placeholder: eb_str})

  def testPBDataset(self):
    self.assertTrue(isinstance(PBDataset(''), FilePBDataset))

    FLAGS.kafka_topics = 'abc,def'
    FLAGS.kafka_group_id = 'test'
    FLAGS.kafka_servers = 'test'
    self.assertTrue(isinstance(PBDataset(), KafkaDataset))
    self.assertTrue(
        isinstance(PBDataset(['ab'], group_id='c', servers='d'), KafkaDataset))
    FLAGS.kafka_topics = None
    FLAGS.kafka_group_id = None
    FLAGS.kafka_servers = None

  def _init_session(self):
    config = tf.compat.v1.ConfigProto()
    config.graph_options.rewrite_options.disable_meta_optimizer = True
    return tf.compat.v1.train.SingularMonitoredSession(
        hooks=[session_hooks.SetCurrentSessionHook()], config=config)

  def testCreateInstanceDatasetHdfs(self):
    with self.session() as sess:
      dataset = PBDataset(topics_or_files=self.patterns,
                          has_sort_id=True,
                          kafka_dump_prefix=False,
                          use_snappy=False)

      def parse(serialized: tf.Tensor):
        return parse_instances(serialized, fidv1_features=list(range(1, 10)))

      dataset = dataset.batch(16, drop_remainder=True).map(parse)
      it = tf.compat.v1.data.make_initializable_iterator(dataset)
      element = it.get_next()
      element_num = None
      self._init_session()
      sess.run(it.initializer)
      for _ in range(10):
        try:
          element_num = sess.run(element)
        except tf.errors.OutOfRangeError:
          break

  def testGenPatterns(self):
    patterns = PBDataset.gen_patterns(input_path='/abc',
                                      start_date='20220901',
                                      end_date='20220920',
                                      is_hourly=False)
    self.assertEqual(len(patterns), 19)


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
