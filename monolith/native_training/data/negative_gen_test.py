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
from random import randint, choice, random

from absl import logging
import numpy as np
from struct import unpack, pack

from monolith.native_training.data.datasets import PBDataset, InstanceReweightDataset, NegativeGenDataset, PbType
from monolith.native_training.data.parsers import parse_instances, parse_examples, parse_example_batch
from monolith.native_training.data.feature_utils import filter_by_fids, filter_by_value, negative_sample, \
  switch_slot, feature_combine, special_strategy
from idl.matrix.proto.example_pb2 import Example
from idl.matrix.proto.proto_parser_pb2 import Instance
from idl.matrix.proto.line_id_pb2 import LineId

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

variant_type = 'instance'
channel_feature = 'f_spm_1'
channel_slot = 301
index_feature = 'f_goods_id'
index_slot = 200
user_id = 'f_user_id'
if variant_type == 'example':
  item_features = [name for name, slot in features.items() if 'goods' in name]
else:
  item_features = [slot for name, slot in features.items() if 'goods' in name]
pos_acts, neg_acts = [1, 2], [3, 4]
num_sample, start_num, neg_num = 1000, 10, 5
cache_only_pos, throw_origin, throw_origin_neg = True, False, False
per_channel = True


def parser(tensor: tf.Tensor):
  if variant_type == 'instance':
    feature_dict = parse_instances(
        tensor,
        fidv1_features=list(features.values()),
        dense_features=['label'],
        dense_feature_shapes=[1],
        dense_feature_types=[tf.float32],
        extra_features=['uid', 'req_time', 'item_id'],
        extra_feature_shapes=[1, 1, 1])
  else:
    feature_dict = parse_examples(
        tensor,
        sparse_features=list(features.keys()),
        dense_features=['label'],
        dense_feature_shapes=[1],
        dense_feature_types=[tf.float32],
        extra_features=['uid', 'req_time', 'item_id', 'actions'],
        extra_feature_shapes=[1, 1, 1, 1])
  return feature_dict


class NegativeGenTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    offset = 48 if variant_type == 'example' else 54
    cids = [(channel_slot << offset) + randint(1, 1 << 32) for _ in range(10)]
    gids = [(index_slot << offset) + randint(1, 1 << 32) for _ in range(100)]
    cls.cid_status, cls.gid_status = {cid: {
        'p': 0,
        'n': 0
    } for cid in cids}, {gid: 0 for gid in gids}

    with open(f'{variant_type}.pb', 'wb') as ostream:
      for _ in range(num_sample):
        uid, gid, cid = None, None, None
        if variant_type == 'example':
          sample = Example()
          for name, slot in features.items():
            named_feature = sample.named_feature.add()
            named_feature.id = slot
            named_feature.name = name
            if name == channel_feature:
              cid = choice(cids)
              fid = cid
            elif name == index_feature:
              gid = choice(gids)
              cls.gid_status[gid] += 1
              fid = gid
            elif name == user_id:
              uid = (slot << offset) + randint(1, 1 << 32)
              fid = uid
            else:
              fid = (slot << offset) + randint(1, 1 << 32)
            named_feature.feature.fid_v2_list.value.append(fid)
        else:
          sample = Instance()
          for name, slot in features.items():
            if name == channel_feature:
              cid = choice(cids)
              fid = cid
            elif name == index_feature:
              gid = choice(gids)
              cls.gid_status[gid] += 1
              fid = gid
            elif name == user_id:
              uid = (slot << offset) + randint(1, 1 << 32)
              fid = uid
            else:
              fid = (slot << offset) + randint(1, 1 << 32)
            sample.fid.append(fid)

        line_id = LineId(uid=uid, item_id=gid, req_time=int(time.time()))
        label = []
        if random() > 0.5:
          line_id.actions.append(choice(pos_acts))
          label.append(1)
          cls.cid_status[cid]['p'] += 1
        else:
          line_id.actions.append(choice(neg_acts))
          label.append(-1)
          cls.cid_status[cid]['n'] += 1

        if variant_type == 'example':
          label_nf = sample.named_feature.add()
          label_nf.name = '__LABEL__'
          label_nf.feature.float_list.value.extend(label)
          lid = sample.named_feature.add()
          lid.name = '__LINE_ID__'
          lid.feature.bytes_list.value.append(line_id.SerializeToString())
        else:
          sample.label.extend(label)
          sample.line_id.CopyFrom(line_id)

        es = sample.SerializeToString()
        ostream.write(pack('<QQ', 0, len(es)))
        ostream.write(es)

    print(cls.cid_status, cls.gid_status)

  @classmethod
  def tearDownClass(cls):
    if tf.io.gfile.exists(f'{variant_type}.pb'):
      tf.io.gfile.remove(f'{variant_type}.pb')

  def test_dataset_target(self):
    with tf.Graph().as_default():
      config = tf.compat.v1.ConfigProto()
      config.graph_options.rewrite_options.disable_meta_optimizer = True
      with self.session(config=config) as sess:
        pb_type = PbType.EXAMPLE if variant_type == 'example' else PbType.INSTANCE
        dataset = PBDataset(file_name=f'{variant_type}.pb',
                            lagrangex_header=True,
                            input_pb_type=pb_type,
                            output_pb_type=pb_type)
        dataset = dataset.negative_gen(
            neg_num=neg_num,
            per_channel=per_channel,
            start_num=start_num,
            max_item_num=1000,
            cache_only_pos=cache_only_pos,
            channel_feature=channel_feature
            if variant_type == 'example' else channel_slot,
            item_features=item_features,
            throw_origin=throw_origin,
            throw_origin_neg=throw_origin_neg,
            variant_type=variant_type)
        dataset = dataset.batch(8, drop_remainder=False).map(parser)
        it = tf.compat.v1.data.make_initializable_iterator(dataset)
        element = it.get_next()
        sess.run(it.initializer)
        count, pos_cnt, neg_cnt = 0, 0, 0
        real_cids = {cid: {'p': 0, 'n': 0} for cid in self.cid_status}
        channel_feature_name = channel_feature if variant_type == 'example' else f'slot_{channel_slot}'
        while True:
          try:
            element_out = sess.run(element)
            # print(element_out, flush=True)
            pos = element_out['label'] > 0
            neg = element_out['label'] < 0
            pos_cnt += np.sum(pos)
            neg_cnt += np.sum(neg)
            count += element_out['label'].shape[0]

            for cid in self.cid_status:
              select_channel = element_out[channel_feature_name] == cid
              np.sum(np.logical_and(select_channel, pos))
              np.sum(np.logical_and(select_channel, neg))

          except tf.errors.OutOfRangeError:
            break
        self.assertEqual(count, pos_cnt + neg_cnt)

        expect_pos, expect_neg = 0, 0
        for pn_dict in self.cid_status.values():
          expect_pos += pn_dict['p']
          expect_neg += pn_dict['n']

        self.assertEqual(expect_pos + expect_neg, num_sample)
        if not throw_origin:
          self.assertEqual(pos_cnt, expect_pos)
        else:
          self.assertEqual(pos_cnt, 0)

        if not throw_origin and not throw_origin_neg:
          if per_channel:
            pass
          else:
            min_gen = (expect_pos - start_num) * neg_num
            max_gen = expect_pos * neg_num
            real_gen = count - num_sample
            self.assertTrue(min_gen <= real_gen <= max_gen)

        print(count, pos_cnt, neg_cnt, expect_pos, expect_neg, flush=True)
        logging.info("The number of batch is: {}".format(count))


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
