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
import os
import getpass
import random
import numpy as np
import tensorflow as tf
from struct import pack, unpack
from datetime import datetime, timedelta

from idl.matrix.proto.proto_parser_pb2 import Instance
from monolith.native_training.data.parsers import parse_instances
from monolith.native_training.data.datasets import PBDataset, PbType

uids = [674432, 9754221, 7665435, 98797865, 778754432]
item_ids = [8767554565, 574220985, 65548979, 5358521231]
actions = [1, 2]
device_types = ['pc', 'mobile', 'cloud']
slots = [1, 200, 5, 7, 9]
NUM_INSTANCE = 4096
MODEL_DIR = os.path.join(os.environ["TEST_TMPDIR"], 'model_dir', 'multi_flow')


class MultiFlowTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    mask = (1 << 54) - 1
    start = int(datetime.now().timestamp())
    stop = int((datetime.now() + timedelta(days=1)).timestamp())
    if not tf.io.gfile.exists(MODEL_DIR):
      tf.io.gfile.makedirs(MODEL_DIR)
    ofile = os.path.join(MODEL_DIR, 'data.pb')
    print(ofile, flush=True)
    if not tf.io.gfile.exists(ofile):
      with tf.io.gfile.GFile(ofile, 'wb') as ostream:
        for _ in range(NUM_INSTANCE):
          inst = Instance()
          for slot in slots:
            h = random.randrange(start, stop)
            fid = (slot << 54) | (h & mask)
            inst.fid.append(fid)

          line_id = inst.line_id
          line_id.uid = random.choice(uids)
          line_id.item_id = random.choice(item_ids)
          line_id.req_time = random.randrange(start, stop)
          line_id.device_type = random.choice(device_types)
          line_id.actions.append(random.choice(actions))

          lgx_header = cls.mk_kgx_header(dataflow=line_id.device_type)
          data = inst.SerializeToString()

          ostream.write(file_content=lgx_header)
          ostream.write(file_content=pack(f'<Q', len(data)))
          ostream.write(file_content=data)

  @classmethod
  def tearDownClass(cls):
    if not tf.io.gfile.exists(MODEL_DIR):
      tf.io.gfile.rmtree(MODEL_DIR)

  @classmethod
  def mk_kgx_header(cls, dataflow: str):
    # calc java hash code
    seed, h = 31, 0
    for c in dataflow:
      h = np.int32(seed * h) + ord(c)

    dfhc = int(np.uint32(h)).to_bytes(4, 'little')
    return pack('4Bi', 0, dfhc[0], dfhc[1], dfhc[2], 0)

  def test_data_flow(self):
    ofile = os.path.join(MODEL_DIR, 'data.pb')
    dataset = PBDataset(file_name=ofile,
                        lagrangex_header=True,
                        input_pb_type=PbType.INSTANCE,
                        output_pb_type=PbType.INSTANCE)
    pc = dataset.split_flow(data_flow=device_types,
                            index=0,
                            variant_type='instance')
    mobile = dataset.split_flow(data_flow=device_types,
                                index=1,
                                variant_type='instance')
    cloud = dataset.split_flow(data_flow=device_types,
                               index=2,
                               variant_type='instance')

    dataset = pc.merge_flow(dataset_to_merge=[mobile, cloud],
                            variant_type='instance')

    def map_fn(tensor: tf.Tensor):
      features = parse_instances(tensor,
                                 fidv1_features=slots,
                                 extra_features=[
                                     'uid', 'item_id', 'req_time',
                                     'device_type', 'actions'
                                 ],
                                 extra_feature_shapes=[1, 1, 1, 1, 1])
      return features

    dataset = dataset.batch(batch_size=512, drop_remainder=True).map(
        map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    cnt = 0
    for feat in dataset:
      cnt += 1
    self.assertEqual(cnt, 8)


if __name__ == "__main__":
  tf.test.main()
