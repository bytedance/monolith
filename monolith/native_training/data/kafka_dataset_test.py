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
import getpass
from absl import flags, app
from random import choice
from struct import pack
from kafka import KafkaProducer

import tensorflow as tf
from monolith.native_training.data.parsers import parse_instances
from monolith.native_training.model_export.data_gen_utils import gen_instance, FeatureMeta
from monolith.native_training.data.datasets import KafkaDataset
from monolith.native_training.data.feature_utils import add_label, filter_by_label

# flags.DEFINE_string('feature_list', None, 'string, feature_list')
flags.DEFINE_bool('lagrangex_header', False, 'bool, lagrangex_header')
flags.DEFINE_bool('sort_id', False, 'bool, sort_id')
flags.DEFINE_bool('kafka_dump', False, 'bool, kafka_dump')
flags.DEFINE_bool('kafka_dump_prefix', False, 'bool, kafka_dump_prefix')

flags.DEFINE_string('topic', 'foobar', 'string, topic')
flags.DEFINE_string('group_id', 'my_favorite_group', 'string, group_id')
flags.DEFINE_string('kafka_servers', 'localhost:9092', 'string, kafka_servers')
flags.DEFINE_bool('data_gen', False, 'bool, data_gen')
flags.DEFINE_integer('num_batch', 2, 'bool, num_batch')

FLAGS = flags.FLAGS
BATCH_SIZE = 128
USE_CLICK_HEAD = False
VALID_FNAMES = [
    1, 2, 3, 4, 5, 6, 7, 8, 81, 82, 83, 84, 86, 87, 88, 89, 92, 93, 110, 115,
    205, 208, 209, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312,
    313, 314, 315, 316, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511,
    512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 526, 527, 528,
    529, 530, 531, 532, 533, 534, 536, 537, 538, 540, 542, 543, 544, 549, 562,
    564, 565, 567, 568, 569, 573, 576, 577, 700, 701, 707, 708, 709, 710, 711,
    712, 719, 720, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811,
    812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826,
    828, 829, 830, 832, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844,
    845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859,
    860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874,
    875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889,
    890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 903, 905, 906,
    907, 908, 909, 910, 911, 912, 913, 914, 915, 918, 924, 925, 926, 927, 928,
    929, 930, 932, 933, 934, 935, 937, 938, 939, 940, 941, 942, 944, 946, 947,
    948, 949, 950, 951, 952, 954, 955, 956, 958, 959, 960, 961, 962, 963, 964,
    965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979,
    980, 981, 982, 983, 984, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1022
]


class KafkaDatasetTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    FLAGS.lagrangex_header = False
    FLAGS.sort_id = False
    FLAGS.kafka_dump = False

    if FLAGS.data_gen:
      producer = KafkaProducer(bootstrap_servers=FLAGS.kafka_servers)
      dense_features = [FeatureMeta(name='label', shape=4, dtype=tf.float32)]
      extra_features = [
          FeatureMeta(name='req_time', shape=1),
          FeatureMeta(name='uid', shape=1),
          FeatureMeta(name='sample_rate', shape=1)
      ]
      actions = [-7, -9, 75, -103, 74, 101, 102, -41]
      for i in range(FLAGS.num_batch):
        for j in range(BATCH_SIZE):
          inst = gen_instance(fidv1_features=VALID_FNAMES,
                              dense_features=dense_features,
                              extra_features=extra_features,
                              actions=actions)
          inst_str = inst.SerializeToString()
          fmt = f'<Q{len(inst_str)}s'
          producer.send(topic=FLAGS.topic,
                        value=pack(fmt, len(inst_str), inst_str))
      producer.close()

  def test_kafka_dataset(self):
    dataset = KafkaDataset(topics=FLAGS.topic,
                           group_id=FLAGS.group_id,
                           servers=FLAGS.kafka_servers,
                           message_poll_timeout=30000)
    # label(0): staytime, not used
    label_vec_size = 4

    # 4 tasks
    # task1: interact, task2: convert, task3: vr, task4: click
    if USE_CLICK_HEAD:
      add_label_config = '-7,-9:-41:0.3;75,-103,74:-41:0.3;101,102:-41:0.3;-41::0.1'
    else:
      add_label_config = '-7,-9:-41:0.3;75,-103,74:-41:0.3;101,102:-41:0.3'
    dataset = dataset.map(lambda variant: add_label(variant,
                                                    config=add_label_config,
                                                    negative_value=0,
                                                    new_sample_rate=1.0,
                                                    variant_type='instance'))

    dataset = dataset.shuffle(256)
    dataset = dataset.batch(batch_size=BATCH_SIZE, drop_remainder=False)

    def map_fn(tensor):
      features = parse_instances(
          tensor,
          fidv1_features=VALID_FNAMES,
          fidv2_features=None,
          dense_features=['label'],
          dense_feature_shapes=[label_vec_size],
          extra_features=['req_time', 'uid', 'sample_rate'],
          extra_feature_shapes=[1, 1, 1])

      if USE_CLICK_HEAD:
        (_, interact_label, convert_label, vr_label,
         clk_label) = tf.split(features['label'],
                               num_or_size_splits=label_vec_size,
                               axis=1)
        features['clk_label'] = tf.reshape(clk_label, shape=(-1,))
      else:
        (_, interact_label, convert_label,
         vr_label) = tf.split(features['label'],
                              num_or_size_splits=label_vec_size,
                              axis=1)
      features['interact_label'] = tf.reshape(interact_label, shape=(-1,))
      features['convert_label'] = tf.reshape(convert_label, shape=(-1,))
      features['vr_label'] = tf.reshape(vr_label, shape=(-1,))
      features['sample_rate'] = tf.reshape(features['sample_rate'], shape=(-1,))
      return features

    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    config = tf.compat.v1.ConfigProto()
    config.graph_options.rewrite_options.disable_meta_optimizer = True
    with tf.compat.v1.Session(config=config) as sess:
      it = tf.compat.v1.data.make_initializable_iterator(dataset)
      element = it.get_next()
      sess.run(it.initializer)
      for x in range(FLAGS.num_batch + 1):
        print(sess.run(fetches=element))


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
