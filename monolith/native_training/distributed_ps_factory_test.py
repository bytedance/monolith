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

os.environ["MONOLITH_WITH_HOROVOD"] = "True"

import tensorflow as tf

from monolith.native_training import distributed_ps_factory
from monolith.native_training import hash_filter_ops
from monolith.native_training import test_utils

import horovod.tensorflow as hvd


def _get_test_slot_to_config():
  config = test_utils.generate_test_hash_table_config(4, learning_rate=0.1)
  return {
      "1": config,
      "2": config,
  }


def _get_test_hash_filters(num):
  return hash_filter_ops.create_hash_filters(num, False)


class FactoryTest(tf.test.TestCase):
  # Since factory itself is very difficult to test. Here we just perform grammar check.

  def test_create_in_worker_multi_type_hash_table(self):
    hvd.init()
    distributed_ps_factory.create_in_worker_multi_type_hash_table(
        1, _get_test_slot_to_config(),
        _get_test_hash_filters(0)[0])

  def test_create_in_worker_multi_type_hash_table_with_reduced_alltoall(self):
    hvd.init()
    distributed_ps_factory.create_in_worker_multi_type_hash_table(
        1, _get_test_slot_to_config(),
        _get_test_hash_filters(0)[0])

  def test_create_multi_type_hash_table_0_ps(self):
    distributed_ps_factory.create_multi_type_hash_table(
        0, _get_test_slot_to_config(), _get_test_hash_filters(0))

  def test_create_multi_type_hash_table_2_ps(self):
    servers, config = test_utils.create_test_ps_cluster(2)
    with tf.compat.v1.Session(servers[0].target, config=config):
      distributed_ps_factory.create_multi_type_hash_table(
          2, _get_test_slot_to_config(), _get_test_hash_filters(2))

  def test_create_multi_type_hash_table_2_ps_with_reduced_packets(self):
    servers, config = test_utils.create_test_ps_cluster(2)
    with tf.compat.v1.Session(servers[0].target, config=config):
      distributed_ps_factory.create_multi_type_hash_table(
          2,
          _get_test_slot_to_config(),
          _get_test_hash_filters(2),
          reduce_network_packets=True)

  def test_create_native_multi_hash_table_0_ps(self):
    distributed_ps_factory.create_native_multi_hash_table(
        0, _get_test_slot_to_config(), _get_test_hash_filters(0))

  def test_create_native_multi_hash_table_2_ps(self):
    servers, config = test_utils.create_test_ps_cluster(2)
    with tf.compat.v1.Session(servers[0].target, config=config):
      distributed_ps_factory.create_native_multi_hash_table(
          2, _get_test_slot_to_config(), _get_test_hash_filters(2))


if __name__ == "__main__":
  tf.compat.v1.disable_v2_behavior()
  tf.test.main()
