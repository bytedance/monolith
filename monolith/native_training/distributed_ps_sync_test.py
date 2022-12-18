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
from absl.testing import parameterized

os.environ["MONOLITH_WITH_HOROVOD"] = "True"

import tensorflow as tf

from monolith.native_training import distributed_ps
from monolith.native_training import distributed_ps_sync
from monolith.native_training import hash_table_ops
from monolith.native_training import learning_rate_functions
from monolith.native_training import multi_type_hash_table
from monolith.native_training import test_utils
from monolith.native_training.multi_hash_table_ops import MultiHashTable

import horovod.tensorflow as hvd


def gen_test_configs():
  return {
      "1":
          test_utils.generate_test_hash_table_config(1, learning_rate=1.0),
      "2":
          test_utils.generate_test_hash_table_config(
              2,
              learning_rate=learning_rate_functions.PolynomialDecay(
                  initial_learning_rate=1.0,
                  decay_steps=10,
                  end_learning_rate=2.0))
  }


def multi_type_table_factory(idx: int):

  def table_factory(name_suffix: str, config):
    return hash_table_ops.hash_table_from_config(config,
                                                 name_suffix=name_suffix +
                                                 str(idx))

  return multi_type_hash_table.MultiTypeHashTable(gen_test_configs(),
                                                  table_factory)


def native_multi_hash_table_factory(idx: int):
  return MultiHashTable.from_configs(configs=gen_test_configs(),
                                     name_suffix=str(idx))


class DistributedMultiTypeHashTableMpiTest(tf.test.TestCase,
                                           parameterized.TestCase):

  @parameterized.parameters([(False,)])
  def testBasic(self, use_native_multi_hash_table):
    table_factory = (native_multi_hash_table_factory if
                     use_native_multi_hash_table else multi_type_table_factory)
    hvd.init()
    with self.session() as sess:
      global_step = tf.compat.v1.train.get_or_create_global_step()
      self.evaluate(tf.compat.v1.global_variables_initializer())
      self.evaluate(tf.compat.v1.assign(global_step, 0))
      table = distributed_ps_sync.DistributedMultiTypeHashTableMpi(
          hvd.size(), table_factory)
      mulitplier = hvd.rank() + 1
      slot_to_ids = {
          "1": tf.constant([1, 1], dtype=tf.int64),
          "2": tf.constant([2], dtype=tf.int64)
      }
      # First lookup, nothing exists, returns 0 simply.
      emb, auxiliary_bundle = table.lookup(slot_to_ids, auxiliary_bundle={})
      emb_value = sess.run(emb)
      self.assertAllClose(emb_value["1"], [[0], [0]])
      self.assertAllClose(emb_value["2"], [[0, 0]])
      updated_table = table.apply_gradients(
          {
              "1": tf.constant([[0.5], [0.5]], dtype=tf.float32),
              "2": tf.constant([[0.5, 1.0]], dtype=tf.float32)
          },
          auxiliary_bundle=auxiliary_bundle,
          global_step=tf.constant(0, dtype=tf.int64),
          req_time=tf.constant(0, dtype=tf.int64))
      emb, auxiliary_bundle = updated_table.lookup(slot_to_ids,
                                                   auxiliary_bundle={})
      emb_value = sess.run(emb)
      sum_multiplier = hvd.size()
      self.assertAllClose(emb_value["1"],
                          [[-1 * sum_multiplier], [-1 * sum_multiplier]])
      self.assertAllClose(emb_value["2"],
                          [[-0.5 * sum_multiplier, -1.0 * sum_multiplier]])


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
