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

import unittest

from monolith.native_training import entry
from monolith.native_training import learning_rate_functions
from monolith.native_training.runtime.hash_table import \
    embedding_hash_table_pb2


def _default_learning_rate_fn():
  return learning_rate_functions.PolynomialDecay(initial_learning_rate=0.01,
                                                 decay_steps=20,
                                                 end_learning_rate=0.05)


class EntryTest(unittest.TestCase):
  """The tests here are for testing complilation."""

  def test_optimizers(self):
    entry.SgdOptimizer(0.01).as_proto()
    entry.AdagradOptimizer(0.01, 0.1).as_proto()
    entry.AdagradOptimizer(0.01, 0.1, 10).as_proto()
    entry.FtrlOptimizer(0.01, 0.1, 1).as_proto()
    entry.DynamicWdAdagradOptimizer(0.01, 0.1, 1).as_proto()
    entry.AdadeltaOptimizer(0.01, 0.0, 0.9, 0.01).as_proto()
    entry.AdamOptimizer(0.01, 0.9, 0.99, False, 0.0, False, 0.01).as_proto()
    entry.AmsgradOptimizer(0.01, 0.9, 0.99, 0.0, False, 0.01).as_proto()
    entry.MomentumOptimizer(0.01, 0.0, False, 0.9).as_proto()
    entry.MovingAverageOptimizer(0.9).as_proto()
    entry.RmspropOptimizer(0.01, 0.0, 0.9).as_proto()
    entry.RmspropV2Optimizer(0.01, 0.0, 0.9).as_proto()
    entry.BatchSoftmaxOptimizer(0.01).as_proto()

  def test_initializer(self):
    entry.ZerosInitializer().as_proto()
    entry.RandomUniformInitializer(-0.5, 0.5).as_proto()
    entry.BatchSoftmaxInitializer(1.0).as_proto()

  def test_compressor(self):
    entry.Fp16Compressor().as_proto()
    entry.Fp32Compressor().as_proto()
    entry.FixedR8Compressor().as_proto()
    entry.OneBitCompressor().as_proto()

  def test_combine(self):
    entry.CombineAsSegment(5, entry.ZerosInitializer(), entry.SgdOptimizer(),
                           entry.Fp16Compressor())

  def test_hashtable_config(self):
    entry.CuckooHashTableConfig()

  def test_hashtable_config_entrance(self):
    table_config1 = embedding_hash_table_pb2.EmbeddingHashTableConfig()
    config1 = entry.HashTableConfigInstance(table_config1, [0.1])
    table_config2 = embedding_hash_table_pb2.EmbeddingHashTableConfig()
    config2 = entry.HashTableConfigInstance(table_config2, [0.1])
    assert (str(config1) == str(config2))

    table_config3 = embedding_hash_table_pb2.EmbeddingHashTableConfig()
    config3 = entry.HashTableConfigInstance(table_config3,
                                            [_default_learning_rate_fn()])
    table_config4 = embedding_hash_table_pb2.EmbeddingHashTableConfig()
    config4 = entry.HashTableConfigInstance(table_config4,
                                            [_default_learning_rate_fn()])
    assert (str(config3) == str(config4))

    assert (str(config1) != str(config3))


if __name__ == "__main__":
  unittest.main()
