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

from copy import deepcopy

import tensorflow as tf
from google.protobuf import text_format

from monolith.native_training import entry
from monolith.native_training import embedding_combiners
from monolith.native_training import feature
from monolith.native_training import learning_rate_functions
from monolith.native_training.runtime.hash_table import \
    embedding_hash_table_pb2


def _default_learning_rate_fn():
  return learning_rate_functions.PolynomialDecay(initial_learning_rate=0.01,
                                                 decay_steps=20,
                                                 end_learning_rate=0.05)


class CollectingConfigTest(tf.test.TestCase):

  def test_basic(self):
    table = feature.DummyFeatureEmbTable(
        batch_size=4, hashtable_config=entry.CuckooHashTableConfig())
    seg = embedding_hash_table_pb2.EntryConfig.Segment()
    seg.dim_size = 5
    seg.opt_config.sgd.SetInParent()
    table.add_feature_slice(seg)
    table.set_feature_metadata("feature_name",
                               feature.FeatureColumn.reduce_sum())
    placeholder = table.embedding_lookup("feature_name", 0, 5)
    self.assertAllEqual(placeholder.shape, [4, 5])

  def test_basic_with_seq_features(self):
    table = feature.DummyFeatureEmbTable(
        batch_size=4, hashtable_config=entry.CuckooHashTableConfig())
    seg = embedding_hash_table_pb2.EntryConfig.Segment()
    seg.dim_size = 5
    seg.opt_config.sgd.SetInParent()
    table.add_feature_slice(seg)
    table.set_feature_metadata("feature_name",
                               feature.FeatureColumn.first_n(10))
    placeholder = table.embedding_lookup("feature_name", 0, 5)
    self.assertAllEqual(placeholder.shape, [4, 10, 5])

  def test_info(self):
    table = feature.DummyFeatureEmbTable(
        batch_size=4, hashtable_config=entry.CuckooHashTableConfig())
    entry1 = embedding_hash_table_pb2.EntryConfig.Segment()
    text_format.Parse(
        "dim_size: 5 opt_config { adagrad { warmup_steps: 10 } } ", entry1)
    table.add_feature_slice(deepcopy(entry1))
    entry2 = embedding_hash_table_pb2.EntryConfig.Segment()
    text_format.Parse("dim_size: 2 opt_config { sgd {} }", entry2)
    table.add_feature_slice(deepcopy(entry2),
                            learning_rate_fn=_default_learning_rate_fn())
    entry3 = embedding_hash_table_pb2.EntryConfig.Segment()
    text_format.Parse("dim_size: 2 opt_config { sgd {} }", entry3)
    table.add_feature_slice(deepcopy(entry3),
                            learning_rate_fn=_default_learning_rate_fn())
    table.add_feature_slice(deepcopy(entry3))
    table.set_feature_metadata("feature1", embedding_combiners.ReduceSum())
    table.embedding_lookup("feature1", 0, 2)
    config = table.get_table_config()
    slices = config.slice_configs
    self.assertEqual(len(slices), 3)
    self.assertEqual(slices[0].segment.SerializeToString(),
                     entry1.SerializeToString())
    self.assertIsInstance(slices[0].learning_rate_fn,
                          learning_rate_functions.LearningRateFunction)
    merged_entry = embedding_hash_table_pb2.EntryConfig.Segment()
    text_format.Parse("dim_size: 4 opt_config { sgd {} }", merged_entry)
    self.assertEqual(slices[1].segment.SerializeToString(),
                     merged_entry.SerializeToString())
    self.assertAllEqual(config.feature_names, ["feature1"])

  def test_factory(self):
    factory = feature.DummyFeatureFactory(5)
    slot_config = feature.FeatureSlotConfig(name="table_name")
    slot = factory.create_feature_slot(slot_config)
    s = slot.add_feature_slice(5)
    fc1 = feature.FeatureColumnV1(slot, "feature1")
    fc1.embedding_lookup(s)
    fc2 = feature.FeatureColumnV1(slot, "feature2")
    fc2.embedding_lookup(s)
    table_name_to_config = factory.get_table_name_to_table_config()
    self.assertTrue("table_name" in table_name_to_config)
    table_config = table_name_to_config["table_name"]
    self.assertSetEqual(set(table_config.feature_names),
                        set(["feature1", "feature2"]))
    self.assertEqual(table_config.slice_configs[0].segment.dim_size, 5)

  def test_factory_with_seq_features(self):
    factory = feature.DummyFeatureFactory(5)
    slot_config = feature.FeatureSlotConfig(name="table_name")
    slot = factory.create_feature_slot(slot_config)
    s = slot.add_feature_slice(5)
    fc1 = feature.FeatureColumnV1(slot,
                                  "feature1",
                                  combiner=embedding_combiners.FirstN(5))
    fc1.embedding_lookup(s)
    fc2 = feature.FeatureColumnV1(slot,
                                  "feature2",
                                  combiner=embedding_combiners.FirstN(10))
    fc2.embedding_lookup(s)
    table_name_to_config = factory.get_table_name_to_table_config()
    self.assertTrue("table_name" in table_name_to_config)
    table_config = table_name_to_config["table_name"]
    self.assertSetEqual(set(table_config.feature_names),
                        set(["feature1", "feature2"]))
    self.assertDictEqual(table_config.feature_to_combiners, {
        "feature1": fc1.combiner,
        "feature2": fc2.combiner
    })
    self.assertEqual(table_config.slice_configs[0].segment.dim_size, 5)

  def test_factory_with_slot_occurrence_threshold(self):
    factory = feature.DummyFeatureFactory(5)
    slot_config_1 = feature.FeatureSlotConfig(name="table_name_1",
                                              slot_id=1,
                                              occurrence_threshold=3)
    slot_1 = factory.create_feature_slot(slot_config_1)
    s_1 = slot_1.add_feature_slice(5)
    fc1 = feature.FeatureColumnV1(slot_1, "feature1")
    fc1.embedding_lookup(s_1)

    slot_config_2 = feature.FeatureSlotConfig(name="table_name_2",
                                              slot_id=2,
                                              occurrence_threshold=7)
    slot_2 = factory.create_feature_slot(slot_config_2)
    s_2 = slot_2.add_feature_slice(5)
    fc2 = feature.FeatureColumnV1(slot_2, "feature2")
    fc2.embedding_lookup(s_2)
    self.assertEqual(factory.slot_to_occurrence_threshold[1], 3)
    self.assertEqual(factory.slot_to_occurrence_threshold[2], 7)

  def test_factory_with_applying_gradients(self):
    factory = feature.DummyFeatureFactory(5)
    slot_config = feature.FeatureSlotConfig(name="table")
    slot = factory.create_feature_slot(slot_config)
    s = slot.add_feature_slice(1)
    fc = feature.FeatureColumnV1(slot, "feature1")
    concat_embedding = fc.get_all_embeddings_concat()
    factory.apply_gradients([(tf.constant([[0.0] * 2] * 5), concat_embedding)])

  def test_bias(self):
    factory = feature.DummyFeatureFactory(5)
    slot_config = feature.FeatureSlotConfig(name="table", has_bias=True)
    slot = factory.create_feature_slot(slot_config)
    fc = feature.FeatureColumnV1(slot, "feature1")
    fc.get_bias()


class EmbeddingTest(tf.test.TestCase):

  def test_factory(self):
    embeddings = {"feature1": tf.constant([[1, 4], [2, 3]], dtype=tf.float32)}
    embedding_ids = {
        "feature1": tf.RaggedTensor.from_row_splits([1, 2], [0, 1, 2])
    }
    slices = feature.create_embedding_slices(
        embeddings, embedding_ids,
        {"feature1": embedding_combiners.ReduceSum()}, {"feature1": [1, 1]})
    factory = feature.FeatureFactoryFromEmbeddings(embeddings, slices)
    slot_config = feature.FeatureSlotConfig(name="table_name")
    slot = factory.create_feature_slot(slot_config)
    s = slot.add_feature_slice(1)
    fc = feature.FeatureColumnV1(slot, "feature1")
    tensor = fc.embedding_lookup(s)
    with self.session() as sess:
      tensor = sess.run(tensor)
    self.assertAllEqual(tensor, [[1], [2]])

  def test_factory_with_seq_features(self):
    embeddings = {
        "feature1":
            tf.constant([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=tf.float32)
    }
    embedding_ids = {
        "feature1": tf.RaggedTensor.from_row_splits([1, 2, 3, 4], [0, 2, 4])
    }
    slices = feature.create_embedding_slices(
        embeddings, embedding_ids, {"feature1": embedding_combiners.FirstN(2)},
        {"feature1": [1, 1]})
    factory = feature.FeatureFactoryFromEmbeddings(embeddings, slices)
    slot_config = feature.FeatureSlotConfig(name="table_name")
    slot = factory.create_feature_slot(slot_config)
    s = slot.add_feature_slice(1)
    fc = feature.FeatureColumnV1(slot, "feature1")
    tensor = fc.embedding_lookup(s)
    with self.session() as sess:
      tensor = sess.run(tensor)
    self.assertAllEqual(tensor, [[[1], [3]], [[5], [7]]])

  def test_fused_factory(self):
    embeddings = {
        "feature1": tf.constant([[1, 2], [2, 3], [3, 5]], dtype=tf.float32)
    }
    embedding_ids = {
        "feature1": tf.RaggedTensor.from_row_splits([1, 2, 3], [0, 1, 1, 3])
    }
    slices = feature.create_embedding_slices(
        embeddings, embedding_ids,
        {"feature1": embedding_combiners.ReduceSum()}, {"feature1": [1, 1]})
    factory = feature.FeatureFactoryFromEmbeddings(embeddings, slices)
    slot_config = feature.FeatureSlotConfig(name="table_name")
    slot = factory.create_feature_slot(slot_config)
    s = slot.add_feature_slice(1)
    s2 = slot.add_feature_slice(1)
    fc = feature.FeatureColumnV1(slot, "feature1")
    tensor = fc.embedding_lookup(s)
    with self.session() as sess:
      tensor = sess.run(tensor)
    self.assertAllClose(tensor, [[1], [0], [5]])
    tensor = fc.embedding_lookup(s2)
    with self.session() as sess:
      tensor = sess.run(tensor)
    self.assertAllClose(tensor, [[2], [0], [8]])

  def test_fused_factory_with_seq_features_larger_than_max_seq_length(self):
    # For rows with bigger number of embeddings than max_seq_length,
    # discard the extra embedding elements.
    embeddings = {
        "feature1":
            tf.constant([[1, 2], [2, 3], [3, 5], [10, 11]], dtype=tf.float32)
    }
    embedding_ids = {
        "feature1": tf.RaggedTensor.from_row_splits([1, 2, 3, 4], [0, 1, 1, 4])
    }
    ragged_ids = embedding_ids["feature1"]
    slices = feature.create_embedding_slices(
        embeddings, embedding_ids, {"feature1": embedding_combiners.FirstN(2)},
        {"feature1": [1, 1]})
    factory = feature.FeatureFactoryFromEmbeddings(embeddings, slices)
    slot_config = feature.FeatureSlotConfig(name="table_name")
    slot = factory.create_feature_slot(slot_config)
    s = slot.add_feature_slice(1)
    s2 = slot.add_feature_slice(1)
    fc = feature.FeatureColumnV1(slot, "feature1")
    tensor = fc.embedding_lookup(s)
    with self.session() as sess:
      tensor = sess.run(tensor)
    self.assertAllEqual(tensor, [[[1], [0]], [[0], [0]], [[2], [3]]])
    tensor = fc.embedding_lookup(s2)
    with self.session() as sess:
      tensor = sess.run(tensor)
    self.assertAllEqual(tensor, [[[2], [0]], [[0], [0]], [[3], [5]]])


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
