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

import math
import itertools
from typing import Dict, List
import numpy as np
from absl.testing import parameterized
import logging
import tensorflow as tf

from monolith.native_training import distributed_ps
from monolith.native_training import hash_table_ops
from monolith.native_training.multi_hash_table_ops import MultiHashTable
from monolith.native_training import learning_rate_functions
from monolith.native_training import multi_type_hash_table
from monolith.native_training import utils
from monolith.native_training import test_utils
from monolith.native_training.model_export import export_context
from monolith.native_training import entry
from idl.matrix.proto.example_pb2 import FeatureConfigs, FeatureConfig, PoolingType, OutType, OutConfig
import monolith.native_training.embedding_combiners as embedding_combiners
from monolith.native_training.runtime.hash_table import embedding_hash_table_pb2
from monolith.native_training.data.feature_utils import string_to_variant
from idl.matrix.proto.example_pb2 import Example
from monolith.native_training.data.parsers import sharding_sparse_fids


def factory(idx: int, config):
  return hash_table_ops.hash_table_from_config(config=config,
                                               name_suffix=str(idx))


class DistributedHashTableTest(tf.test.TestCase):

  def test_basic(self):
    servers, config = test_utils.create_test_ps_cluster(2)
    table_config = test_utils.generate_test_hash_table_config(2)
    with tf.compat.v1.Session(servers[0].target, config=config) as sess:
      hash_table = distributed_ps.DistributedHashTable(2, table_config, factory)
      hash_table = hash_table.assign_add(
          tf.constant([1, 2, 3], dtype=tf.int64),
          tf.constant([[1, 1], [2, 2], [3, 3]], dtype=tf.float32))
      values = hash_table.lookup(tf.constant([1, 2, 3], dtype=tf.int64))
      values = sess.run(values)
    self.assertAllEqual(values, [[1, 1], [2, 2], [3, 3]])

  def test_assign(self):
    servers, config = test_utils.create_test_ps_cluster(2)
    table_config = test_utils.generate_test_hash_table_config(2)
    with tf.compat.v1.Session(servers[0].target, config=config) as sess:
      hash_table = distributed_ps.DistributedHashTable(2, table_config, factory)
      hash_table = hash_table.assign(
          tf.constant([1, 2, 3], dtype=tf.int64),
          tf.constant([[1, 1], [2, 2], [3, 3]], dtype=tf.float32))
      values1 = hash_table.lookup(tf.constant([1, 2, 3], dtype=tf.int64))

      # Ensure the second assign happens after the first lookup
      with tf.control_dependencies([values1]):
        hash_table = hash_table.assign(
            tf.constant([2, 3, 4], dtype=tf.int64),
            tf.constant([[1, 1], [2, 2], [3, 3]], dtype=tf.float32))
        values2 = hash_table.lookup(tf.constant([1, 2, 3, 4], dtype=tf.int64))
      values1, values2 = sess.run([values1, values2])
    self.assertAllEqual(values1, [[1, 1], [2, 2], [3, 3]])
    self.assertAllEqual(values2, [[1, 1], [1, 1], [2, 2], [3, 3]])

  def test_lookup_dedup(self):
    servers, config = test_utils.create_test_ps_cluster(2)
    table_config = test_utils.generate_test_hash_table_config(2)
    with tf.compat.v1.Session(servers[0].target, config=config) as sess:
      hash_table = distributed_ps.DistributedHashTable(2, table_config, factory)
      hash_table = hash_table.assign_add(
          tf.constant([1, 2, 3], dtype=tf.int64),
          tf.constant([[1, 1], [2, 2], [3, 3]], dtype=tf.float32))
      values = hash_table.lookup(tf.constant([1, 1, 3], dtype=tf.int64))
      values = sess.run(values)
    self.assertAllEqual(values, [[1, 1], [1, 1], [3, 3]])

  def test_apply_gradients(self):
    table_config = test_utils.generate_test_hash_table_config(1)
    g = tf.Graph()
    with g.as_default():
      hash_table = distributed_ps.DistributedHashTable(2, table_config, factory)
      ids = tf.constant([0, 1], dtype=tf.int64)
      values = hash_table.lookup(ids)
      loss = 2 * values
      grads = tf.gradients(loss, values)
      global_step = tf.constant(0, dtype=tf.int64)
      hash_table = hash_table.apply_gradients(ids, grads[0], global_step)

      new_values = hash_table.lookup(ids)

    servers, config = test_utils.create_test_ps_cluster(2)

    with tf.compat.v1.Session(servers[0].target, config=config,
                              graph=g) as sess:
      new_values = sess.run(new_values)
    self.assertAllEqual(new_values, [[-2], [-2]])

  def test_apply_gradients_with_learning_rate_function(self):
    table_config = test_utils.generate_test_hash_table_config(
        1,
        learning_rate=learning_rate_functions.PolynomialDecay(
            initial_learning_rate=1.0, decay_steps=10, end_learning_rate=2.0))

    g = tf.Graph()
    with g.as_default():
      global_step = tf.compat.v1.train.get_or_create_global_step()
      hash_table = distributed_ps.DistributedHashTable(2, table_config, factory)
      ids = tf.constant([0, 1], dtype=tf.int64)
      values = hash_table.lookup(ids)
      loss = 2 * values
      grads = tf.gradients(loss, values)
      hash_table = hash_table.apply_gradients(ids, grads[0], global_step)

      new_values = hash_table.lookup(ids)

    servers, config = test_utils.create_test_ps_cluster(2)

    with tf.compat.v1.Session(servers[0].target, config=config,
                              graph=g) as sess:
      self.evaluate(tf.compat.v1.global_variables_initializer())
      values_eval = sess.run(new_values)
      self.assertAllEqual(values_eval, [[-2], [-2]])

      self.evaluate(tf.compat.v1.assign_add(global_step, 1))
      values_eval = sess.run(new_values)
      self.assertAllClose(values_eval, [[-4.2], [-4.2]])

  def test_apply_gradients_with_duplicates(self):
    table_config = test_utils.generate_test_hash_table_config(1)
    g = tf.Graph()
    with g.as_default():
      hash_table = distributed_ps.DistributedHashTable(2, table_config, factory)
      ids = tf.constant([0, 3, 0, 1], dtype=tf.int64)
      values = hash_table.lookup(ids)
      loss = 2 * values
      grads = tf.gradients(loss, values)
      global_step = tf.constant(0, dtype=tf.int64)
      hash_table = hash_table.apply_gradients(ids, grads[0], global_step)

      new_values = hash_table.lookup(ids)

    servers, config = test_utils.create_test_ps_cluster(2)

    with tf.compat.v1.Session(servers[0].target, config=config,
                              graph=g) as sess:
      new_values = sess.run(new_values)
    self.assertAllEqual(new_values, [[-4], [-2], [-4], [-2]])

  def test_apply_gradients_with_different_ids(self):
    table_config = test_utils.generate_test_hash_table_config(1)
    g = tf.Graph()
    with g.as_default():
      hash_table = distributed_ps.DistributedHashTable(2, table_config, factory)
      ids = tf.constant([1, 0], dtype=tf.int64)
      bp_ids = tf.constant([1, 1], dtype=tf.int64)
      values = hash_table.lookup(ids)
      loss = -2 * values
      grads = tf.gradients(loss, values)
      global_step = tf.constant(0, dtype=tf.int64)
      hash_table = hash_table.apply_gradients(bp_ids, grads[0], global_step)

      new_values = hash_table.lookup(ids)

    servers, config = test_utils.create_test_ps_cluster(2)

    with tf.compat.v1.Session(servers[0].target, config=config,
                              graph=g) as sess:
      new_values = sess.run(new_values)
    self.assertAllEqual(new_values, [[4], [0]])


def multi_type_table_factory(ps_num: int, slot_to_config_on_ps):

  def table_factory(name_suffix: str, config):
    return hash_table_ops.hash_table_from_config(config,
                                                 name_suffix=name_suffix +
                                                 str(ps_num))

  return multi_type_hash_table.MultiTypeHashTable(slot_to_config_on_ps,
                                                  table_factory)


def native_multi_hash_table_factory(ps_num: int, slot_to_config):
  return MultiHashTable.from_configs(configs=slot_to_config,
                                     name_suffix=str(ps_num))


class DistributedMultiTypeHashTableTest(tf.test.TestCase,
                                        parameterized.TestCase):

  @parameterized.parameters([(True,), (False,)])
  def testBasic(self, use_native_multi_hash_table):
    servers, config = test_utils.create_test_ps_cluster(2)
    with tf.compat.v1.Session(servers[0].target, config=config) as sess:
      slot_to_config = {
          "1":
              test_utils.generate_test_hash_table_config(1),
          "2":
              test_utils.generate_test_hash_table_config(
                  2, learning_rate=lambda: 1.0)
      }
      table_factory = (native_multi_hash_table_factory
                       if use_native_multi_hash_table else
                       multi_type_table_factory)
      hash_table = distributed_ps.DistributedMultiTypeHashTable(
          2, slot_to_config, table_factory)
      ids1 = tf.constant([1, 2], dtype=tf.int64)
      values1 = tf.constant([[-1], [-2]], dtype=tf.float32)
      ids2 = tf.constant([3], dtype=tf.int64)
      values2 = tf.constant([[-3, -3]], dtype=tf.float32)
      updated_hash_table = hash_table.assign_add({
          "1": (ids1, values1),
          "2": (ids2, values2)
      })
      values = updated_hash_table.lookup({"1": ids1, "2": ids2})
      sess.run(tf.compat.v1.global_variables_initializer())
      sess.run(tf.compat.v1.local_variables_initializer())
      values = sess.run(values)
      self.assertAllEqual(values["1"], [[-1], [-2]])
      self.assertAllEqual(values["2"], [[-3, -3]])
      global_step = tf.constant(0, dtype=tf.int64)
      updated_hash_table = hash_table.apply_gradients(
          {
              "1": (ids1, values1 / 2),
              "2": (ids2, values2 / 2)
          },
          global_step,
          req_time=tf.constant(0, dtype=tf.int64))
      values = updated_hash_table.lookup({"1": ids1, "2": ids2})
      values, _ = sess.run([values, updated_hash_table.as_op()])
      self.assertAllEqual(values["1"], [[-0.5], [-1]])
      self.assertAllEqual(values["2"], [[-1.5, -1.5]])

  @parameterized.parameters([(True,), (False,)])
  def test_assign(self, use_native_multi_hash_table):
    servers, config = test_utils.create_test_ps_cluster(2)
    with tf.compat.v1.Session(servers[0].target, config=config) as sess:
      slot_to_config = {
          "1": test_utils.generate_test_hash_table_config(1),
          "2": test_utils.generate_test_hash_table_config(2)
      }
      table_factory = (native_multi_hash_table_factory
                       if use_native_multi_hash_table else
                       multi_type_table_factory)
      hash_table = distributed_ps.DistributedMultiTypeHashTable(
          2, slot_to_config, table_factory)
      ids1 = tf.constant([1, 2], dtype=tf.int64)
      values1 = tf.constant([[-1], [-2]], dtype=tf.float32)
      ids2 = tf.constant([3], dtype=tf.int64)
      values2 = tf.constant([[-3, -3]], dtype=tf.float32)
      updated_hash_table = hash_table.assign({
          "1": (ids1, values1),
          "2": (ids2, values2)
      })
      values = updated_hash_table.lookup({"1": ids1, "2": ids2})
      sess.run(tf.compat.v1.global_variables_initializer())
      sess.run(tf.compat.v1.local_variables_initializer())
      values = sess.run(values)
      self.assertAllEqual(values["1"], [[-1], [-2]])
      self.assertAllEqual(values["2"], [[-3, -3]])

      updated_hash_table = hash_table.assign({
          "1": (ids1, values1 / 2),
          "2": (ids2, values2 / 2)
      })
      values = updated_hash_table.lookup({"1": ids1, "2": ids2})
      values = sess.run(values)
      self.assertAllEqual(values["1"], [[-0.5], [-1]])
      self.assertAllEqual(values["2"], [[-1.5, -1.5]])

  @parameterized.parameters([(True,), (False,)])
  def test_apply_gradients_with_learning_rate_function(
      self, use_native_multi_hash_table):
    servers, config = test_utils.create_test_ps_cluster(2)
    with tf.compat.v1.Session(servers[0].target, config=config) as sess:
      global_step = tf.compat.v1.train.get_or_create_global_step()
      slot_to_config = {
          "1":
              test_utils.generate_test_hash_table_config(
                  1,
                  learning_rate=learning_rate_functions.PolynomialDecay(
                      initial_learning_rate=1.0,
                      decay_steps=10,
                      end_learning_rate=2.0)),
          "2":
              test_utils.generate_test_hash_table_config(
                  2, learning_rate=lambda: 1.0)
      }
      table_factory = (native_multi_hash_table_factory
                       if use_native_multi_hash_table else
                       multi_type_table_factory)
      hash_table = distributed_ps.DistributedMultiTypeHashTable(
          2, slot_to_config, table_factory)
      ids1 = tf.constant([1, 2], dtype=tf.int64)
      values1 = tf.constant([[-1], [-2]], dtype=tf.float32)
      ids2 = tf.constant([3], dtype=tf.int64)
      values2 = tf.constant([[-3, -3]], dtype=tf.float32)
      updated_hash_table = hash_table.assign_add({
          "1": (ids1, values1),
          "2": (ids2, values2)
      })
      values = updated_hash_table.lookup({"1": ids1, "2": ids2})
      global_step = tf.compat.v1.train.get_or_create_global_step()
      sess.run(tf.compat.v1.global_variables_initializer())
      sess.run(tf.compat.v1.local_variables_initializer())
      values = sess.run(values)
      self.assertAllEqual(values["1"], [[-1], [-2]])
      self.assertAllEqual(values["2"], [[-3, -3]])
      updated_hash_table = hash_table.apply_gradients(
          {
              "1": (ids1, values1 / 2),
              "2": (ids2, values2 / 2)
          }, global_step)
      values = updated_hash_table.lookup({"1": ids1, "2": ids2})
      values, _ = sess.run([values, updated_hash_table.as_op()])
      self.assertAllEqual(values["1"], [[-0.5], [-1]])
      self.assertAllEqual(values["2"], [[-1.5, -1.5]])

      self.evaluate(tf.compat.v1.assign_add(global_step, 1))
      updated_hash_table = hash_table.apply_gradients(
          {
              "1": (ids1, values1 / 2),
              "2": (ids2, values2 / 2)
          }, global_step)
      values = updated_hash_table.lookup({"1": ids1, "2": ids2})
      values, _ = sess.run([values, updated_hash_table.as_op()])
      self.assertAllClose(values["1"], [[0.05], [0.1]])
      self.assertAllClose(values["2"], [[0, 0]])

  @parameterized.parameters([(True,), (False,)])
  def test_apply_gradients_float16(self, use_native_multi_hash_table):
    servers, config = test_utils.create_test_ps_cluster(2)
    with tf.compat.v1.Session(servers[0].target, config=config) as sess:
      slot_to_config = {
          "1":
              test_utils.generate_test_hash_table_config(dim=1,
                                                         use_float16=True),
          "2":
              test_utils.generate_test_hash_table_config(dim=2,
                                                         use_float16=True),
      }
      table_factory = (native_multi_hash_table_factory
                       if use_native_multi_hash_table else
                       multi_type_table_factory)
      hash_table = distributed_ps.DistributedMultiTypeHashTable(
          num_ps=2,
          slot_to_config=slot_to_config,
          table_factory=table_factory,
          transfer_float16=True)
      ids1 = tf.constant([1, 2], dtype=tf.int64)
      values1 = tf.constant([[-1], [-2]], dtype=tf.float32)
      ids2 = tf.constant([3], dtype=tf.int64)
      values2 = tf.constant([[-3, -3]], dtype=tf.float32)
      loss1 = 2 * values1
      loss2 = 3 * values2
      grads1 = tf.gradients(loss1, values1)
      grads2 = tf.gradients(loss2, values2)
      hash_table = hash_table.apply_gradients(
          {
              '1': {ids1, grads1[0]},
              '2': {ids2, grads2[0]},
          },
          global_step=tf.constant(1, dtype=tf.int64))
      values = hash_table.lookup({"1": ids1, "2": ids2})
      sess.run(tf.compat.v1.global_variables_initializer())
      sess.run(tf.compat.v1.local_variables_initializer())
      res = sess.run(values)
      self.assertAllEqual(res["1"], [[-2.], [-2.]])
      self.assertAllEqual(res["2"], [[-3., -3.]])


class DistributedMultiTypeHashTableServingTest(tf.test.TestCase,
                                               parameterized.TestCase):

  @parameterized.parameters([(True,), (False,)])
  def test_export_model(self, use_native_multi_hash_table):
    table_factory = (native_multi_hash_table_factory if
                     use_native_multi_hash_table else multi_type_table_factory)
    servers, config = test_utils.create_test_ps_cluster(2)
    with tf.compat.v1.Session(servers[0].target, config=config) as sess:
      slot_to_config = {
          "1": test_utils.generate_test_hash_table_config(1),
          "2": test_utils.generate_test_hash_table_config(2)
      }
      # Exporting distributed saved model
      with tf.Graph().as_default():
        export_ctx = export_context.ExportContext()
        with export_context.enter_export_mode(
            export_context.ExportMode.DISTRIBUTED, export_ctx):
          hash_table = distributed_ps.DistributedMultiTypeHashTable(
              2, slot_to_config, table_factory)
          self.assertAllEqual(export_ctx.sub_graph_num, 2)
          result = hash_table.lookup({"1": tf.constant([1, 2], dtype=tf.int64)})
        self.assertEqual(result["1"].shape, [2, 1])

      # Exporting standalone saved model
      with tf.Graph().as_default():
        export_ctx = export_context.ExportContext()
        with export_context.enter_export_mode(
            export_context.ExportMode.STANDALONE, export_ctx):
          hash_table = distributed_ps.DistributedMultiTypeHashTable(
              2, slot_to_config, table_factory)
          self.assertAllEqual(export_ctx.sub_graph_num, 0)
          hash_table.lookup({"1": tf.constant([1], dtype=tf.int64)})

      # Normal training
      with tf.Graph().as_default():
        hash_table = distributed_ps.DistributedMultiTypeHashTable(
            2, slot_to_config, table_factory)
        hash_table.lookup({"1": tf.constant([1], dtype=tf.int64)})


def multi_table_factory(idx: int, configs: Dict[str,
                                                entry.HashTableConfigInstance]):

  def factory(name_suffix, config):
    return hash_table_ops.hash_table_from_config(config,
                                                 name_suffix="_".join(
                                                     [name_suffix,
                                                      str(idx)]))

  return multi_type_hash_table.MultiTypeHashTable(configs, factory)


class PartitionedHashTableTest(tf.test.TestCase):
  use_native_multi_hash_table = False

  @classmethod
  def gen_table_config(cls,
                       dims: List[int],
                       use_float16: bool = False,
                       learning_rate: float = 1.0):
    assert len(dims) >= 1
    table_config = embedding_hash_table_pb2.EmbeddingHashTableConfig()
    table_config.cuckoo.SetInParent()
    for i, dim in enumerate(dims):
      segment = table_config.entry_config.segments.add()
      segment.dim_size = dim
      segment.init_config.zeros.SetInParent()
      segment.comp_config.fp32.SetInParent()

      if i == 0:
        segment.opt_config.ftrl.SetInParent()
        segment.opt_config.stochastic_rounding_float16 = use_float16
      else:
        segment.opt_config.adagrad.SetInParent()
        segment.opt_config.stochastic_rounding_float16 = use_float16

    return entry.HashTableConfigInstance(table_config,
                                         [learning_rate] * len(dims))

  @classmethod
  def gen_out_config(cls, feature_to_unmerged_slice_dims: Dict[str, List[int]],
                     layout_names: List[str]):
    features = list(feature_to_unmerged_slice_dims.keys())
    feature_stats = {name: 0 for name in features}

    layout_configs = {}
    for i, layout in enumerate(
        itertools.zip_longest(*feature_to_unmerged_slice_dims.values())):
      out_conf = OutConfig(out_type=OutType.CONCAT)
      assert len(features) == len(layout)
      out_dim = 0
      for name, dim_size in zip(features, layout):
        if dim_size is None:
          continue
        out_dim += dim_size
        slice_config = out_conf.slice_configs.add()
        slice_config.feature_name = name
        slice_config.start = feature_stats[name]
        slice_config.end = slice_config.start + dim_size
        pooling_type = PoolingType.SUM
        feature_stats[name] += dim_size

      shape = out_conf.shape.add()
      shape.dims.extend([-1, out_dim])

      layout_configs[layout_names[i]] = out_conf

    return layout_configs

  @classmethod
  def setUpClass(cls):
    cls.num_ps = 2

    cls.feature_to_unmerged_slice_dims = {
        "uid": [1, 4],
        'gid': [1, 4, 8],
        "cid": [1, 4, 8],
    }

    cls.feature_to_combiner = {
        'uid': embedding_combiners.ReduceSum(),
        'gid': embedding_combiners.ReduceSum(),
        'cid': embedding_combiners.ReduceSum(),
    }

    cls.feature_name_to_config = {
        name: cls.gen_table_config(dims=dims)
        for name, dims in cls.feature_to_unmerged_slice_dims.items()
    }

    if cls.use_native_multi_hash_table:
      cls.sub_table_name_to_config, cls.feature_to_sub_table = \
        distributed_ps.PartitionedHashTable.no_merge_feature_config(cls.feature_name_to_config)
    else:
      cls.sub_table_name_to_config, cls.feature_to_sub_table = \
        distributed_ps.PartitionedHashTable.merge_feature_config(cls.feature_name_to_config)

    cls.layout_configs = cls.gen_out_config(
        cls.feature_to_unmerged_slice_dims,
        layout_names=['bias', 'vec', 'deep'])
    feature_info = {}
    for feature_name, sub_table in cls.feature_to_sub_table.items():
      feature_info[feature_name] = distributed_ps.FeatureInfo(
          cls.feature_to_unmerged_slice_dims[feature_name],
          cls.feature_to_combiner[feature_name], sub_table)
    cls.feature_configs = distributed_ps.PartitionedHashTable.gen_feature_configs(
        feature_info, cls.layout_configs)

  @classmethod
  def gen_data(cls,
               with_emb: bool = False,
               method: str = 'random',
               value: float = 1.0):
    assert method in {'random', 'const'}
    data_tf, data_np = {}, {}
    for i in range(cls.num_ps):
      data_tf[i], data_np[i] = {}, {}
      for tbname, conf in cls.sub_table_name_to_config.items():
        size = sum(
            seg.dim_size for seg in conf._table_config.entry_config.segments)
        fids_np = np.array([i + cls.num_ps * j for j in range(size)],
                           dtype=np.int64)
        fids_tf = tf.constant(value=fids_np,
                              dtype=tf.int64,
                              name=f'{tbname}_fids')

        if with_emb:
          emb_size = sum(
              segment.dim_size
              for segment in conf._table_config.entry_config.segments)
          if method == 'random':
            embs_np = np.random.uniform(-1, 1, size=(size, emb_size))
          else:
            embs_np = np.ones(
                shape=(size, emb_size),
                dtype=np.float32) * value  #fids_np.reshape([size, 1])
          embs_tf = tf.constant(value=embs_np,
                                dtype=tf.float32,
                                name=f'{tbname}_embss')
          data_tf[i][tbname] = (fids_tf, embs_tf)
          data_np[i][tbname] = (fids_np, embs_np)
        else:
          data_tf[i][tbname] = fids_tf
          data_np[i][tbname] = fids_np
    return data_tf, data_np

  @classmethod
  def gen_variant_tensor(cls, batch_size: int):
    examples = []
    for i in range(batch_size):
      example = Example()
      start = i * 3
      named_feature = example.named_feature.add()
      named_feature.name = 'uid'
      named_feature.feature.fid_v2_list.value.append(start)

      named_feature = example.named_feature.add()
      named_feature.name = 'gid'
      named_feature.feature.fid_v2_list.value.append(start + 1)

      named_feature = example.named_feature.add()
      named_feature.name = 'cid'
      named_feature.feature.fid_v2_list.value.append(start + 2)

      #logging.info(f" {i}/{batch_size}:{example}")
      examples.append(example.SerializeToString())

    example_strs = tf.constant(value=examples, dtype=tf.string, name='examples')
    return string_to_variant(example_strs, variant_type='example')

  def test_basic(self):
    servers, config = test_utils.create_test_ps_cluster(self.num_ps)
    config.share_cluster_devices_in_session = True
    config.experimental.share_session_state_in_clusterspec_propagation = True
    # grappler doesn't really understand RaggedTensor.
    config.graph_options.rewrite_options.disable_meta_optimizer = True
    with tf.compat.v1.Session(servers[0].target, config=config) as sess:
      hash_table = distributed_ps.PartitionedHashTable(
          self.num_ps,
          self.feature_name_to_config,
          native_multi_hash_table_factory
          if self.use_native_multi_hash_table else multi_table_factory,
          self.layout_configs,
          self.feature_to_unmerged_slice_dims,
          self.feature_to_combiner,
          use_native_multi_hash_table=self.use_native_multi_hash_table)

      if self.use_native_multi_hash_table:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())
      assign_data, assign_data_np = self.gen_data(with_emb=True)
      hash_table = hash_table.assign(assign_data)
      assign_add_data, assign_add_data_np = self.gen_data(with_emb=True)
      hash_table = hash_table.assign_add(assign_add_data)
      lookup_data, _ = self.gen_data(with_emb=False)
      values = hash_table._lookup_raw(lookup_data)

      real_result = sess.run(values)
      #logging.info(
      #    f"xxx {lookup_data} {assign_data_np} {assign_add_data_np} {real_result}"
      #)
      for part in assign_data_np:
        for tbname in assign_data_np[part]:
          fid1, emb1 = assign_data_np[part][tbname]
          fid2, emb2 = assign_add_data_np[part][tbname]
          self.assertAllClose(real_result[part][tbname], emb1 + emb2)

  def test_lookup(self):
    servers, config = test_utils.create_test_ps_cluster(self.num_ps)
    config.share_cluster_devices_in_session = True
    config.experimental.share_session_state_in_clusterspec_propagation = True
    # grappler doesn't really understand RaggedTensor.
    config.graph_options.rewrite_options.disable_meta_optimizer = True
    with tf.compat.v1.Session(servers[0].target, config=config) as sess:
      hash_table = distributed_ps.PartitionedHashTable(
          self.num_ps,
          self.feature_name_to_config,
          native_multi_hash_table_factory
          if self.use_native_multi_hash_table else multi_table_factory,
          self.layout_configs,
          self.feature_to_unmerged_slice_dims,
          self.feature_to_combiner,
          use_native_multi_hash_table=self.use_native_multi_hash_table)
      hash_table._inner_data_type = 'example'

      if self.use_native_multi_hash_table:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())
      x = 2.0
      assign_data, assign_data_np = self.gen_data(with_emb=True,
                                                  method='const',
                                                  value=x)
      hash_table = hash_table.assign(assign_data)

      sparse_features = self.gen_variant_tensor(batch_size=self.num_ps * 3)
      auxiliary_bundle = {}

      layouts = hash_table.lookup(sparse_features,
                                  auxiliary_bundle=auxiliary_bundle)
      layouts = sess.run(layouts)
      auxiliary_bundle_ret = sess.run(auxiliary_bundle)

      expect = {
          'bias':
              np.array([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.],
                        [0., 1., 1.], [0., 1., 1.]],
                       dtype=np.float32),
          'deep':
              np.array([[
                  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.
              ], [
                  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.
              ], [
                  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.
              ], [
                  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.
              ], [
                  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.
              ], [
                  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.
              ]],
                       dtype=np.float32),
          'vec':
              np.array([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                        [0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
                        [0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.]],
                       dtype=np.float32)
      }

      for key, value in layouts.items():
        #logging.info(f" {key} {value}  ---  {expect[key] * x}")
        self.assertAllClose(value, expect[key] * x)

  def test_apply_gradients(self):
    servers, config = test_utils.create_test_ps_cluster(self.num_ps)
    config.share_cluster_devices_in_session = True
    config.experimental.share_session_state_in_clusterspec_propagation = True
    # grappler doesn't really understand RaggedTensor.
    config.graph_options.rewrite_options.disable_meta_optimizer = True
    with tf.compat.v1.Session(servers[0].target, config=config) as sess:
      hash_table = distributed_ps.PartitionedHashTable(
          self.num_ps,
          self.feature_name_to_config,
          native_multi_hash_table_factory
          if self.use_native_multi_hash_table else multi_table_factory,
          self.layout_configs,
          self.feature_to_unmerged_slice_dims,
          self.feature_to_combiner,
          use_native_multi_hash_table=self.use_native_multi_hash_table)
      hash_table._inner_data_type = 'example'

      if self.use_native_multi_hash_table:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())
      init_val = 3.0
      assign_data, assign_data_np = self.gen_data(with_emb=True,
                                                  method='const',
                                                  value=init_val)
      hash_table = hash_table.assign(assign_data)
      sparse_features = self.gen_variant_tensor(batch_size=self.num_ps * 3)
      auxiliary_bundle = {'sparse_features': sparse_features}

      layouts = hash_table.lookup(sparse_features,
                                  auxiliary_bundle=auxiliary_bundle)
      layout_grads_and_vars, init_grad = [], 2.0
      for name in sorted(hash_table._feature_configs.out_configs):
        layout = layouts[name]
        layout_grads_and_vars.append(
            (tf.ones_like(layout, dtype=tf.float32) * init_grad, layout))

      global_step = tf.constant(0, dtype=tf.int64)
      hash_table = hash_table.apply_gradients(layout_grads_and_vars,
                                              global_step,
                                              auxiliary_bundle=auxiliary_bundle)
      lookup_data, lookup_data_np = self.gen_data(with_emb=False)
      values = hash_table._lookup_raw(lookup_data)
      values = sess.run(values)
      #logging.info(f"xx values: {lookup_data_np} {values}")

      if self.use_native_multi_hash_table:
        shards = {
            'uid:0': [0, 6, 12],
            'uid:1': [3, 9, 15],
            'cid:0': [2, 8, 14],
            'cid:1': [5, 11, 17],
            'gid:0': [4, 10, 16],
            'gid:1': [1, 7, 13]
        }
      else:
        shards = {
            '9871d3a2c554b27151cacf1422eec048:0': [0, 6, 12],
            '9871d3a2c554b27151cacf1422eec048:1': [3, 9, 15],
            'c4a398dea30b21551ae4c09454001dba:0': [2, 8, 14, 4, 10, 16],
            'c4a398dea30b21551ae4c09454001dba:1': [5, 11, 17, 1, 7, 13]
        }

      learning_rate = 1.0
      initial_accumulator_value = 0.1
      beta = 0.0
      ada_grad = learning_rate / math.sqrt(
          init_grad * init_grad + initial_accumulator_value) * init_grad

      n = initial_accumulator_value + init_grad * init_grad
      sigma = (math.sqrt(n) -
               math.sqrt(initial_accumulator_value)) / learning_rate
      z = init_grad - sigma * init_val
      ftrl_grad = -learning_rate * z / (math.sqrt(n) + beta)

      for name, value in shards.items():
        tb_name, idx = name.split(':')
        lu_value = values[int(idx)][tb_name]
        for i in value:
          k = int(i / self.num_ps)
          if k < lu_value.shape[0]:
            for j, x in enumerate(lu_value[k]):
              if j == 0:
                self.assertAlmostEqual(x, ftrl_grad, delta=1e-6)
              else:
                self.assertAlmostEqual(init_val - ada_grad, x, delta=1e-6)


class PartitionedHashTableWithNativeHashTableTest(PartitionedHashTableTest):

  def __init__(self, *args, **kwargs) -> None:
    super(PartitionedHashTableWithNativeHashTableTest,
          self).__init__(*args, **kwargs)

  @classmethod
  def setUpClass(cls):
    cls.use_native_multi_hash_table = True
    super(PartitionedHashTableWithNativeHashTableTest, cls).setUpClass()


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
