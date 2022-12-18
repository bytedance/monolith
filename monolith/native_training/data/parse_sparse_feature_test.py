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

from collections import defaultdict
from copy import deepcopy
import os
import time

import tensorflow as tf

from absl import logging
import numpy as np
from struct import unpack

from monolith.native_training.data.parsers import parse_instances, parse_examples, parse_example_batch, \
    sharding_sparse_fids, get_default_parser_ctx
from monolith.native_training.model_export.data_gen_utils import gen_fids_v1, gen_fids_v2, fill_line_id, \
  gen_instance, FeatureMeta
from idl.matrix.proto.example_pb2 import Example, ExampleBatch, FeatureConfigs, FeatureConfig, FeatureListType
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


class DataOpsV2Test(tf.test.TestCase):

  def __init__(self, *args, **kwargs):
    super(DataOpsV2Test, self).__init__(*args, **kwargs)
    self.mask = (1 << 48) - 1
    #self.version = 2

  def fid_v1_to_v2(self, fid_v1):
    slot_id = (fid_v1 >> 54)
    fid_v2 = ((slot_id << 48) | (self.mask & fid_v1))
    return slot_id, fid_v2

  def fill_row_split(self, ps_num, t_cfg, fid_list, row_split):
    for ps_i in range(ps_num):
      lenth = 0
      for feature_name in t_cfg["feature_list"]:
        key = feature_name + ":" + str(ps_i)
        if key in fid_list:
          lenth += len(fid_list[key])
      row_split[t_cfg["table_name"] + ":" + str(ps_i)].append(lenth)

  def get_pre_output_offset(self, shard, f_cfg):
    return f_cfg["pre_output_index"] + shard * f_cfg[
        "table_feature_count"] + f_cfg["feature_in_table_index"]

  def get_feature_cfg(self, raw_feature_cfgs, ps_num):
    feature_cfg = defaultdict(dict)
    table_cfg = defaultdict(dict)
    for feature_name, cfg in raw_feature_cfgs.feature_configs.items():
      feature_cfg[feature_name] = {
          "feature_name": feature_name,
          "feature_index": -1,
          "table_name": cfg.table,
          "table_index": -1,
          "feature_in_table_index": -1,
          "table_feature_count": 0,
          "pre_output_index": 0,
      }
      if cfg.table not in table_cfg:
        table_cfg[cfg.table] = {
            "table_name": cfg.table,
            "feature_list": [],
            "table_index": -1,
            "feature_count": 0,
        }

    table_name_sort = sorted(table_cfg.keys())
    for idx, name in enumerate(table_name_sort):
      table_cfg[name]["table_index"] = idx

    feature_name_sort = sorted(feature_cfg.keys())
    for idx, name in enumerate(feature_name_sort):
      f_cfg = feature_cfg[name]
      t_cfg = table_cfg[f_cfg["table_name"]]

      f_cfg["feature_index"] = idx
      f_cfg["table_index"] = t_cfg["table_index"]
      f_cfg["feature_in_table_index"] = len(t_cfg["feature_list"])

      t_cfg["feature_list"].append(name)

    pre_index = 0
    for idx, name in enumerate(table_name_sort):
      t_cfg = table_cfg[name]
      t_cfg["feature_count"] = len(t_cfg["feature_list"])
      for feature_name in t_cfg["feature_list"]:
        f_cfg = feature_cfg[feature_name]
        f_cfg["pre_output_index"] = pre_index
        f_cfg["table_feature_count"] = t_cfg["feature_count"]
      pre_index += max(t_cfg["feature_count"], 1) * ps_num
    return feature_cfg, table_cfg, feature_name_sort, table_name_sort

  def handle_feature(self, fid_v1_list, fid_v2_list, f_cfg, t_cfg, ps_num,
                     fid_offset_list, fid_offset_list2, fid_map_t,
                     fid_map_unique_map, fid_map_unique_t):
    value_list = []
    if len(fid_v1_list) != 0:
      for fid in fid_v1_list:
        slot_id, fid_v2 = self.fid_v1_to_v2(fid)
        value_list.append(fid_v2)
        #print("aaaaaa {} -> {}".format(fid, fid_v2))
    elif len(fid_v2_list) != 0:
      value_list = fid_v2_list
    for value in value_list:
      shard = value % ps_num
      key = f_cfg["feature_name"] + ":" + str(shard)
      fid_offset = self.get_pre_output_offset(shard, f_cfg) << 32
      fid_offset_list.append(fid_offset | len(fid_map_t[key]))
      fid_map_t[key].append(value)
      # print("aaaaaa {} {} {}".format(table_name, shard, value))
      if value not in fid_map_unique_map[key]:
        fid_map_unique_map[key][value] = len(fid_map_unique_map[key])
        fid_map_unique_t[key].append(value)
      fid_offset_list2.append(fid_offset | fid_map_unique_map[key][value])

  def get_offset_result(self,
                        feature_name_sort,
                        table_name_sort,
                        ps_num,
                        feature_cfg,
                        table_cfg,
                        fid_offset_map,
                        fid_offset_map_unique,
                        fid_map_t_in,
                        fid_map_unique_t_in,
                        sparse_feature_shared=set()):
    #print('xxxxx', sparse_feature_shared)
    feature_offset_t = []
    nfl_offset_t = []
    fid_offset_list = []
    fid_offset_list_unique = []
    for sparse_key in feature_name_sort:
      if sparse_key in sparse_feature_shared:
        nfl_offset = len(feature_offset_t) | 1 << 31
        #pass
      else:
        nfl_offset = len(feature_offset_t)
      nfl_offset_t.append(nfl_offset)
      if sparse_key not in fid_offset_map:
        continue
      for fid_list in fid_offset_map[sparse_key]:
        feature_offset_t.append(len(fid_offset_list))
        fid_offset_list.extend(fid_list)
      for fid_list in fid_offset_map_unique[sparse_key]:
        fid_offset_list_unique.extend(fid_list)

    fid_map_t = defaultdict(list)
    fid_map_unique_t = defaultdict(list)
    for table_name in table_name_sort:
      t_cfg = table_cfg[table_name]
      for ps_i in range(ps_num):
        to_key = table_name + ":" + str(ps_i)
        for feature_name in t_cfg["feature_list"]:
          key = feature_name + ":" + str(ps_i)
          fid_map_t[to_key].extend(fid_map_t_in[key])
          fid_map_unique_t[to_key].extend(fid_map_unique_t_in[key])

    print('==' * 10 + "fid_map_t" + '==' * 10)
    #print(fid_map_t)
    print('==' * 10 + "fid_map_unique_t" + '==' * 10)
    #print(fid_map_unique_t)
    print('==' * 10 + "fid_offset_map" + '==' * 10)
    #print(fid_offset_map)
    return nfl_offset_t, feature_offset_t, fid_offset_list, fid_offset_list_unique, fid_map_t, fid_map_unique_t

  def diff_test(self, input_type, parse_func, input_str_list, ps_num,
                feature_cfgs, sparse_features, dense_features, extra_features,
                nfl_offset_t, feature_offset_t, fid_offset_list,
                fid_offset_list_unique, fid_map_t, fid_map_row_split_t,
                fid_map_unique_t, fid_map_row_split_unique_t):
    with tf.Graph().as_default():
      config = tf.compat.v1.ConfigProto()
      config.graph_options.rewrite_options.disable_meta_optimizer = True
      input_placeholder = tf.compat.v1.placeholder(dtype=tf.string,
                                                   shape=(None,))
      parsed_results_base, parsed_results = parse_func(input_placeholder)
      example_batch_varint = parsed_results.pop("sparse_features")

      parallel_flag_list = [1, 2, 3, 4]
      fid_map_list = []
      fid_map_unique_list = []
      for parallel_flag in parallel_flag_list:
        fid_map, fid_offset, feature_offset, nfl_offset, batch_size, fid_map_row_split = sharding_sparse_fids(
            example_batch_varint, ps_num, feature_cfgs, False, input_type,
            parallel_flag)
        fid_map_list.append([
            fid_map, fid_offset, feature_offset, nfl_offset, fid_map_row_split
        ])
        fid_map_unique, fid_offset, feature_offset, nfl_offset, batch_size, fid_map_unique_row_split = sharding_sparse_fids(
            example_batch_varint, ps_num, feature_cfgs, True, input_type,
            parallel_flag)
        fid_map_unique_list.append([
            fid_map_unique, fid_offset, feature_offset, nfl_offset,
            fid_map_unique_row_split
        ])

      with self.session(config=config) as sess:
        parsed_results_base1, parsed_results1 = sess.run(
            fetches=[parsed_results_base, parsed_results],
            feed_dict={input_placeholder: input_str_list})

        def diff(k, a, b, sort=False):

          if not isinstance(a[0], list) and sort:
            a.sort()
            b.sort()
          #print("diff:a {} {}".format(k, a), flush=True)
          #print("diff:b {} {}".format(k, b), flush=True)
          assert (len(a) == len(b))
          if (len(a) == 0):
            return
          for i in range(len(a)):
            if isinstance(a[i], list):
              assert isinstance(b[i], list)
              diff(k + "/" + str(i), a[i], b[i], sort)
            else:
              assert (a[i] == b[i]), f"{i}: {a[i]} / {b[i]}"

        #print('==' * 10 + "parsed_results_base1" + '==' * 10, flush=True)
        #print('==' * 10 + "parsed_results1" + '==' * 10, flush=True)
        # .numpy()
        for k, v in parsed_results_base1.items():
          if k in sparse_features:
            continue
          if k in dense_features + extra_features:
            if k not in parsed_results1:
              print("no find {} in parse_example_batch_v2".format(k))
              assert (False)
            diff(k, v.tolist(), parsed_results1[k].tolist())  #.numpy()
          else:
            print("no need {}".format(k), flush=True)
            assert (False)
        for k, v in parsed_results1.items():
          if k not in dense_features + extra_features:
            print("no need {}".format(k), flush=True)
            assert (False)

        for fid_map_index in range(len(parallel_flag_list)):
          fid_map = fid_map_list[fid_map_index]
          fid_map_unique = fid_map_unique_list[fid_map_index]
          fid_map1_list, fid_map_unique1_list = sess.run(
              fetches=[fid_map, fid_map_unique],
              feed_dict={input_placeholder: input_str_list})

          #print('==' * 10 + "fid_map1" + '==' * 10, flush=True)
          #print(fid_map1, flush=True)
          #print('==' * 10 + "fid_map_unique1" + '==' * 10, flush=True)
          #print(fid_map_unique1, flush=True)
          fid_map1, fid_offset1, feature_offset1, nfl_offset1, fid_map_row_split1 = fid_map1_list
          fid_map2, fid_offset2, feature_offset2, nfl_offset2, fid_map_unique_row_split1 = fid_map_unique1_list
          print('==' * 10 + "diff fidoffset " +
                str(parallel_flag_list[fid_map_index]) + '==' * 10,
                flush=True)
          diff("nfl_offset", nfl_offset1, nfl_offset_t)
          diff("nfl_offset2", nfl_offset2, nfl_offset_t)
          diff("feature_offset", feature_offset1, feature_offset_t)
          diff("feature_offset2", feature_offset2, feature_offset_t)
          #print(f"xxxx fid_offset_list_unique {fid_offset_list_unique}",
          #      flush=True)
          #print(f"xxxx fid_offset2 {list(fid_offset2)}", flush=True)
          diff("fid_offset2", fid_offset2, fid_offset_list_unique)
          diff("fid_offset", fid_offset1, fid_offset_list)

          assert (len(fid_map_t) == len(fid_map1))
          assert (len(fid_map_unique_t) == len(fid_map2))

          def fid_diff(a, b):
            for k, v in a.items():
              assert (k in b)
              diff(k, v.tolist(), b[k])  #.numpy()
            for k, v in b.items():
              assert (k in a)

          print('==' * 10 + "diff fid_map1 " +
                str(parallel_flag_list[fid_map_index]) + '==' * 10,
                flush=True)
          #print(f"xxxx fid_map1 {fid_map1} ", flush=True)
          #print(f"xxxx fid_map_t {fid_map_t} ", flush=True)
          fid_diff(fid_map1, fid_map_t)
          print(f"xxxx fid_map_row_split1 {fid_map_row_split1}", flush=True)
          print(f"xxxx fid_map_row_split_t {fid_map_row_split_t}", flush=True)
          fid_diff(fid_map_row_split1, fid_map_row_split_t)
          print('==' * 10 + "diff fid_map_unique1 " +
                str(parallel_flag_list[fid_map_index]) + '==' * 10,
                flush=True)
          #print(f"xxxx fid_map2 {fid_map2} ", flush=True)
          #print(f"xxxx fid_map_unique_t {fid_map_unique_t} ", flush=True)
          #print(f"xxxx fid_map_unique_row_split {fid_map_unique_row_split} ",
          #      flush=True)
          fid_diff(fid_map2, fid_map_unique_t)
          fid_diff(fid_map_unique_row_split1, fid_map_row_split_unique_t)

  def testExampleBatchSharding(self):
    file_name = "monolith/native_training/data/training_instance/examplebatch.data"

    sparse_features = list(features.keys())
    with open(file_name, 'rb') as stream:
      stream.read(8)  # strip lagrangex_header
      size = unpack("<Q", stream.read(8))[0]
      eb_str = stream.read(size)
      example_batch = ExampleBatch()
      example_batch.ParseFromString(eb_str)
      #add shared
      named_feature_list = example_batch.named_feature_list.add()
      named_feature_list.type = FeatureListType.SHARED
      named_feature_list.id = 990
      named_feature_list.name = "test_shared1"
      feature = named_feature_list.feature.add()
      feature.fid_v2_list.value.extend(gen_fids_v2(990, 10))
      named_feature_list = example_batch.named_feature_list.add()
      named_feature_list.type = FeatureListType.SHARED
      named_feature_list.id = 991
      named_feature_list.name = "test_shared2"
      feature = named_feature_list.feature.add()
      feature.fid_v1_list.value.extend(gen_fids_v1(991, 10))
      named_feature_list = example_batch.named_feature_list.add()
      named_feature_list.type = FeatureListType.SHARED
      named_feature_list.id = 991
      named_feature_list.name = "test_shared3"
      feature = named_feature_list.feature.add()

      eb_str = example_batch.SerializeToString()

    sparse_features += ["test_shared1", "test_shared2", "test_shared3"]
    dense_features = ['label']
    dense_feature_shapes = [2]
    dense_feature_types = [tf.float32]
    extra_features = ['uid', 'req_time', 'item_id']
    extra_feature_shapes = [1, 1, 1]

    print('==' * 10 + "sparse_features" + '==' * 10)
    #print(sparse_features)

    feature_cfgs = FeatureConfigs()
    index = 0
    ps_num = 3
    for sparse_key in sparse_features:
      cfg = FeatureConfig()
      cfg.table = 'table_{}'.format(index % 3)
      #print(sparse_key, cfg.table)
      feature_cfgs.feature_configs[sparse_key].CopyFrom(cfg)
      index += 1

    feature_cfg, table_cfg, feature_name_sort, table_name_sort = self.get_feature_cfg(
        feature_cfgs, ps_num)

    #f_goods_title_terms
    fid_map_t = defaultdict(list)
    fid_map_unique_t = defaultdict(list)
    fid_map_unique_map = defaultdict(dict)

    fid_offset_map = defaultdict(list)
    fid_offset_map_unique = defaultdict(list)

    fid_map_row_split_t = defaultdict(list)
    fid_map_row_split_unique_t = defaultdict(list)

    sparse_feature_shared = set()
    example_batch_feature_map = {}
    for named_feature_list in example_batch.named_feature_list:
      if named_feature_list.name not in feature_cfgs.feature_configs or \
              named_feature_list.name not in feature_name_sort:
        continue
      example_batch_feature_map[named_feature_list.name] = named_feature_list

    for sparse_key in feature_name_sort:
      f_cfg = feature_cfg[sparse_key]
      table_name = f_cfg["table_name"]
      t_cfg = table_cfg[table_name]
      self.fill_row_split(ps_num, t_cfg, fid_map_t, fid_map_row_split_t)
      self.fill_row_split(ps_num, t_cfg, fid_map_unique_t,
                          fid_map_row_split_unique_t)
      if sparse_key not in example_batch_feature_map:
        continue
      named_feature_list = example_batch_feature_map[sparse_key]
      table_index = table_cfg[table_name]["table_index"]
      if named_feature_list.type == FeatureListType.SHARED:
        sparse_feature_shared.add(named_feature_list.name)
        feature = named_feature_list.feature[0]
        fid_offset_list = []
        fid_offset_list2 = []
        self.handle_feature(feature.fid_v1_list.value,
                            feature.fid_v2_list.value, f_cfg, t_cfg, ps_num,
                            fid_offset_list, fid_offset_list2, fid_map_t,
                            fid_map_unique_map, fid_map_unique_t)
        fid_offset_map[named_feature_list.name].append(fid_offset_list)
        fid_offset_map_unique[named_feature_list.name].append(fid_offset_list2)
      else:
        for feature in named_feature_list.feature:
          fid_offset_list = []
          fid_offset_list2 = []
          self.handle_feature(feature.fid_v1_list.value,
                              feature.fid_v2_list.value, f_cfg, t_cfg, ps_num,
                              fid_offset_list, fid_offset_list2, fid_map_t,
                              fid_map_unique_map, fid_map_unique_t)
          fid_offset_map[named_feature_list.name].append(fid_offset_list)
          fid_offset_map_unique[named_feature_list.name].append(
              fid_offset_list2)
    for table_name in table_name_sort:
      t_cfg = table_cfg[table_name]
      self.fill_row_split(ps_num, t_cfg, fid_map_t, fid_map_row_split_t)
      self.fill_row_split(ps_num, t_cfg, fid_map_unique_t,
                          fid_map_row_split_unique_t)

    nfl_offset_t, feature_offset_t, fid_offset_list, fid_offset_list_unique, \
      fid_map_t, fid_map_unique_t = self.get_offset_result(feature_name_sort,
                        table_name_sort,
                        ps_num,
                        feature_cfg,
                        table_cfg,
                        fid_offset_map,
                        fid_offset_map_unique,
                        fid_map_t,
                        fid_map_unique_t,
                        sparse_feature_shared
    )

    #print('==' * 10 + "example_batch" + '==' * 10)
    #print(example_batch)

    def parse_func(input_placeholder):
      get_default_parser_ctx().enable_fused_layout = False
      parsed_results_base = parse_example_batch(
          input_placeholder,
          sparse_features=sparse_features,
          dense_features=dense_features,
          dense_feature_shapes=dense_feature_shapes,
          dense_feature_types=dense_feature_types,
          extra_features=extra_features,
          extra_feature_shapes=extra_feature_shapes)
      get_default_parser_ctx().enable_fused_layout = True
      parsed_results = parse_example_batch(
          input_placeholder,
          sparse_features=[],
          dense_features=dense_features,
          dense_feature_shapes=dense_feature_shapes,
          dense_feature_types=dense_feature_types,
          extra_features=extra_features,
          extra_feature_shapes=extra_feature_shapes)
      return parsed_results_base, parsed_results

    self.diff_test("examplebatch", parse_func, [eb_str], ps_num, feature_cfgs,
                   sparse_features, dense_features, extra_features,
                   nfl_offset_t, feature_offset_t, fid_offset_list,
                   fid_offset_list_unique, fid_map_t, fid_map_row_split_t,
                   fid_map_unique_t, fid_map_row_split_unique_t)
    #assert (False)

  def testExampleSharding(self):
    sparse_features_set = set()
    dense_features = ['label']
    dense_feature_shapes = [2]
    dense_feature_types = [tf.float32]
    extra_features = ['uid', 'req_time', 'item_id', 'actions']
    extra_feature_shapes = [1, 1, 1, 1]
    example_str_list = []
    example_list = []
    file_name = "monolith/native_training/data/training_instance/example.pb"
    feature_fid_map = defaultdict(list)
    with open(file_name, 'rb') as stream:
      while (True):
        if len(example_str_list) > 10:
          break
        try:
          stream.read(8)  # strip has_sort_id
          stream.read(8)  # strip kafka_dump
          size = unpack("<Q", stream.read(8))[0]
          example_str = stream.read(size)

          example = Example()
          example.ParseFromString(example_str)

          for feature_index in range(1, 3):
            named_feature = example.named_feature.add()
            named_feature.name = 'fc_slot_9999{}'.format(feature_index)
            fid_list = gen_fids_v2(9999 + feature_index, 10)
            named_feature.feature.fid_v2_list.value.extend(fid_list)
            named_feature.feature.fid_v2_list.value.extend(fid_list)

          for named_feature in example.named_feature:
            feature_fids = feature_fid_map[named_feature.name]
            value_list = []
            if len(named_feature.feature.fid_v1_list.value) != 0:
              for fid in named_feature.feature.fid_v1_list.value:
                slot_id, fid_v2 = self.fid_v1_to_v2(fid)
                value_list.append(fid_v2)
            elif len(named_feature.feature.fid_v2_list.value) != 0:
              value_list = named_feature.feature.fid_v2_list.value
            '''
            for k, v in feature_fid_map.items():
              if k != named_feature.name:
                for fid in value_list:
                  if fid in v:
                    print(
                        f"xxxxx hit fid {fid} {named_feature.name} in other feat {k}",
                        flush=False)
            feature_fids.extend(value_list)
            '''

          #print('xxxxx example_str_list,', example, flush=True)
          for named_feature in example.named_feature:
            if "fc_slot_" in named_feature.name:
              sparse_features_set.add(named_feature.name)
          example_list.append(example)
          example_str_list.append(example.SerializeToString())
        except:
          break

    #for k, v in feature_fid_map.items():
    #  print(f"xxxxx show fid  {k} {v}", flush=False)

    sparse_features = sorted(sparse_features_set)
    #print(sparse_features, flush=True)

    feature_cfgs = FeatureConfigs()
    index = 0
    ps_num = 3
    table_name_index_map = {}
    for sparse_key in sparse_features:
      cfg = FeatureConfig()
      cfg.table = 'table_{}'.format(index % 3)
      table_name_index_map[cfg.table] = -1
      feature_cfgs.feature_configs[sparse_key].CopyFrom(cfg)
      index += 1

    feature_cfg, table_cfg, feature_name_sort, table_name_sort = self.get_feature_cfg(
        feature_cfgs, ps_num)

    fid_map_t = defaultdict(list)
    fid_map_unique_t = defaultdict(list)
    fid_map_unique_map = defaultdict(dict)

    fid_offset_map = defaultdict(list)
    fid_offset_map_unique = defaultdict(list)

    fid_map_row_split_t = defaultdict(list)
    fid_map_row_split_unique_t = defaultdict(list)

    for sparse_key in feature_name_sort:
      f_cfg = feature_cfg[sparse_key]
      table_name = f_cfg["table_name"]
      t_cfg = table_cfg[table_name]
      self.fill_row_split(ps_num, t_cfg, fid_map_t, fid_map_row_split_t)
      self.fill_row_split(ps_num, t_cfg, fid_map_unique_t,
                          fid_map_row_split_unique_t)
      for example in example_list:
        find_named_feature = None
        for named_feature in example.named_feature:
          if named_feature.name == sparse_key:
            find_named_feature = named_feature
            break
        fid_offset_list = []
        fid_offset_list2 = []
        if find_named_feature:
          table_index = table_name_index_map[table_name]
          self.handle_feature(find_named_feature.feature.fid_v1_list.value,
                              find_named_feature.feature.fid_v2_list.value,
                              f_cfg, t_cfg, ps_num, fid_offset_list,
                              fid_offset_list2, fid_map_t, fid_map_unique_map,
                              fid_map_unique_t)
        fid_offset_map[sparse_key].append(fid_offset_list)
        fid_offset_map_unique[sparse_key].append(fid_offset_list2)
    for table_name in table_name_sort:
      t_cfg = table_cfg[table_name]
      self.fill_row_split(ps_num, t_cfg, fid_map_t, fid_map_row_split_t)
      self.fill_row_split(ps_num, t_cfg, fid_map_unique_t,
                          fid_map_row_split_unique_t)

    nfl_offset_t, feature_offset_t, fid_offset_list, fid_offset_list_unique, \
      fid_map_t, fid_map_unique_t = self.get_offset_result(feature_name_sort,
                                                            table_name_sort,
                                                            ps_num,
                                                            feature_cfg,
                                                            table_cfg,
                                                            fid_offset_map,
                                                            fid_offset_map_unique,
                                                            fid_map_t,
                                                            fid_map_unique_t
                                                         )

    def parse_func(input_placeholder):
      get_default_parser_ctx().enable_fused_layout = False
      parsed_results_base = parse_examples(
          input_placeholder,
          sparse_features=sparse_features,
          dense_features=dense_features,
          dense_feature_shapes=dense_feature_shapes,
          dense_feature_types=dense_feature_types,
          extra_features=extra_features,
          extra_feature_shapes=extra_feature_shapes)
      get_default_parser_ctx().enable_fused_layout = True
      parsed_results = parse_examples(input_placeholder,
                                      sparse_features=[],
                                      dense_features=dense_features,
                                      dense_feature_shapes=dense_feature_shapes,
                                      dense_feature_types=dense_feature_types,
                                      extra_features=extra_features,
                                      extra_feature_shapes=extra_feature_shapes)
      return parsed_results_base, parsed_results

    #example_tensor = tf.convert_to_tensor(example_str_list)
    self.diff_test("example", parse_func, example_str_list, ps_num,
                   feature_cfgs, sparse_features, dense_features,
                   extra_features, nfl_offset_t, feature_offset_t,
                   fid_offset_list, fid_offset_list_unique, fid_map_t,
                   fid_map_row_split_t, fid_map_unique_t,
                   fid_map_row_split_unique_t)
    #assert (False)

  def testInstanceSharding(self):
    fidv1_features = [1, 200, 3, 5, 9, 203, 205]
    fidv2_features = ["fc_v2_1", "fc_v2_2", "fc_v2_3"]
    dense_features = ['label']
    dense_feature_shapes = [2]
    dense_feature_types = [tf.float32]
    extra_features = ['uid', 'req_time', 'item_id', 'actions']
    extra_feature_shapes = [1, 1, 1, 2]

    instance_str_list = []
    instance_list = []
    while (len(instance_str_list) < 128):
      instance = gen_instance(
          fidv1_features=fidv1_features,
          fidv2_features=[],
          dense_features=[FeatureMeta('label', shape=2, dtype=tf.float32)],
          extra_features=[
              FeatureMeta('actions', shape=2),
              FeatureMeta('uid'),
              FeatureMeta('req_time', dtype=tf.int32),
              FeatureMeta('item_id'),
          ])
      #print("aaaaaa:", instance)
      instance_list.append(instance)
      instance_str_list.append(instance.SerializeToString())

      instance2 = deepcopy(instance)
      for slot, feature_name in enumerate(fidv2_features):
        feature = instance2.feature.add()
        feature.name = feature_name
        feature.fid.extend(gen_fids_v2(1000 + slot, 10))
      instance_list.append(instance2)
      instance_str_list.append(instance2.SerializeToString())

      instance3 = deepcopy(instance2)
      del instance3.fid[:]
      instance_list.append(instance3)
      instance_str_list.append(instance3.SerializeToString())

    def gen_slot_feature_name(slot_id):
      return f"slot_{slot_id}"

    sparse_features = sorted(
        fidv2_features +
        [gen_slot_feature_name(slot_id) for slot_id in fidv1_features])
    print(sparse_features, flush=True)

    feature_cfgs = FeatureConfigs()
    index = 0
    ps_num = 3
    table_name_index_map = {}
    for sparse_key in sparse_features:
      cfg = FeatureConfig()
      cfg.table = 'table_{}'.format(index % 3)
      table_name_index_map[cfg.table] = -1
      feature_cfgs.feature_configs[sparse_key].CopyFrom(cfg)
      index += 1

    feature_cfg, table_cfg, feature_name_sort, table_name_sort = self.get_feature_cfg(
        feature_cfgs, ps_num)

    fid_map_t = defaultdict(list)
    fid_map_unique_t = defaultdict(list)
    fid_map_unique_map = defaultdict(dict)

    intance_tmp_dict = defaultdict(list)
    for instance in instance_list:
      fid_v2_list = defaultdict(list)
      for fid in instance.fid:
        slot_id, fid_v2 = self.fid_v1_to_v2(fid)
        sparse_key = gen_slot_feature_name(slot_id)
        if sparse_key not in feature_name_sort:
          continue
        fid_v2_list[sparse_key].append(fid_v2)

      for feature in instance.feature:
        sparse_key = feature.name
        if sparse_key not in feature_name_sort:
          continue
        fid_v2_list[sparse_key] = feature.fid

      for sparse_key in feature_name_sort:
        if sparse_key not in fid_v2_list:
          fid_v2_list[sparse_key] = []

        fid_list = fid_v2_list[sparse_key]
        intance_tmp_dict[sparse_key].append(fid_list)

    fid_offset_map = defaultdict(list)
    fid_offset_map_unique = defaultdict(list)
    fid_map_row_split_t = defaultdict(list)
    fid_map_row_split_unique_t = defaultdict(list)

    for sparse_key in feature_name_sort:
      f_cfg = feature_cfg[sparse_key]
      table_name = f_cfg["table_name"]
      t_cfg = table_cfg[table_name]
      self.fill_row_split(ps_num, t_cfg, fid_map_t, fid_map_row_split_t)
      self.fill_row_split(ps_num, t_cfg, fid_map_unique_t,
                          fid_map_row_split_unique_t)
      for fid_list in intance_tmp_dict[sparse_key]:
        fid_offset_list = []
        fid_offset_list2 = []
        table_index = table_name_index_map[table_name]
        self.handle_feature([], fid_list, f_cfg, t_cfg, ps_num, fid_offset_list,
                            fid_offset_list2, fid_map_t, fid_map_unique_map,
                            fid_map_unique_t)
        #print("bbbbb {} {} {}".format(sparse_key, fid_offset_list,
        #                              fid_offset_list2))
        fid_offset_map[sparse_key].append(fid_offset_list)
        fid_offset_map_unique[sparse_key].append(fid_offset_list2)
    for table_name in table_name_sort:
      t_cfg = table_cfg[table_name]
      self.fill_row_split(ps_num, t_cfg, fid_map_t, fid_map_row_split_t)
      self.fill_row_split(ps_num, t_cfg, fid_map_unique_t,
                          fid_map_row_split_unique_t)

    nfl_offset_t, feature_offset_t, fid_offset_list, fid_offset_list_unique, \
      fid_map_t, fid_map_unique_t = self.get_offset_result(feature_name_sort,
                                                            table_name_sort,
                                                            ps_num,
                                                            feature_cfg,
                                                            table_cfg,
                                                            fid_offset_map,
                                                            fid_offset_map_unique,
                                                            fid_map_t,
                                                            fid_map_unique_t
                                                          )

    def parse_func(input_placeholder):
      get_default_parser_ctx().enable_fused_layout = False
      parsed_results_base = parse_instances(
          input_placeholder,
          fidv1_features=fidv1_features,
          fidv2_features=fidv2_features,
          dense_features=dense_features,
          dense_feature_shapes=dense_feature_shapes,
          dense_feature_types=dense_feature_types,
          extra_features=extra_features,
          extra_feature_shapes=extra_feature_shapes)
      get_default_parser_ctx().enable_fused_layout = True
      parsed_results = parse_instances(
          input_placeholder,
          fidv1_features=fidv1_features,
          fidv2_features=fidv2_features,
          dense_features=dense_features,
          dense_feature_shapes=dense_feature_shapes,
          dense_feature_types=dense_feature_types,
          extra_features=extra_features,
          extra_feature_shapes=extra_feature_shapes)
      return parsed_results_base, parsed_results

    self.diff_test("instance", parse_func, instance_str_list, ps_num,
                   feature_cfgs, sparse_features, dense_features,
                   extra_features, nfl_offset_t, feature_offset_t,
                   fid_offset_list, fid_offset_list_unique, fid_map_t,
                   fid_map_row_split_t, fid_map_unique_t,
                   fid_map_row_split_unique_t)
    #assert (False)


class DataOpsV2TestFitPre(tf.test.TestCase):  #DataOpsV2Test

  def __init__(self, *args, **kwargs):
    super(DataOpsV2TestFitPre, self).__init__(*args, **kwargs)

  def testExampleBatchSharding(self):
    file_name = "monolith/native_training/data/training_instance/examplebatch.data"

    sparse_features = list(features.keys())
    with open(file_name, 'rb') as stream:
      stream.read(8)  # strip lagrangex_header
      size = unpack("<Q", stream.read(8))[0]
      eb_str = stream.read(size)
      example_batch = ExampleBatch()
      example_batch.ParseFromString(eb_str)
      #add shared
      named_feature_list = example_batch.named_feature_list.add()
      named_feature_list.type = FeatureListType.SHARED
      named_feature_list.id = 990
      named_feature_list.name = "test_shared1"
      feature = named_feature_list.feature.add()
      feature.fid_v2_list.value.extend(gen_fids_v2(990, 10))
      named_feature_list = example_batch.named_feature_list.add()
      named_feature_list.type = FeatureListType.SHARED
      named_feature_list.id = 991
      named_feature_list.name = "test_shared2"
      feature = named_feature_list.feature.add()
      feature.fid_v1_list.value.extend(gen_fids_v1(991, 10))
      named_feature_list = example_batch.named_feature_list.add()
      named_feature_list.type = FeatureListType.SHARED
      named_feature_list.id = 991
      named_feature_list.name = "test_shared3"
      feature = named_feature_list.feature.add()

      eb_str = example_batch.SerializeToString()

    sparse_features += ["test_shared1", "test_shared2", "test_shared3"]
    dense_features = ['label']
    dense_feature_shapes = [2]
    dense_feature_types = [tf.float32]
    extra_features = ['uid', 'req_time', 'item_id']
    extra_feature_shapes = [1, 1, 1]

    print('==' * 10 + "sparse_features" + '==' * 10)
    print(sparse_features)

    feature_cfgs = FeatureConfigs()
    index = 0
    ps_num = 3
    table_name_index_map = {}
    for sparse_key in sparse_features:
      cfg = FeatureConfig()
      cfg.table = 'table_{}'.format(index % 3)
      table_name_index_map[cfg.table] = -1
      feature_cfgs.feature_configs[sparse_key].CopyFrom(cfg)
      index += 1

    sparse_features.sort()
    table_name_list = list(table_name_index_map.keys())
    table_name_list.sort()
    for index, table_name in enumerate(table_name_list):
      table_name_index_map[table_name] = index

    #f_goods_title_terms
    fid_map_t = defaultdict(list)
    fid_map_unique_t = defaultdict(list)
    fid_map_unique_map = defaultdict(dict)
    mask = (1 << 48) - 1

    def handle_feature(feature, table_name, table_index, fid_offset_list,
                       fid_offset_list2):
      value_list = []
      if len(feature.fid_v1_list.value) != 0:
        for fid in feature.fid_v1_list.value:
          slot_id = (fid >> 54)
          fid_v2 = ((slot_id << 48) | (mask & fid))
          value_list.append(fid_v2)
      elif len(feature.fid_v2_list.value) != 0:
        value_list = feature.fid_v2_list.value
      for value in value_list:
        shard = value % ps_num
        key = table_name + ":" + str(shard)
        fid_offset = (table_index * ps_num + shard) << 32
        fid_offset_list.append(fid_offset | len(fid_map_t[key]))
        fid_map_t[key].append(value)
        if value not in fid_map_unique_map[key]:
          fid_map_unique_map[key][value] = len(fid_map_unique_map[key])
          fid_map_unique_t[key].append(value)
        fid_offset_list2.append(fid_offset | fid_map_unique_map[key][value])

    fid_offset_map = defaultdict(list)
    fid_offset_map_unique = defaultdict(list)
    sparse_feature_shared = set()
    example_batch_feature_map = {}
    for named_feature_list in example_batch.named_feature_list:
      if named_feature_list.name not in feature_cfgs.feature_configs or \
              named_feature_list.name not in sparse_features:
        continue
      example_batch_feature_map[named_feature_list.name] = named_feature_list

    for sparse_key in sparse_features:
      if sparse_key not in example_batch_feature_map:
        continue
      named_feature_list = example_batch_feature_map[sparse_key]
      table_name = feature_cfgs.feature_configs[named_feature_list.name].table
      table_index = table_name_index_map[table_name]
      if named_feature_list.type == FeatureListType.SHARED:
        sparse_feature_shared.add(named_feature_list.name)
        feature = named_feature_list.feature[0]
        fid_offset_list = []
        fid_offset_list2 = []
        handle_feature(feature, table_name, table_index, fid_offset_list,
                       fid_offset_list2)
        fid_offset_map[named_feature_list.name].append(fid_offset_list)
        fid_offset_map_unique[named_feature_list.name].append(fid_offset_list2)
      else:
        for feature in named_feature_list.feature:
          fid_offset_list = []
          fid_offset_list2 = []
          handle_feature(feature, table_name, table_index, fid_offset_list,
                         fid_offset_list2)
          fid_offset_map[named_feature_list.name].append(fid_offset_list)
          fid_offset_map_unique[named_feature_list.name].append(
              fid_offset_list2)

    feature_offset_t = []
    nfl_offset_t = []
    fid_offset_list = []
    fid_offset_list_unique = []
    for sparse_key in sparse_features:
      if sparse_key in sparse_feature_shared:
        nfl_offset = len(feature_offset_t) | 1 << 31
        #pass
      else:
        nfl_offset = len(feature_offset_t)
      nfl_offset_t.append(nfl_offset)
      if sparse_key not in fid_offset_map:
        continue
      for fid_list in fid_offset_map[sparse_key]:
        feature_offset_t.append(len(fid_offset_list))
        fid_offset_list.extend(fid_list)
      for fid_list in fid_offset_map_unique[sparse_key]:
        fid_offset_list_unique.extend(fid_list)

    print('==' * 10 + "fid_map_t" + '==' * 10)
    #print(fid_map_t)
    print('==' * 10 + "fid_map_unique_t" + '==' * 10)
    #print(fid_map_unique_t)
    print('==' * 10 + "fid_offset_map" + '==' * 10)
    #print(fid_offset_map)
    #print('==' * 10 + "example_batch" + '==' * 10)
    #print(example_batch)

    with tf.Graph().as_default():
      config = tf.compat.v1.ConfigProto()
      config.graph_options.rewrite_options.disable_meta_optimizer = True
      examples_placeholder = tf.compat.v1.placeholder(dtype=tf.string,
                                                      shape=(None,))
      get_default_parser_ctx().enable_fused_layout = False
      parsed_results_base = parse_example_batch(
          examples_placeholder,
          sparse_features=sparse_features,
          dense_features=dense_features,
          dense_feature_shapes=dense_feature_shapes,
          dense_feature_types=dense_feature_types,
          extra_features=extra_features,
          extra_feature_shapes=extra_feature_shapes)
      get_default_parser_ctx().enable_fused_layout = True
      parsed_results = parse_example_batch(
          examples_placeholder,
          sparse_features=[],
          dense_features=dense_features,
          dense_feature_shapes=dense_feature_shapes,
          dense_feature_types=dense_feature_types,
          extra_features=extra_features,
          extra_feature_shapes=extra_feature_shapes)
      example_batch_varint = parsed_results.pop("sparse_features")

      parallel_flag_list = [1, 2, 3, 4]
      fid_map_list = []
      fid_map_unique_list = []
      for parallel_flag in parallel_flag_list:
        fid_map, fid_offset, feature_offset, nfl_offset, batch_size, fid_list_row_splits = sharding_sparse_fids(
            example_batch_varint,
            ps_num,
            feature_cfgs,
            False,
            "examplebatch",
            parallel_flag,
            version=1)
        fid_map_list.append([fid_map, fid_offset, feature_offset, nfl_offset])
        fid_map_unique, fid_offset, feature_offset, nfl_offset, batch_size, fid_list_row_splits = sharding_sparse_fids(
            example_batch_varint,
            ps_num,
            feature_cfgs,
            True,
            "examplebatch",
            parallel_flag,
            version=1)
        fid_map_unique_list.append(
            [fid_map_unique, fid_offset, feature_offset, nfl_offset])

      with self.session(config=config) as sess:
        parsed_results_base1, parsed_results1 = sess.run(
            fetches=[parsed_results_base, parsed_results],
            feed_dict={examples_placeholder: [eb_str]})

        def diff(k, a, b, sort=False):

          if not isinstance(a[0], list) and sort:
            a.sort()
            b.sort()
          #print("diff:a {} {}".format(k, a), flush=True)
          #print("diff:b {} {}".format(k, b), flush=True)
          assert (len(a) == len(b))
          if (len(a) == 0):
            return
          for i in range(len(a)):
            if isinstance(a[i], list):
              assert isinstance(b[i], list)
              diff(k + "/" + str(i), a[i], b[i], sort)
            else:
              assert (a[i] == b[i]), f"{i}: {a[i]} / {b[i]}"

        #print('==' * 10 + "parsed_results_base1" + '==' * 10, flush=True)
        #print('==' * 10 + "parsed_results1" + '==' * 10, flush=True)
        # .numpy()
        for k, v in parsed_results_base1.items():
          if k in sparse_features:
            continue
          if k in dense_features + extra_features:
            if k not in parsed_results1:
              print("no find {} in parse_example_batch_v2".format(k))
              assert (False)
            diff(k, v.tolist(), parsed_results1[k].tolist())  #.numpy()
          else:
            print("no need {}".format(k), flush=True)
            assert (False)
        for k, v in parsed_results1.items():
          if k not in dense_features + extra_features:
            print("no need {}".format(k), flush=True)
            assert (False)

        for fid_map_index in range(len(parallel_flag_list)):
          fid_map = fid_map_list[fid_map_index]
          fid_map_unique = fid_map_unique_list[fid_map_index]
          fid_map1_list, fid_map_unique1_list = sess.run(
              fetches=[fid_map, fid_map_unique],
              feed_dict={examples_placeholder: [eb_str]})

          #print('==' * 10 + "fid_map1" + '==' * 10, flush=True)
          #print(fid_map1, flush=True)
          #print('==' * 10 + "fid_map_unique1" + '==' * 10, flush=True)
          #print(fid_map_unique1, flush=True)
          fid_map1, fid_offset1, feature_offset1, nfl_offset1 = fid_map1_list
          fid_map2, fid_offset2, feature_offset2, nfl_offset2 = fid_map_unique1_list
          print('==' * 10 + "diff fidoffset " +
                str(parallel_flag_list[fid_map_index]) + '==' * 10,
                flush=True)
          diff("nfl_offset", nfl_offset1, nfl_offset_t)
          diff("nfl_offset2", nfl_offset2, nfl_offset_t)
          diff("feature_offset", feature_offset1, feature_offset_t)
          diff("feature_offset2", feature_offset2, feature_offset_t)
          diff("fid_offset", fid_offset1, fid_offset_list)
          diff("fid_offset2", fid_offset2, fid_offset_list_unique)

          assert (len(fid_map_t) == len(fid_map1))
          assert (len(fid_map_unique_t) == len(fid_map2))

          def fid_diff(a, b):
            for k, v in a.items():
              assert (k in b)
              diff(k, v.tolist(), b[k], True)  #.numpy()
            for k, v in b.items():
              assert (k in a)

          print('==' * 10 + "diff fid_map1 " +
                str(parallel_flag_list[fid_map_index]) + '==' * 10,
                flush=True)
          fid_diff(fid_map1, fid_map_t)
          print('==' * 10 + "diff fid_map_unique1 " +
                str(parallel_flag_list[fid_map_index]) + '==' * 10,
                flush=True)
          fid_diff(fid_map2, fid_map_unique_t)
        #assert (False)

  def testExampleSharding(self):
    sparse_features_set = set()
    dense_features = ['label']
    dense_feature_shapes = [2]
    dense_feature_types = [tf.float32]
    extra_features = ['uid', 'req_time', 'item_id', 'actions']
    extra_feature_shapes = [1, 1, 1, 1]
    example_str_list = []
    example_list = []
    file_name = "monolith/native_training/data/training_instance/example.pb"
    with open(file_name, 'rb') as stream:
      while (True):
        if len(example_str_list) > 10:
          break
        try:
          stream.read(8)  # strip has_sort_id
          stream.read(8)  # strip kafka_dump
          size = unpack("<Q", stream.read(8))[0]
          example_str = stream.read(size)

          example = Example()
          example.ParseFromString(example_str)
          for feature_index in range(1, 3):
            named_feature = example.named_feature.add()
            named_feature.name = 'fc_slot_9999{}'.format(feature_index)
            fid_list = gen_fids_v2(feature_index, 10)
            named_feature.feature.fid_v2_list.value.extend(fid_list)
            named_feature.feature.fid_v2_list.value.extend(fid_list)
          for named_feature in example.named_feature:
            if "fc_slot_" in named_feature.name:
              sparse_features_set.add(named_feature.name)
          example_list.append(example)
          example_str_list.append(example.SerializeToString())
        except:
          break

    sparse_features = sorted(sparse_features_set)
    print(sparse_features, flush=True)

    feature_cfgs = FeatureConfigs()
    index = 0
    ps_num = 3
    table_name_index_map = {}
    for sparse_key in sparse_features:
      cfg = FeatureConfig()
      cfg.table = 'table_{}'.format(index % 3)
      table_name_index_map[cfg.table] = -1
      feature_cfgs.feature_configs[sparse_key].CopyFrom(cfg)
      index += 1

    sparse_features.sort()
    table_name_list = list(table_name_index_map.keys())
    table_name_list.sort()
    for index, table_name in enumerate(table_name_list):
      table_name_index_map[table_name] = index

    fid_map_t = defaultdict(list)
    fid_map_unique_t = defaultdict(list)
    fid_map_unique_map = defaultdict(dict)
    mask = (1 << 48) - 1

    def handle_feature(feature, table_name, table_index, fid_offset_list,
                       fid_offset_list2):
      value_list = []
      if len(feature.fid_v1_list.value) != 0:
        for fid in feature.fid_v1_list.value:
          slot_id = (fid >> 54)
          fid_v2 = ((slot_id << 48) | (mask & fid))
          value_list.append(fid_v2)
      elif len(feature.fid_v2_list.value) != 0:
        value_list = feature.fid_v2_list.value
      for value in value_list:
        shard = value % ps_num
        key = table_name + ":" + str(shard)
        fid_offset = (table_index * ps_num + shard) << 32
        fid_offset_list.append(fid_offset | len(fid_map_t[key]))
        fid_map_t[key].append(value)
        if value not in fid_map_unique_map[key]:
          fid_map_unique_map[key][value] = len(fid_map_unique_map[key])
          fid_map_unique_t[key].append(value)
        fid_offset_list2.append(fid_offset | fid_map_unique_map[key][value])

    fid_offset_map = defaultdict(list)
    fid_offset_map_unique = defaultdict(list)
    for sparse_key in sparse_features:
      for example in example_list:
        find_named_feature = None
        for named_feature in example.named_feature:
          if named_feature.name == sparse_key:
            find_named_feature = named_feature
            break
        fid_offset_list = []
        fid_offset_list2 = []
        if find_named_feature:
          table_name = feature_cfgs.feature_configs[sparse_key].table
          table_index = table_name_index_map[table_name]
          handle_feature(find_named_feature.feature, table_name, table_index,
                         fid_offset_list, fid_offset_list2)
        fid_offset_map[sparse_key].append(fid_offset_list)
        fid_offset_map_unique[sparse_key].append(fid_offset_list2)

    feature_offset_t = []
    nfl_offset_t = []
    fid_offset_list = []
    fid_offset_list_unique = []
    for sparse_key in sparse_features:
      nfl_offset = len(feature_offset_t)
      nfl_offset_t.append(nfl_offset)
      if sparse_key not in fid_offset_map:
        continue
      for fid_list in fid_offset_map[sparse_key]:
        feature_offset_t.append(len(fid_offset_list))
        fid_offset_list.extend(fid_list)
      for fid_list in fid_offset_map_unique[sparse_key]:
        fid_offset_list_unique.extend(fid_list)

    print('==' * 10 + "fid_map_t" + '==' * 10, flush=True)
    #print(fid_map_t, flush=True)
    print('==' * 10 + "fid_map_unique_t" + '==' * 10, flush=True)
    #print(fid_map_unique_t, flush=True)
    print('==' * 10 + "fid_offset_map" + '==' * 10)
    #print(fid_offset_map)

    #example_tensor = tf.convert_to_tensor(example_str_list)
    with tf.Graph().as_default():
      config = tf.compat.v1.ConfigProto()
      config.graph_options.rewrite_options.disable_meta_optimizer = True
      examples_placeholder = tf.compat.v1.placeholder(dtype=tf.string,
                                                      shape=(None))

      get_default_parser_ctx().enable_fused_layout = False
      parsed_results_base = parse_examples(
          examples_placeholder,
          sparse_features=sparse_features,
          dense_features=dense_features,
          dense_feature_shapes=dense_feature_shapes,
          dense_feature_types=dense_feature_types,
          extra_features=extra_features,
          extra_feature_shapes=extra_feature_shapes)
      get_default_parser_ctx().enable_fused_layout = True
      parsed_results = parse_examples(examples_placeholder,
                                      sparse_features=[],
                                      dense_features=dense_features,
                                      dense_feature_shapes=dense_feature_shapes,
                                      dense_feature_types=dense_feature_types,
                                      extra_features=extra_features,
                                      extra_feature_shapes=extra_feature_shapes)
      examples_varint = parsed_results.pop("sparse_features")
      parallel_flag_list = [1, 2, 3, 4]
      fid_map_list = []
      fid_map_unique_list = []
      for parallel_flag in parallel_flag_list:
        fid_map, fid_offset, feature_offset, nfl_offset, batch_size, fid_list_row_splits = sharding_sparse_fids(
            examples_varint,
            ps_num,
            feature_cfgs,
            False,
            "example",
            parallel_flag,
            version=1)
        fid_map_list.append([fid_map, fid_offset, feature_offset, nfl_offset])
        fid_map_unique, fid_offset, feature_offset, nfl_offset, batch_size, fid_list_row_splits = sharding_sparse_fids(
            examples_varint,
            ps_num,
            feature_cfgs,
            True,
            "example",
            parallel_flag,
            version=1)
        fid_map_unique_list.append(
            [fid_map_unique, fid_offset, feature_offset, nfl_offset])

      with self.session(config=config) as sess:
        parsed_results_base1, parsed_results1 = sess.run(
            fetches=[parsed_results_base, parsed_results],
            feed_dict={examples_placeholder: example_str_list})

        def diff(k, a, b, sort=False):
          if not isinstance(a[0], list) and sort:
            a.sort()
            b.sort()

          def print_func():
            print("diff:a {} {}".format(k, a), flush=True)
            print("diff:b {} {}".format(k, b), flush=True)
            return "{}, {}".format(len(a), len(b))

          assert (len(a) == len(b)), print_func()
          if (len(a) == 0):
            return
          for i in range(len(a)):
            if isinstance(a[i], list):
              assert isinstance(b[i], list), print_func()
              diff(k + "/" + str(i), a[i], b[i], sort)
            else:
              assert (a[i] == b[i]), print_func()

        #print('==' * 10 + "parsed_results_base1" + '==' * 10, flush=True)
        #print(parsed_results_base1, flush=True)
        #print('==' * 10 + "parsed_results1" + '==' * 10, flush=True)
        #print(parsed_results1, flush=True)
        # .numpy()
        for k, v in parsed_results_base1.items():
          if k in sparse_features:
            continue
          if k in dense_features + extra_features:
            if k not in parsed_results1:
              print("no find {} in parse_example_batch_v2".format(k),
                    flush=True)
              assert (False)
            diff(k, v.tolist(), parsed_results1[k].tolist())  #.numpy()
          else:
            print("no need {}".format(k), flush=True)
            assert (False)
        for k, v in parsed_results1.items():
          if k not in dense_features + extra_features:
            print("no need {}".format(k), flush=True)
            assert (False)

        for fid_map_index in range(len(parallel_flag_list)):
          fid_map = fid_map_list[fid_map_index]
          fid_map_unique = fid_map_unique_list[fid_map_index]
          fid_map1_list, fid_map_unique1_list = sess.run(
              fetches=[fid_map, fid_map_unique],
              feed_dict={examples_placeholder: example_str_list})

          #print('==' * 10 + "fid_map1" + '==' * 10, flush=True)
          #print(fid_map1, flush=True)
          #print('==' * 10 + "fid_map_unique1" + '==' * 10, flush=True)
          #print(fid_map_unique1, flush=True)
          fid_map1, fid_offset1, feature_offset1, nfl_offset1 = fid_map1_list
          fid_map2, fid_offset2, feature_offset2, nfl_offset2 = fid_map_unique1_list
          print('==' * 10 + "diff fidoffset " +
                str(parallel_flag_list[fid_map_index]) + '==' * 10,
                flush=True)
          diff("nfl_offset", list(nfl_offset1), nfl_offset_t)
          diff("nfl_offset2", list(nfl_offset2), nfl_offset_t)
          diff("feature_offset", list(feature_offset1), feature_offset_t)
          diff("feature_offset2", list(feature_offset2), feature_offset_t)
          diff("fid_offset", list(fid_offset1), fid_offset_list)
          diff("fid_offset2", list(fid_offset2), fid_offset_list_unique)

          print('==' * 10 + "fid_map1" + '==' * 10, flush=True)
          print('==' * 10 + "fid_map_unique1" + '==' * 10, flush=True)

          assert (len(fid_map_t) == len(fid_map1))
          assert (len(fid_map_unique_t) == len(fid_map2))

          def fid_diff(a, b):
            for k, v in a.items():
              assert (k in b)
              diff(k, v.tolist(), b[k], True)  #.numpy()
            for k, v in b.items():
              assert (k in a)

          print('==' * 10 + "diff fid_map1 " +
                str(parallel_flag_list[fid_map_index]) + '==' * 10,
                flush=True)
          fid_diff(fid_map1, fid_map_t)
          print('==' * 10 + "diff fid_map_unique1 " +
                str(parallel_flag_list[fid_map_index]) + '==' * 10,
                flush=True)
          fid_diff(fid_map2, fid_map_unique_t)
        #assert (False)

  def testInstanceSharding(self):
    fidv1_features = [1, 200, 3, 5, 9, 203, 205]
    fidv2_features = ["fc_v2_1", "fc_v2_2", "fc_v2_3"]
    dense_features = ['label']
    dense_feature_shapes = [2]
    dense_feature_types = [tf.float32]
    extra_features = ['uid', 'req_time', 'item_id', 'actions']
    extra_feature_shapes = [1, 1, 1, 2]

    instance_str_list = []
    instance_list = []
    while (len(instance_str_list) < 128):
      instance = gen_instance(
          fidv1_features=fidv1_features,
          fidv2_features=[],
          dense_features=[FeatureMeta('label', shape=2, dtype=tf.float32)],
          extra_features=[
              FeatureMeta('actions', shape=2),
              FeatureMeta('uid'),
              FeatureMeta('req_time', dtype=tf.int32),
              FeatureMeta('item_id'),
          ])
      instance_list.append(instance)
      instance_str_list.append(instance.SerializeToString())

      instance2 = deepcopy(instance)
      for slot, feature_name in enumerate(fidv2_features):
        feature = instance2.feature.add()
        feature.name = feature_name
        feature.fid.extend(gen_fids_v2(1000 + slot, 10))
      instance_list.append(instance2)
      instance_str_list.append(instance2.SerializeToString())

      instance3 = deepcopy(instance2)
      del instance3.fid[:]
      instance_list.append(instance3)
      instance_str_list.append(instance3.SerializeToString())

    def gen_slot_feature_name(slot_id):
      return f"slot_{slot_id}"

    sparse_features = sorted(
        fidv2_features +
        [gen_slot_feature_name(slot_id) for slot_id in fidv1_features])
    print(sparse_features, flush=True)

    feature_cfgs = FeatureConfigs()
    index = 0
    ps_num = 3
    table_name_index_map = {}
    for sparse_key in sparse_features:
      cfg = FeatureConfig()
      cfg.table = 'table_{}'.format(index % 3)
      table_name_index_map[cfg.table] = -1
      feature_cfgs.feature_configs[sparse_key].CopyFrom(cfg)
      index += 1

    sparse_features.sort()
    table_name_list = list(table_name_index_map.keys())
    table_name_list.sort()
    for index, table_name in enumerate(table_name_list):
      table_name_index_map[table_name] = index

    fid_map_t = defaultdict(list)
    fid_map_unique_t = defaultdict(list)
    fid_map_unique_map = defaultdict(dict)
    mask = (1 << 48) - 1

    def handle_feature(value_list, table_name, table_index, fid_offset_list,
                       fid_offset_list2):
      for value in value_list:
        shard = value % ps_num
        key = table_name + ":" + str(shard)
        fid_offset = (table_index * ps_num + shard) << 32
        fid_offset_list.append(fid_offset | len(fid_map_t[key]))
        fid_map_t[key].append(value)
        if value not in fid_map_unique_map[key]:
          fid_map_unique_map[key][value] = len(fid_map_unique_map[key])
          fid_map_unique_t[key].append(value)
        fid_offset_list2.append(fid_offset | fid_map_unique_map[key][value])

    def slot_id_v1(fid):
      return fid >> 54

    intance_tmp_dict = defaultdict(list)
    for instance in instance_list:
      fid_v2_list = defaultdict(list)
      for fid in instance.fid:
        slot_id = slot_id_v1(fid)
        sparse_key = gen_slot_feature_name(slot_id)
        if sparse_key not in sparse_features:
          continue
        fid_v2 = ((slot_id << 48) | (mask & fid))
        fid_v2_list[sparse_key].append(fid_v2)

      for feature in instance.feature:
        sparse_key = feature.name
        if sparse_key not in sparse_features:
          continue
        fid_v2_list[sparse_key] = feature.fid

      for sparse_key in sparse_features:
        if sparse_key not in fid_v2_list:
          fid_v2_list[sparse_key] = []

        fid_list = fid_v2_list[sparse_key]
        intance_tmp_dict[sparse_key].append(fid_list)

    fid_offset_map = defaultdict(list)
    fid_offset_map_unique = defaultdict(list)
    for sparse_key in sparse_features:
      for fid_list in intance_tmp_dict[sparse_key]:
        fid_offset_list = []
        fid_offset_list2 = []
        table_name = feature_cfgs.feature_configs[sparse_key].table
        table_index = table_name_index_map[table_name]
        handle_feature(fid_list, table_name, table_index, fid_offset_list,
                       fid_offset_list2)
        fid_offset_map[sparse_key].append(fid_offset_list)
        fid_offset_map_unique[sparse_key].append(fid_offset_list2)

    feature_offset_t = []
    nfl_offset_t = []
    fid_offset_list = []
    fid_offset_list_unique = []
    for sparse_key in sparse_features:
      nfl_offset = len(feature_offset_t)
      nfl_offset_t.append(nfl_offset)
      if sparse_key not in fid_offset_map:
        continue
      for fid_list in fid_offset_map[sparse_key]:
        feature_offset_t.append(len(fid_offset_list))
        fid_offset_list.extend(fid_list)
      for fid_list in fid_offset_map_unique[sparse_key]:
        fid_offset_list_unique.extend(fid_list)

    print('==' * 10 + "fid_map_t" + '==' * 10, flush=True)
    #print(fid_map_t, flush=True)
    print('==' * 10 + "fid_map_unique_t" + '==' * 10, flush=True)
    #print(fid_map_unique_t, flush=True)
    print('==' * 10 + "fid_offset_map" + '==' * 10)
    print("xxxxx:", len(feature_offset_t), len(instance_list))
    #print(fid_offset_map)

    #example_tensor = tf.convert_to_tensor(example_str_list)
    with tf.Graph().as_default():
      config = tf.compat.v1.ConfigProto()
      config.graph_options.rewrite_options.disable_meta_optimizer = True
      examples_placeholder = tf.compat.v1.placeholder(dtype=tf.string,
                                                      shape=(None))

      get_default_parser_ctx().enable_fused_layout = False
      parsed_results_base = parse_instances(
          examples_placeholder,
          fidv1_features=fidv1_features,
          fidv2_features=fidv2_features,
          dense_features=dense_features,
          dense_feature_shapes=dense_feature_shapes,
          dense_feature_types=dense_feature_types,
          extra_features=extra_features,
          extra_feature_shapes=extra_feature_shapes)
      get_default_parser_ctx().enable_fused_layout = True
      parsed_results = parse_instances(
          examples_placeholder,
          fidv1_features=fidv1_features,
          fidv2_features=fidv2_features,
          dense_features=dense_features,
          dense_feature_shapes=dense_feature_shapes,
          dense_feature_types=dense_feature_types,
          extra_features=extra_features,
          extra_feature_shapes=extra_feature_shapes)
      examples_varint = parsed_results.pop("sparse_features")
      parallel_flag_list = [1, 2, 3, 4]
      fid_map_list = []
      fid_map_unique_list = []
      for parallel_flag in parallel_flag_list:
        fid_map, fid_offset, feature_offset, nfl_offset, batch_size, fid_list_row_splits = sharding_sparse_fids(
            examples_varint,
            ps_num,
            feature_cfgs,
            False,
            "instance",
            parallel_flag,
            version=1)
        fid_map_list.append([fid_map, fid_offset, feature_offset, nfl_offset])
        fid_map_unique, fid_offset, feature_offset, nfl_offset, batch_size, fid_list_row_splits = sharding_sparse_fids(
            examples_varint,
            ps_num,
            feature_cfgs,
            True,
            "instance",
            parallel_flag,
            version=1)
        fid_map_unique_list.append(
            [fid_map_unique, fid_offset, feature_offset, nfl_offset])

      with self.session(config=config) as sess:
        parsed_results_base1, parsed_results1 = sess.run(
            fetches=[parsed_results_base, parsed_results],
            feed_dict={examples_placeholder: instance_str_list})

        def diff(k, a, b, sort=False):
          if not isinstance(a[0], list) and sort:
            a.sort()
            b.sort()

          def print_func():
            print("diff:a {} {}".format(k, a), flush=True)
            print("diff:b {} {}".format(k, b), flush=True)
            return "{}, {}".format(len(a), len(b))

          assert (len(a) == len(b)), print_func()
          if (len(a) == 0):
            return
          for i in range(len(a)):
            if isinstance(a[i], list):
              assert isinstance(b[i], list), print_func()
              diff(k + "/" + str(i), a[i], b[i], sort)
            else:
              assert (a[i] == b[i]), print_func()

        print('==' * 10 + "parsed_results_base1" + '==' * 10, flush=True)
        print(parsed_results_base1, flush=True)
        print('==' * 10 + "parsed_results1" + '==' * 10, flush=True)
        print(parsed_results1, flush=True)
        # .numpy()
        for k, v in parsed_results_base1.items():
          if k in sparse_features:
            continue
          if k in dense_features + extra_features:
            if k not in parsed_results1:
              print("no find {} in parse_example_batch_v2".format(k),
                    flush=True)
              assert (False)
            diff(k, v.tolist(), parsed_results1[k].tolist())  #.numpy()
          else:
            print("no need {}".format(k), flush=True)
            assert (False)
        for k, v in parsed_results1.items():
          if k not in dense_features + extra_features:
            print("no need {}".format(k), flush=True)
            assert (False)

        for fid_map_index in range(len(parallel_flag_list)):
          fid_map = fid_map_list[fid_map_index]
          fid_map_unique = fid_map_unique_list[fid_map_index]
          fid_map1_list, fid_map_unique1_list = sess.run(
              fetches=[fid_map, fid_map_unique],
              feed_dict={examples_placeholder: instance_str_list})

          #print('==' * 10 + "fid_map1" + '==' * 10, flush=True)
          #print('==' * 10 + "fid_map_unique1" + '==' * 10, flush=True)
          fid_map1, fid_offset1, feature_offset1, nfl_offset1 = fid_map1_list
          fid_map2, fid_offset2, feature_offset2, nfl_offset2 = fid_map_unique1_list
          print('==' * 10 + "diff fidoffset " +
                str(parallel_flag_list[fid_map_index]) + '==' * 10,
                flush=True)
          diff("nfl_offset", list(nfl_offset1), nfl_offset_t)
          diff("nfl_offset2", list(nfl_offset2), nfl_offset_t)
          diff("feature_offset", list(feature_offset1), feature_offset_t)
          diff("feature_offset2", list(feature_offset2), feature_offset_t)
          diff("fid_offset", list(fid_offset1), fid_offset_list)
          diff("fid_offset2", list(fid_offset2), fid_offset_list_unique)

          print('==' * 10 + "fid_map1" + '==' * 10, flush=True)
          #print(fid_map1, flush=True)
          print('==' * 10 + "fid_map_unique1" + '==' * 10, flush=True)
          #print(fid_map_unique1, flush=True)

          assert (len(fid_map_t) == len(fid_map1))
          assert (len(fid_map_unique_t) == len(fid_map2))

          def fid_diff(a, b):
            for k, v in a.items():
              assert (k in b)
              diff(k, v.tolist(), b[k], True)  #.numpy()
            for k, v in b.items():
              assert (k in a)

          print('==' * 10 + "diff fid_map1 " +
                str(parallel_flag_list[fid_map_index]) + '==' * 10,
                flush=True)
          fid_diff(fid_map1, fid_map_t)
          print('==' * 10 + "diff fid_map_unique1 " +
                str(parallel_flag_list[fid_map_index]) + '==' * 10,
                flush=True)
          fid_diff(fid_map2, fid_map_unique_t)
        #assert (False)


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
