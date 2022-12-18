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

import logging
import string
import numpy as np

np.random.seed(2)
from random import randint
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

from collections import defaultdict
from tensorflow.python.framework import test_util
from monolith.native_training import distribution_ops
from idl.matrix.proto.example_pb2 import ExampleBatch, Example, FeatureListType, \
  SliceConfig, PoolingType, OutType, OutConfig, FeatureConfig, FeatureConfigs, TensorShape
from monolith.native_training.data.parsers import parse_instances, parse_examples, parse_example_batch, \
    sharding_sparse_fids, get_default_parser_ctx

SHARD_BIT = 0x80000000


def infer_shape(out_conf: OutConfig,
                out_type: OutType,
                max_sequence_length: int = 0):
  out_conf.out_type = out_type
  if out_type == OutType.NONE:
    for sc in out_conf.slice_configs:
      shape = out_conf.shape.add()
      if max_sequence_length > 0:
        shape.dims.extend([-1, max_sequence_length, sc.end - sc.start])
      else:
        shape.dims.extend([-1, sc.end - sc.start])
  elif out_type == OutType.CONCAT:
    shape = out_conf.shape.add()
    last_dim = 0
    for sc in out_conf.slice_configs:
      last_dim += sc.end - sc.start
    if max_sequence_length > 0:
      shape.dims.extend([-1, max_sequence_length, last_dim])
    else:
      shape.dims.extend([-1, last_dim])
  elif out_type == OutType.STACK:
    shape = out_conf.shape.add()
    last_dim = None
    for sc in out_conf.slice_configs:
      if last_dim is None:
        last_dim = sc.end - sc.start
      else:
        assert last_dim == sc.end - sc.start
    if max_sequence_length > 0:
      shape.dims.extend(
          [-1, len(out_conf.slice_configs), max_sequence_length, last_dim])
    else:
      shape.dims.extend([-1, len(out_conf.slice_configs), last_dim])
  elif out_type == OutType.ADDN:
    shape = out_conf.shape.add()
    last_dim = None
    for sc in out_conf.slice_configs:
      if last_dim is None:
        last_dim = sc.end - sc.start
      else:
        assert last_dim == sc.end - sc.start

    if max_sequence_length > 0:
      shape.dims.extend([-1, max_sequence_length, last_dim])
    else:
      shape.dims.extend([-1, last_dim])
  else:
    raise ValueError('out_type error')


def get_key(ln: str, sc: SliceConfig) -> str:
  return f"{ln}_{sc.feature_name}_{sc.start}_{sc.end}"


def pooling(pooling_type, in_data, max_length):
  if max_length and len(in_data) > max_length:
    data = in_data[0:max_length]
  else:
    data = in_data
  if pooling_type == PoolingType.SUM:
    result = np.zeros_like(data[0])
    for d in data:
      result += d
    return result
  if pooling_type == PoolingType.MEAN:
    result = np.zeros_like(data[0])
    for d in data:
      result += d
    result /= len(data)
    return result
  else:
    last_dim = int(data[0].shape[-1])
    result = np.zeros(shape=(max_length, last_dim), dtype=np.float32)
    for i, d in enumerate(data):
      result[i, :] = d
      if i < max_length:
        result[i, :] = d
      else:
        break
    return result


class FusedEmbeddingToLayoutTest(tf.test.TestCase):

  def get_pre_output_offset(self, shard, f_cfg):
    return f_cfg["pre_output_index"] + shard * f_cfg[
        "table_feature_count"] + f_cfg["feature_in_table_index"]

  def get_feature_cfg(self, raw_feature_cfgs, ps_num):
    feature_cfg = defaultdict(dict)
    table_cfg = defaultdict(dict)
    for feature_name, cfg in raw_feature_cfgs.feature_configs.items():
      dim = 0
      for slice_dim in cfg.slice_dims:
        dim += slice_dim
      feature_cfg[feature_name] = {
          "feature_name": feature_name,
          "feature_index": -1,
          "table_name": cfg.table,
          "table_index": -1,
          "feature_in_table_index": -1,
          "table_feature_count": 0,
          "pre_output_index": 0,
          "dim_sum": dim,
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

  def test_fused_embedding_to_layout(self,
                                     use_shard_op=False,
                                     parallel_flag=0x11):
    batch_size = 10
    num_ps = 5
    slot_count = 12
    slot_table_split = [5,
                        8]  #slot split for [table_one, table_two, table_three]
    max_sequence_length = 3
    feature_cfgs = FeatureConfigs()
    bias = OutConfig()
    vec = OutConfig()
    ffm1 = OutConfig()
    ffm2 = OutConfig()
    firstN = OutConfig()

    for slot in range(1, slot_count):
      feature_name = f"fc_slot_{slot}"
      fconf = FeatureConfig()
      if slot >= slot_table_split[1]:
        table_name = "table_one"  #table_three, but now test for table with different dim
        slice_dims = [1, 4, 16]
        sequence_length = max_sequence_length
        pooling_type = PoolingType.FIRSTN
        slice_config = firstN.slice_configs.add()
        slice_config.feature_name = feature_name
        slice_config.start = 1
        slice_config.end = 21
      else:
        sequence_length = 0
        if slot < slot_table_split[0]:
          table_name = "table_one"
          slice_dims = [1, 4, 8]
          pooling_type = PoolingType.SUM
          slice_config = ffm1.slice_configs.add()
          slice_config.feature_name = feature_name
          slice_config.start = 5
          slice_config.end = 13
        else:
          table_name = "table_two"
          slice_dims = [1, 4, 16]
          pooling_type = PoolingType.MEAN
          slice_config = ffm2.slice_configs.add()
          slice_config.feature_name = feature_name
          slice_config.start = 5
          slice_config.end = 21

        slice_config = bias.slice_configs.add()
        slice_config.feature_name = feature_name
        slice_config.start = 0
        slice_config.end = 1
        slice_config = vec.slice_configs.add()
        slice_config.feature_name = feature_name
        slice_config.start = 1
        slice_config.end = 5

      fconf.table = table_name
      fconf.slice_dims.extend(slice_dims)
      fconf.max_sequence_length = sequence_length
      fconf.pooling_type = pooling_type
      feature_cfgs.feature_configs[feature_name].CopyFrom(fconf)

    infer_shape(bias, OutType.ADDN)
    feature_cfgs.out_configs['bias'].CopyFrom(bias)
    infer_shape(vec, OutType.CONCAT)
    feature_cfgs.out_configs['vec'].CopyFrom(vec)
    infer_shape(ffm1, OutType.STACK)
    feature_cfgs.out_configs['ffm1'].CopyFrom(ffm1)
    infer_shape(ffm2, OutType.NONE)
    feature_cfgs.out_configs['ffm2'].CopyFrom(ffm2)
    infer_shape(firstN, OutType.NONE, max_sequence_length)
    feature_cfgs.out_configs['firstN'].CopyFrom(firstN)

    logging.info(f"feature_cfgs : {feature_cfgs} ")
    feature_cfg, table_cfg, feature_name_sort, table_name_sort = self.get_feature_cfg(
        feature_cfgs, num_ps)

    fid_offset_list = list()
    feature_offset_list = [0]
    nfl_offset_list = [0]
    nfl_offset_list2 = [0]

    sparse_features = ExampleBatch(batch_size=batch_size)
    std_features = defaultdict(list)
    fids_dict = {}
    fid_row_split_list = [[0] for _ in range(num_ps * len(table_name_sort))]

    for feature_name in feature_name_sort:
      slot = int(feature_name.split("fc_slot_")[-1])
      named_feature_list = sparse_features.named_feature_list.add()
      named_feature_list.id = slot
      named_feature_list.name = feature_name
      is_shared = True if slot % 2 == 0 else False
      logging.info(f"show shared {named_feature_list.name} {is_shared}")
      named_feature_list.type = FeatureListType.SHARED if is_shared else FeatureListType.INDIVIDUAL
      f_cfg = feature_cfg[feature_name]
      table_name = f_cfg["table_name"]
      t_cfg = table_cfg[table_name]
      table_index = t_cfg["table_index"]
      dim_sum = f_cfg["dim_sum"]
      if table_name not in fids_dict:
        fids_dict[table_name] = defaultdict(list)
      index2 = [0] * num_ps * len(table_cfg)

      def make_fids(feature):
        std_features[named_feature_list.name].append(feature)
        '''
        fids = list(
            set([(slot << 48) + randint(100, 1000000)
                 for _ in range(randint(1, 5))]))
        '''
        fids = list(
            set([(slot * 10000) + (i + 1) * 1000 + randint(1, 9) * 100
                 for i in range(randint(1, max_sequence_length * 2))]))
        logging.info(f"show fids {fids}")
        feature.fid_v2_list.value.extend(fids)
        for fid in fids:
          idx = fid % num_ps
          full_index = self.get_pre_output_offset(idx, f_cfg)
          index1 = table_index * num_ps + idx
          fid_offset = full_index << 32 | index2[index1]
          index2[index1] += 1

          fid_offset_list.append(fid_offset)
          fids_dict[table_name][idx].append((dim_sum, fid))
        feature_offset_list.append(len(fid_offset_list))

      if is_shared:
        feature = named_feature_list.feature.add()
        make_fids(feature)
      else:
        for _ in range(batch_size):
          feature = named_feature_list.feature.add()
          make_fids(feature)

      for ps_i in range(num_ps):
        fid_row_split_list[table_index * num_ps + ps_i].append(
            len(fids_dict[table_name][ps_i]))

      nfl_index = len(feature_offset_list) - 1
      if is_shared:  # add shared encode, 向前一位
        nfl_offset_list[-1] |= SHARD_BIT
      nfl_offset_list.append(nfl_index)

    logging.info(f"show fid_row_split_list: {fid_row_split_list}")

    logging.info(f"sparse_features : {sparse_features} ")

    fid_to_emb = {}
    embeddings_list = []
    for table_name, table in fids_dict.items():
      for idx in sorted(table):
        values = table[idx]
        #emb = np.random.uniform(size=size)
        #logging.info(f"show emb {emb}")
        emb = []
        for i, (dim, fid) in enumerate(values):
          fid_emb = []
          for j in range(dim):
            fid_emb.append(fid + j)
          fid_to_emb[fid] = np.array(fid_emb, dtype=float)
          emb.extend(fid_emb)
        emb = np.array(emb, dtype=float)
        logging.info(f"show emb2 {emb}")
        embeddings_list.append(
            tf.reshape(tf.constant(value=emb, dtype=tf.float32), [-1]))
    #sparse_features_str = tf.constant(value=sparse_features.SerializeToString(),
    #                                  dtype=tf.string)

    if use_shard_op:
      get_default_parser_ctx().enable_fused_layout = True
      parsed_results = parse_example_batch(sparse_features.SerializeToString(),
                                           sparse_features=[],
                                           dense_features=[],
                                           dense_feature_shapes=[],
                                           dense_feature_types=[],
                                           extra_features=[],
                                           extra_feature_shapes=[])
      sparse_varint = parsed_results.pop("sparse_features")
      fid_list, fid_offset_list_ts, feature_offset_list_ts, nfl_offset_list_ts, batch_size_ts, fid_row_split_list_ts = sharding_sparse_fids(
          sparse_varint,
          num_ps,
          feature_cfgs,
          False,
          "examplebatch",
          parallel_flag=0,
          fid_list_ret_list=True)
    else:
      fid_row_split_list_ts = fid_row_split_list
      fid_offset_list_ts = tf.constant(fid_offset_list, dtype=tf.uint64)
      feature_offset_list_ts = tf.constant(feature_offset_list[:-1],
                                           dtype=tf.int32)
      nfl_offset_list_ts = tf.constant(nfl_offset_list[:-1], dtype=tf.uint32)
      batch_size_ts = tf.constant([batch_size], dtype=tf.int32)

    variant_type = 'example_batch'
    layouts_op = distribution_ops.fused_embedding_to_layout(
        embeddings_list,
        fid_row_split_list_ts,
        fid_offset_list_ts,
        feature_offset_list_ts,
        nfl_offset_list_ts,
        batch_size_ts,
        variant_type,
        feature_cfgs,
        num_ps,
        parallel_flag=parallel_flag)
    with self.session() as sess:
      layouts = sess.run(layouts_op)
    #logging.info(f"show layouts: {layouts}")

    layout_names = sorted([x for x in feature_cfgs.out_configs.keys()])
    out_tensors = {}
    layout_info = {}
    out_tensor_list = []
    out_tensor_name_list = []

    # get layout configs.
    for ln in layout_names:
      out_config = feature_cfgs.out_configs[ln]
      out_tensors[ln] = []
      info = {}
      if len(out_config.shape) == 1:
        for shape in out_config.shape:
          real_shape = list(shape.dims)
          real_shape[0] = batch_size
          ts = np.zeros(shape=real_shape, dtype=np.float32)
          #logging.info(f" {ln} {ts} ")
          out_tensors[ln].append(ts)
          out_tensor_list.append(ts)
          out_tensor_name_list.append(ln + ":" + str(len(out_tensors[ln])))

          offset = 0
          for i, sc in enumerate(out_config.slice_configs):
            key = get_key(ln, sc)
            dim = sc.end - sc.start
            if out_config.out_type == OutType.CONCAT:
              info[key] = (ts, offset)
              offset += dim
            elif out_config.out_type == OutType.STACK:
              info[key] = (ts, i)
            elif out_config.out_type == OutType.ADDN:
              info[key] = (ts, 0)
            else:
              raise Exception("error")
      else:
        for sc, shape in zip(out_config.slice_configs, out_config.shape):
          real_shape = list(shape.dims)
          real_shape[0] = batch_size
          ts = np.zeros(shape=real_shape, dtype=np.float32)
          out_tensors[ln].append(ts)
          out_tensor_list.append(ts)
          out_tensor_name_list.append(ln + ":" + str(len(out_tensors[ln])))

          key = get_key(ln, sc)
          info[key] = (ts, 0)

      layout_info[ln] = info
    # {name: (out, offset)}

    for ln in layout_names:
      out_config = feature_cfgs.out_configs[ln]
      out_type = out_config.out_type
      for slice_conf in out_config.slice_configs:
        name = slice_conf.feature_name
        features = std_features[name]
        feature_config = feature_cfgs.feature_configs[name]
        pooling_type = feature_config.pooling_type
        max_length = feature_config.max_sequence_length
        key = get_key(ln, slice_conf)
        dim = slice_conf.end - slice_conf.start
        (ts, offset) = layout_info[ln][key]
        if out_type == OutType.ADDN:
          tmp_addn = np.zeros(ts.shape)  # per slice tmp out

        #logging.info(f" {ln} {ts} ")
        for i in range(batch_size):
          if i < len(features):
            tmp = []
            for fid in features[i].fid_v2_list.value:
              fid_emb = fid_to_emb[fid]
              emb_slice = fid_emb[slice_conf.start:slice_conf.end]
              tmp.append(emb_slice)

            if out_type == OutType.CONCAT:
              ts[i, offset:offset + dim] = pooling(pooling_type, tmp,
                                                   max_length)
            elif out_type == OutType.STACK:
              ts[i, offset, :] = pooling(pooling_type, tmp, max_length)
            elif out_type == OutType.ADDN:
              ret = pooling(pooling_type, tmp, max_length)
              tmp_addn[i, :] = ret
            else:
              ts[i, :] = pooling(pooling_type, tmp, max_length)
          else:
            # shared & copy
            if out_type == OutType.CONCAT:
              ts[i, offset:offset + dim] = ts[i - 1, offset:offset + dim]
            elif out_type == OutType.STACK:
              ts[i, offset, :] = ts[i - 1, offset, :]
            elif out_type == OutType.ADDN:
              tmp_addn[i, :] = tmp_addn[i - 1, :]
            else:
              ts[i, :] = ts[i - 1, :]

        if out_type == OutType.ADDN:
          ts += tmp_addn

        #logging.info(f" {ln} {ts} ")

    #logging.info(f"xxx out_tensor_list: {out_tensor_list}")
    for name, t, p in zip(out_tensor_name_list, out_tensor_list, layouts):
      #logging.info(f"xxx show result: {name} \n ans:{t} \n res:{p}")
      flag = np.allclose(t, p, rtol=1e-04, atol=1e-07, equal_nan=False)
      if not flag:
        logging.error(f"xxx show result: {name} \n ans:{t} \n res:{p}")
      else:
        logging.info(f"show result: {name} \n ans:{t} \n res:{p}")
      assert flag

  def test_fused_embedding_to_layout_use_shard_op(self):
    self.test_fused_embedding_to_layout(use_shard_op=True)

  def test_fused_embedding_to_layout_parallel(self):
    self.test_fused_embedding_to_layout(parallel_flag=0x00)

  def test_fused_embedding_to_layout_grad(self, parallel_flag=0x11):
    batch_size = 4
    num_ps = 2
    slot_num = 30
    slot_table_split = [10,
                        20]  #slot split for [table_one, table_two, table_three]
    max_sequence_length = 3
    alphabet_name = list(string.ascii_lowercase) + ['za', 'zb', 'zc', 'zd']
    feature_cfgs = FeatureConfigs()
    #sparse_features = list()

    bias = OutConfig()
    vec = OutConfig()
    ffm1 = OutConfig()
    ffm2 = OutConfig()
    firstN = OutConfig()

    slot2fid = dict()
    slot2fid_offset = defaultdict(list)
    for slot in range(1, slot_num):
      feature_name = f"fc_slot_{alphabet_name[slot - 1]}"
      fconf = FeatureConfig()
      if slot >= slot_table_split[1]:
        table_name = "table_one"  #table_three, but now test for table with different dim
        slice_dims = [1, 4, 16]
        sequence_length = max_sequence_length
        pooling_type = PoolingType.FIRSTN
        slice_config = firstN.slice_configs.add()
        slice_config.feature_name = feature_name
        slice_config.start = 0
        slice_config.end = 21
      else:
        sequence_length = 0
        if slot < slot_table_split[0]:
          table_name = "table_one"
          slice_dims = [1, 4, 8]
          pooling_type = PoolingType.SUM
          slice_config = ffm1.slice_configs.add()
          slice_config.feature_name = feature_name
          slice_config.start = 5
          slice_config.end = 13
        else:
          table_name = "table_two"
          slice_dims = [1, 4, 16]
          pooling_type = PoolingType.MEAN
          slice_config = ffm2.slice_configs.add()
          slice_config.feature_name = feature_name
          slice_config.start = 5
          slice_config.end = 21
        slice_config = bias.slice_configs.add()
        slice_config.feature_name = feature_name
        slice_config.start = 0
        slice_config.end = 1
        slice_config = vec.slice_configs.add()
        slice_config.feature_name = feature_name
        slice_config.start = 1
        slice_config.end = 5

      fconf.table = table_name
      fconf.slice_dims.extend(slice_dims)
      fconf.max_sequence_length = sequence_length
      fconf.pooling_type = pooling_type
      feature_cfgs.feature_configs[feature_name].CopyFrom(fconf)

    infer_shape(bias, OutType.ADDN)
    feature_cfgs.out_configs['bias'].CopyFrom(bias)
    infer_shape(vec, OutType.CONCAT)
    feature_cfgs.out_configs['vec'].CopyFrom(vec)
    infer_shape(ffm1, OutType.STACK)
    feature_cfgs.out_configs['ffm1'].CopyFrom(ffm1)
    infer_shape(ffm2, OutType.NONE)
    feature_cfgs.out_configs['ffm2'].CopyFrom(ffm2)
    infer_shape(firstN, OutType.NONE, max_sequence_length)
    feature_cfgs.out_configs['firstN'].CopyFrom(firstN)

    logging.info(f"feature_cfgs : {feature_cfgs} ")
    feature_cfg, table_cfg, feature_name_sort, table_name_sort = self.get_feature_cfg(
        feature_cfgs, num_ps)

    embedding_fid_list = [[] for _ in range(num_ps * len(table_cfg))]
    fid_row_split_list = [[0] for _ in range(num_ps * len(table_cfg))]

    # record the truth
    truth = defaultdict(lambda: defaultdict(list))
    for slot in range(1, slot_num):
      feature_name = f"fc_slot_{alphabet_name[slot - 1]}"
      f_cfg = feature_cfg[feature_name]
      dim_sum = f_cfg["dim_sum"]
      table_idx = f_cfg["table_index"]
      fids = list(
          set([(slot << 48) + randint(100, 1000000)
               for _ in range(randint(2, 10))]))
      slot2fid[slot] = fids
      embedding_fid_list_tmp = [[] for _ in range(num_ps)]

      for fid in fids:
        ps_index = fid % num_ps
        index1 = table_idx * num_ps + ps_index
        embedding_fid_list[index1].append((fid, dim_sum))
        index2 = len(embedding_fid_list[index1]) - 1
        embedding_fid_list_tmp[ps_index].append(fid)
        feature_index = len(embedding_fid_list_tmp[ps_index]) - 1
        full_index = self.get_pre_output_offset(ps_index, f_cfg)
        #[index1(table_index), index2(fid in table index)
        # , full_index(all_feature_index), feature_index(fid in all_feature index)]
        slot2fid_offset[slot].append(
            [index1, index2, full_index, feature_index])
        truth[index1][index2] = [0, dim_sum]
      for ps_i in range(num_ps):
        index1 = table_idx * num_ps + ps_i
        fid_row_split_list[index1].append(len(embedding_fid_list[index1]))

    fid_offset_list = list()
    feature_offset_list = [0]
    nfl_offset_list = [0]
    embeddings_list = list()
    #sparse_features = [Example() for i in range(batch_size)]

    for slot in range(1, slot_num):
      feature_name = f"fc_slot_{alphabet_name[slot - 1]}"
      feature_config = feature_cfgs.feature_configs[feature_name]
      pooling_type = feature_config.pooling_type
      max_length = feature_config.max_sequence_length
      for bi in range(batch_size):
        #sparse_feature = sparse_features[bi]
        #named_feature = sparse_feature.named_feature.add()
        #named_feature.name = f"fc_slot_{alphabet_name[slot - 1]}"
        all_fids = slot2fid[slot]
        fid_num = randint(0, len(all_fids))
        fid_idx_list = [randint(1, len(all_fids) - 1) for i in range(fid_num)]
        #fid_list = [all_fids[idx] for idx in fid_idx_list]
        #named_feature.feature.fid_v2_list.value.extend(fid_list)

        for i, idx in enumerate(fid_idx_list):
          index1, index2, full_index, feature_index = slot2fid_offset[slot][idx]
          fid_offset = full_index << 32 | feature_index
          fid_offset_list.append(fid_offset)
          if pooling_type == PoolingType.FIRSTN and i >= max_length:
            pass
          else:
            truth[index1][index2][0] += 1

        feature_offset_list.append(len(fid_offset_list))
      nfl_index = len(feature_offset_list) - 1
      nfl_offset_list.append(nfl_index)

    for idx, embedding_fid in enumerate(embedding_fid_list):
      dim_sum = 0
      for fid, dim in embedding_fid:
        dim_sum += dim
      size = (dim_sum, 1)
      emb = np.random.uniform(size=size)
      embeddings_list.append(
          tf.reshape(tf.constant(value=emb, dtype=tf.float32), [-1]))

    with self.session() as sess:
      fid_offset_list_ts = tf.constant(fid_offset_list, dtype=tf.uint64)
      feature_offset_list_ts = tf.constant(feature_offset_list[:-1],
                                           dtype=tf.int32)
      nfl_offset_list_ts = tf.constant(nfl_offset_list[:-1], dtype=tf.uint32)
      batch_size_ts = tf.constant([batch_size], dtype=tf.int32)

      variant_type = 'example'
      layouts = distribution_ops.fused_embedding_to_layout(
          embeddings_list,
          fid_row_split_list,
          fid_offset_list_ts,
          feature_offset_list_ts,
          nfl_offset_list_ts,
          batch_size_ts,
          variant_type,
          feature_cfgs,
          num_ps,
          parallel_flag=parallel_flag)
      #layouts_ret = sess.run(layouts)
      #logging.info(f"show result: {layouts_ret}")
      test_grads = tf.gradients(layouts, embeddings_list)
      grads = sess.run(test_grads)
      logging.info(f"show result: {grads}")
      logging.info(f"show truth: {truth}")
      assert len(grads) == len(truth)
      for i in range(len(truth)):
        part_truth = truth[i]
        grad = grads[i]
        offset = 0
        for j in range(len(part_truth)):
          t, dim = part_truth[j]
          # There is no slice use twice in the UT data, so the grads of one fid embedding should be the same
          assert len(np.unique(grad[offset: offset + dim])) == 1, \
                    f"Alert All The Same! [{i}, {j}] [{(t, dim)}, {grad[offset: offset + dim]}]"
          # The gound truth should be the fid used times
          assert t == grad[offset], \
            f"Alert Equal! [{i}, {j}] [{t} {grad[offset]}]"
          offset += dim

  def test_fused_embedding_to_layout_grad_no_parallel(self):
    self.test_fused_embedding_to_layout_grad(parallel_flag=0x00)


class FusedEmbeddingToLayoutFitPreTest(tf.test.TestCase):

  def test_fused_embedding_to_layout(self):
    batch_size = 10
    feature_cfgs = FeatureConfigs()
    sparse_features = ExampleBatch(batch_size=batch_size)
    std_features = defaultdict(list)
    fids_dict = {}
    bias = OutConfig()
    vec = OutConfig()
    ffm1 = OutConfig()
    ffm2 = OutConfig()

    index1 = 0
    index2 = [0] * 10
    fid_offset_list = list()
    feature_offset_list = [0]
    nfl_offset_list = [0]
    feature_names_list = list()
    slot_to_nfl_map = dict()

    for slot in range(1, 50):
      named_feature_list = sparse_features.named_feature_list.add()
      named_feature_list.id = slot
      named_feature_list.name = f"fc_slot_{slot}"
      feature_names_list.append(named_feature_list.name)
      is_shared = True if slot % 2 == 0 else False
      named_feature_list.type = FeatureListType.SHARED if is_shared else FeatureListType.INDIVIDUAL
      slot_to_nfl_map[slot] = named_feature_list

    # set all offset_list by sorted sorted_feature_names_list
    sorted_feature_names_list = sorted(feature_names_list)

    for feature_name in sorted_feature_names_list:
      slot = int(feature_name.split("fc_slot_")[-1])
      named_feature_list = slot_to_nfl_map[slot]
      fconf = FeatureConfig()
      if slot < 25:
        table_name = "table_one"
        slice_dims = [1, 4, 8]
        pooling_type = PoolingType.SUM
        slice_config = ffm1.slice_configs.add()
        slice_config.feature_name = f"fc_slot_{slot}"
        slice_config.start = 5
        slice_config.end = 13
      else:
        table_name = "table_two"
        slice_dims = [1, 4, 16]
        pooling_type = PoolingType.MEAN
        slice_config = ffm2.slice_configs.add()
        slice_config.feature_name = f"fc_slot_{slot}"
        slice_config.start = 5
        slice_config.end = 21

      slice_config = bias.slice_configs.add()
      slice_config.feature_name = f"fc_slot_{slot}"
      slice_config.start = 0
      slice_config.end = 1
      slice_config = vec.slice_configs.add()
      slice_config.feature_name = f"fc_slot_{slot}"
      slice_config.start = 1
      slice_config.end = 5

      fconf.table = table_name
      fconf.slice_dims.extend(slice_dims)
      fconf.pooling_type = pooling_type
      feature_cfgs.feature_configs[f"fc_slot_{slot}"].CopyFrom(fconf)
      table_name = "table_one" if slot < 25 else "table_two"
      if table_name not in fids_dict:
        fids_dict[table_name] = defaultdict(list)
      if named_feature_list.type == FeatureListType.SHARED:
        feature = named_feature_list.feature.add()
        std_features[named_feature_list.name].append(feature)
        fids = list(
            set([(slot << 48) + randint(100, 1000000)
                 for _ in range(randint(1, 5))]))
        feature.fid_v2_list.value.extend(fids)
        for fid in fids:
          idx = fid % 5
          if table_name == "table_one":
            index1 = 0 + idx
            fid_offset = index1 << 32 | index2[index1]
            index2[index1] += 1
          else:
            index1 = 5 + idx
            fid_offset = index1 << 32 | index2[index1]
            index2[index1] += 1

          fid_offset_list.append(fid_offset)
          fids_dict[table_name][idx].append(fid)
        feature_offset_list.append(len(fid_offset_list))

      else:
        for _ in range(batch_size):
          feature = named_feature_list.feature.add()
          fids = list(
              set([(slot << 48) + randint(100, 1000000)
                   for _ in range(randint(1, 5))]))
          feature.fid_v2_list.value.extend(fids)
          for fid in fids:
            idx = fid % 5
            if table_name == "table_one":
              index1 = 0 + idx
              fid_offset = index1 << 32 | index2[index1]
              index2[index1] += 1
            else:
              index1 = 5 + idx
              fid_offset = index1 << 32 | index2[index1]
              index2[index1] += 1

            fid_offset_list.append(fid_offset)
            fids_dict[table_name][idx].append(fid)
          feature_offset_list.append(len(fid_offset_list))
          std_features[named_feature_list.name].append(feature)

      nfl_index = len(feature_offset_list) - 1
      nfl_offset_list.append(nfl_index)

    # add shared encode
    nfl_idx = 0
    for feature_name in sorted_feature_names_list:
      slot = int(feature_name.split("fc_slot_")[-1])
      is_shared = True if slot % 2 == 0 else False
      head_bit = SHARD_BIT if is_shared else 0
      nfl_offset_list[nfl_idx] |= head_bit
      nfl_idx += 1

    infer_shape(bias, OutType.ADDN)
    feature_cfgs.out_configs['bias'].CopyFrom(bias)
    infer_shape(vec, OutType.CONCAT)
    feature_cfgs.out_configs['vec'].CopyFrom(vec)
    infer_shape(ffm1, OutType.STACK)
    feature_cfgs.out_configs['ffm1'].CopyFrom(ffm1)
    infer_shape(ffm2, OutType.NONE)
    feature_cfgs.out_configs['ffm2'].CopyFrom(ffm2)

    embeddings_dict, fid_to_emb = {}, {}
    fids_list, embeddings_list, embeddings_np_list = [], [], []
    for table_name, table in fids_dict.items():
      embeddings_dict[table_name] = {}
      for idx in sorted(table):
        values = table[idx]
        fids_list.append(tf.constant(value=values, dtype=tf.int64))
        if table_name == "table_one":
          length = 13
        else:
          length = 21

        size = (len(values), length)
        emb = np.random.uniform(size=size)
        embeddings_dict[table_name][idx] = emb
        embeddings_list.append(
            tf.constant(value=emb, shape=size, dtype=tf.float32))
        embeddings_np_list.append(emb)

        for i, fid in enumerate(values):
          fid_to_emb[fid] = (len(embeddings_np_list) - 1, i)
    sparse_features_str = tf.constant(value=sparse_features.SerializeToString(),
                                      dtype=tf.string)
    variant_type = 'example_batch'
    # layouts = distribution_ops.fused_embedding_to_layout(sparse_features_str, fids_list, embeddings_list, variant_type, feature_cfgs)

    fid_offset_list_ts = tf.constant(fid_offset_list, dtype=tf.uint64)
    feature_offset_list_ts = tf.constant(feature_offset_list[:-1],
                                         dtype=tf.int32)
    nfl_offset_list_ts = tf.constant(nfl_offset_list[:-1], dtype=tf.uint32)
    batch_size_ts = tf.constant([batch_size], dtype=tf.int32)

    layouts_op = distribution_ops.fused_embedding_to_layout(
        embeddings_list,
        None,
        fid_offset_list_ts,
        feature_offset_list_ts,
        nfl_offset_list_ts,
        batch_size_ts,
        variant_type,
        feature_cfgs,
        -1,
        version=1)
    with self.session() as sess:
      layouts = sess.run(layouts_op)

    layout_names = sorted(['bias', 'vec', 'ffm1', 'ffm2'])
    out_tensors = {}
    layout_info = {}
    out_tensor_list = []

    # get layout configs.
    for ln in layout_names:
      out_config = feature_cfgs.out_configs[ln]
      out_tensors[ln] = []
      info = {}
      if len(out_config.shape) == 1:
        for shape in out_config.shape:
          real_shape = list(shape.dims)
          real_shape[0] = batch_size
          ts = np.zeros(shape=real_shape, dtype=np.float32)
          out_tensors[ln].append(ts)
          out_tensor_list.append(ts)

          offset = 0
          for i, sc in enumerate(out_config.slice_configs):
            key = get_key(ln, sc)
            dim = sc.end - sc.start
            if out_config.out_type == OutType.CONCAT:
              info[key] = (ts, offset)
              offset += dim
            elif out_config.out_type == OutType.STACK:
              info[key] = (ts, i)
            elif out_config.out_type == OutType.ADDN:
              info[key] = (ts, 0)
            else:
              raise Exception("error")
      else:
        for sc, shape in zip(out_config.slice_configs, out_config.shape):
          real_shape = list(shape.dims)
          real_shape[0] = batch_size
          ts = np.zeros(shape=real_shape, dtype=np.float32)
          out_tensors[ln].append(ts)
          out_tensor_list.append(ts)

          key = get_key(ln, sc)
          info[key] = (ts, 0)

      layout_info[ln] = info
    # {name: (out, offset)}

    for ln in layout_names:
      out_config = feature_cfgs.out_configs[ln]
      out_type = out_config.out_type
      for slice_conf in out_config.slice_configs:
        name = slice_conf.feature_name
        features = std_features[name]
        feature_config = feature_cfgs.feature_configs[name]
        pooling_type = feature_config.pooling_type
        max_length = feature_config.max_sequence_length
        key = get_key(ln, slice_conf)
        dim = slice_conf.end - slice_conf.start
        (ts, offset) = layout_info[ln][key]
        if out_type == OutType.ADDN:
          tmp_addn = np.zeros(ts.shape)  # per slice tmp out

        for i in range(batch_size):
          if i < len(features):
            tmp = []
            for fid in features[i].fid_v2_list.value:
              (idx, row) = fid_to_emb[fid]
              emb_slice = embeddings_np_list[idx][
                  row, slice_conf.start:slice_conf.end]
              tmp.append(emb_slice)

            if out_type == OutType.CONCAT:
              ts[i, offset:offset + dim] = pooling(pooling_type, tmp,
                                                   max_length)
            elif out_type == OutType.STACK:
              ts[i, offset, :] = pooling(pooling_type, tmp, max_length)
            elif out_type == OutType.ADDN:
              ret = pooling(pooling_type, tmp, max_length)
              tmp_addn[i, :] = ret
            else:
              ts[i, :] = pooling(pooling_type, tmp, max_length)
          else:
            # shared & copy
            if out_type == OutType.CONCAT:
              ts[i, offset:offset + dim] = ts[i - 1, offset:offset + dim]
            elif out_type == OutType.STACK:
              ts[i, offset, :] = ts[i - 1, offset, :]
            elif out_type == OutType.ADDN:
              tmp_addn[i, :] = tmp_addn[i - 1, :]
            else:
              ts[i, :] = ts[i - 1, :]

        if out_type == OutType.ADDN:
          ts += tmp_addn

    for t, p in zip(out_tensor_list, layouts):
      logging.info(f"fused_embedding_to_layout show {t} {p}")
      assert np.allclose(t, p, rtol=1e-04, atol=1e-07, equal_nan=False)

  def test_fused_embedding_to_layout_grad(self):
    batch_size = 4
    slot_num = 30
    alphabet_name = list(string.ascii_lowercase) + ['za', 'zb', 'zc', 'zd']
    feature_cfgs = FeatureConfigs()
    sparse_features = list()
    # 13, 13, 21, 21
    embedding_fid_list = [[], [], [], []]

    bias = OutConfig()
    vec = OutConfig()
    ffm1 = OutConfig()
    ffm2 = OutConfig()

    slot2fid = dict()
    slot2fid_offset = defaultdict(list)
    for slot in range(1, slot_num):
      table_idx = 0 if slot < slot_num / 2 else 2
      fids = list(
          set([(slot << 48) + randint(100, 1000000)
               for _ in range(randint(2, 10))]))
      slot2fid[slot] = fids
      for fid in fids:
        if fid % 2:
          index1 = table_idx
        else:
          index1 = table_idx + 1
        embedding_fid_list[index1].append(fid)
        index2 = len(embedding_fid_list[index1]) - 1
        slot2fid_offset[slot].append([index1, index2])

      fconf = FeatureConfig()
      table_name = "table_one" if slot < slot_num / 2 else "table_two"
      if slot < slot_num / 2:
        slice_dims = [1, 4, 8]
        pooling_type = PoolingType.SUM
        slice_config = ffm1.slice_configs.add()
        slice_config.feature_name = f"fc_slot_{alphabet_name[slot - 1]}"
        slice_config.start = 5
        slice_config.end = 13
      else:
        slice_dims = [1, 4, 16]
        pooling_type = PoolingType.MEAN
        slice_config = ffm2.slice_configs.add()
        slice_config.feature_name = f"fc_slot_{alphabet_name[slot - 1]}"
        slice_config.start = 5
        slice_config.end = 21
      slice_config = bias.slice_configs.add()
      slice_config.feature_name = f"fc_slot_{alphabet_name[slot - 1]}"
      slice_config.start = 0
      slice_config.end = 1
      slice_config = vec.slice_configs.add()
      slice_config.feature_name = f"fc_slot_{alphabet_name[slot - 1]}"
      slice_config.start = 1
      slice_config.end = 5
      fconf.table = table_name
      fconf.slice_dims.extend(slice_dims)
      fconf.pooling_type = pooling_type
      feature_cfgs.feature_configs[
          f"fc_slot_{alphabet_name[slot - 1]}"].CopyFrom(fconf)

    infer_shape(bias, OutType.ADDN)
    feature_cfgs.out_configs['bias'].CopyFrom(bias)
    infer_shape(vec, OutType.CONCAT)
    feature_cfgs.out_configs['vec'].CopyFrom(vec)
    infer_shape(ffm1, OutType.STACK)
    feature_cfgs.out_configs['ffm1'].CopyFrom(ffm1)
    infer_shape(ffm2, OutType.NONE)
    feature_cfgs.out_configs['ffm2'].CopyFrom(ffm2)

    # record the truth
    truth = defaultdict(lambda: defaultdict(int))

    fid_offset_list = list()
    feature_offset_list = [0]
    nfl_offset_list = [0]
    embeddings_list = list()
    sparse_features = [Example() for i in range(batch_size)]

    for slot in range(1, slot_num):
      for bi in range(batch_size):
        sparse_feature = sparse_features[bi]
        named_feature = sparse_feature.named_feature.add()
        named_feature.name = f"fc_slot_{alphabet_name[slot - 1]}"
        all_fids = slot2fid[slot]
        fid_num = randint(0, len(all_fids))
        fid_idx_list = [randint(1, len(all_fids) - 1) for i in range(fid_num)]
        fid_list = [all_fids[idx] for idx in fid_idx_list]
        named_feature.feature.fid_v2_list.value.extend(fid_list)

        for idx in fid_idx_list:
          index1, index2 = slot2fid_offset[slot][idx]
          fid_offset = index1 << 32 | index2
          fid_offset_list.append(fid_offset)
          truth[index1][index2] += 1

        feature_offset_list.append(len(fid_offset_list))
      nfl_index = len(feature_offset_list) - 1
      nfl_offset_list.append(nfl_index)

    idx = 0
    for embedding_fid in embedding_fid_list:
      if idx < 2:
        embeddings_dim = [13 for i in embedding_fid]
      else:
        embeddings_dim = [21 for i in embedding_fid]
      size = (len(embeddings_dim), embeddings_dim[0])
      emb = np.random.uniform(size=size)
      embeddings_list.append(
          tf.constant(value=emb, shape=size, dtype=tf.float32))
      idx += 1
    variant_type = 'example'

    with self.session() as sess:
      fid_offset_list_ts = tf.constant(fid_offset_list, dtype=tf.uint64)
      feature_offset_list_ts = tf.constant(feature_offset_list[:-1],
                                           dtype=tf.int32)
      nfl_offset_list_ts = tf.constant(nfl_offset_list[:-1], dtype=tf.uint32)
      batch_size_ts = tf.constant([batch_size], dtype=tf.int32)

      layouts = distribution_ops.fused_embedding_to_layout(
          embeddings_list,
          None,
          fid_offset_list_ts,
          feature_offset_list_ts,
          nfl_offset_list_ts,
          batch_size_ts,
          variant_type,
          feature_cfgs,
          -1,
          version=1)
      test_grads = tf.gradients(layouts, embeddings_list)
      grads = sess.run(test_grads)
      for i in range(len(grads)):
        grad = grads[i]
        for j in range(grad.shape[0]):
          # There is no slice use twice in the UT data, so the grads of one fid embedding should be the same
          assert len(np.unique(
              grad[j, :])) == 1, "Alert All The Same! [{}, {}]".format(i, j)
          # The gound truth should be the fid used times
          logging.info(
              f"fused_embedding_to_layout grad show {truth[i][j]} {grad[j, 0]}")
          assert truth[i][j] == grad[j, 0], "Alert Equal! [{}, {}]".format(i, j)


if __name__ == "__main__":
  # tf.compat.v1.disable_eager_execution()
  tf.test.main()
