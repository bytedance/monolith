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
import sys
from absl import app
from absl import flags
from absl import logging
from random import randint, uniform, choice
from copy import deepcopy
import numpy as np
from struct import pack, unpack
from functools import singledispatch
from typing import List, Iterable, Tuple, Dict, Any, get_type_hints
from datetime import datetime
from dataclasses import dataclass
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_log_pb2
from tensorflow.python.saved_model.signature_constants import DEFAULT_SERVING_SIGNATURE_DEF_KEY
from tensorflow.python.saved_model.model_utils.export_output import PredictOutput
from monolith.native_training import env_utils
from monolith.native_training.data.feature_list import Feature, FeatureList, get_feature_name_and_slot
from idl.matrix.proto.example_pb2 import Example, ExampleBatch
from idl.matrix.proto.example_pb2 import Feature as EFeature
from idl.matrix.proto.line_id_pb2 import LineId
from idl.matrix.proto.proto_parser_pb2 import Instance
from idl.matrix.proto.feature_pb2 import Feature as IFeature
from google.protobuf.descriptor import FieldDescriptor
from monolith.native_training.utils import get_collection
from monolith.native_training.model_export import export_context

MASK_V1 = (1 << 54) - 1
MAX_SLOT_V1 = (1 << (64 - 54)) - 1
MASK_V2 = (1 << 48) - 1
MAX_SLOT_V2 = (1 << (64 - 48)) - 1


class FeatureMeta(object):
  line_id_fields = {f.name: f for f in LineId.DESCRIPTOR.fields}
  dtypes = {
      FieldDescriptor.CPPTYPE_FLOAT: tf.float32,
      FieldDescriptor.CPPTYPE_DOUBLE: tf.float32,
      FieldDescriptor.CPPTYPE_UINT32: tf.int32,
      FieldDescriptor.CPPTYPE_INT32: tf.int32,
      FieldDescriptor.CPPTYPE_UINT64: tf.int64,
      FieldDescriptor.CPPTYPE_INT64: tf.int64,
      FieldDescriptor.CPPTYPE_BOOL: tf.bool,
      FieldDescriptor.CPPTYPE_STRING: tf.string
  }

  def __init__(self,
               name: str,
               slot: int = None,
               shape: int = None,
               dtype: tf.compat.v1.dtypes.DType = None,
               extra=None):
    self.name = name
    self.slot = slot

    if shape is None:
      self.shape = 1 if self.slot is None else -1
    else:
      self.shape = shape

    # infer data type
    self.dtype = dtype
    if self.dtype is None:
      if name in self.line_id_fields:
        cpp_type = self.line_id_fields[name].cpp_type
        if cpp_type in self.dtypes:
          self.dtype = self.dtypes[cpp_type]
    if self.dtype is None:
      if slot is None:
        self.dtype = tf.float32
      else:
        self.dtype = tf.int64

    self.extra = extra


@dataclass
class ParserArgs(object):
  model_name: str = 'entry'
  fidv1_features: List[int] = None
  fidv2_features: List[str] = None
  sparse_features: List[str] = None
  dense_features: List[str] = None
  dense_feature_shapes: List[int] = None
  dense_feature_types: List[tf.compat.v1.dtypes.DType] = None
  extra_features: List[str] = None
  extra_feature_shapes: List[int] = None
  feature_list: FeatureList = None
  batch_size: int = 64
  max_records: int = 1000
  signature_name: List[str] = None
  variant_type: str = None
  warmup_file: str = None
  drop_rate: float = 0.5

  def __post_init__(self):
    self.model_name = self.model_name or self.get('model_name')
    self.fidv1_features = self.fidv1_features or self.get('fidv1_features')
    self.fidv2_features = self.fidv2_features or self.get('fidv2_features')
    self.sparse_features = self.sparse_features or self.get('sparse_features')
    self.dense_features = self.dense_features or self.get('dense_features')
    self.dense_feature_shapes = self.dense_feature_shapes or self.get(
        'dense_feature_shapes')
    self.dense_feature_types = self.dense_feature_types or self.get(
        'dense_feature_types')
    self.extra_features = self.extra_features or self.get('extra_features')
    self.extra_feature_shapes = self.extra_feature_shapes or self.get(
        'extra_feature_shapes')
    self.feature_list = self.feature_list or self.get('feature_list')
    if self.feature_list is None:
      try:
        self.feature_list = FeatureList.parse()
      except:
        logging.info('cannot get feature_list, pls check!')

    self.signature_name = self.signature_name or self.get('signature_name')
    if self.signature_name is None:
      self.signature_name = [DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    else:
      if DEFAULT_SERVING_SIGNATURE_DEF_KEY not in self.signature_name:
        self.signature_name.append(DEFAULT_SERVING_SIGNATURE_DEF_KEY)
      self.signature_name = list(set(self.signature_name))
    self.variant_type = self.variant_type or self.get('variant_type')

  @classmethod
  def get(cls, name):
    collection = get_collection(name)

    if collection is None:
      return None
    elif name == 'signature_name':
      return list(set(collection))
    else:
      return collection[-1]


def gen_fids_v1(slot: int, size: int = 1) -> List[int]:
  if 0 < slot < MAX_SLOT_V1:
    return [
        (slot << 54) | (randint(1, sys.maxsize) & MASK_V1) for _ in range(size)
    ]
  else:
    logging.log_first_n(logging.INFO,
                        f"enconter slot bigger the 1023 in fid v1 {slot}", 10)
    return []


def gen_fids_v2(slot: int, size: int = 1) -> List[int]:
  assert 0 < slot < MAX_SLOT_V2
  return [
      (slot << 48) | (randint(1, sys.maxsize) & MASK_V2) for _ in range(size)
  ]


@singledispatch
def fill_features():
  raise NotImplementedError("Not implemented fill_features")


@fill_features.register(EFeature)
def _(feature: EFeature, meta: FeatureMeta, drop_rate: float = 0):
  (name, size, dtype, feat) = meta.name, meta.shape, meta.dtype, meta.extra
  if size == -1:  # sparse
    if '_recent' in name or '_cp' in name or feat.method in {
        'Combine', 'VectorTopString'
    }:
      if drop_rate > 0 and uniform(0, 1) > drop_rate:
        feature.fid_v2_list.value.extend(gen_fids_v2(feat.slot, randint(0, 2)))
    elif feat.slot not in {1, 200}:  # user_id, item_id
      if uniform(0, 1) > drop_rate:
        feature.fid_v2_list.value.extend(gen_fids_v2(feat.slot, 1))
    else:
      feature.fid_v2_list.value.extend(gen_fids_v2(feat.slot, 1))
  elif dtype == tf.float64:
    data = [uniform(0, 1) for _ in range(size)]
    feature.double_list.value.extend(data)
  elif dtype == tf.float32:
    data = [uniform(0, 1) for _ in range(size)]
    feature.float_list.value.extend(data)
  elif dtype == tf.int64:
    data = [randint(sys.maxsize // 2, sys.maxsize) for _ in range(size)]
    feature.int64_list.value.extend(data)
  else:
    logging.warning(f'{name} is empty')


@fill_features.register(IFeature)
def _(feature: IFeature, meta: FeatureMeta, drop_rate: float = 0):
  (name, size, dtype) = meta.name, meta.shape, meta.dtype
  feature.name = name
  if size == -1:  # sparse
    if '_recent' in name or '_cp' in name or '-' in name:
      if drop_rate > 0 and uniform(0, 1) > drop_rate:
        feature.fid.extend(gen_fids_v2(meta.slot, randint(0, 2)))
    else:
      feature.fid.extend(gen_fids_v2(meta.slot, 1))
  elif dtype in {tf.float64, tf.float32}:
    data = [uniform(0, 1) for _ in range(size)]
    feature.float_value.extend(data)
  elif dtype == tf.int64:
    data = [randint(sys.maxsize // 2, sys.maxsize) for _ in range(size)]
    feature.int64_value.extend(data)
  else:
    logging.warning(f'{name} is empty')


def fill_line_id(line_id,
                 features: List[FeatureMeta] = None,
                 hash_len: int = 48,
                 actions: List[int] = None):
  MASK = MASK_V1 if hash_len == 54 else MASK_V2
  if features:
    for meta in features:
      name, shape = meta.name, meta.shape
      if name == 'uid':
        line_id.uid = (1 << hash_len) | (
            randint(sys.maxsize // 2, sys.maxsize) & MASK)
      elif name == 'item_id':
        line_id.item_id = (200 << hash_len) | (
            randint(sys.maxsize // 2, sys.maxsize) & MASK)
      elif name == 'req_time':
        line_id.req_time = int(datetime.now().timestamp())
        line_id.sample_rate = 1.0
      elif name == 'actions':
        if actions:
          line_id.actions.extend([choice(actions) for _ in range(shape)])
        else:
          line_id.actions.extend([randint(0, 10) for _ in range(shape)])
      elif name == 'stay_time':
        line_id.stay_time = uniform(a=0, b=1) * 1000
      elif hasattr(LineId, name):
        desc = getattr(LineId, name).DESCRIPTOR
        if desc.label == FieldDescriptor.LABEL_REPEATED:
          value_list = getattr(line_id, name)
          if desc.cpp_type in {
              FieldDescriptor.CPPTYPE_DOUBLE, FieldDescriptor.CPPTYPE_FLOAT
          }:
            value_list.extend([uniform(0, 1) for _ in range(shape)])
          elif desc.cpp_type in {
              FieldDescriptor.CPPTYPE_INT32, FieldDescriptor.CPPTYPE_INT64,
              FieldDescriptor.CPPTYPE_UINT32, FieldDescriptor.CPPTYPE_UINT64
          }:
            value_list.extend([randint(0, 10) for _ in range(shape)])
          elif desc.cpp_type == FieldDescriptor.CPPTYPE_STRING:
            value_list.extend(['hello world' for _ in range(shape)])
          elif desc.cpp_type == FieldDescriptor.CPPTYPE_BOOL:
            value_list.extend([False for _ in range(shape)])
        else:
          if desc.cpp_type in {
              FieldDescriptor.CPPTYPE_DOUBLE, FieldDescriptor.CPPTYPE_FLOAT
          }:
            setattr(line_id, name, uniform(0, 1))
          elif desc.cpp_type in {
              FieldDescriptor.CPPTYPE_INT32, FieldDescriptor.CPPTYPE_INT64,
              FieldDescriptor.CPPTYPE_UINT32, FieldDescriptor.CPPTYPE_UINT64
          }:
            setattr(line_id, name, randint(0, 10))
          elif desc.cpp_type == FieldDescriptor.CPPTYPE_STRING:
            setattr(line_id, name, 'hello world')
          elif desc.cpp_type == FieldDescriptor.CPPTYPE_BOOL:
            setattr(line_id, name, False)
  else:
    line_id.uid = (1 << hash_len) | (randint(sys.maxsize // 2, sys.maxsize) &
                                     MASK)
    line_id.item_id = (200 << hash_len) | (
        randint(sys.maxsize // 2, sys.maxsize) & MASK)
    line_id.req_time = int(datetime.now().timestamp())
    line_id.sample_rate = 1.0
    line_id.actions.append(randint(0, 10))


def lg_header(source: str):
  # calc java hash code
  if source:
    seed, h = 31, 0
    for c in source:
      h = np.int32(seed * h) + ord(c)

    dfhc = int(np.uint32(h)).to_bytes(4, 'little')
    return pack('4Bi', 0, dfhc[0], dfhc[1], dfhc[2], 0)
  else:
    return int.to_bytes(0, 8, byteorder='little')


def sort_header(sort_id: bool, kafka_dump: bool, kafka_dump_prefix: bool):
  # kafka_dump_prefix: [size: 8 bytes][aggregate_page_sortid_size: 8 bytes]
  # sort_id: [size: 8 bytes][sort_id: size bytes]
  # kafka_dump: [kafka_dump: 8 bytes]
  if sort_id and not (kafka_dump or kafka_dump_prefix):
    return pack('<Q', 0)
  elif sort_id and kafka_dump and not kafka_dump_prefix:
    return pack('<2Q', 0, 0)
  elif sort_id and kafka_dump_prefix and not kafka_dump:
    return pack('<3Q', 0, 0, 0)
  elif not (sort_id or kafka_dump or kafka_dump_prefix):
    return b''
  else:
    raise Exception(
        'kafka_dump_prefix={kafka_dump_prefix}, sort_id={sort_id}, kafka_dump={kafka_dump} not support'
    )


def gen_example(sparse_features: List[str],
                dense_features: List[FeatureMeta] = None,
                extra_features: List[FeatureMeta] = None,
                feature_list: FeatureList = None,
                drop_rate: float = 0,
                actions: List[int] = None) -> Example:
  assert len(sparse_features) > 0 and len(sparse_features) == len(
      set(sparse_features))

  name_to_info = {}
  for name in sparse_features:
    try:
      feat = feature_list.get(name)
      if feat is not None:
        name_to_info[name] = FeatureMeta(name,
                                         slot=feat.slot,
                                         dtype=tf.int64,
                                         extra=feat)
    except:
      _, slot = get_feature_name_and_slot(name)
      feat = Feature(feature_name=name, slot=slot)
      name_to_info[name] = FeatureMeta(name,
                                       slot=feat.slot,
                                       dtype=tf.int64,
                                       extra=feat)
      # logging.warning(f'cannot find name {name} in feature_list')

  if dense_features:
    name_to_info.update({meta.name: meta for meta in dense_features})

  assert len(name_to_info) > 0

  example = Example()
  label_meta = name_to_info.pop('label', None)
  for name, meta in name_to_info.items():
    named_feature = example.named_feature.add()
    if meta.slot:
      named_feature.id = meta.slot
    named_feature.name = name
    fill_features(named_feature.feature, meta, drop_rate)

  fill_line_id(example.line_id, extra_features, actions=actions)
  if label_meta:
    example.label.extend([choice([0, 1]) for _ in range(label_meta.shape)])
  else:
    example.label.append(choice([0, 1]))
  return example


def gen_instance(fidv1_features: List[int] = None,
                 fidv2_features: List[str] = None,
                 dense_features: List[FeatureMeta] = None,
                 extra_features: List[FeatureMeta] = None,
                 feature_list: FeatureList = None,
                 drop_rate: float = 0,
                 actions: List[int] = None) -> Instance:
  inst = Instance()
  if fidv1_features is not None:
    assert len(fidv1_features) > 0 and len(fidv1_features) == len(
        set(fidv1_features))
    for slot in fidv1_features:
      size = 1 if slot in {1, 200} else randint(0, 3)
      fids_v1 = gen_fids_v1(slot, size)
      if fids_v1:
        inst.fid.extend(fids_v1)

  name_to_info = {}
  if fidv2_features:
    assert len(fidv2_features) > 0 and len(fidv2_features) == len(
        set(fidv2_features))
    for name in fidv2_features:
      try:
        feat = feature_list.get(name)
        if feat is not None:
          name_to_info[name] = FeatureMeta(name,
                                           slot=feat.slot,
                                           dtype=tf.int64,
                                           extra=feat)
      except:
        logging.warning(f'cannot find name {name} in feature_list')

  if dense_features:
    name_to_info.update({meta.name: meta for meta in dense_features})

  label_meta = name_to_info.pop('label', None)
  for name, meta in name_to_info.items():
    feature = inst.feature.add()
    fill_features(feature, meta, drop_rate)

  fill_line_id(inst.line_id, extra_features, actions=actions)
  if label_meta:
    inst.label.extend([choice([0, 1]) for _ in range(label_meta.shape)])
  else:
    inst.label.append(choice([0, 1]))
  return inst


def gen_example_batch(sparse_features: List[str],
                      dense_features: List[FeatureMeta] = None,
                      extra_features: List[FeatureMeta] = None,
                      feature_list: FeatureList = None,
                      batch_size: int = 64,
                      drop_rate: float = 0,
                      actions: List[int] = None) -> ExampleBatch:
  assert len(sparse_features) > 0 and len(sparse_features) == len(
      set(sparse_features)) and batch_size > 0
  name_to_info = {}
  for name in sparse_features:
    try:
      feat = feature_list.get(name)
      if feat is not None:
        name_to_info[name] = FeatureMeta(name,
                                         slot=feat.slot,
                                         dtype=tf.int64,
                                         extra=feat)
    except:
      _, slot = get_feature_name_and_slot(name)
      feat = Feature(feature_name=name, slot=slot)
      name_to_info[name] = FeatureMeta(name,
                                       slot=feat.slot,
                                       dtype=tf.int64,
                                       extra=feat)
      # logging.warning(f'cannot find name {name} in feature_list')

  if dense_features:
    name_to_info.update({meta.name: meta for meta in dense_features})
  assert len(name_to_info) > 0

  example_batch = ExampleBatch(batch_size=batch_size)
  label_meta = name_to_info.pop('label', None)
  for name, meta in name_to_info.items():
    named_feature_list = example_batch.named_feature_list.add()
    if meta.slot:
      named_feature_list.id = meta.slot
    named_feature_list.name = name

    for _ in range(batch_size):
      feature = named_feature_list.feature.add()
      fill_features(feature, meta, drop_rate)

  named_feature_list = example_batch.named_feature_list.add()
  named_feature_list.name = '__LINE_ID__'
  for _ in range(batch_size):
    feature = named_feature_list.feature.add()
    line_id = LineId()
    fill_line_id(line_id, extra_features, hash_len=48, actions=actions)
    feature.bytes_list.value.append(line_id.SerializeToString())

  named_feature_list = example_batch.named_feature_list.add()
  named_feature_list.name = '__LABEL__'
  for i in range(batch_size):
    feature = named_feature_list.feature.add()
    if label_meta:
      feature.float_list.value.extend(
          [choice([0, 1]) for _ in range(label_meta.shape)])
    else:
      feature.float_list.value.append(i % 2)

  return example_batch


def gen_prediction_log(
    args: ParserArgs) -> Iterable[prediction_log_pb2.PredictionLog]:
  assert args.variant_type in {
      'example', 'instance', 'example_batch', 'examplebatch'
  } and args.batch_size < args.max_records
  if args.variant_type == 'example':
    input_name = 'examples'
  elif args.variant_type == 'instance':
    input_name = 'instances'
  else:
    input_name = 'example_batch'

  dense_feature_meta = []
  if args.dense_features:
    for name, shape, dtype in zip(args.dense_features,
                                  args.dense_feature_shapes,
                                  args.dense_feature_types):
      try:
        assert shape >= 1
        try:
          feat = args.feature_list.get(name)
          if feat is not None:
            dense_feature_meta.append(
                FeatureMeta(name, shape=shape, dtype=dtype, extra=feat))
          else:
            dense_feature_meta.append(
                FeatureMeta(name, shape=shape, dtype=dtype))
        except:
          dense_feature_meta.append(FeatureMeta(name, shape=shape, dtype=dtype))
      except:
        logging.warning(f'cannot find name {name} in feature_list')
  else:
    dense_feature_meta = None

  extra_meta = None
  if args.extra_features:
    extra_meta = [
        FeatureMeta(name=name, shape=shape)
        for name, shape in zip(args.extra_features, args.extra_feature_shapes)
    ]

  if args.signature_name is None:
    args.signature_name = [DEFAULT_SERVING_SIGNATURE_DEF_KEY]
  num_log = args.max_records // args.batch_size
  assert num_log >= len(args.signature_name)

  export_ctx = export_context.get_current_export_ctx()
  graph = tf.compat.v1.get_default_graph()
  if export_ctx is None:
    signatures = None
  else:
    signatures = {
        signature.name: signature for signature in export_ctx.signatures(graph)
    }
    for name in signatures:
      if name not in args.signature_name:
        args.signature_name.append(name)

  for i in range(num_log):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = args.model_name
    signature_name = args.signature_name[i % len(args.signature_name)]
    request.model_spec.signature_name = signature_name

    if signatures is None or signatures[signature_name].inputs:
      if signatures is not None:
        assert input_name in signatures[signature_name].inputs
      if args.variant_type == 'example':
        instances = [
            gen_example(args.sparse_features, dense_feature_meta, extra_meta,
                        args.feature_list, args.drop_rate).SerializeToString()
            for _ in range(args.batch_size)
        ]
      elif args.variant_type == 'instance':
        instances = [
            gen_instance(args.fidv1_features, args.fidv2_features,
                         dense_feature_meta, extra_meta, args.feature_list,
                         args.drop_rate).SerializeToString()
            for _ in range(args.batch_size)
        ]
      else:
        instances = [
            gen_example_batch(args.sparse_features, dense_feature_meta,
                              extra_meta, args.feature_list, args.batch_size,
                              args.drop_rate).SerializeToString()
        ]
      request.inputs[input_name].CopyFrom(tf.make_tensor_proto(instances))

    log = prediction_log_pb2.PredictionLog(
        predict_log=prediction_log_pb2.PredictLog(request=request))
    yield log
    if signatures:
      outputs = signatures[signature_name].outputs
      if signature_name == DEFAULT_SERVING_SIGNATURE_DEF_KEY and outputs is not None:
        if len(outputs) > 1 or (len(outputs) == 1 and
                                PredictOutput._SINGLE_OUTPUT_DEFAULT_NAME
                                not in outputs):
          for head_name in outputs:
            request.output_filter.append(head_name)
            log = prediction_log_pb2.PredictionLog(
                predict_log=prediction_log_pb2.PredictLog(request=request))
            yield log
            del request.output_filter[:]


def gen_warmup_file(warmup_file: str = None, drop_rate: float = None):
  warmup_args = ParserArgs(warmup_file=warmup_file)

  if drop_rate is not None:
    warmup_args.drop_rate = drop_rate

  if not warmup_args.warmup_file:
    logging.info(f'warmup_file is None, skip')
    return None
  elif tf.io.gfile.exists(warmup_args.warmup_file):
    logging.info(f'{warmup_args.warmup_file} exists, return directly')
    return warmup_args.warmup_file
  else:
    features = warmup_args.fidv1_features or warmup_args.fidv2_features or \
      warmup_args.sparse_features or warmup_args.dense_features or warmup_args.extra_features
    if features is None:
      logging.warning('features is None, pls. check!')
      return None

    # if warmup_args.variant_type != 'instance' and warmup_args.feature_list is None:
    #   logging.warning('feature_list is None, pls. check!')
    #   return None

    # remove label if exists
    if warmup_args.dense_features is not None and 'label' in warmup_args.dense_features:
      dense_features = deepcopy(warmup_args.dense_features)
      dense_feature_shapes = deepcopy(warmup_args.dense_feature_shapes)
      dense_feature_types = deepcopy(warmup_args.dense_feature_types)

      idx = warmup_args.dense_features.index('label')
      if idx is not None and idx >= 0:
        try:
          del dense_features[idx]
          del dense_feature_shapes[idx]
          del dense_feature_types[idx]
        except:
          pass
    else:
      dense_features = None
      dense_feature_shapes = None
      dense_feature_types = None

    warmup_args.dense_features = dense_features
    warmup_args.dense_feature_shapes = dense_feature_shapes
    warmup_args.dense_feature_types = dense_feature_types

    try:
      logging.info(
          f'begin to write prediction log to {warmup_args.warmup_file}')
      dirname = os.path.dirname(warmup_args.warmup_file)
      if not tf.io.gfile.exists(dirname):
        tf.io.gfile.makedirs(dirname)

      with tf.io.TFRecordWriter(warmup_args.warmup_file) as writer:
        for log in gen_prediction_log(warmup_args):
          writer.write(log.SerializeToString())
      logging.info(
          f'finish to write prediction log to {warmup_args.warmup_file}')
      return warmup_args.warmup_file
    except Exception as e:
      logging.warning(f'{type(e)}: {str(e)}')
      raise e


def gen_random_data_file(data_file_name: str,
                         args: ParserArgs,
                         num_batch: int = 128,
                         source: str = None,
                         sort_id: bool = True,
                         kafka_dump: bool = False,
                         kafka_dump_prefix: bool = False,
                         actions: List[int] = None):
  dense_feature_meta = []
  if args.dense_features:
    for name, shape, dtype in zip(args.dense_features,
                                  args.dense_feature_shapes,
                                  args.dense_feature_types):
      try:
        assert shape >= 1
        try:
          feat = args.feature_list.get(name)
          if feat is not None:
            dense_feature_meta.append(
                FeatureMeta(name, shape=shape, dtype=dtype, extra=feat))
          else:
            dense_feature_meta.append(
                FeatureMeta(name, shape=shape, dtype=dtype))
        except:
          dense_feature_meta.append(FeatureMeta(name, shape=shape, dtype=dtype))
      except:
        logging.warning(f'cannot find name {name} in feature_list')
  else:
    dense_feature_meta = None

  extra_meta = None
  if args.extra_features:
    extra_meta = [
        FeatureMeta(name=name, shape=shape)
        for name, shape in zip(args.extra_features, args.extra_feature_shapes)
    ]

  instances = []
  for i in range(num_batch):
    if args.variant_type == 'example':
      instances.extend([
          gen_example(args.sparse_features,
                      dense_feature_meta,
                      extra_meta,
                      args.feature_list,
                      args.drop_rate,
                      actions=actions).SerializeToString()
          for _ in range(args.batch_size)
      ])
    elif args.variant_type == 'instance':
      instances.extend([
          gen_instance(args.fidv1_features,
                       args.fidv2_features,
                       dense_feature_meta,
                       extra_meta,
                       args.feature_list,
                       args.drop_rate,
                       actions=actions).SerializeToString()
          for _ in range(args.batch_size)
      ])
    else:
      instances.extend([
          gen_example_batch(args.sparse_features,
                            dense_feature_meta,
                            extra_meta,
                            args.feature_list,
                            args.batch_size,
                            args.drop_rate,
                            actions=actions).SerializeToString()
      ])
  if sort_id:
    header = sort_header(sort_id, kafka_dump, kafka_dump_prefix)
  else:
    header = lg_header(source)
  with open(data_file_name, 'wb') as ostream:
    for inst in instances:
      ostream.write(header)
      ostream.write(int.to_bytes(len(inst), 8, byteorder='little'))
      ostream.write(inst)
