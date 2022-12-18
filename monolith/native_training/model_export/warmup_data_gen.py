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


import sys

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_log_pb2
from monolith.native_training import env_utils
from monolith.native_training.model_export.data_gen_utils import gen_prediction_log
from monolith.native_training.data.feature_list import FeatureList

FLAGS = flags.FLAGS

flags.DEFINE_string("file_name", None, "input file name")
flags.DEFINE_integer("batch_size", 256, "Batch size of prediction request.")
flags.DEFINE_bool("lagrangex_header", False, "kafka_dump_prefix")
flags.DEFINE_bool("kafka_dump_prefix", False, "kafka_dump_prefix")
flags.DEFINE_bool("has_sort_id", True, "has_sort_id")
flags.DEFINE_bool("kafka_dump", False, "kafka_dump")
flags.DEFINE_integer("max_records", 1000, "Maximum number of warmup records.")
flags.DEFINE_string("model_name", "default", "mode name")
flags.DEFINE_string("signature_names", "serving_default", "signature names")
flags.DEFINE_string("output_path", "/tmp/tf_warmup_data",
                    "output path of warmup data.")
flags.DEFINE_enum("variant_type", "instance",
                  ['instance', 'example', 'example_batch'], "variant_type")
flags.DEFINE_string("sparse_features", None, "sparse_features")
flags.DEFINE_string("dense_features", None, "dense_features")
flags.DEFINE_integer("dense_feature_shapes", None, "dense_feature_shapes")
flags.DEFINE_integer("dense_feature_types", None, "dense_feature_types")
flags.DEFINE_string("extra_features", None, "extra_features")
flags.DEFINE_integer("extra_feature_shapes", None, "extra_feature_shapes")
flags.DEFINE_string("feature_list", None, "feature_list")
flags.DEFINE_enum("gen_type", "file", ['file', 'random'], "gen_type")
flags.DEFINE_integer("drop_rate", 0, "drop_rate")


class PBReader(object):

  def __init__(self,
               file_name: str,
               batch_size: int,
               lagrangex_header: bool = False,
               has_sort_id: bool = False,
               kafka_dump_prefix: bool = False,
               kafka_dump: bool = False,
               variant_type: str = 'instance'):
    self.file_name = file_name
    assert batch_size > 0
    self.batch_size = batch_size

    if self.file_name is None or len(self.file_name) == 0:
      self._stream = sys.stdin.buffer
    else:
      self._stream = tf.io.gfile.GFile(self.file_name)

    self.lagrangex_header = lagrangex_header
    self.has_sort_id = has_sort_id
    self.kafka_dump_prefix = kafka_dump_prefix
    self.kafka_dump = kafka_dump
    self.variant_type = variant_type

    self._curr = 0
    self._max_iter = None

  def __iter__(self):
    return self

  def __next__(self):
    try:
      self._curr += 1
      if self._max_iter is not None and self._curr > self._max_iter:
        raise StopIteration

      pb_items = []
      if self.variant_type == 'example_batch':
        # example_batch
        self._read_header()
        bin_string = self._stream.read(self._read_size())
        pb_items.append(bin_string)
      else:
        # example/instance
        for _ in range(self.batch_size):
          self._read_header()
          bin_string = self._stream.read(self._read_size())
          pb_items.append(bin_string)

      return tf.make_tensor_proto(pb_items)
    except:
      if self.file_name:
        self._stream.close()

      raise StopIteration

  def _read_size(self) -> int:
    size_t = 8
    try:
      size_binary = self._stream.read(size_t)
      if len(size_binary) != size_t:
        raise EOFError
    except Exception as e:
      raise e

    return int.from_bytes(size_binary, byteorder="little")

  def _read_header(self):
    size, aggregate_page_sortid_size = 0, 0
    if self.lagrangex_header:
      size = self._read_size()
    else:
      if self.kafka_dump_prefix:
        size = self._read_size()
        if size == 0:
          size = self._read_size()
        else:
          aggregate_page_sortid_size = size

      if self.has_sort_id:
        if aggregate_page_sortid_size == 0:
          size = self._read_size()
        else:
          size = aggregate_page_sortid_size
        sort_id = self._stream.read(size)

      if self.kafka_dump:
        size = self._read_size()

  def set_max_iter(self, max_records):
    if self.variant_type == 'example_batch':
      assert self.batch_size < max_records
      self._max_iter = (max_records // self.batch_size)
    else:
      self._max_iter = max_records


def gen_prediction_log_from_file(file_name: str = None,
                                 batch_size: int = 64,
                                 lagrangex_header: bool = False,
                                 kafka_dump_prefix=False,
                                 has_sort_id=True,
                                 kafka_dump=False,
                                 max_records=1000,
                                 variant_type: str = 'instance'):
  assert variant_type in {'instance', 'example', 'example_batch'}
  if variant_type == 'instance':
    input_name = 'instances'
  elif variant_type == 'example':
    input_name = 'examples'
  else:
    assert lagrangex_header == True
    input_name = 'example_batch'

  reader = PBReader(file_name, batch_size, lagrangex_header, has_sort_id,
                    kafka_dump_prefix, kafka_dump, variant_type)
  reader.set_max_iter(max_records)
  signature_names = [name.strip() for name in FLAGS.signature_names.split(',')]
  if 'serving_default' not in signature_names:
    signature_names.append('serving_default')

  for i, batch in enumerate(reader):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = FLAGS.model_name
    request.model_spec.signature_name = signature_names[i %
                                                        len(signature_names)]
    request.inputs[input_name].CopyFrom(batch)
    log = prediction_log_pb2.PredictionLog(
        predict_log=prediction_log_pb2.PredictLog(request=request))
    yield log


def tf_dtype(dtype: str) -> tf.compat.v1.dtypes.DType:
  if dtype in {'int', 'int32', 'short', 'uint', 'uint32', '3', '22'}:
    return tf.int32
  elif dtype in {'int64', 'long', 'uint64', '9', '23'}:
    return tf.int46
  elif dtype in {'float', 'float32', '1'}:
    return tf.float32
  elif dtype in {'float64', 'double', '2'}:
    return tf.float64
  elif dtype in {'bool', 'boolean', '10'}:
    return tf.bool
  elif dtype in {'str', 'string', 'char', '7'}:
    return tf.string
  else:
    raise Exception(f'{dtype} error')


def main(_):
  env_utils.setup_hdfs_env()
  with tf.io.TFRecordWriter(FLAGS.output_path) as writer:
    if FLAGS.gen_type == 'file':
      for log in gen_prediction_log_from_file(
          FLAGS.file_name, FLAGS.batch_size, FLAGS.lagrangex_header,
          FLAGS.kafka_dump_prefix, FLAGS.has_sort_id, FLAGS.kafka_dump,
          FLAGS.max_records, FLAGS.variant_type):
        writer.write(log.SerializeToString())
    else:
      assert FLAGS.sparse_features is not None
      sparse_features = FLAGS.sparse_features.split(',')

      if FLAGS.dense_features is not None:
        dense_features = FLAGS.dense_features.split(',')
        dense_feature_shapes = [
            int(shape) for shape in FLAGS.dense_feature_shapes.split(',')
        ]
        dense_feature_types = [
            tf_dtype(dtype) for dtype in FLAGS.dense_feature_types.split(',')
        ]
      else:
        dense_features = FLAGS.dense_features
        dense_feature_shapes = FLAGS.dense_feature_shapes
        dense_feature_types = FLAGS.dense_feature_types

      if FLAGS.extra_features is not None:
        extra_features = FLAGS.extra_features.split(',')
        extra_feature_shapes = [
            int(shape) for shape in FLAGS.extra_feature_shapes.split(',')
        ]
      else:
        extra_features = FLAGS.extra_features
        extra_feature_shapes = FLAGS.extra_feature_shapes

      feature_list = FeatureList.parse(FLAGS.feature_list)
      for log in gen_prediction_log(FLAGS.model_name, sparse_features,
                                    dense_features, dense_feature_shapes,
                                    dense_feature_types, extra_features,
                                    extra_feature_shapes, feature_list,
                                    FLAGS.batch_size, FLAGS.max_records, None,
                                    FLAGS.variant_type, FLAGS.drop_rate):
        writer.write(log.SerializeToString())


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  app.run(main)
