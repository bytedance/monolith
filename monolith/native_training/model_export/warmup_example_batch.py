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

import tensorflow as tf

from tensorflow.python.util import compat

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_log_pb2
from monolith.native_training import env_utils

flags.DEFINE_string('input_folder', '', '')
flags.DEFINE_string('output_path', '', '')

FLAGS = flags.FLAGS


def gen_prediction_log(input_folder):
  filenames = tf.io.gfile.listdir(input_folder)
  for filename in filenames:
    with tf.io.gfile.GFile(os.path.join(input_folder, filename), 'rb') as f:
      request = predict_pb2.PredictRequest()
      print(request.ParseFromString(compat.as_bytes(f.read())))
      request.model_spec.name = "default"
      request.model_spec.signature_name = "serving_default"
      log = prediction_log_pb2.PredictionLog(
          predict_log=prediction_log_pb2.PredictLog(request=request))
      yield log


def main(_):
  with tf.io.TFRecordWriter(FLAGS.output_path) as writer:
    for log in gen_prediction_log(FLAGS.input_folder):
      writer.write(log.SerializeToString())


if __name__ == "__main__":
  env_utils.setup_hdfs_env()
  app.run(main)