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

from absl import app
from absl import flags
from absl import logging

import re
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_log_pb2
from monolith.native_training import env_utils

FLAGS = flags.FLAGS

flags.DEFINE_string("file_name", None, "input file name")


def main(_):
  try:
    env_utils.setup_hdfs_env()
  except:
    pass
  tf.compat.v1.enable_eager_execution()
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  logging.set_verbosity(logging.INFO)

  def decode_fn(record_bytes):
    log = prediction_log_pb2.PredictionLog()
    log.ParseFromString(record_bytes)
    return log

  for i, batch in enumerate(tf.data.TFRecordDataset([FLAGS.file_name])):
    prediction_log = decode_fn(batch.numpy())
    predict_log = prediction_log.predict_log
    request = predict_log.request
    simple_request_string = re.sub('string_val:.*', 'string_val: ...',
                                   str(request))
    logging.info('%dth model_spec:\n%s', i, simple_request_string)


if __name__ == "__main__":
  app.run(main)
