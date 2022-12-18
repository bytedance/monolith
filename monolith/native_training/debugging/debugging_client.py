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

import json
import requests
from google.protobuf import text_format

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf
from monolith.native_training.debugging.debugging_server import \
    STATUS, SUCCESS, FAIL, MSG
from monolith.native_training.runtime.hash_table import \
    embedding_hash_table_pb2

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "type", "",
    "The debugging type: debugging_variables or debugging_features.")
flags.DEFINE_list("variable_names", "", "The variable names for debugging.")
flags.DEFINE_list("feature_ids", "", "The feature ids for debugging.")
flags.DEFINE_string("feature_name", "", "The feature name of all ids.")
flags.DEFINE_list(
    "feature_names", "",
    "The feature names of all ids. If use this flag, the size of names and the size of ids must be equal."
)


def main(_):
  if FLAGS.type == "debugging_variables":
    if not FLAGS.variable_names:
      logging.info("Empty variable names")
      return

    data = json.dumps({"variable_names": FLAGS.variable_names})
    res = requests.post("http://127.0.0.1:%s/debugging/variables" % FLAGS.port,
                        data=data)
    res = json.loads(res.text)
    if res[STATUS] == FAIL:
      logging.info("Request fail! Reason: %s" % res[MSG])
      return
    msg = json.loads(res[MSG])
    for name in FLAGS.variable_names:
      logging.info("Variables: name[%s], value[%s]" %
                   (name, msg.get(name, "Not exist")))

  elif FLAGS.type == "debugging_features":
    if FLAGS.feature_name and FLAGS.feature_names:
      raise Exception("Can not provide both feature_name and feature_names.")
    if not FLAGS.feature_ids:
      logging.info("Empty feature ids")
      return
    if FLAGS.feature_name:
      FLAGS.feature_names = [FLAGS.feature_name] * len(FLAGS.feature_ids)
    if len(FLAGS.feature_names) != len(FLAGS.feature_ids):
      raise Exception(
          "Size of feature names [%s] and size of feature ids [%s] must be equal."
          % (len(FLAGS.feature_names), len(FLAGS.feature_ids)))

    data = json.dumps({
        "feature_names": FLAGS.feature_names,
        "feature_ids": FLAGS.feature_ids
    })
    res = requests.post("http://127.0.0.1:%s/debugging/features" % FLAGS.port,
                        data=data)
    res = json.loads(res.text)
    if res[STATUS] == FAIL:
      logging.info("Request fail! Reason: %s" % res[MSG])
      return
    msg = json.loads(res[MSG])
    for fname, fid in zip(FLAGS.feature_names, FLAGS.feature_ids):
      if fname in msg and fid in msg[fname]:
        entry_dump = embedding_hash_table_pb2.EntryDump()
        text_format.Parse(msg[fname][fid], entry_dump)
        logging.info("Features: name[%s], id[%s], value[\n%s]" %
                     (fname, fid, entry_dump))
      else:
        logging.info("Features: name[%s], id[%s], value[Not exist]" %
                     (fname, fid))


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.compat.v1.disable_eager_execution()
  app.run(main)
