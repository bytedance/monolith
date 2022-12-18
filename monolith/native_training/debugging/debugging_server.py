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
import traceback
from collections import defaultdict
from google.protobuf import text_format

import flask
from flask import request
from typing import List
from urllib.parse import urlparse

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf
from tensorflow.python.framework import meta_graph
from tensorflow.python.lib.io import file_io

from monolith.utils import get_libops_path
from monolith.native_training import cluster_manager
from monolith.native_training import env_utils
from monolith.native_training import multi_type_hash_table
from monolith.native_training import utils
from monolith.native_training.proto import debugging_info_pb2
from monolith.native_training.runtime.ops import gen_monolith_ops
from monolith.native_training.runtime.hash_table import \
    embedding_hash_table_pb2

STATUS = "status"
SUCCESS = "success"
FAIL = "fail"

MSG = "msg"

FLAGS = flags.FLAGS
flags.DEFINE_string("host", "", "The host of server")
flags.DEFINE_integer("port", 0, "The port of server")
flags.DEFINE_string(
    "model_dir",
    default="",
    help="Directory where model parameters, graph, etc are saved.")

clip_ops = gen_monolith_ops

filter_ops = gen_monolith_ops
filter_save_op = gen_monolith_ops
filter_restore_op = gen_monolith_ops

monolith_custom_ops = gen_monolith_ops

training_ops = gen_monolith_ops

pb_datasource_ops = gen_monolith_ops


class DebuggingWorker:
  """A debugging worker that connected to training servers."""

  def __init__(self, model_dir):

    debugging_info_str = file_io.read_file_to_string(
        utils.get_debugging_info_file_name(model_dir), binary_mode=True)
    self._debugging_info = debugging_info_pb2.DebuggingInfo()
    self._debugging_info.ParseFromString(debugging_info_str)

    self._server = tf.distribute.Server(
        {"local": [utils.get_local_host() + ":0"]})
    self._addr = urlparse(self._server.target).netloc

    fake_worker_list = [
        "0.0.0.0:{}".format(i)
        for i in range(1, self._debugging_info.num_workers)
    ]
    fake_worker_list.append(self._addr)
    cluster = {
        "chief": [self._debugging_info.cluster.chief_addr],
        "ps": [addr for addr in self._debugging_info.cluster.ps_addrs],
        "worker": fake_worker_list,
    }
    task = {"type": "worker", "index": self._debugging_info.num_workers - 1}

    feature_name_config_map = {}
    for feature_name_config in self._debugging_info.feature_name_configs:
      feature_name_config_map[
          feature_name_config.feature_name] = feature_name_config.config_str

    def dummy_factory(*args, **kwargs):
      pass

    self._dummy_merged_table = multi_type_hash_table.MergedMultiTypeHashTable(
        feature_name_config_map, dummy_factory)

    self._session_config = cluster_manager.generate_session_config(
        (cluster, task))
    logging.info(self._session_config)
    self._graph = tf.Graph()
    with tf.compat.v1.Session(self._server.target,
                              config=self._session_config,
                              graph=self._graph) as sess:
      self._imported_vars, self._imported_return_elements = (
          meta_graph.import_scoped_meta_graph_with_return_elements(
              utils.get_meta_graph_file_name(model_dir)))
    logging.info("Finish import meta graph.")

  def _get_table_name(self, merged_name, index):
    return "MonolithHashTable_%s_%s" % (merged_name, index)

  def fetch_variables(self, variable_names: List[str]):
    with tf.compat.v1.Session(self._server.target,
                              config=self._session_config,
                              graph=self._graph) as sess:
      req_variables, req_names = [], []
      for var_name in variable_names:
        if var_name in self._imported_vars:
          req_variables.append(self._imported_vars[var_name])
          req_names.append(var_name)
      resp_variables = sess.run(req_variables)
    return {k: str(v) for k, v in zip(req_names, resp_variables)}

  def fetch_features(self, feature_names: List[str], feature_ids: List[str]):
    merged_name_to_features = defaultdict(list)
    for fea_name, fid in zip(feature_names, feature_ids):
      if fea_name not in self._dummy_merged_table.slot_mapping:
        continue
      merged_name_to_features[
          self._dummy_merged_table.slot_mapping[fea_name]].append(
              (fea_name, int(fid)))

    feature_to_entry_dump_str = defaultdict(dict)
    with tf.compat.v1.Session(self._server.target,
                              config=self._session_config,
                              graph=self._graph) as sess:
      for merged_name, features in merged_name_to_features.items():
        indices = tf.math.floormod(
            [f[1] for f in features],
            len(self._debugging_info.cluster.ps_addrs)).eval()
        index_to_features = defaultdict(list)
        for index, fea in zip(indices, features):
          index_to_features[index].append(fea)

        for index, features in index_to_features.items():
          table_name = self._get_table_name(merged_name, index)
          table = self._graph.get_tensor_by_name(table_name + ":0")
          entry_dump_strs = sess.run(
              monolith_custom_ops.monolith_hash_table_lookup_entry(
                  table, tf.cast([f[1] for f in features], dtype=tf.int64)))
          for (fname, fid), entry_dump_str in zip(features, entry_dump_strs):
            if entry_dump_str:
              entry_dump = embedding_hash_table_pb2.EntryDump()
              entry_dump.ParseFromString(entry_dump_str)
              feature_to_entry_dump_str[fname][str(
                  fid)] = text_format.MessageToString(entry_dump)

    return feature_to_entry_dump_str


def create_app() -> flask.Flask:
  app = flask.Flask("Monolith_Debugging_Server")
  worker = DebuggingWorker(FLAGS.model_dir)

  @app.route("/debugging/variables", methods=["POST"])
  def fetch_variables():
    try:
      data = request.get_data()
      data = json.loads(data)
      logging.info("Fetch variables req: %s" % data)
      result = worker.fetch_variables(data.get("variable_names", []))
      resp = {STATUS: SUCCESS, MSG: json.dumps(result)}
    except:
      resp = {STATUS: FAIL, MSG: traceback.format_exc()}
    logging.info("Fetch variables resp: %s" % resp)
    return resp

  @app.route("/debugging/features", methods=["POST"])
  def fetch_features():
    try:
      data = request.get_data()
      data = json.loads(data)
      logging.info("Fetch features req: %s" % data)
      feature_names = data.get("feature_names", [])
      feature_ids = data.get("feature_ids", [])
      if len(feature_names) != len(feature_ids):
        raise Exception(
            "Size of feature names [%s] and size of feature ids [%s] must be equal."
            % (len(feature_names), len(feature_ids)))
      result = worker.fetch_features(feature_names, feature_ids)
      resp = {STATUS: SUCCESS, MSG: json.dumps(result)}
    except:
      resp = {STATUS: FAIL, MSG: traceback.format_exc()}
    logging.info("Fetch features resp: %s" % resp)
    return resp

  return app


def main(_):
  env_utils.setup_hdfs_env()
  server_app = create_app()
  server_app.run(host=FLAGS.host, port=FLAGS.port)


if __name__ == "__main__":
  app.run(main)
