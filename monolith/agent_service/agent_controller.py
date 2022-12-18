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
import os
import fnmatch
from absl import app, flags, logging

import tensorflow as tf
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat

from monolith.native_training import env_utils
from monolith.agent_service.backends import CtrlBackend, ZKBackend, SavedModel, SavedModelDeployConfig

SUPPORTED_CMDS = "decl|pub|unpub|bzid_info"
flags.DEFINE_string(
    "zk_servers",
    "",
    "zk connection string")
flags.DEFINE_string("bzid", "test", "namespace")
flags.DEFINE_string("export_base", "", "exported model base path")
flags.DEFINE_integer("overwrite", 0, "overwrite existing saved_model configs")
flags.DEFINE_string("model_name", "", "model_name")
flags.DEFINE_string("layout", "", "layout base")
flags.DEFINE_string("arch", "entry_ps", "serving architecture")
flags.DEFINE_string("cmd", "bzid_info", SUPPORTED_CMDS)

FLAGS = flags.FLAGS


def find_model_name(exported_models_path: str):
  # find model name used in remote predict op
  entry_path = os.path.join(exported_models_path, 'entry')
  latest_timestamp = sorted(tf.io.gfile.listdir(entry_path))[0]
  sm_file = os.path.join(entry_path, latest_timestamp, "saved_model.pb")
  logging.info(f"loading: {sm_file}")
  with tf.io.gfile.GFile(sm_file, 'rb') as f:
    sm = saved_model_pb2.SavedModel()
    sm.ParseFromString(compat.as_bytes(f.read()))
    remote_predict_model_names = [
        node.attr['model_name'].s.decode('utf-8')
        for node in sm.meta_graphs[0].graph_def.node
        if node.op == 'TfServingRemotePredict'
    ]
    if not remote_predict_model_names:
      return None
    else:
      return remote_predict_model_names[0].split(":")[0]


def declare_saved_model(bd: CtrlBackend,
                        export_base: str,
                        model_name: str = None,
                        overwrite=False,
                        arch="entry_ps"):
  assert arch == "entry_ps", "only entry + ps architecture supported"
  model_name_from_export = find_model_name(export_base)
  if not model_name:
    model_name = model_name_from_export
  if model_name != model_name_from_export:
    logging.error(
        f"user model_name: {model_name}, exported_model_name: {model_name_from_export}"
    )
  assert model_name is not None, "Model name is None"
  assert not bd.list_saved_models(
      model_name) or overwrite, f"{model_name} exists and not in overwrite mode"

  sub_graphs = tf.io.gfile.listdir(export_base)
  for sub_graph in sub_graphs:
    deploy_config = SavedModelDeployConfig(
        model_base_path=os.path.join(export_base, sub_graph),
        version_policy='latest' if sub_graph == 'entry' else 'latest_once')
    bd.decl_saved_model(SavedModel(model_name, sub_graph), deploy_config)
  logging.info(
      f"declare saved_model for {model_name} on path {export_base} success")
  return model_name


def map_model_to_layout(bd: CtrlBackend, model_pattern: str, layout_path: str,
                        action: str):
  model_name, sub_graph_pattern = model_pattern.split(":", 1)
  sub_graphs = [
      saved_model.sub_graph for saved_model in bd.list_saved_models(model_name)
  ]
  matched_sub_graphs = fnmatch.filter(sub_graphs, sub_graph_pattern)
  for sub_graph in matched_sub_graphs:
    saved_model = SavedModel(model_name, sub_graph)
    if action == 'pub':
      logging.info(f"publishing {saved_model} to {layout_path}")
      bd.add_to_layout(layout_path, saved_model)
    elif action == 'unpub':
      logging.info(f"deleting {saved_model} from {layout_path}")
      bd.remove_from_layout(layout_path, saved_model)


def bzid_info(bd: CtrlBackend):
  print(json.dumps(bd.bzid_info(), indent=2))


def main(_):
  if FLAGS.cmd not in SUPPORTED_CMDS.split("|"):
    raise ValueError(
        f"unsupported cmd {FLAGS.cmd}, options are {SUPPORTED_CMDS}")
  print()
  bd = ZKBackend(FLAGS.bzid, FLAGS.zk_servers)
  try:
    bd.start()
    if FLAGS.cmd == 'decl':
      assert FLAGS.export_base is not None and len(FLAGS.export_base) > 0
      declare_saved_model(bd,
                          FLAGS.export_base,
                          overwrite=FLAGS.overwrite,
                          arch=FLAGS.arch)
    elif FLAGS.cmd == 'pub' or FLAGS.cmd == 'unpub':
      assert len(FLAGS.layout) > 0 and len(FLAGS.model_name) > 0
      layout_path = f"/{FLAGS.bzid}/layouts/{FLAGS.layout}"
      map_model_to_layout(bd, FLAGS.model_name, layout_path, action=FLAGS.cmd)
    elif FLAGS.cmd == 'bzid_info':
      bzid_info(bd)
    else:
      raise ValueError(
          f"unsupported cmd {FLAGS.cmd}, options are {SUPPORTED_CMDS}")
  finally:
    bd.stop()


if __name__ == "__main__":
  try:
    env_utils.setup_hdfs_env()
  except Exception as e:
    logging.error('setup_hdfs_env fail {}!'.format(e))
  logging.set_verbosity(logging.INFO)
  app.run(main)
