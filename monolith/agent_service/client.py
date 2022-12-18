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

from absl import flags, logging, app
from monolith.agent_service.zk_mirror import ZKMirror
from monolith.native_training import env_utils

from monolith.native_training.zk_utils import MonolithKazooClient
from monolith.agent_service.data_def import ModelMeta, ReplicaMeta
from typing import Dict
from dataclasses import dataclass, field

from tensorflow_serving.apis.get_model_status_pb2 import ModelVersionStatus

ModelState = ModelVersionStatus.State

FLAGS = flags.FLAGS
flags.DEFINE_enum("cmd_type", "hb", [
    "hb", "gr", "addr", "get", "clean", "load", "unload", "meta", "status",
    "profile"
], "cmd_type: hb, gr, addr, res, status")
flags.DEFINE_string('zk_servers', None, 'zk servers')
flags.DEFINE_string('bzid', None, 'business id')
flags.DEFINE_string('model_name', None, 'model name')
flags.DEFINE_string("target", None, "host:port")
flags.DEFINE_enum("input_type", 'dump', [
    "json", "pbtext", "dump", "binary", "instance", "example_batch",
    "example_batch_to_instance"
], "inputs type for prediction")
flags.DEFINE_string("input_file", None, "The input file name")


@dataclass
class LoadSate:
  portal: bool = None
  publish: bool = None
  service: dict = field(default_factory=dict)  # Dict[str, ModelState]


class ServingClient(object):

  def __init__(self, zk_servers: str, bzid: str):
    self.kazoo = MonolithKazooClient(hosts=zk_servers)
    self.bzid = bzid
    self._zk = ZKMirror(self.kazoo, bzid)
    self._zk.start(is_client=True)

  def load(self,
           model_name: str,
           model_dir: str,
           ckpt: str = None,
           num_shard: int = -1):
    mm = ModelMeta(model_name=model_name,
                   model_dir=model_dir,
                   ckpt=ckpt,
                   num_shard=num_shard)
    path = mm.get_path(self._zk.portal_base_path)
    if self._zk.exists(path):
      raise RuntimeError(f'{model_name} has exists')
    self._zk.create(path=path, value=mm.serialize(), include_data=True)

  def unload(self, model_name: str):
    mm = ModelMeta(model_name=model_name)
    path = mm.get_path(self._zk.portal_base_path)
    if self._zk.exists(path):
      self._zk.delete(path)
    else:
      logging.warning(f'{model_name} not exists, nothing to do!')

  def get_status(self, model_name: str) -> LoadSate:
    state = LoadSate()
    if self.kazoo.exists(f'/{self.bzid}/portal/{model_name}'):
      state.portal = True

    for node in self.kazoo.get_children(f'/{self.bzid}/publish'):
      shard_id, replica_id, name = node.split(':')
      if name == model_name:
        state.publish = True
        break

    service = {}
    for node in self.kazoo.get_children(f'/{self.bzid}/service/{model_name}'):
      for replica in self.kazoo.get_children(
          f'/{self.bzid}/service/{model_name}/{node}'):
        path = f'/{self.bzid}/service/{model_name}/{node}/{replica}'
        value, _ = self.kazoo.get(path)
        rm = ReplicaMeta.deserialize(value)
        service[f'{node}:{replica}'] = rm.stat
    state.service = service

    return state


def main(_):
  env_utils.setup_host_ip()
  if FLAGS.zk_servers is None:
    raise ValueError(f'zk_servers is {FLAGS.zk_servers}')

  if FLAGS.bzid is None:
    raise ValueError(f'bzid is {FLAGS.bzid}')

  client = ServingClient(FLAGS.zk_servers, FLAGS.bzid)

  assert FLAGS.model_name is not None
  if FLAGS.cmd_type == 'load':
    assert FLAGS.model_dir is not None
    client.load(FLAGS.model_name, FLAGS.model_dir, FLAGS.ckpt, FLAGS.num_shard)
  elif FLAGS.cmd_type == 'unload':
    client.unload(FLAGS.model_name)
  else:
    print(client.get_status(FLAGS.model_name))


if __name__ == '__main__':
  app.run(main)
