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

from absl import app, flags, logging

import grpc
import socket
import os, re

from monolith.agent_service import utils
from monolith.agent_service.agent_service_pb2_grpc import AgentServiceStub
from monolith.agent_service.agent_service_pb2 import HeartBeatRequest, ServerType, \
  GetReplicasRequest
from monolith.agent_service.data_def import ModelMeta, ReplicaMeta, ModelState
from monolith.agent_service.resource_utils import cal_model_info_v2
from monolith.agent_service.client import FLAGS
from monolith.native_training import env_utils
from monolith.native_training.zk_utils import MonolithKazooClient
from kazoo.exceptions import NoNodeError

flags.DEFINE_integer("port", 0, "agent_port")
flags.DEFINE_enum("args", "addr",
                  ["addr", "portal", "pub", "res", "lock", "elect", "info"],
                  "args: addr, portal, pub, res, lock, elect")
flags.DEFINE_enum("server_type", "ps", ["ps", "entry", "dense"],
                  "server_type, ps or entry or dense")
flags.DEFINE_integer("task", 0, "task id of given server_type")
flags.DEFINE_string('model_dir', None, 'saved model dir')
flags.DEFINE_string('ckpt', None, 'ckpt name')
flags.DEFINE_integer('num_shard', -1,
                     'number of shard will use of current model')


def main(_):
  env_utils.setup_hdfs_env()
  agent_conf = utils.AgentConfig.from_file(FLAGS.conf)
  if FLAGS.port != 0:
    agent_conf.agent_port = FLAGS.port

  host = os.environ.get("MY_HOST_IP",
                        socket.gethostbyname(socket.gethostname()))
  channel = grpc.insecure_channel(f"{host}:{agent_conf.agent_port}")
  stub = AgentServiceStub(channel)
  model_name = agent_conf.base_name or FLAGS.model_name

  if FLAGS.server_type == "ps":
    server_type = ServerType.PS
  elif FLAGS.server_type == "dense":
    server_type = ServerType.DENSE
  else:
    server_type = ServerType.ENTRY

  if FLAGS.cmd_type == 'hb':
    request = HeartBeatRequest(server_type=server_type)
    addresses = stub.HeartBeat(request).addresses
    for k, v in addresses.items():
      addrs = f"{v}".strip().split("\n")
      print("{k}  -> ({length}) \n\t{addrs}".format(k=k,
                                                    length=len(addrs),
                                                    addrs="\n\t".join(addrs)))
  elif FLAGS.cmd_type == 'gr':
    assert model_name is not None
    request = GetReplicasRequest(server_type=server_type,
                                 task=FLAGS.task,
                                 model_name=model_name)
    print(ServerType.Name(server_type), FLAGS.task, " => ",
          stub.GetReplicas(request).address_list.address)
  elif (FLAGS.cmd_type == 'get' and
        FLAGS.args == 'addr') or FLAGS.cmd_type == 'addr':
    assert model_name is not None
    zk = MonolithKazooClient(hosts=agent_conf.zk_servers)
    zk.start()
    # bzid/service/model_name/idc:cluster/server_type:task/replica_id
    path_prefix = f'/{agent_conf.bzid}/service/{model_name}'
    servers = []
    TASK = re.compile(r'^(\w+):(\d+)$')
    try:
      ics_or_svrs = zk.get_children(path_prefix)
      for ic_svr in ics_or_svrs:
        matched = TASK.match(ic_svr)
        if matched:
          svr = ic_svr
          servers.append(svr)
        else:
          ic = ic_svr
          svrs = zk.get_children(f'{path_prefix}/{ic}')
          if svrs:
            servers.extend([f'{ic}/{svr}' for svr in svrs])
    except NoNodeError as e:
      print(f'{model_name} has not load !')
      zk.stop()
      return

    entry_id = 0
    for_print = []
    if servers:
      for server in servers:
        replicas = zk.get_children(f"{path_prefix}/{server}")
        if replicas:
          for replica in replicas:
            data, _ = zk.get(f"{path_prefix}/{server}/{replica}")
            data = ReplicaMeta.deserialize(data)

            replica_id = replica
            for_print.append(
                f"{path_prefix}/{server}/{replica_id}\tarchon_address: {data.archon_address}\t"
                f"address: {data.address}\tstate: {ModelState.Name(data.stat)}")

    for_print.sort()
    print("\n".join(for_print))
    zk.stop()
  elif FLAGS.cmd_type == 'get' and FLAGS.args == 'info':
    print(cal_model_info_v2(FLAGS.model_dir, FLAGS.ckpt))
  elif FLAGS.cmd_type == 'get':
    zk = MonolithKazooClient(hosts=agent_conf.zk_servers)
    zk.start()
    # /{bzid}/resource/{shard_id}:{replica_id}  -> ResourceSpec
    if FLAGS.args == 'res':
      path_prefix = f'/{agent_conf.bzid}/resource'
    elif FLAGS.args == 'pub':
      path_prefix = f'/{agent_conf.bzid}/publish'
    elif FLAGS.args == 'portal':
      path_prefix = f'/{agent_conf.bzid}/portal'
    elif FLAGS.args == 'lock':
      path_prefix = f'/{agent_conf.bzid}/lock'
    elif FLAGS.args == 'elect':
      path_prefix = f'/{agent_conf.bzid}/election'
    else:
      return

    try:
      servers = zk.get_children(path_prefix)
    except NoNodeError as e:
      print(f'no {FLAGS.args} found !')
      zk.stop()
      return

    resources = {}
    if servers:
      for server in servers:
        data, _ = zk.get(f"{path_prefix}/{server}")
        resources[server] = data

    if resources:
      keys = list(resources.keys())
      keys.sort()
      for key in keys:
        print(key, resources[key])
    else:
      print(resources)
    zk.stop()
  elif FLAGS.cmd_type == 'load':
    assert model_name is not None
    zk = MonolithKazooClient(hosts=agent_conf.zk_servers)
    zk.start()
    mm = ModelMeta(model_name=model_name,
                   model_dir=FLAGS.model_dir,
                   ckpt=FLAGS.ckpt,
                   num_shard=FLAGS.num_shard)
    path = f'/{agent_conf.bzid}/portal/{model_name}'
    try:
      zk.create(path, value=mm.serialize(), include_data=True, makepath=True)
    except Exception as e:
      logging.info(e)
      zk.set(path, value=mm.serialize())
    zk.stop()
  elif FLAGS.cmd_type == 'unload':
    zk = MonolithKazooClient(hosts=agent_conf.zk_servers)
    zk.start()

    path = f'/{agent_conf.bzid}/portal/{model_name}'
    try:
      zk.delete(path)
    except Exception as e:
      logging.info(e)

    zk.stop()
  elif FLAGS.cmd_type == 'clean':
    zk = MonolithKazooClient(hosts=agent_conf.zk_servers)
    zk.start()

    if FLAGS.args == 'portal':
      path = f'/{agent_conf.bzid}/portal'
      for node in zk.get_children(path):
        zk.delete(os.path.join(path, node))
    elif FLAGS.args == 'pub':
      path = f'/{agent_conf.bzid}/publish'
      for node in zk.get_children(path):
        zk.delete(os.path.join(path, node))
    elif FLAGS.args == 'addr':
      path = f'/{agent_conf.bzid}/service'
      for node in zk.get_children(path):
        zk.delete(os.path.join(path, node), recursive=True)
    elif FLAGS.args == 'res':
      path = f'/{agent_conf.bzid}/resource'
      for node in zk.get_children(path):
        zk.delete(os.path.join(path, node), recursive=True)
    else:
      raise RuntimeError(f"{FLAGS.args} is not support!")

    zk.stop()


if __name__ == "__main__":
  app.run(main)
