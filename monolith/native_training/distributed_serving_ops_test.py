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
import tensorflow as tf

from absl import logging
from monolith.agent_service import utils
from monolith.agent_service import backends
from monolith.agent_service.replica_manager import ReplicaWatcher
from monolith.agent_service.mocked_zkclient import FakeKazooClient
from monolith.agent_service.data_def import ReplicaMeta
from monolith.native_training import hash_table_ops
from monolith.native_training import distributed_serving_ops
from monolith.native_training.distributed_serving_ops import ParameterSyncClient, \
  DummySyncServer
from monolith.native_training.runtime.parameter_sync import \
  parameter_sync_pb2


def test_dummy_sync_server(server_num: int):
  return [DummySyncServer("localhost:0") for _ in range(server_num)]


def test_parameter_sync_client(targets):
  config = parameter_sync_pb2.ClientConfig()
  config.targets.extend(targets)
  return ParameterSyncClient(
      distributed_serving_ops.parameter_sync_client_from_config(config=config))


def _get_id_tensor(x):
  return tf.constant(x, dtype=tf.int64)


class ParameterSyncOpsTest(tf.test.TestCase):

  def test_parameter_sync_client(self):
    servers = test_dummy_sync_server(2)
    ports = [server.get_port() for server in servers]
    with self.session() as sess:
      ports = sess.run(ports)
      targets = ["localhost:{}".format(port[0]) for port in ports]
      client = test_parameter_sync_client(targets)
      dim = 3
      hash_table = hash_table_ops.test_hash_table(dim,
                                                  learning_rate=0.1,
                                                  sync_client=client.handle)
      id_tensor = _get_id_tensor([0, 0, 1])
      embeddings = hash_table.lookup(id_tensor)
      loss = -embeddings
      grads = tf.gradients(loss, embeddings)
      global_step = _get_id_tensor(0)
      hash_table = hash_table.apply_gradients(id_tensor,
                                              grads[0],
                                              global_step=global_step)
      new_embeddings = hash_table.lookup(_get_id_tensor([0, 1]))
      new_embeddings = sess.run(new_embeddings)
      self.assertAllClose(new_embeddings, [[0.2, 0.2, 0.2], [0.1, 0.1, 0.1]])

      config = parameter_sync_pb2.ClientConfig()
      config.targets.extend(targets)
      result = sess.run(client.create_sync_op(config.SerializeToString()))
      print(json.dumps(json.loads(result[0]), indent=2))
      sess.run([server.shutdown() for server in servers])

  def test_refresh_sync_config_1(self):

    def mock_replica_watcher(ps_index: int):
      zk = FakeKazooClient()
      zk.start()
      config = utils.AgentConfig(bzid="demo",
                                 base_name="test_ffm_model",
                                 deploy_type='ps',
                                 replica_id=0,
                                 num_ps=10)
      path_prefix = f'/{config.bzid}/service/{config.base_name}'
      replica_path = f'{path_prefix}/ps:{ps_index}/{config.replica_id}'
      replica_meta = ReplicaMeta(address="localhost:8500",
                                 stat=utils.ModelState.AVAILABLE)
      replica_meta_bytes = bytes(replica_meta.to_json(), encoding='utf-8')
      zk.ensure_path(replica_path)
      zk.set(replica_path, replica_meta_bytes)
      replica_watcher = ReplicaWatcher(zk, config)
      replica_watcher.watch_data()
      return replica_watcher, zk

    replica_watcher, zk = mock_replica_watcher(1)
    config_str = distributed_serving_ops.refresh_sync_config(
        replica_watcher.to_sync_wrapper(), 1)
    config = parameter_sync_pb2.ClientConfig()
    config.ParseFromString(config_str)
    self.assertEqual(config.model_name, "ps_1")
    logging.info('targets: %s', config.targets)
    self.assertEqual(config.targets, ["localhost:8500"])
    replica_watcher.stop()
    zk.stop()

  def test_refresh_sync_config_2(self):
    # prepare envs
    bd = backends.ZKBackend('demo', zk_servers='127.0.0.1:9999')
    bd._zk = FakeKazooClient()
    bd.start()
    container = backends.Container("default", "asdf")
    service_info = backends.ContainerServiceInfo(grpc="localhost:8888",
                                                 http="localhost:8889",
                                                 archon="localhost:8890",
                                                 agent="localhost:8891",
                                                 idc="lf")
    bd.report_service_info(container, service_info)
    bd.sync_available_saved_models(
        container, {
            backends.SavedModel("test_ffm_model", "ps_0"),
            backends.SavedModel("test_ffm_model", "ps_1"),
            backends.SavedModel("test_ffm_model", "ps_2"),
        })

    # test sync targets
    bd.subscribe_model("test_ffm_model")
    config = parameter_sync_pb2.ClientConfig()
    config_str = distributed_serving_ops.refresh_sync_config(bd, 1)
    config.ParseFromString(config_str)
    self.assertEqual(config.model_name, "test_ffm_model:ps_1")
    self.assertEqual(config.targets, ["localhost:8888"])
    bd.stop()


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
