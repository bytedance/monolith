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

import unittest

from monolith.agent_service import utils


class ServingUtilsTest(unittest.TestCase):

  def test_gen_model_spec(self):
    name, version, signature_name = 'model', 1, 'predict'
    model_spec = utils.gen_model_spec(name, version, signature_name)
    self.assertEqual(model_spec.name, name)
    self.assertEqual(model_spec.version.value, version)
    self.assertEqual(model_spec.signature_name, signature_name)

  def test_gen_model_config(self):
    name, base_path, num_versions = 'model', '/tmp/model/saved_model', 2
    version_labels = {'v0': 0, 'v1': 1}
    model_config = utils.gen_model_config(name,
                                          base_path,
                                          version_data=num_versions,
                                          version_labels=version_labels)
    self.assertEqual(model_config.name, name)
    self.assertEqual(model_config.base_path, base_path)
    self.assertEqual(model_config.model_version_policy.latest.num_versions,
                     num_versions)

  def test_gen_status_proto(self):
    status_proto = utils.gen_status_proto(utils.ErrorCode.CANCELLED,
                                          error_message='CANCELLED')
    self.assertEqual(status_proto.error_code, utils.ErrorCode.CANCELLED)
    self.assertEqual(status_proto.error_message, 'CANCELLED')

  def test_gen_model_version_status(self):
    version, state = 1, utils.ModelState.START
    error_code, error_message = utils.ErrorCode.NOT_FOUND, "NOT_FOUND"
    model_version_status = utils.gen_model_version_status(
        version, state, error_code, error_message)
    self.assertEqual(model_version_status.version, version)
    self.assertEqual(model_version_status.state, state)

  def test_gen_from_file(self):
    conf = utils.AgentConfig.from_file(
        fname='monolith/agent_service/agent.conf')
    self.assertTrue(conf.stand_alone_serving)

  def test_list_field(self):
    conf = utils.AgentConfig.from_file(
        fname='monolith/agent_service/agent.conf')
    self.assertEqual(conf.layout_filters, ['ps_0', 'ps_1'])

  def test_instance_wrapper_from_json(self):
    iw = utils.InstanceFormater.from_json(
        'monolith/agent_service/test_data/inst.json')
    tensor_proto = iw.to_tensor_proto(5)
    self.assertEqual(tensor_proto.dtype, 7)
    self.assertEqual(tensor_proto.tensor_shape.dim[0].size, 5)

  def test_instance_wrapper_from_pbtext(self):
    iw = utils.InstanceFormater.from_pb_text(
        'monolith/agent_service/test_data/inst.pbtext')
    tensor_proto = iw.to_tensor_proto(5)
    self.assertEqual(tensor_proto.dtype, 7)
    self.assertEqual(tensor_proto.tensor_shape.dim[0].size, 5)

  def test_instance_wrapper_from_dump(self):
    iw = utils.InstanceFormater.from_dump(
        'monolith/agent_service/test_data/inst.dump')
    tensor_proto = iw.to_tensor_proto(5)
    self.assertEqual(tensor_proto.dtype, 7)
    self.assertEqual(tensor_proto.tensor_shape.dim[0].size, 5)

  def test_get_cmd_and_port(self):
    conf = utils.AgentConfig.from_file(
        fname='monolith/agent_service/agent.conf')
    conf.agent_version = 2
    cmd, port = conf.get_cmd_and_port(binary='tensorflow_model_server',
                                      server_type='ps')
    self.assertTrue('model_config_file_poll_wait_seconds' in cmd)

  def test_zk_path_full(self):
    zk_pzth = utils.ZKPath(
        '/bzid/service/base_name/idc:cluster/server_type:0/1')
    self.assertEqual(zk_pzth.bzid, 'bzid')
    self.assertEqual(zk_pzth.base_name, 'base_name')
    self.assertEqual(zk_pzth.idc, 'idc')
    self.assertEqual(zk_pzth.cluster, 'cluster')
    self.assertEqual(zk_pzth.server_type, 'server_type')
    self.assertEqual(zk_pzth.index, '0')
    self.assertEqual(zk_pzth.replica_id, '1')
    self.assertEqual(zk_pzth.location, 'idc:cluster')
    self.assertEqual(zk_pzth.task, 'server_type:0')
    self.assertTrue(zk_pzth.ship_in(None, None))

  def test_zk_path_partial(self):
    zk_pzth = utils.ZKPath('/bzid/service/base_name/idc:cluster/server_type:0')
    self.assertEqual(zk_pzth.bzid, 'bzid')
    self.assertEqual(zk_pzth.base_name, 'base_name')
    self.assertEqual(zk_pzth.idc, 'idc')
    self.assertEqual(zk_pzth.cluster, 'cluster')
    self.assertEqual(zk_pzth.server_type, 'server_type')
    self.assertEqual(zk_pzth.index, '0')
    self.assertEqual(zk_pzth.replica_id, None)
    self.assertEqual(zk_pzth.location, 'idc:cluster')
    self.assertEqual(zk_pzth.task, 'server_type:0')
    self.assertTrue(zk_pzth.ship_in('idc', 'cluster'))

  def test_zk_path_old_full(self):
    zk_pzth = utils.ZKPath('/bzid/service/base_name/server_type:0/1')
    self.assertEqual(zk_pzth.bzid, 'bzid')
    self.assertEqual(zk_pzth.base_name, 'base_name')
    self.assertEqual(zk_pzth.idc, None)
    self.assertEqual(zk_pzth.cluster, None)
    self.assertEqual(zk_pzth.server_type, 'server_type')
    self.assertEqual(zk_pzth.index, '0')
    self.assertEqual(zk_pzth.replica_id, '1')
    self.assertEqual(zk_pzth.location, None)
    self.assertEqual(zk_pzth.task, 'server_type:0')
    self.assertTrue(zk_pzth.ship_in(None, None))

  def test_zk_path_old_partial(self):
    zk_pzth = utils.ZKPath('/bzid/service/base_name/server_type:0')
    self.assertEqual(zk_pzth.bzid, 'bzid')
    self.assertEqual(zk_pzth.base_name, 'base_name')
    self.assertEqual(zk_pzth.idc, None)
    self.assertEqual(zk_pzth.cluster, None)
    self.assertEqual(zk_pzth.server_type, 'server_type')
    self.assertEqual(zk_pzth.index, '0')
    self.assertEqual(zk_pzth.replica_id, None)
    self.assertEqual(zk_pzth.location, None)
    self.assertEqual(zk_pzth.task, 'server_type:0')
    self.assertTrue(zk_pzth.ship_in(None, None))

  def test_zk_path_old_partial2(self):
    zk_pzth = utils.ZKPath(
        '/1_20001223_44ce735e-d05c-11ec-ba29-00163e356637/service/20001223_zm_test_realtime_training_1328_v4_r982567_0/ps:1'
    )
    self.assertEqual(zk_pzth.bzid,
                     '1_20001223_44ce735e-d05c-11ec-ba29-00163e356637')
    self.assertEqual(zk_pzth.base_name,
                     '20001223_zm_test_realtime_training_1328_v4_r982567_0')
    self.assertEqual(zk_pzth.idc, None)
    self.assertEqual(zk_pzth.cluster, None)
    self.assertEqual(zk_pzth.server_type, 'ps')
    self.assertEqual(zk_pzth.index, '1')
    self.assertEqual(zk_pzth.replica_id, None)
    self.assertEqual(zk_pzth.location, None)
    self.assertEqual(zk_pzth.task, 'ps:1')
    self.assertTrue(zk_pzth.ship_in(None, None))


if __name__ == "__main__":
  unittest.main()
