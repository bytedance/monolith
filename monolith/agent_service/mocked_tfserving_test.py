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

import grpc
import socket
import time
import threading
import unittest

from tensorflow_serving.apis.get_model_metadata_pb2 import GetModelMetadataRequest, \
  GetModelMetadataResponse
from tensorflow_serving.apis.get_model_status_pb2 import GetModelStatusRequest, \
  GetModelStatusResponse
from tensorflow_serving.apis.model_management_pb2 import ReloadConfigRequest, \
  ReloadConfigResponse
from tensorflow_serving.apis.model_service_pb2_grpc import ModelServiceStub
from tensorflow_serving.apis.prediction_service_pb2_grpc import PredictionServiceStub

from monolith.agent_service import utils
from monolith.agent_service.mocked_tfserving import FakeTFServing

MODEL_NAME = 'test_model_test'
BASE_PATH = '/tmp/test_model/monolith'
PORT = utils.find_free_port()

Address = f'{socket.gethostbyname(socket.gethostname())}:{PORT}'


class MockedTFSTest(unittest.TestCase):
  tfs: FakeTFServing = None

  @classmethod
  def setUpClass(cls) -> None:
    cls.tfs = FakeTFServing(MODEL_NAME, BASE_PATH, num_versions=2, port=PORT)
    # cls.tfs.start()
    thread = threading.Thread(target=lambda: cls.tfs.start())
    thread.start()
    time.sleep(5)

  @classmethod
  def tearDownClass(cls) -> None:
    cls.tfs.stop()

  def test_get_model_metadata(self):
    request = GetModelMetadataRequest()
    request.model_spec.CopyFrom(
        utils.gen_model_spec(MODEL_NAME, 2, signature_name='predict'))
    request.metadata_field.extend(
        ['base_path', 'num_versions', 'signature_name'])

    stub = PredictionServiceStub(grpc.insecure_channel(Address))
    self.assertTrue(
        isinstance(stub.GetModelMetadata(request), GetModelMetadataResponse))

  def test_get_model_status(self):
    stub = ModelServiceStub(grpc.insecure_channel(Address))
    request = GetModelStatusRequest()
    request.model_spec.CopyFrom(
        utils.gen_model_spec(MODEL_NAME, 1, signature_name='predict'))
    self.assertTrue(
        isinstance(stub.GetModelStatus(request), GetModelStatusResponse))

  def test_handle_reload_config_request(self):
    stub = ModelServiceStub(grpc.insecure_channel(Address))
    request = ReloadConfigRequest()
    model_config_list = request.config.model_config_list.config
    model_config_list.extend([
        utils.gen_model_config(name='test_model',
                               base_path='/tmp/test_model/ctr/saved_model',
                               version_data=2),
        utils.gen_model_config(name='test_model',
                               base_path='/tmp/test_model/cvr/saved_model',
                               version_data=1),
    ])
    self.assertTrue(
        isinstance(stub.HandleReloadConfigRequest(request),
                   ReloadConfigResponse))


if __name__ == "__main__":
  unittest.main()
