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

import grpc
import tensorflow as tf

from monolith.native_training import barrier_ops
from monolith.native_training import save_utils
from monolith.native_training.hooks.server import client_lib
from monolith.native_training.hooks.server import server_lib
from monolith.native_training.hooks.server import service_pb2
from monolith.native_training.hooks.server import service_pb2_grpc


class ServerTest(tf.test.TestCase):

  def test_basic(self):
    model_dir = os.path.join(os.environ["TEST_TMPDIR"], "basic")
    barrier = barrier_ops.BarrierOp(1)
    saver_hook = save_utils.NoFirstSaveCheckpointSaverHook(model_dir,
                                                           save_secs=10000)
    server_hook = server_lib.ServerHook(model_dir, barrier, saver_hook)
    tf.compat.v1.train.create_global_step()
    with tf.compat.v1.train.SingularMonitoredSession(
        hooks=[server_hook, saver_hook]) as sess:
      stub = client_lib.get_stub_from_model_dir(model_dir)
      stub.StopTraining(service_pb2.StopTrainingRequest())
      with self.assertRaises(grpc.RpcError):
        stub.StopTraining(service_pb2.StopTrainingRequest())
      resp = stub.GetBlockStatus(service_pb2.GetBlockStatusRequest())
      self.assertAllEqual(resp.blocked_indices, [0])
      stub.ResumeTraining(service_pb2.ResumeTrainingRequest())
      resp = stub.GetBlockStatus(service_pb2.GetBlockStatusRequest())
      self.assertAllEqual(resp.blocked_indices, [])
      stub.SaveCheckpoint(service_pb2.SaveCheckpointRequest())
      stub.GetTrainingStatus(service_pb2.GetTrainingStatusRequest())


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
