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

from monolith.native_training.hooks.server import service_pb2
from monolith.native_training.hooks.server import service_pb2_grpc
from monolith.native_training.hooks.server import constants


def get_stub_from_model_dir(model_dir: str):
  with tf.io.gfile.GFile(
      os.path.join(model_dir, constants.SERVER_ADDR_FILENAME), "r") as f:
    addr = f.read()
  channel = grpc.insecure_channel(addr)
  return service_pb2_grpc.ControllerStub(channel)
