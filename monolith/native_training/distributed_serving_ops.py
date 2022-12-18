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

from contextlib import nullcontext
from typing import List

from absl import logging
import tensorflow as tf
from monolith.agent_service.agent_service_pb2 import ServerType
from monolith.agent_service.backends import SyncBackend
from monolith.utils import get_libops_path
from monolith.native_training.runtime.ops import gen_monolith_ops
from monolith.native_training import utils
from monolith.native_training.model_export.export_context import is_exporting_standalone
from monolith.native_training.runtime.parameter_sync import \
  parameter_sync_pb2

gen_distributed_serving_ops = gen_monolith_ops


def remote_predict(input_tensor_alias,
                   input_tensors,
                   output_tensor_alias,
                   model_name,
                   task,
                   old_model_name,
                   model_version=-1,
                   fail_op_on_rpc_error=True,
                   max_rpc_deadline_millis=30,
                   output_types=None,
                   name=None,
                   signature_name='serving_default'):
  """Runs a predict in remote process through rpc.
  Args:
    input_tensor_alias: input tensor alias for Predict
    input_tensors: input tensors for Predict
    output_tensor_alias: output tensor alias for Predict
    task: Parameter Server index
    model_name: model_name that the Predict is running on
    model_version: the model version for the Predict call. If unset, the highest
      version available for serving will be targeted.
    max_rpc_deadline_millis: rpc deadline in millis
    output_types: output types for Predict
    name: name for the op in the graph
    signature_name: the signature def for remote graph inference
  Returns:
    output_tensors as a result of the Predict.
  Raises ValueError if model_name value is missing.
  """
  if model_name is None:
    raise ValueError('model_name must be specified.')
  return (gen_distributed_serving_ops.tf_serving_remote_predict(
      input_tensor_alias,
      input_tensors,
      output_tensor_alias,
      model_name=model_name,
      old_model_name=old_model_name,
      task=task,
      model_version=model_version,
      fail_op_on_rpc_error=fail_op_on_rpc_error,
      max_rpc_deadline_millis=max_rpc_deadline_millis,
      signature_name=signature_name,
      output_types=output_types,
      name=name))[2]


def create_parameter_sync_clients(ps_num: int,) -> List[tf.Tensor]:
  logging.info("Create parameter sync clients.")
  if ps_num == 0:
    return [parameter_sync_client_from_config()]

  sync_clients = list()
  for i in range(ps_num):
    ps_device_name = utils.ps_device(i)
    with nullcontext() if is_exporting_standalone() else tf.device(
        ps_device_name):
      sync_clients.append(parameter_sync_client_from_config(name_suffix=str(i)))
  return sync_clients


def parameter_sync_client_from_config(
    config: parameter_sync_pb2.ClientConfig = None,
    name_suffix: str = "") -> tf.Tensor:
  return gen_distributed_serving_ops.MonolithParameterSyncClient(
      config=config.SerializeToString() if config else '',
      shared_name="MonolithSyncClient_" + name_suffix)


def refresh_sync_config(sync_backend: SyncBackend, ps_index: int) -> bytes:
  saved_model, online_ps_replicas = sync_backend.get_sync_targets(
      f"ps_{ps_index}")
  config = parameter_sync_pb2.ClientConfig()
  config.targets.extend(online_ps_replicas)
  config.model_name = saved_model
  config.signature_name = "hashtable_assign"
  config.timeout_in_ms = 3000
  return config.SerializeToString()


def create_dummy_sync_client() -> tf.Tensor:
  return gen_distributed_serving_ops.MonolithDummySyncClient(
      shared_name="MonolithDummySyncClient")


def create_dummy_sync_server(address: str) -> tf.Tensor:
  return gen_distributed_serving_ops.MonolithDummySyncServer(address=address)


class ParameterSyncClient(object):

  def __init__(self, client: tf.Tensor):
    self._client = client

  def create_sync_op(self, config_str: tf.Tensor):
    return gen_distributed_serving_ops.monolith_parameter_sync(
        self._client, config_str)

  def as_op(self):
    return tf.group(self._client)

  @property
  def handle(self):
    return self._client


class DummySyncServer(object):

  def __init__(self, address: str):
    self._server = create_dummy_sync_server(address)

  def shutdown(self):
    return gen_distributed_serving_ops.monolith_dummy_sync_server_shutdown(
        self._server)

  def get_port(self):
    return gen_distributed_serving_ops.monolith_dummy_sync_server_get_port(
        self._server)

  def as_op(self):
    return tf.group(self._server)

  @property
  def handle(self):
    return self._server
