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

from absl import logging
import os
import grpc
from threading import RLock
from functools import singledispatchmethod
from typing import Dict, Iterable, Union, List

from tensorflow.core.protobuf.error_codes_pb2 import Code
from tensorflow_serving.util.status_pb2 import StatusProto
from tensorflow_serving.apis.get_model_status_pb2 import ModelVersionStatus, GetModelStatusRequest
from tensorflow_serving.apis.model_management_pb2 import ReloadConfigRequest
from tensorflow_serving.apis.prediction_service_pb2_grpc import PredictionServiceStub
from tensorflow_serving.apis.model_service_pb2_grpc import ModelServiceStub
from tensorflow_serving.config.model_server_config_pb2 import ModelServerConfig

from monolith.agent_service.data_def import PublishMeta, SubModelName, TFSModelName, \
  VersionPath, PublishType as PType
from monolith.agent_service.utils import AgentConfig, gen_model_spec, gen_model_config, \
  DeployType, TFSServerType, get_local_ip, DEFAULT_MODEL_CONFIG

State = ModelVersionStatus.State


class TFSMonitor(object):

  def __init__(self, config: AgentConfig):
    self._conf: AgentConfig = config
    self._host = None
    self._lock = RLock()
    self.stubs = {
        TFSServerType.ENTRY: {},
        TFSServerType.PS: {},
        TFSServerType.DENSE: {}
    }

  @property
  def host(self):
    if self._host is None or self._host in {'', 'localhost', '127.0.0.1'}:
      self._host = get_local_ip()

    return self._host

  def get_addr(self, sub_model_name: SubModelName) -> str:
    if self._conf.deploy_type == DeployType.MIXED:
      if self.is_entry(sub_model_name):
        return f"{self.host}:{self._conf.tfs_entry_port}"
      elif self.is_ps(sub_model_name):
        return f"{self.host}:{self._conf.tfs_ps_port}"
      else:
        return f"{self.host}:{self._conf.tfs_dense_port}"
    elif self._conf.deploy_type == DeployType.ENTRY:
      assert self.is_entry(sub_model_name)
      return f"{self.host}:{self._conf.tfs_entry_port}"
    elif self._conf.deploy_type == DeployType.PS:
      assert self.is_ps(sub_model_name)
      return f"{self.host}:{self._conf.tfs_ps_port}"
    elif self._conf.deploy_type == DeployType.DENSE:
      assert self.is_dense(sub_model_name)
      return f"{self.host}:{self._conf.tfs_dense_port}"
    else:
      raise RuntimeError(f'deploy_type {self._conf.deploy_type} is error')

  def get_service_type(self, sub_model_name: SubModelName):
    if self._conf.deploy_type == DeployType.ENTRY:
      return TFSServerType.ENTRY if sub_model_name.startswith(
          TFSServerType.ENTRY) else None
    elif self._conf.deploy_type == DeployType.PS:
      return TFSServerType.PS if sub_model_name.startswith(
          TFSServerType.PS) else None
    elif self._conf.deploy_type == DeployType.DENSE:
      return TFSServerType.DENSE if sub_model_name.startswith(
          TFSServerType.DENSE) else None
    else:
      assert self._conf.deploy_type == DeployType.MIXED
      if sub_model_name.startswith(TFSServerType.ENTRY):
        return TFSServerType.ENTRY
      elif sub_model_name.startswith(TFSServerType.PS):
        return TFSServerType.PS
      elif sub_model_name.startswith(TFSServerType.DENSE):
        return TFSServerType.DENSE
      else:
        return None

  def is_entry(self, sub_model_name: str):
    return sub_model_name.startswith('entry')

  def is_ps(self, sub_model_name: str):
    return sub_model_name.startswith('ps')

  def is_dense(self, sub_model_name: str):
    return sub_model_name.startswith('dense')

  def connect(self):
    if self._conf.deploy_type in {DeployType.MIXED, DeployType.ENTRY}:
      entry_channel = grpc.insecure_channel(
          f'{self.host}:{self._conf.tfs_entry_port}')
      self.stubs[TFSServerType.ENTRY]['channel'] = entry_channel
      self.stubs[TFSServerType.ENTRY]['model_service'] = ModelServiceStub(
          entry_channel)
      self.stubs[TFSServerType.ENTRY][
          'prediction_service'] = PredictionServiceStub(entry_channel)

    if self._conf.deploy_type in {DeployType.MIXED, DeployType.PS}:
      ps_channel = grpc.insecure_channel(
          f'{self.host}:{self._conf.tfs_ps_port}')
      self.stubs[TFSServerType.PS]['channel'] = ps_channel
      self.stubs[TFSServerType.PS]['model_service'] = ModelServiceStub(
          ps_channel)
      self.stubs[TFSServerType.
                 PS]['prediction_service'] = PredictionServiceStub(ps_channel)

    if self._conf.dense_alone and self._conf.deploy_type in {
        DeployType.MIXED, DeployType.DENSE
    }:
      dense_channel = grpc.insecure_channel(
          f'{self.host}:{self._conf.tfs_dense_port}')
      self.stubs[TFSServerType.DENSE]['channel'] = dense_channel
      self.stubs[TFSServerType.DENSE]['model_service'] = ModelServiceStub(
          dense_channel)
      self.stubs[TFSServerType.DENSE][
          'prediction_service'] = PredictionServiceStub(dense_channel)

  def start(self):
    self.stubs = {
        TFSServerType.ENTRY: {},
        TFSServerType.PS: {},
        TFSServerType.DENSE: {}
    }
    self.connect()

  def stop(self):
    if len(self.stubs) > 0:
      for stub in self.stubs.values():
        if 'channel' in stub:
          try:
            stub['channel'].close()
            stub['model_service'] = None
            stub['prediction_service'] = None
          except:
            logging.error('stop channel fail!')
      self.stubs.clear()

  @singledispatchmethod
  def get_model_status(self, arg):
    raise NotImplementedError("get_model_status is not implemented!")

  @get_model_status.register
  def _(self, pm: PublishMeta, fix_dense_version: bool = False):
    # return 'Dict[TFSModelName, (VersionPath, ModelVersionStatus)]'
    with self._lock:
      model_status: Dict[str, State] = {}
      for sub_model_name, smvpath in pm.sub_models.items():
        service_type = self.get_service_type(sub_model_name)
        if service_type is None:
          continue

        tfs_model_name = f'{pm.model_name}:{sub_model_name}'
        request = GetModelStatusRequest()
        # TODO(ltli): 这步修改不确定
        is_dense_node = (
            (not self._conf.dense_alone and self.is_entry(sub_model_name)) or
            (self._conf.dense_alone and self.is_dense(sub_model_name)))
        if not fix_dense_version and is_dense_node:
          request.model_spec.CopyFrom(gen_model_spec(tfs_model_name))
        else:
          version = int(os.path.basename(smvpath))
          request.model_spec.CopyFrom(gen_model_spec(tfs_model_name, version))

        stub: ModelServiceStub = self.stubs[service_type]['model_service']

        try:
          model_version_status = stub.GetModelStatus(
              request).model_version_status
          if model_version_status is None or len(model_version_status) == 0:
            status = ModelVersionStatus(
                state=State.UNKNOWN,
                status=StatusProto(error_code=Code.NOT_FOUND,
                                   error_message=f'{tfs_model_name} not found'))
          else:
            # if there are more than one version, select the latest one
            model_version_status = sorted(model_version_status,
                                          key=lambda mvs: mvs.version)
            status = model_version_status[-1]
        except grpc._channel._InactiveRpcError as e:
          logging.info(repr(e))
          status = ModelVersionStatus(state=State.UNKNOWN,
                                      status=StatusProto(
                                          error_code=e.code().value[0],
                                          error_message=e.details()))

        model_status[tfs_model_name] = (smvpath, status)

      return model_status

  @get_model_status.register
  def _(self,
        name: str,
        version: Union[int, str] = None,
        signature_name: str = None) -> List[ModelVersionStatus]:
    """Get model version status
    
      :param name: The model name
      :param version: The version of model. If  not specify version,
                      information about all versions of the model will be returned.
                      
      :return a list of ModelVersionStatus, which has three attribute:
        - version: int, Model version.
        - state: State, Model state, A Enum of UNKNOWN, START, LOADING, AVAILABLE, UNLOADING, END.
        - status: StatusProto, Model status.
    """

    with self._lock:
      service_type = self.get_service_type(SubModelName(name))
      if service_type is None:
        return []
      else:
        request = GetModelStatusRequest()
        request.model_spec.CopyFrom(
            gen_model_spec(name, version, signature_name))
        stub = self.stubs[service_type]['model_service']
        return stub.GetModelStatus(request).model_version_status

  def gen_model_config(
      self,
      pms: Iterable[PublishMeta],
      fix_dense_version: bool = False) -> Dict[str, ModelServerConfig]:
    model_configs = {
        TFSServerType.ENTRY: ModelServerConfig(),
        TFSServerType.PS: ModelServerConfig(),
        TFSServerType.DENSE: ModelServerConfig()
    }

    for pm in pms:
      if pm.ptype == PType.UNLOAD:
        continue

      for sub_model_name, smv_path in pm.sub_models.items():
        tfs_model_name = f'{pm.model_name}:{sub_model_name}'
        service_type = self.get_service_type(sub_model_name)
        if service_type is None:
          continue

        base_path = os.path.dirname(smv_path)
        # TODO(ltli): 这步修改不确定
        is_dense_node = (
            (not self._conf.dense_alone and self.is_entry(sub_model_name)) or
            (self._conf.dense_alone and self.is_dense(sub_model_name)))
        if is_dense_node:
          version_policy = 'specific' if fix_dense_version else 'latest'
          version_data = int(
              os.path.basename(smv_path)) if fix_dense_version else 1
        else:
          version_policy = 'specific'
          version_data = int(os.path.basename(smv_path))

        model_config = gen_model_config(tfs_model_name, base_path,
                                        version_policy, version_data)
        model_configs[service_type].model_config_list.config.append(
            model_config)

    return model_configs

  def handle_reload_config_request(
      self, service_type: str, model_configs: ModelServerConfig) -> StatusProto:
    with self._lock:
      request = ReloadConfigRequest()
      # keep default model in memory, incase no model in tfs
      model_config_list = model_configs.model_config_list.config
      if not any(mc.name == 'default' for mc in model_config_list):
        model_config = model_config_list.add()
        model_config.CopyFrom(DEFAULT_MODEL_CONFIG)

      request.config.CopyFrom(model_configs)
      if service_type == TFSServerType.ENTRY:
        port = self._conf.tfs_entry_port
      elif service_type == TFSServerType.PS:
        port = self._conf.tfs_ps_port
      else:
        port = self._conf.tfs_dense_port
      logging.info(f'{service_type} load ({port}): \n{request}')
      try:
        response = self.stubs[service_type][
            'model_service'].HandleReloadConfigRequest(request)
        logging.info(f'{service_type} load done!')
      except Exception as e:
        logging.info(repr(e))
        raise e

      return response.status
