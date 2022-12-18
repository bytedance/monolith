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
from concurrent import futures
from dataclasses import dataclass
import grpc
from google.protobuf.any_pb2 import Any
from queue import Queue
import random
import threading
import time
from typing import List, Tuple, Union, Optional

from google.protobuf import text_format

from tensorflow_serving.apis.get_model_status_pb2 import GetModelStatusRequest, \
  GetModelStatusResponse
from tensorflow_serving.apis.get_model_metadata_pb2 import GetModelMetadataRequest, \
  GetModelMetadataResponse
from tensorflow_serving.apis.model_management_pb2 import ReloadConfigRequest, \
  ReloadConfigResponse
from tensorflow_serving.apis.model_service_pb2_grpc import ModelServiceServicer
from tensorflow_serving.apis.model_service_pb2_grpc import add_ModelServiceServicer_to_server
from tensorflow_serving.apis.prediction_service_pb2_grpc import PredictionServiceServicer
from tensorflow_serving.apis.prediction_service_pb2_grpc import add_PredictionServiceServicer_to_server
from tensorflow_serving.config import model_server_config_pb2



from monolith.agent_service.utils import gen_model_version_status, gen_status_proto, gen_model_config, \
  ModelState, ErrorCode


@dataclass
class ModelConf:
  model_name: str = None
  base_path: str = None
  version_policy: str = 'latest'
  version_data: Union[int, List[int]] = None
  model_platform: str = 'tensorflow'
  signature_name: Tuple = ('update', 'predict')


@dataclass
class ModelVersion:
  version: int = 0
  version_label: str = None
  state: int = ModelState.UNKNOWN


class ModelMeta:

  def __init__(self, conf: ModelConf, versions: List[ModelVersion] = None):
    self.conf = conf
    self.versions = versions or [ModelVersion()]
    self._unloading = False

  def is_unloading(self):
    return self._unloading

  def set_unloading(self):
    self._unloading = True


@dataclass
class Event:
  model_name: str = None
  version: int = 0
  state: int = ModelState.UNKNOWN


class ModelMgr:

  def __init__(self, model_config_list=None):
    self._models = {}
    self._lock = threading.Lock()
    self._queue = Queue()
    self._has_stopped = False
    self._thread: Optional[threading.Thread] = None

    if model_config_list is not None:
      self.load(model_config_list)

  def load(self, model_config_list):
    for config in model_config_list:
      if config.model_version_policy.HasField('latest'):
        version_policy = 'latest'
        version_data = config.model_version_policy.latest.num_versions
        versions = [ModelVersion(i + 1) for i in range(version_data)]
      elif config.model_version_policy.HasField('all'):
        version_policy = 'latest'
        version_data = None
        versions = [ModelVersion(1)]
      else:
        version_policy = 'specific'
        version_data = config.model_version_policy.specific.versions
        version_data.sort()
        versions = [ModelVersion(i) for i in version_data]

      model_conf = ModelConf(config.name,
                             config.base_path,
                             version_policy=version_policy,
                             version_data=version_data)
      self._models[config.name] = ModelMeta(model_conf, versions)

      logging.info('start load a new model {}'.format(config.name))
      for v in versions:
        self._queue.put(
            Event(model_name=config.name,
                  version=v.version,
                  state=ModelState.START))

  def remove(self, model_name_list):
    for model_name in model_name_list:
      model: ModelMeta = self._models[model_name]
      model.set_unloading()
      logging.info('start remove the model {}'.format(model_name))
      for version in model.versions:
        self._queue.put(Event(model_name, version.version,
                              ModelState.UNLOADING))

  def get_status(self, model_spec):
    model_version_status = []
    with self._lock:
      if model_spec.name in self._models:
        model_meta: ModelMeta = self._models[model_spec.name]
        if model_spec.WhichOneof('version_choice') is None:
          for version in model_meta.versions:
            mvs = gen_model_version_status(version.version, version.state)
            model_version_status.append(mvs)
        else:
          if model_spec.HasField('version'):
            value = model_spec.version.value
            for version in model_meta.versions:
              if version.version == value:
                mvs = gen_model_version_status(version.version, version.state)
                model_version_status.append(mvs)
                break
          else:
            value = model_spec.version_label
            for version in model_meta.versions:
              if version.version_label == value:
                mvs = gen_model_version_status(version.version, version.state)
                model_version_status.append(mvs)
                break

      if len(model_version_status) == 0:
        mvs = gen_model_version_status(
            -1,
            error_code=ErrorCode.NOT_FOUND,
            error_message=f'{model_spec.name} is not found')
        model_version_status.append(mvs)

    return model_version_status

  def get_metadata(self, model_spec, metadata_field):
    metadata = {}
    if metadata_field is not None and len(metadata_field) > 0:
      with self._lock:
        model_meta: ModelMeta = self._models[model_spec.name]
        conf = model_meta.conf
        for field in metadata_field:
          if hasattr(conf, field):
            metadata[field] = getattr(conf, field)

        if model_spec.HasField('version'):
          version = model_spec.version.value
          for v in model_meta.versions:
            if v.version == version:
              for field in metadata_field:
                if hasattr(v, field):
                  metadata[field] = getattr(v, field)
              break

    return metadata

  def get_alive_model_names(self):
    with self._lock:
      return {k for k, v in self._models.items() if not v.is_unloading()}

  def start(self):
    self._thread = threading.Thread(target=self._poll,)
    self._thread.start()

  def stop(self):
    self._has_stopped = True
    if self._thread is not None:
      self._thread.join()
      self._thread = None

  def _poll(self):
    start_time = time.time()
    while not self._has_stopped:
      if not self._queue.empty():
        event = self._queue.get()
        self._event_handler(event)

      end_time = time.time()
      if end_time - start_time > 30:
        start_time = end_time
        model_names = list(self._models.keys())
        if len(model_names) == 0:
          continue
        model_name = random.choice(list(self._models.keys()))
        model_conf: ModelConf = self._models[model_name].conf
        if model_conf.version_policy != 'specific':
          versions = self._models[model_name].versions
          version = versions[-1].version + 1
          versions.append(ModelVersion(version))
          logging.info(
              'start load a new version of model {}'.format(model_name))
          self._queue.put(Event(model_name, version, ModelState.START))

      # time.sleep(random.uniform(0, 0.1))

  def _event_handler(self, event: Event):
    with self._lock:
      model: ModelMeta = self._models.get(event.model_name, None)
      if model is None:
        logging.error(f'{event.model_name} has removed!')
        return

      log_flag = False
      if event.state == ModelState.START:
        for version in model.versions:
          if version.version == event.version:
            if version.state == ModelState.UNKNOWN:
              version.state = event.state
              self._queue.put(
                  Event(event.model_name, event.version, ModelState.LOADING))
              log_flag = True
            break
      elif event.state == ModelState.LOADING:
        for version in model.versions:
          if version.version == event.version:
            if version.state == ModelState.START:
              version.state = event.state
              self._queue.put(
                  Event(event.model_name, event.version, ModelState.AVAILABLE))
              log_flag = True
            break
      elif event.state == ModelState.AVAILABLE:
        for version in model.versions:
          if version.version == event.version:
            if version.state == ModelState.LOADING:
              version.state = event.state
              log_flag = True

              if model.conf.version_policy == 'latest':
                if len(model.versions) > model.conf.version_data:
                  self._queue.put(
                      Event(event.model_name, model.versions[0].version,
                            ModelState.UNLOADING))
            break
      elif event.state == ModelState.UNLOADING:
        for version in model.versions:
          if version.version == event.version:
            # in case unloading in unloading
            if version.state not in {ModelState.UNLOADING, ModelState.END}:
              version.state = event.state
              self._queue.put(
                  Event(event.model_name, event.version, ModelState.END))
              log_flag = True
            break
      elif event.state == ModelState.END:
        index = -1
        for i, version in enumerate(model.versions):
          if version.version == event.version:
            if version.state == ModelState.UNLOADING:
              version.state = event.state
              logging.info(
                  f'{event.model_name}-{event.version}: state is {ModelState.Name(event.state)}'
              )
            index = i
            break

        if index >= 0:
          logging.info(
              f'{event.model_name}-{model.versions[index].version} is removed!')
          del model.versions[index]

        if len(model.versions) == 0:
          logging.info(f'{event.model_name} is removed!')
          del self._models[event.model_name]
      else:
        logging.error('unknown event')

      if log_flag:
        logging.info(
            f'{event.model_name}-{event.version}: state is {event.state}')


class ModelServiceImpl(ModelServiceServicer):

  def __init__(self, model_mgr: ModelMgr):
    self._model_mgr = model_mgr

  def GetModelStatus(self, request: GetModelStatusRequest, context):
    response = GetModelStatusResponse()
    model_version_status = self._model_mgr.get_status(request.model_spec)
    response.model_version_status.extend(model_version_status)
    return response

  def HandleReloadConfigRequest(self, request: ReloadConfigRequest, context):
    model_config_list = request.config.model_config_list.config

    old_names = self._model_mgr.get_alive_model_names()
    new_names = {config.name for config in model_config_list}

    to_remove = old_names - new_names
    self._model_mgr.remove(to_remove)

    to_load = new_names - old_names
    self._model_mgr.load(
        [config for config in model_config_list if config.name in to_load])

    response = ReloadConfigResponse()
    response.status.CopyFrom(gen_status_proto())
    return response


class PredictionServiceImpl(PredictionServiceServicer):

  def __init__(self, model_mgr: ModelMgr):
    self._model_mgr = model_mgr

  def Predict(self, request, context):
    pass

  def GetModelMetadata(self, request: GetModelMetadataRequest, context):
    model_spec = request.model_spec
    metadata_field = set(request.metadata_field)

    response = GetModelMetadataResponse()
    response.model_spec.CopyFrom(model_spec)
    metadata = self._model_mgr.get_metadata(model_spec, metadata_field)
    for k, v in metadata.items():
      value = bytes(repr(v), encoding='utf-8')
      response.metadata[k].CopyFrom(Any(value=value))

    return response


class FakeTFServing:

  def __init__(self,
               model_name: str = None,
               base_path: str = None,
               num_versions: int = 1,
               port: int = 8500,
               max_workers: int = 10,
               model_config_file=None):
    if model_config_file is None:
      self._model_mgr = ModelMgr(
          [gen_model_config(model_name, base_path, version_data=num_versions)])
    elif isinstance(model_config_file, str):
      msc = model_server_config_pb2.ModelServerConfig()
      with open(model_config_file, 'r') as fp:
        text = ''.join(fp.readlines())
        text_format.Parse(text, msc)
      self._model_mgr = ModelMgr(msc.model_config_list.config)
    else:
      assert isinstance(model_config_file,
                        model_server_config_pb2.ModelServerConfig)
      self._model_mgr = ModelMgr(model_config_file.model_config_list.config)

    self._server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers))
    add_ModelServiceServicer_to_server(ModelServiceImpl(self._model_mgr),
                                       self._server)
    add_PredictionServiceServicer_to_server(
        PredictionServiceImpl(self._model_mgr), self._server)
    self._server.add_insecure_port(f'[::]:{port}')

  def start(self):
    self._model_mgr.start()
    self._server.start()
    self._server.wait_for_termination()

  def stop(self, grace=None):
    self._server.stop(grace=grace)
    self._model_mgr.stop()


if __name__ == '__main__':
  tfs = FakeTFServing('model_test', '/tmp/model/monolith', num_versions=1)
  tfs.start()
