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


from collections import defaultdict
from functools import partial
from absl import app, flags, logging
import json
import os
import re
import time
import signal
import threading
import tempfile
from threading import RLock
from typing import Callable, Dict, List, Tuple
from subprocess import Popen, STDOUT

from google.protobuf import text_format, json_format

from monolith.agent_service.utils import AgentConfig, DeployType, gen_model_config, get_local_ip, \
  normalize_regex
from monolith.agent_service.tfs_wrapper import TFSWrapper, State
from monolith.agent_service.agent_service import AgentService, AgentDataProvider
from monolith.agent_service.agent_base import AgentBase
from monolith.agent_service.backends import ZKBackend, Container, SavedModel, \
  ContainerServiceInfo, SavedModelDeployConfig

from tensorflow_serving.config import model_server_config_pb2


def gen_empty_model_config_file():
  tmp_file = tempfile.mktemp()
  with open(tmp_file, "w") as f:
    f.write("model_config_list {}")
  return tmp_file


class AgentV3(AgentBase):

  _lock = RLock()

  def __init__(self, config: AgentConfig, conf_path: str, tfs_log: str):
    super(AgentV3, self).__init__(config)
    assert config.deploy_type == DeployType.UNIFIED, "agent v3 only supports unifed deploy_type"
    assert config.agent_version == 3, f"agent version {config.agent_version} unexpected"
    self._conf_path = conf_path
    self._tfs_log = tfs_log

    self._exit_event = threading.Event()
    signal.signal(signal.SIGTERM, self.signal_handler)
    signal.signal(signal.SIGINT, self.signal_handler)

    self._model_config_path = gen_empty_model_config_file()

    self._tfs_wrapper = TFSWrapper(config.tfs_port_archon, config.tfs_port_grpc,
                                   config.tfs_port_http,
                                   self._model_config_path, config,
                                   self._tfs_log)

    self._layout_filters = []
    if config.layout_filters:
      shard_id = max(config.shard_id, 0)
      shard_num = max(config.num_shard, 1)
      for raw_filter in config.layout_filters:
        for k, v in [('${shard_id}', shard_id), ('${shard_num}', shard_num)]:
          raw_filter = raw_filter.replace(k, str(v))
        match, cond = raw_filter.split(";", 1)
        self._layout_filters.append((normalize_regex(match), cond))

    self._container = Container(self.config.container_cluster,
                                self.config.container_id)

    local_ip = get_local_ip()
    self._service_info = ContainerServiceInfo(
        grpc=f"{local_ip}:{self.config.tfs_port_grpc}",
        http=f"{local_ip}:{self.config.tfs_port_http}",
        archon=f"{local_ip}:{self.config.tfs_port_archon}",
        agent=f"{local_ip}:{self.config.agent_port}",
        idc=self.config.idc,
        debug_info=json.dumps({
            'layout_path':
                config.layout_path,
            'layout_filters': [
                f"{match};{cond}" for match, cond in self._layout_filters
            ]
        }))
    self._backend = ZKBackend(bzid=config.bzid, zk_servers=config.zk_servers)

    self._threads = []

    self._agent_service = AgentService(
        AgentDataProvider(addrs_fn=self._gen_addrs_map), conf=config)

  def _gen_addrs_map(self):
    service_map = self._backend.get_service_map()
    addrs_map = {}
    for model_name in service_map:
      for sub_graph, service_infos in service_map[model_name].items():
        addrs_map[f"{model_name}:{sub_graph}"] = [
            service_info.grpc
            if self._tfs_wrapper.is_grpc_remote_op else service_info.archon
            for service_info in service_infos
        ]
    return addrs_map

  def sync_available_saved_models(self):
    saved_model_status = self._tfs_wrapper.list_saved_models_status()
    available_saved_models = set()
    for saved_model_name, status in saved_model_status.items():
      if status.state == State.AVAILABLE:
        model_name, sub_graph = saved_model_name.split(":")[:2]
        available_saved_models.add(SavedModel(model_name, sub_graph))
    self._backend.sync_available_saved_models(self._container,
                                              available_saved_models)

  def layout_update_callback(
      self, saved_models: List[Tuple[SavedModel,
                                     SavedModelDeployConfig]]) -> bool:
    logging.info(f"layout callback with saved_models: {saved_models}")
    model_server_config = model_server_config_pb2.ModelServerConfig()
    model_server_config.model_config_list.SetInParent()
    model_config_list = model_server_config.model_config_list.config

    for saved_model, deploy_config in saved_models:
      accepted = len(self._layout_filters) == 0
      for match, cond in self._layout_filters:
        m = re.match(match, saved_model.sub_graph)
        if m:
          if eval(cond, None, {k: int(v) for k, v in m.groupdict().items()}):
            accepted = True
            logging.info(f"loading {str(saved_model)} with rule {match}:{cond}")
            break
      if not accepted:
        continue
      tfs_model_config = gen_model_config(
          name=str(saved_model),
          base_path=deploy_config.model_base_path,
          version_policy=deploy_config.version_policy)
      model_config_list.add().CopyFrom(tfs_model_config)

    logging.info(
        f"writing model server_config: {text_format.MessageToString(model_server_config)}"
    )
    with open(self._model_config_path, 'w') as f:
      f.write(text_format.MessageToString(model_server_config))
    return True

  def signal_handler(self, signum, frame):
    logging.info(f"catch signal {signum}, frame {frame}")
    self._exit_event.set()

  def start_bg_thread(self, fn, interval=10):

    def target():
      while not self._exit_event.is_set():
        try:
          fn()
        except Exception as e:
          logging.error(f"error in bg thread: {e}")
        time.sleep(interval)

    bg_thread = threading.Thread(target=target)
    bg_thread.start()
    self._threads.append(bg_thread)

  def start(self):
    self._tfs_wrapper.start()
    self._backend.start()
    self._agent_service.start()

    self.start_bg_thread(partial(self._backend.report_service_info,
                                 self._container, self._service_info),
                         interval=60)
    self.start_bg_thread(self.sync_available_saved_models, interval=30)
    self._backend.register_layout_callback(self.config.layout_path,
                                           self.layout_update_callback)

  def stop(self):
    self._exit_event.set()
    try:
      for t in self._threads:
        t.join()
      self._agent_service.stop()
      self._backend.stop()
      self._tfs_wrapper.stop()
    except Exception as e:
      logging.warning(e)

  def wait_for_termination(self):
    while not self._exit_event.is_set():
      time.sleep(1)
      ret_code = self._tfs_wrapper.poll()
      if ret_code is not None:
        self._exit_event.set()

    self.stop()
    time.sleep(1)
    os.kill(os.getpid(), signal.SIGKILL)
