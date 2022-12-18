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

import abc
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from threading import RLock, Event

from absl import logging
from dataclasses_json import dataclass_json
from typing import Callable, Dict, List, Tuple, Any, Set

from kazoo.client import KazooState
from kazoo.recipe.watchers import ChildrenWatch
from kazoo.exceptions import NodeExistsError, ZookeeperError, \
    NoNodeError, NotEmptyError, ConnectionClosedError
from monolith.native_training.zk_utils import MonolithKazooClient


@dataclass(frozen=True)
class SavedModel:
  model_name: str = None
  sub_graph: str = None

  def __repr__(self):
    return str(self)

  def __str__(self):
    return f"{self.model_name}:{self.sub_graph}"


@dataclass_json
@dataclass(frozen=True)
class SavedModelDeployConfig:
  model_base_path: str = None
  version_policy: str = None

  def serialize(self) -> bytes:
    return bytes(self.to_json(), encoding='utf-8')

  @classmethod
  def deserialize(cls, serialized: bytes) -> 'SavedModelDeployConfig':
    return cls.from_json(str(serialized, encoding='utf-8'))


@dataclass(frozen=True)
class Container:
  ctx_cluster: str = None
  ctx_id: str = None

  def __repr__(self):
    return str(self)

  def __str__(self):
    return f"{self.ctx_cluster}:{self.ctx_id}"


@dataclass_json
@dataclass(frozen=True)
class ContainerServiceInfo:
  grpc: str = None  # grpc ip:port
  http: str = None  # http ip:port
  archon: str = None  # archon ip:port
  agent: str = None  # agent ip:port
  idc: str = None  # dc name
  debug_info: str = None

  def serialize(self) -> bytes:
    return bytes(self.to_json(), encoding='utf-8')

  @classmethod
  def deserialize(cls, serialized: bytes) -> 'ContainerServiceInfo':
    return cls.from_json(str(serialized, encoding='utf-8'))


class AgentBackend(abc.ABC):

  def __init__(self):
    pass

  @abc.abstractmethod
  def register_layout_callback(
      self, layout_path: str,
      callback: Callable[[List[Tuple[SavedModel, SavedModelDeployConfig]]],
                         None]
  ) -> None:
    """
    Invoke {callback} on layout updates(adding or removing saved_models)
    """
    pass

  @abc.abstractmethod
  def sync_available_saved_models(self, saved_models: List[SavedModel]) -> None:
    """
    Report available saved models serving in localhost
    """
    pass

  @abc.abstractmethod
  def report_service_info(self, container: Container,
                          service_info: ContainerServiceInfo) -> None:
    pass

  @abc.abstractmethod
  def get_service_map(self) -> Dict[str, Dict[str, List[ContainerServiceInfo]]]:
    """
    Get service info map
    {
      "model_name": {
        "sub_graph: [
          {
            "idc": "LQ",
            "archon": "10.xx.xx.1:9876",
            "grpc": "10.xx.xx.1:8765",
            "http": "10.xx.xx.1:6789",
            "agent": "10.xx.xx.1:6787"
          }
        ]
      }
    }
    """
    pass

  @abc.abstractmethod
  def report_service_info(self, container: Container,
                          service_info: ContainerServiceInfo) -> None:
    pass

  @abc.abstractmethod
  def get_service_info(self, container) -> ContainerServiceInfo:
    pass

  @abc.abstractmethod
  def start(self):
    pass

  @abc.abstractmethod
  def stop(self):
    pass


class CtrlBackend(abc.ABC):

  def __init__(self):
    pass

  @abc.abstractmethod
  def list_saved_models(self, model_name: str) -> List[SavedModel]:
    pass

  @abc.abstractmethod
  def decl_saved_model(self, saved_model: SavedModel,
                       deploy_config: SavedModelDeployConfig):
    pass

  @abc.abstractmethod
  def add_to_layout(self, layout: str, saved_model: SavedModel):
    pass

  @abc.abstractmethod
  def remove_from_layout(self, layout: str, saved_model: SavedModel):
    pass

  @abc.abstractmethod
  def bzid_info(self):
    pass

  @abc.abstractmethod
  def start(self):
    pass

  @abc.abstractmethod
  def stop(self):
    pass


class SyncBackend(abc.ABC):

  def __init__(self):
    pass

  @abc.abstractmethod
  def subscribe_model(self, model_name: str):
    pass

  @abc.abstractmethod
  def get_sync_targets(self, sub_graph: str) -> Tuple[str, List[str]]:
    pass

  @abc.abstractmethod
  def start(self):
    pass

  @abc.abstractmethod
  def stop(self):
    pass


class ZKBackend(AgentBackend, CtrlBackend, SyncBackend):

  _lock = RLock()

  def __init__(self, bzid: str, zk_servers: str):
    super(ZKBackend, self).__init__()
    self._bzid = bzid
    self._zk = MonolithKazooClient(hosts=zk_servers)
    self._available_saved_model = set()
    self._service_info_map = {}
    self._children_watcher_map: Dict[str, ChildrenWatch] = {}
    self._sync_model_name = None
    self._is_lost = Event()

    def zk_listener(state):
      if state == KazooState.LOST:
        logging.error("zk state lost, set lost flag")
        self._is_lost.set()
      else:
        logging.warning(f"zk state changed to {state}, unset lost flag")
        self._is_lost.clear()
      return False

    self._zk.add_listener(zk_listener)

  def sync_available_saved_models(self, container: Container,
                                  saved_models: Set[SavedModel]) -> None:
    """
    Report available saved models serving in tensorflow serving
    """
    with self._lock:
      if self._is_lost.is_set():
        self._available_saved_model.clear()
        logging.warning("zk is lost, try restarting")
        self._zk.restart()
        return
      add_saved_models = saved_models - self._available_saved_model
      remove_saved_models = self._available_saved_model - saved_models
      logging.info(
          f"available saved models updating, add: {add_saved_models}, remove: {remove_saved_models}"
      )
      for saved_model in add_saved_models:
        bind_path = f"/{self._bzid}/binding/{saved_model.model_name}/{saved_model.sub_graph}:{container}"
        self.create_znode(bind_path, b"", ephemeral=True, makepath=True)

      for saved_model in remove_saved_models:
        bind_path = f"/{self._bzid}/binding/{saved_model.model_name}/{saved_model.sub_graph}:{container}"
        self.delete_znode(bind_path)
      logging.info(f"available saved models updated: {saved_models}")
      self._available_saved_model = saved_models

  def register_layout_callback(
      self, layout_path: str,
      callback: Callable[[List[Tuple[SavedModel, SavedModelDeployConfig]]],
                         bool]):
    """
    Invoke {callback} on layout updates(adding or removing saved_models)
    """

    def callback_wrap(children: List[str]):
      with self._lock:
        logging.info(f"layout updated: {children}")
        model_names = set()
        saved_models = []
        for child in children:
          model_name, sub_graph = child.split(":")[:2]
          saved_model = SavedModel(model_name, sub_graph)
          fetch_path = f"/{self._bzid}/saved_models/{model_name}/{sub_graph}"
          data = self.get_znode(fetch_path)
          if data is None:
            logging.error("missing deploy config for saved model")
            continue
          saved_models.append(
              (saved_model, SavedModelDeployConfig.deserialize(data)))
          model_names.add(model_name)

        self._service_info_map = {
            model_name: self._service_info_map.get(model_name, {})
            for model_name in model_names
        }
        for model_name in model_names:
          binding_watch_path = f"/{self._bzid}/binding/{model_name}"
          self._children_watch(binding_watch_path,
                               partial(self._bind_callback, model_name))
        ret = callback(saved_models)
        return ret

    with self._lock:
      self._zk.ensure_path(layout_path)
      self._children_watch(layout_path, callback_wrap)

  def get_service_map(self) -> Dict[str, Dict[str, List[ContainerServiceInfo]]]:
    """
    Get service info map
    {
      "model_name": {
        "sub_graph: [
          {
            "idc": "LQ",
            "archon": "10.xx.xx.1:9876",
            "grpc": "10.xx.xx.1:8765",
            "http": "10.xx.xx.1:6789",
            "agent": "10.xx.xx.1:6787"
          }
        ]
      }
    }
    """
    return self._service_info_map.copy()

  def _bind_callback(self, model_name, children):
    with self._lock:
      if model_name not in self._service_info_map:
        logging.info(f"model {model_name} no longer subscribed.")
        return False
      new_binding = defaultdict(list)
      for child in children:
        sub_graph, ctx_cluster, ctx_id = child.split(":")[:3]
        saved_model = SavedModel(model_name, sub_graph)
        container = Container(ctx_cluster, ctx_id)
        service_info = self.get_service_info(container)
        if service_info is None:
          logging.error(f"no serivice info of {child}")
          continue
        new_binding[sub_graph].append(service_info)
      self._service_info_map[model_name] = new_binding

  def report_service_info(self, container: Container,
                          service_info: ContainerServiceInfo) -> None:
    service_info_path = f"/{self._bzid}/container_service/{container}"
    self.create_znode(service_info_path,
                      service_info.serialize(),
                      ephemeral=True,
                      makepath=True)

  def get_service_info(self, container) -> ContainerServiceInfo:
    service_info_path = f"/{self._bzid}/container_service/{container}"
    data = self.get_znode(service_info_path)
    if data is None:
      return None
    else:
      return ContainerServiceInfo.deserialize(data)

  def _children_watch(self, path, callback):
    with self._lock:
      if path in self._children_watcher_map and not self._children_watcher_map[
          path]._stopped:
        logging.info(f"active watcher exists on path {path}")
      else:
        self._zk.ensure_path(
            path
        )  # make sure the path exists otherwise the watcher may not be effective
        self._children_watcher_map[path] = self._zk.ChildrenWatch(
            path, callback)
        logging.info(f"registered new watcher on {path}")

  def list_saved_models(self, model_name: str) -> List[SavedModel]:
    model_path = f"/{self._bzid}/saved_models/{model_name}"
    try:
      sub_graphs = self._zk.get_children(model_path)
      return [SavedModel(model_name, sub_graph) for sub_graph in sub_graphs]
    except NoNodeError:
      return []

  def decl_saved_model(self, saved_model: SavedModel,
                       deploy_config: SavedModelDeployConfig):
    saved_model_path = f"/{self._bzid}/saved_models/{saved_model.model_name}/{saved_model.sub_graph}"
    logging.info(f"publishing {saved_model} -> {deploy_config}")
    self.create_znode(saved_model_path,
                      deploy_config.serialize(),
                      makepath=True)

  def add_to_layout(self, layout: str, saved_model: SavedModel):
    path = f"{layout}/{saved_model}"
    self._zk.ensure_path(path)

  def remove_from_layout(self, layout: str, saved_model: SavedModel):
    path = f"{layout}/{saved_model}"
    try:
      self._zk.delete(path)
    except NoNodeError:
      pass

  def bzid_info(self):
    # model deploy configs
    model_info = defaultdict(lambda: defaultdict(dict))
    if self._zk.exists(f"/{self._bzid}/saved_models"):
      model_names = self._zk.get_children(f"/{self._bzid}/saved_models")
      for model_name in model_names:
        sub_graphs = self._zk.get_children(
            f"/{self._bzid}/saved_models/{model_name}")
        model_info[model_name]['sub_graphs_total'] = len(sub_graphs)
        for sub_graph in sub_graphs:
          model_info[model_name][sub_graph]['deploy_config'] = self.get_znode(
              f"/{self._bzid}/saved_models/{model_name}/{sub_graph}").decode(
                  'utf-8')

    container_info = defaultdict(lambda: defaultdict(dict))

    # container service info
    if self._zk.exists(f"/{self._bzid}/container_service"):
      containers = self._zk.get_children(f"/{self._bzid}/container_service")
      for container in containers:
        cluster, container_id = container.split(":")[:2]
        container_info[cluster][container_id]['service_info'] = self.get_znode(
            f"/{self._bzid}/container_service/{container}").decode('utf-8')

    # layout info
    layout_info = defaultdict(lambda: defaultdict(dict))
    if self._zk.exists(f"/{self._bzid}/layouts"):
      layouts = self._zk.get_children(f"/{self._bzid}/layouts")
      for layout in layouts:
        saved_models = self._zk.get_children(f"/{self._bzid}/layouts/{layout}")
        if saved_models:
          layout_info[layout] = sorted(saved_models)
        else:
          layout_info[layout] = []

    # bindings
    if self._zk.exists(f"/{self._bzid}/binding"):
      model_names = self._zk.get_children(f"/{self._bzid}/binding")
      for model_name in model_names:
        bindings = self._zk.get_children(f"/{self._bzid}/binding/{model_name}")
        for binding in bindings:
          sub_graph, cluster, container_id = binding.split(":")[:3]
          if 'bindings' not in model_info[model_name][sub_graph]:
            model_info[model_name][sub_graph]['bindings'] = []
            model_info[model_name][
                'sub_graphs_available'] = model_info[model_name].get(
                    'sub_graphs_available', 0) + 1
          model_info[model_name][sub_graph]['bindings'].append(
              f"{cluster}:{container_id}")
          if 'saved_models' not in container_info[cluster][container_id]:
            container_info[cluster][container_id]['saved_models'] = []
          container_info[cluster][container_id]['saved_models'].append(
              f"{model_name}:{sub_graph}")

    def sorted_dict(d):
      return dict(sorted(d.items()))

    return {
        'model_info': {
            model_name: sorted_dict(model_info[model_name])
            for model_name in model_info
        },
        'container_info': {
            cluster: sorted_dict(container_info[cluster])
            for cluster in container_info
        },
        'layout_info': {layout: layout_info[layout] for layout in layout_info}
    }

  # sync backend
  def subscribe_model(self, model_name: str):
    if model_name == self._sync_model_name:
      return
    assert self._sync_model_name is None
    self._sync_model_name = model_name
    self._service_info_map[model_name] = self._service_info_map.get(
        model_name, {})
    binding_watch_path = f"/{self._bzid}/binding/{model_name}"
    self._children_watch(binding_watch_path,
                         partial(self._bind_callback, model_name))

  def get_sync_targets(self, sub_graph: str) -> Tuple[str, List[str]]:
    with self._lock:
      if self._is_lost.is_set():
        self._available_saved_model.clear()
        logging.warning("zk is lost, try restarting")
        self._zk.restart()

      sub_graph_map = self._service_info_map.get(self._sync_model_name, {})
      service_infos = sub_graph_map.get(sub_graph, [])
      return f"{self._sync_model_name}:{sub_graph}", [
          service_info.grpc for service_info in service_infos
      ]

  def create_znode(self, path, value, ephemeral=False, makepath=False) -> None:
    with self._lock:
      try:
        self._zk.create(path,
                        value=value,
                        ephemeral=ephemeral,
                        makepath=makepath)
      except NodeExistsError as e:
        self._zk.retry(self._zk.set, path=path, value=value)
      except Exception as e:
        logging.error(f"exception in create_znode: {e}")

  def delete_znode(self, path) -> None:
    with self._lock:
      try:
        self._zk.delete(path)
      except Exception as e:
        logging.error(f"exception in delete_znode: {e}")

  def get_znode(self, path) -> bytes:
    try:
      return self._zk.get(path)[0]
    except NoNodeError:
      return None

  def start(self):
    self._zk.start()

  def stop(self):
    self._zk.stop()
