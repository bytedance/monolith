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
import socket
import time
import traceback
from queue import Queue
from absl import logging
from collections import defaultdict
from threading import Thread, RLock
from typing import List, Optional, Dict, Union, Set
from monolith.agent_service.data_def import ResourceSpec, ReplicaMeta
from monolith.agent_service.data_def import PublishMeta, PublishType as PType
from monolith.agent_service.data_def import ModelState, ModelName, ModelMeta
from monolith.agent_service.data_def import Event, EventType as EType
from kazoo.client import Election
from kazoo.protocol.states import WatchedEvent, EventType, ZnodeStat, KazooState
from kazoo.exceptions import NodeExistsError, ZookeeperError, \
    NoNodeError, NotEmptyError, ConnectionClosedError
from monolith.native_training.zk_utils import MonolithKazooClient

from monolith.agent_service.utils import get_local_ip, DeployType, replica_id_from_pod_name


class ZKMirror(object):
  _data_lock = RLock()
  _zk_lock = RLock()
  _sep = '/'

  def __init__(
      self,
      zk: MonolithKazooClient,
      bzid: str,
      queue: Queue = None,
      tce_shard_id: int = -1,  # for entry deploy mode, tce_shard_id is -1
      num_tce_shard: int = 1,
      deploy_type: str = DeployType.MIXED):
    self._data: Dict[str, bytes] = {}
    self._zk: MonolithKazooClient = zk
    self.queue: Queue = queue
    self._bzid: str = bzid
    self._is_leader = False
    self.tce_shard_id: int = tce_shard_id
    self.num_tce_shard: int = num_tce_shard
    self._local_host = get_local_ip()
    self._deploy_type = deploy_type
    self._leader = None

    self._zk_lock_path = f"/{bzid}/locks/"
    self._zk_election_path = f"/{bzid}/election/"

    # /{bzid}/resource/{shard_id}:{replica_id}  -> ResourceSpec
    self.resource_path: str = f'/{bzid}/resource'

    # /{bzid}/portal/{model_name} -> ModelMeta
    self.portal_base_path: str = f'/{bzid}/portal'

    # /{bzid}/publish/{shard_id}:{model_name} -> PublishMeta
    self.publish_base_path: str = f'/{bzid}/publish'

    # /{bzid}/service/{model_name}/deploy_type:task_id/replica -> ReplicaMeta
    self.service_base_path: str = f'/{bzid}/service'

  @property
  def is_leader(self) -> bool:
    return self._is_leader

  def set_leader(self):
    self._is_leader = True

  def create(self,
             path,
             value=b"",
             acl=None,
             ephemeral=False,
             sequence=False,
             makepath=True,
             include_data=False):
    with self._zk_lock:
      try:
        self._zk.create(path,
                        value,
                        acl=acl,
                        ephemeral=ephemeral,
                        sequence=sequence,
                        makepath=makepath,
                        include_data=include_data)
      except NodeExistsError as e:
        self._zk.retry(self._zk.set, path=path, value=value)
      except Exception as e:
        raise e

  def ensure_path(self, path):
    with self._zk_lock:
      self._zk.retry(self._zk.ensure_path, path=path)

  def set(self, path: str, value: bytes = b''):
    with self._zk_lock:
      try:
        self._zk.retry(self._zk.set, path=path, value=value)
      except NoNodeError as e:
        self._zk.create(path=path, value=value, makepath=True)
      except Exception as e:
        raise e

  def exists(self, path: str) -> bool:
    with self._zk_lock:
      try:
        status = self._zk.exists(path=path)
        if isinstance(status, bool):
          return status
        else:
          return status is not None
      except ZookeeperError as e:
        return path in self._data

  def delete(self, path: str, recursive: bool = True):
    with self._zk_lock:
      try:
        self._zk.retry(self._zk.delete, path=path, recursive=recursive)
      except NoNodeError as e:
        logging.info(e)
      except NotEmptyError as e:
        self._zk.retry(self._zk.delete, path=path, recursive=True)
      except Exception as e:
        raise e

  def get(self, path) -> Optional[bytes]:
    with self._data_lock:
      return self._data.get(path)

  def get_children(self, path: str) -> List[str]:
    with self._data_lock:
      length = len(path.split(self._sep))
      children = []
      for p in self._data:
        if p.startswith(path):
          tl = p.split(self._sep)
          if len(tl) > length:
            children.append(tl[length])
      return children

  def report_resource(self, recource: ResourceSpec):
    path = recource.get_path(self.resource_path)
    value = b'' if recource is None else recource.serialize()
    self.create(path=path, value=value, makepath=True, ephemeral=True)

  @property
  def resources(self) -> List[ResourceSpec]:
    # /{bzid}/resource/{shard_id}:{replica_id}  -> ResourceSpec
    with self._data_lock:
      resource_paths = [
          os.path.join(self.resource_path, child)
          for child in self.get_children(self.resource_path)
      ]
      return [ResourceSpec.deserialize(self.get(p)) for p in resource_paths]

  @property
  def num_tce_replica(self) -> Optional[int]:

    def get_replica_cnt():
      replica_cnt = {}
      with self._data_lock:
        for path in self._data:
          if path.startswith(self.resource_path):
            replica_id = int(os.path.basename(path).split(':')[-1])
            if replica_id == -1:
              continue  # skip entry
            elif replica_id in replica_cnt:
              replica_cnt[replica_id] += 1
            else:
              replica_cnt[replica_id] = 1

    replicas = get_replica_cnt()
    while not all(cnt == self.tce_shard_id for cnt in replicas.values()):
      time.sleep(5)
      # log every minute
      logging.log_every_n(level=logging.INFO,
                          msg='cluster autoscaler or broken node, keep waiting',
                          n=12)
      replicas = get_replica_cnt()

    return len(replica_cnt)

  @property
  def tce_replica_id(self) -> int:
    replica_id = int(os.environ.get('REPLICA_ID', -1))
    if replica_id == -1:
      replica_id = replica_id_from_pod_name()
    return replica_id

  def publish_loadding(self, info: Union[PublishMeta, List[PublishMeta]]):
    if isinstance(info, (list, tuple)):
      for pm in info:
        path = pm.get_path(self.publish_base_path)
        value = pm.serialize()
        loc = self.get(path)
        if loc is None or loc != value:
          self.create(path=path, value=value, makepath=True)
    else:
      path = info.get_path(self.publish_base_path)
      value = info.serialize()
      loc = self.get(path)
      if loc is None or loc != value:
        self.create(path=path, value=value, makepath=True)

  def expected_loading(self) -> Dict[ModelName, PublishMeta]:
    # /{bzid}/publish/{shard_id}:{model_name} -> PublishMeta
    with self._data_lock:
      nodes = self.get_children(self.publish_base_path)
      models, select, shortest_sub_model_pm = {}, [], {}
      for node in nodes:
        path = os.path.join(self.publish_base_path, node)
        pm = PublishMeta.deserialize(self.get(path))
        shard_id, replica_id, model_name = node.split(':')
        if model_name in models:
          models[model_name] += 1
        else:
          models[model_name] = 1

        # record the most sub_model pm
        if model_name not in shortest_sub_model_pm:
          shortest_sub_model_pm[model_name] = pm
        elif len(shortest_sub_model_pm[model_name].sub_models) > len(
            pm.sub_models):
          shortest_sub_model_pm[model_name] = pm

        # the last one
        if models[model_name] == pm.total_publish_num:
          select.append(shortest_sub_model_pm[model_name])

      expected: Dict[str, PublishMeta] = {}
      for pm in select:
        path = os.path.join(
            self.publish_base_path,
            f'{self.tce_shard_id}:{self.tce_replica_id}:{pm.model_name}')
        data = self.get(path)
        # for new replica or entry, data is None
        pm = pm if data is None else PublishMeta.deserialize(data)

        model_name = pm.model_name
        if pm.ptype != PType.LOAD:
          logging.info("ptype is not load, skip!")
          continue

        # note: the sceduler will not scedule entry submodel alone,
        # all the submodels are sceduled with ps submodel.
        # so there is an entry submodel in every PublishMeta.
        # the shard_id of every service is -1.

        if pm.shard_id == self.tce_shard_id and pm.replica_id == self.tce_replica_id:
          # if service type is 'entry', then shard_id is -1,
          # and no PublishMeta will fall in this branch
          # only ps/dense/mixed service type will hit this branch
          expected[model_name] = pm
        elif pm.shard_id == self.tce_shard_id and not pm.is_spec:  # for autoscalar,  new replica
          if model_name not in expected:
            pm.replica_id = self.tce_replica_id
            expected[model_name] = pm
        else:
          # all entry/ps/dense/mixed service type can hit this branch
          # and ps/dense submodels were filtered
          if model_name not in expected:
            pm.shard_id = self.tce_shard_id
            pm.replica_id = self.tce_replica_id
            pm.sub_models = {
                sub_model_name: vp
                for sub_model_name, vp in pm.sub_models.items()
                if sub_model_name.startswith('entry')
            }
            expected[model_name] = pm

      return expected

  def get_published_path(self, model_name: str) -> List[str]:
    with self._data_lock:
      paths = []
      for path in self._data:
        if path.startswith(
            self.publish_base_path) and path.endswith(model_name):
          paths.append(path)
      return paths

  def update_service(self, replicas: List[ReplicaMeta]):
    # /{bzid}/service/{model_name}/deploy_type:task_id/replica -> ReplicaMeta
    need_create_or_update, local_load_paths = {}, set()
    for rm in replicas:
      path = rm.get_path(self._bzid, self._sep)
      value = rm.serialize()
      local_load_paths.add(path)

      loc = self.get(path)
      if loc is None or loc != value:  # not exists or changed
        need_create_or_update[path] = value

    # only care about local replicas, remove first
    need_remove_paths = self.local_replica_paths - local_load_paths
    for path in need_remove_paths:
      self.delete(path)

    # create or update replicas
    if need_create_or_update:
      logging.info(f'need_create_or_update: {need_create_or_update}')
    for path, value in need_create_or_update.items():
      try:
        self.create(path=path, value=value, ephemeral=True, makepath=True)
      except Exception as e:
        logging.info(repr(e))

  @property
  def local_replica_paths(self) -> Set[str]:
    with self._data_lock:
      local_replicas = set()
      for path in self._data:
        if path.startswith(self.service_base_path):
          rm = ReplicaMeta.deserialize(self._data[path])
          host = rm.address.split(':')[0]
          if host == self._local_host and rm.replica == self.tce_replica_id:
            local_replicas.add(path)
      return local_replicas

  def get_all_replicas(self, server_type: str) -> Dict[str, List[ReplicaMeta]]:
    # f'{model_name}:{server_type}:{task_id}' -> ReplicaMeta
    with self._data_lock:
      result: Dict[str, List[ReplicaMeta]] = defaultdict(list)
      for path, value in self._data.items():
        if path.startswith(self.service_base_path):
          raw_key = path[len(self.service_base_path):].strip(self._sep)
          model_name, st, task, _ = raw_key.replace(self._sep, ':').split(':')
          if st == server_type:
            key = ':'.join([model_name, server_type, task])
            rm = ReplicaMeta.deserialize(value)
            if rm.stat == ModelState.AVAILABLE:
              result[key].append(rm)

      return result

  def get_model_replicas(self, model_name: str,
                         server_type: str) -> Dict[str, List[ReplicaMeta]]:
    # f'{model_name}:{server_type}:{task_id}' -> ReplicaMeta
    with self._data_lock:
      result: Dict[str, List[ReplicaMeta]] = defaultdict(list)
      base_path = os.path.join(self.service_base_path, model_name)
      for task in self.get_children(base_path):
        if task.startswith(server_type.lower()):
          task_path = os.path.join(base_path, task)
          for replica in self.get_children(task_path):
            path = os.path.join(task_path, replica)
            content = self._data.get(path)
            if content is not None:
              rm = ReplicaMeta.deserialize(content)
              if rm.stat == ModelState.AVAILABLE:
                result[f'{model_name}:{task}'].append(rm)
      return result

  def get_task_replicas(self, model_name: str, server_type: str,
                        task: int) -> List[ReplicaMeta]:
    with self._data_lock:
      path = os.path.join(self.service_base_path, model_name,
                          f'{server_type.lower()}:{task}')
      result: List[ReplicaMeta] = []
      for child in self.get_children(path):
        content = self._data.get(os.path.join(path, child))
        if content is not None:
          rm = ReplicaMeta.deserialize(content)
          if rm.stat == ModelState.AVAILABLE:
            result.append(rm)
      return result

  def get_replica(self, model_name: str, server_type: str, task: int,
                  replica: int) -> Optional[ReplicaMeta]:
    with self._data_lock:
      path = os.path.join(self.service_base_path, model_name,
                          f'{server_type.lower()}:{task}', str(replica))
      content = self._data.get(path)
      if content is None:
        return None
      else:
        rm = ReplicaMeta.deserialize(content)
        if rm.stat == ModelState.AVAILABLE:
          return rm
        else:
          return None

  def watch_portal(self):
    # 1) check portal/publish conscience
    self._zk.ensure_path(path=self.portal_base_path)
    self._zk.ensure_path(path=self.publish_base_path)
    models_in_portal = set(
        self._zk.get_children(path=self.portal_base_path) or [])
    models_in_publish = {
        item.split(':')[-1]  # {shard_id}:{model_name} -> model_name
        for item in (self._zk.get_children(path=self.publish_base_path) or [])
    }
    if len(models_in_publish) > 0:
      if len(models_in_portal) == 0:
        # just remove all
        remove = models_in_publish
      else:
        remove = models_in_publish - models_in_portal

      for model in remove:
        for key in self._zk.get_children(path=self.publish_base_path):
          if key.endswith(model):
            self.delete(path=os.path.join(self.publish_base_path, key),
                        recursive=True)

    # 2) watch portal
    models = set()

    def create_data_watch(data_path: str):
      logging.info(
          f"add data watch for model {os.path.basename(data_path)} in portal")

      def data_watch(data: bytes, state: ZnodeStat, event: WatchedEvent):
        # info = ModelMeta.deserialize(data)
        with self._data_lock:
          if event is None or event.type in {
              EventType.CREATED, EventType.DELETED
          }:
            # in the first call, event is None
            if event is None:
              logging.info(f'call watch_portal when restart {data}')
            if event is None and data is None:
              action = EventType.DELETED
            else:
              action = EventType.NONE if event is None else event.type
            if data is not None and len(data) > 0:
              mm = ModelMeta.deserialize(data)
              mm.action = action
            else:
              mm = ModelMeta(model_name=os.path.basename(data_path),
                             action=action)
            self.queue.put(Event(data_path, mm.serialize(), EType.PORTAL))
          else:
            assert event.type in {
                EventType.CHILD, EventType.CHANGED, EventType.NONE
            }

      return data_watch

    def children_watch(children: List[str]):
      if children is None or len(children) == 0:
        return
      else:
        for model in children:
          if model not in models:
            models.add(model)
            path = os.path.join(self.portal_base_path, model)
            self._zk.DataWatch(path=path, func=create_data_watch(path))

    logging.info(f"add children watch in portal")
    self._zk.ChildrenWatch(path=self.portal_base_path, func=children_watch)

  def watch_publish(self):
    publishs = set()
    self._zk.ensure_path(path=self.publish_base_path)

    def get_publish_cnt(model_name: str):
      cnt = 0
      for path in self._data:
        if path.startswith(
            self.publish_base_path) and path.endswith(model_name):
          cnt += 1

      return cnt

    def create_data_watch(data_path: str):

      def data_watch(data: bytes, state: ZnodeStat, event: WatchedEvent):
        data = data or self._data.get(data_path, None)
        if data is not None and len(data) > 0:
          pm = PublishMeta.deserialize(data)
        else:
          logging.info(f'watch_publish: data is None, {event}')
          return

        with self._data_lock:
          if event is None or event.type == EventType.CREATED:
            # in the first call, event is None
            if pm.ptype == PType.LOAD:
              self._data[data_path] = data
            else:
              del self._data[data_path]
          elif event.type == EventType.DELETED:
            if data_path in self._data:
              del self._data[data_path]
          else:
            assert event.type in {
                EventType.CHILD, EventType.CHANGED, EventType.NONE
            }

          cnt = get_publish_cnt(pm.model_name)
          if cnt == 0 or cnt == pm.total_publish_num:
            logging.info(f"all the publish of model {pm.model_name} arrived, "
                         f"send event to {'unload' if cnt == 0 else 'load'}")
            load_path = pm.get_path(self.publish_base_path)
            data = self._data.get(load_path, data)
            self.queue.put(Event(data_path, data, EType.PUBLISH))

      return data_watch

    def children_watch(children: List[str]):
      if children is None or len(children) == 0:
        return
      else:
        for pub in children:
          if pub not in publishs:
            publishs.add(pub)
            path = os.path.join(self.publish_base_path, pub)
            self._zk.DataWatch(path=path, func=create_data_watch(path))

    self._zk.ChildrenWatch(path=self.publish_base_path, func=children_watch)

  def watch_resource(self):
    instances = set()

    def create_data_watch(data_path: str):

      def data_watch(data: bytes, state: ZnodeStat, event: WatchedEvent):
        data = data or self._data.get(data_path, None)
        with self._data_lock:
          if event is None or event.type == EventType.CREATED:
            # in the first call, event is None
            self._data[data_path] = data
          elif event.type == EventType.DELETED:
            del self._data[data_path]
          elif event.type == EventType.CHANGED:
            self._data[data_path] = data
          else:
            assert event.type in {EventType.CHILD, EventType.NONE}

      return data_watch

    def children_watch(children: List[str]):
      if children is None or len(children) == 0:
        return
      else:
        for inst in children:
          if inst not in instances:
            instances.add(inst)
            path = os.path.join(self.resource_path, inst)
            self._zk.DataWatch(path=path, func=create_data_watch(path))

    self._zk.ChildrenWatch(path=self.resource_path, func=children_watch)

  def watch_service(self):
    # /{bzid}/service/{model_name}/deploy_type:task_id/replica -> ReplicaMeta

    children_set = set()
    self._zk.ensure_path(path=self.service_base_path)

    def create_data_watch(data_path: str):
      logging.info(f'data_path: {data_path}')

      def data_watch(data: bytes, state: ZnodeStat, event: WatchedEvent):
        logging.info(f'service data_watch: {data_path}: {data}, {event}')
        data = data or self._data.get(data_path, None)
        with self._data_lock:
          if event is None or event.type == EventType.CREATED:
            # in the first call, event is None
            self._data[data_path] = data
          elif event.type == EventType.DELETED:
            del self._data[data_path]
          elif event.type == EventType.CHANGED:
            self._data[data_path] = data
          else:
            assert event.type in {EventType.CHILD, EventType.NONE}

      return data_watch

    def create_replica_watch(task_path: str):
      logging.info(f'task_path: {task_path}')
      model = os.path.basename(os.path.dirname(task_path))
      task = os.path.basename(task_path)

      def replica_watch(children: List[str]):
        if children is None or len(children) == 0:
          return
        else:
          for replica in children:
            key = f"{model}:{task}:{replica}"
            if key not in children_set:
              children_set.add(key)
              path = os.path.join(task_path, replica)
              self._zk.DataWatch(path=path, func=create_data_watch(path))

      return replica_watch

    def create_task_watch(model_path: str):
      logging.info(f'model_path: {model_path}')
      model = os.path.basename(model_path)

      def task_watch(children: List[str]):
        if children is None or len(children) == 0:
          return
        else:
          for task in children:
            key = f"{model}:{task}"
            if key not in children_set:
              children_set.add(key)
              path = os.path.join(model_path, task)
              self._zk.ChildrenWatch(path=path, func=create_replica_watch(path))

      return task_watch

    def model_watch(children: List[str]):
      if children is None or len(children) == 0:
        return
      else:
        for model in children:
          key = model
          if key not in children_set:
            children_set.add(key)
            path = os.path.join(self.service_base_path, model)
            self._zk.ChildrenWatch(path=path, func=create_task_watch(path))

    self._zk.ChildrenWatch(path=self.service_base_path, func=model_watch)

  def election(self, leader, sched, identifier: str = None):
    self._leader = leader
    identifier = identifier or os.environ.get('MY_POD_NAME')

    if self._deploy_type == DeployType.ENTRY:
      logging.info('entry cannot be leader!')
      return

    def target():
      try:
        election: Election = self._zk.Election(self._zk_election_path,
                                               identifier)
        election.run(leader, zk=self, sched=sched)
      except ConnectionClosedError as e:
        if self._zk.state in {KazooState.CONNECTED, KazooState.SUSPENDED}:
          logging.info(f"ConnectionClosedError, state is {self._zk.state}")
          pass
        else:
          logging.info(f"kazo {self._zk.state}, restart!")
          with self._data_lock:
            self._data = {}
            while not self.queue.empty():
              self.queue.get_nowait()
          self.start()
      except Exception as e:
        logging.info(e)

    thread = Thread(target=target)
    thread.start()

  def start(self, is_client: bool = False):
    self._zk.start()
    self.watch_service()
    if not is_client:
      self.watch_publish()

  def stop(self):
    if self._leader is not None:
      self._leader.cancel()

    self._zk.stop()
