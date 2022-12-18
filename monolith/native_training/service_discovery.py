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
import abc
from collections import namedtuple, defaultdict
from enum import Enum
import socket
import os
import threading
import time
from typing import Dict, NamedTuple

import numpy as np
from kazoo.retry import KazooRetry
from monolith.native_training.zk_utils import MonolithKazooClient
from kazoo.client import KazooState
from kazoo.exceptions import NoNodeError, NodeExistsError

from monolith.native_training import consul
from monolith.native_training.zk_utils import default_zk_servers


class ServiceDiscoveryType(Enum):
  PRIMUS = 1
  CONSUL = 2
  ZK = 3


class ServiceDiscovery(abc.ABC):

  @abc.abstractmethod
  def register(self, name: str, index: int, addr: str):
    """Register the port to the index"""

  @abc.abstractmethod
  def deregister(self, name: str, index: int, addr: str):
    """Deregister the port from index."""

  def query(self, name) -> Dict[int, str]:
    """Returns a dict that maps index to str"""

  def close(self):
    pass


_HostAndPort = namedtuple("_HostAndPort", ["host", "port"])

_RETRY_MAX_BACKOFF_SECS = 5


def retry_with_socket_error(http_call):
  tries = 5
  for i in range(tries):
    try:
      return http_call()
    except socket.error:
      if i < tries - 1:
        time.sleep(np.random.rand() * _RETRY_MAX_BACKOFF_SECS)
        continue
      raise


class ConsulServiceDiscovery(ServiceDiscovery):

  def __init__(self, consul_id: str, retry_time_secs: float = 3.0):
    self._consul_id = consul_id
    self._client = consul.Client()
    self._retry_time_secs = retry_time_secs

  def register(self, name: str, index: int, addr: str):
    # This is best effort, deregister the address with the same name & index.
    while True:
      index_to_addr = self.query(name)
      if index in index_to_addr:
        self.deregister(name, index, index_to_addr[index])
      else:
        break
      time.sleep(self._retry_time_secs)

    host, port = self._get_host_and_port(addr)
    retry_with_socket_error(lambda: self._client.register(
        self._consul_id, port, tags={
            "index": index,
            "name": name,
            "ip": host,
        }))

    # We need to make sure we can be registered

    # We wait upto 180 secs before we think the machine is blacklisted.
    max_retries = max(2, 180 / max(1, _RETRY_MAX_BACKOFF_SECS))
    retries = 0
    while True:
      index_to_addr = self.query(name)
      if index in index_to_addr:
        break
      retries += 1
      if retries > max_retries:
        raise OSError("This machine is blacklisted by consul.")
      time.sleep(_RETRY_MAX_BACKOFF_SECS)

  def deregister(self, name: str, index: int, addr: str):
    del name
    del index
    host, port = self._get_host_and_port(addr)
    retry_with_socket_error(
        lambda: self._client.deregister(self._consul_id, port))

  def query_all(self) -> Dict[str, Dict[int, str]]:
    elements = retry_with_socket_error(
        lambda: self._client.lookup(self._consul_id, timeout=15))
    addrs = defaultdict(dict)

    for element in elements:
      name = element["Tags"]["name"]
      addr = "{}:{}".format(element["Tags"]["ip"], element["Port"])
      index = int(element["Tags"]["index"])
      addrs[name][index] = addr

    return addrs

  def query(self, name: str):
    named_addrs = self.query_all()
    return named_addrs[name]

  def _get_host_and_port(self, addr: str) -> _HostAndPort:
    components = addr.split(":")
    if len(components) != 2:
      raise ValueError("Invalid addr: {}".format(addr))
    return _HostAndPort(host=components[0], port=int(components[1]))


class TfConfigServiceDiscovery(ServiceDiscovery):

  def __init__(self, tf_config):
    self._tf_config = tf_config

  def register(self, name: str, index: int, addr: str):
    pass

  def deregister(self, name: str, index: int, addr: str):
    pass

  def query(self, name: str):
    if name == 'ps':
      addr_list = self._tf_config['cluster'][name]
    elif name == 'worker':
      if 'chief' in self._tf_config['cluster']:
        addr_list = self._tf_config['cluster']['chief'] + \
                    self._tf_config['cluster'][name]
      else:
        addr_list = self._tf_config['cluster'][name]
    else:
      raise ValueError('name must be ps/worker')

    return {i: addr for i, addr in enumerate(addr_list)}

  @property
  def server_type(self):
    task = self._tf_config['task']
    return 'worker' if task['type'] == 'chief' else task['type']

  @property
  def addr(self):
    task = self._tf_config['task']
    return self._tf_config['cluster'][task['type']][task['index']]

  @property
  def index(self):
    task = self._tf_config['task']
    if 'chief' in self._tf_config['cluster']:
      return task['index'] + 1 if task['type'] == 'worker' else task['index']
    else:
      return task['index']


class ZKListener(object):

  def __init__(self, zkds: 'ZKServiceDiscovery'):
    self._zksd = zkds
    self._has_lost = False

  def __call__(self, state: KazooState) -> None:
    if state == KazooState.LOST:
      # The connection has been confirmed dead
      logging.warning(
          "Any ephemeral nodes will need to be recreated upon re-establishing a connection."
      )
      self._has_lost = True
    elif state == KazooState.SUSPENDED:
      # Handle being disconnected from Zookeeper
      return
    else:
      # Handle being connected/reconnected to Zookeeper
      if self._has_lost:
        logging.info(
            "connected/reconnected after lost, restart updater and watcher")
        self._zksd.do_all_registrations()
        self._has_lost = False


_ZK_REGISTRATION_PERIOD = 30 * 60


class ZKServiceDiscovery(ServiceDiscovery):

  class ThreadSet(NamedTuple):
    stop: threading.Event
    wakeup: threading.Event
    thread: threading.Thread

    def stop_and_join(self):
      self.stop.set()
      self.wakeup.set()
      self.thread.join()

  def __init__(self,
               job_name: str,
               zk_server: str = None,
               max_tries: int = 3,
               delay: int = 5):
    self._max_tries = max_tries
    self._delay = delay
    # /monolith/{job_name}/server_type.index -> host:port
    self._path_prefix = '/monolith/{}'.format(job_name)
    self._lock = threading.Lock()
    self._cluster = {}
    # Maps (name, index) to thread set.
    self._threads: Dict[Tuple[str, int], ZKServiceDiscovery.ThreadSet] = {}

    try:
      zk_server = zk_server or default_zk_servers()
      self._client = MonolithKazooClient(zk_server)
      self._client.add_listener(ZKListener(self))
      self._client.start()
      self._client.ensure_path(self._path_prefix)
    except Exception as e:
      logging.error("cannot create zk client, {}".format(e))
      raise e

    self._watch_data()

  def do_all_registrations(self):
    for ts in self._threads.values():
      ts.wakeup.set()

  def _get_node_name(self, server_type: str, index: int):
    return '{}.{}'.format(server_type, index)

  def _get_path(self, server_type: str, index: int):
    return "{}/{}".format(self._path_prefix,
                          self._get_node_name(server_type, index))

  def _try_create(self, path: str, value: str):
    value_bytes = bytes(value, 'utf-8')
    try:
      self._client.create(path,
                          value=value_bytes,
                          makepath=True,
                          ephemeral=True)
    except NodeExistsError:
      self._client.set(path, value_bytes)

  def _try_delete(self, path):
    try:
      self._client.delete(path=path, recursive=True)
    except NoNodeError:
      logging.info("{path} is not exist, no need to delete".format(path=path))

  def _children_watch(self, children):
    with self._lock:
      old_children = set(
          self._get_node_name(serve_type, index)
          for serve_type in self._cluster
          for index in self._cluster[serve_type])
      new_children = set(child for child in children if child)
      added = new_children - old_children

    for node in added:
      path = '{}/{}'.format(self._path_prefix, node)
      self._client.DataWatch(path, func=self._get_data_watch(path))

  def _get_data_watch(self, path):

    def data_watch(data, stat):
      server_type, index = os.path.basename(path).split('.')
      index = int(index)

      with self._lock:
        if data is not None and len(data) > 0:
          addr = data.decode("utf-8")
          if server_type in self._cluster:
            self._cluster[server_type][index] = addr
          else:
            self._cluster[server_type] = {index: addr}
        else:
          if server_type in self._cluster:
            if index in self._cluster[server_type]:
              del self._cluster[server_type][index]

    return data_watch

  def _watch_data(self):
    self._client.ChildrenWatch(self._path_prefix, self._children_watch)

  def register(self, name: str, index: int, addr: str):
    self._internal_register(name, index, addr, register_periodically=True)

  def _periodically_register(self, name: str, index: int, addr: str,
                             stop: threading.Event, wakeup: threading.Event):
    while True:
      if wakeup.wait(_ZK_REGISTRATION_PERIOD):
        wakeup.clear()
      if stop.is_set():
        break
      try:
        self._internal_register(name, index, addr, register_periodically=False)
      except Exception:
        # This is the best effort.
        pass

  def _internal_register(self, name: str, index: int, addr: str,
                         register_periodically: bool):
    path = self._get_path(name, index)
    retry = KazooRetry(max_tries=self._max_tries, delay=self._delay)
    try:
      retry(self._try_create, path, addr)
    except Exception as e:
      logging.error("server_type: {} , index:{} register fail".format(
          name, index))
      raise e

    if register_periodically:
      stop = threading.Event()
      wakeup = threading.Event()
      thread = threading.Thread(target=self._periodically_register,
                                args=(name, index, addr, stop, wakeup),
                                daemon=True)
      thread.start()
      self._threads[(name, index)] = ZKServiceDiscovery.ThreadSet(stop=stop,
                                                                  wakeup=wakeup,
                                                                  thread=thread)

  def deregister(self, name: str, index: int, addr: str):
    path = self._get_path(name, index)
    retry = KazooRetry(max_tries=self._max_tries, delay=self._delay)
    try:
      retry(self._try_delete, path)
    except Exception as e:
      logging.error("server_type: {} , index:{} deregister fail".format(
          name, index))
      raise e
    key = (name, index)
    ts = self._threads[key]
    ts.stop_and_join()
    del self._threads[key]

  def query(self, name) -> Dict[int, str]:
    with self._lock:
      if name in self._cluster:
        return self._cluster[name]
      else:
        return {}

  def close(self):
    if self._client is not None:
      self._client.stop()
      self._client.close()
      self._client = None

    for ts in self._threads.values():
      ts.stop_and_join()

  def __del__(self):
    self.close()


def deregister_all(consul_id: str):
  """Deregisters all records in the given consul_id."""
  discovery = ConsulServiceDiscovery(consul_id)
  named_addrs = discovery.query_all()
  for name, addrs in named_addrs.items():
    for idx, addr in addrs.items():
      discovery.deregister(name, idx, addr)
