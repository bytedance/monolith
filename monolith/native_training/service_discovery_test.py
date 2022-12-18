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

import json
from collections import defaultdict
from functools import partial
import os
import socket
import threading
import time
import unittest
from unittest import mock
from kazoo.client import KazooState
from kazoo.exceptions import NoNodeError, NodeExistsError, NotEmptyError
from typing import List, Callable

from monolith.native_training import service_discovery


class FakeConsul:

  def __init__(self, blacklist=[]):
    """The host in blacklist will not be registered to consul"""
    self._name_to_dict = defaultdict(dict)
    self._blacklist = blacklist

  def lookup(self, name: str, **kwargs):
    return list(self._name_to_dict[name].values())

  def register(self,
               name: str,
               port: int,
               tags={},
               host: str = None,
               check_script: str = None):
    del check_script
    if tags["ip"] in self._blacklist:
      return
    key = str(host) + ":" + str(port)
    d = self._name_to_dict[name]
    d[key] = {"Host": host, "Port": port, "Tags": tags}

  def deregister(self, name: str, port: int, host: str = None):
    key = str(host) + ":" + str(port)
    d = self._name_to_dict[name]
    del d[key]


class FakeKazooClient:

  def __init__(self, zk_server: str):
    self._lock = threading.RLock()
    self._zk_server = zk_server
    self._data = None

    self._children_watches = []
    self._data_watches = []
    self.DataWatch = partial(DataWatch, self)
    self.ChildrenWatch = partial(ChildrenWatch, self)
    self._listeners = []

  def ensure_path(self, path: str):
    with self._lock:
      if path not in self._data:
        self._data[path] = None

  def start(self):
    with self._lock:
      self._data = {}

  def create(self,
             path: str,
             value: bytes = b'',
             makepath: bool = False,
             ephemeral: bool = False):
    with self._lock:
      if path in self._data:
        raise NodeExistsError('node {} exists'.format(path))
      else:
        prefix = os.path.dirname(path)
        if prefix in self._data or makepath:
          self._data[path] = value

          for dw in self._data_watches:
            if dw.path == path:
              dw(value, None)

          for cw in self._children_watches:
            dirname = os.path.dirname(path)
            if dirname == cw.path:
              children = [
                  os.path.basename(key)
                  for key in self._data
                  if os.path.dirname(key) == cw.path
              ]
              cw(children)
        else:
          raise NoNodeError('No Node {}'.format(prefix))

  def delete(self, path: str, recursive: bool = True):
    with self._lock:
      if path in self._data:
        del self._data[path]
        for dw in self._data_watches:
          if dw.path == path:
            dw(None, None)

        for cw in self._children_watches:
          dirname = os.path.dirname(path)
          if dirname == cw.path:
            children = [
                os.path.basename(key)
                for key in self._data
                if os.path.dirname(key) == cw.path
            ]
            cw(children)
      else:
        collected = []
        for key in self._data:
          if key.startswith(path):
            collected.append(key)
        if collected:
          if recursive:
            for key in collected:
              del self._data[key]
              for dw in self._data_watches:
                if dw.path == key:
                  dw(None, None)

              for cw in self._children_watches:
                dirname = os.path.dirname(key)
                if dirname == cw.path:
                  children = [
                      os.path.basename(key)
                      for key in self._data
                      if os.path.dirname(key) == cw.path
                  ]
                  cw(children)
          else:
            raise NotEmptyError('node {} has children'.format(path))
        else:
          raise NoNodeError('node {} not found'.format(path))

  def set(self, path: str, value: bytes):
    with self._lock:
      if path in self._data:
        self._data[path] = value
        for dw in self._data_watches:
          if dw.path == path:
            dw(value, None)
      else:
        raise NoNodeError('node {} is not exist'.format(path))

  def get(self, path: str):
    with self._lock:
      if path in self._data:
        return self._data[path], None
      else:
        raise NoNodeError('node {} is not exist'.format(path))

  def get_children(self, path: str):
    with self._lock:
      if path in self._data:
        return []
      else:
        collected = []
        for key in self._data:
          if key.startswith(path):
            child = key[len(path) + 1:].split('/')[0]
            collected.append(child)

        if collected:
          return collected
        else:
          raise NoNodeError('node {} is not exist'.format(path))

  def stop(self):
    with self._lock:
      self._data = None

  def close(self):
    with self._lock:
      if self._data is not None:
        self.stop()

  def add_listener(self, listener):
    self._listeners.append(listener)

  @property
  def listeners(self):
    return self._listeners


class ChildrenWatch:

  def __init__(self, client: FakeKazooClient, path: str,
               func: Callable[[List[str]], None]):
    self._client = client
    client._children_watches.append(self)
    self.path = path
    self._func = func

    children = []
    for key in self._client._data:
      dirname = os.path.dirname(key)
      if dirname == path:
        children.append(os.path.basename(key))
    self._func(children)

  def __call__(self, children: List[str]):
    self._func(children)


class DataWatch:

  def __init__(self, client: FakeKazooClient, path: str, func: str):
    self._client = client
    client._data_watches.append(self)
    self.path = path
    self._func = func

    for key, value in self._client._data.items():
      if key == path:
        self._func(value, None)

  def __call__(self, data: str, stat=None):
    self._func(data, stat)


_CONSUL_CLIENT = "monolith.native_training.service_discovery.consul.Client"
_ZK_CLIENT = "monolith.native_training.service_discovery.MonolithKazooClient"


class ConsultServiceDiscovery(unittest.TestCase):

  def test_basic(self):
    with mock.patch(_CONSUL_CLIENT) as MockClient:
      MockClient.return_value = FakeConsul()

      discovery = service_discovery.ConsulServiceDiscovery("unique_id")
      discovery.register("server", 0, "192.168.0.1:1001")
      discovery.register("server", 1, "192.168.0.2:1002")
      self.assertDictEqual(discovery.query("server"), {
          0: "192.168.0.1:1001",
          1: "192.168.0.2:1002"
      })
      discovery.deregister("server", 0, "192.168.0.1:1001")
      discovery.deregister("server", 1, "192.168.0.2:1002")
      self.assertDictEqual(discovery.query("server"), {})

  def test_duplicate_registration(self):
    with mock.patch(_CONSUL_CLIENT) as MockClient:
      MockClient.return_value = FakeConsul()
      discovery = service_discovery.ConsulServiceDiscovery("unique_id",
                                                           retry_time_secs=0.0)
      discovery.register("server", 0, "192.168.0.1:1001")
      discovery.register("server", 0, "192.168.0.1:1002")
      self.assertDictEqual(discovery.query("server"), {0: "192.168.0.1:1002"})

  def test_multi_names(self):
    with mock.patch(_CONSUL_CLIENT) as MockClient:
      MockClient.return_value = FakeConsul()
      discovery = service_discovery.ConsulServiceDiscovery("unique_id")
      discovery.register("ps", 0, "192.168.0.1:1001")
      discovery.register("worker", 0, "192.168.0.1:1002")
      self.assertDictEqual(discovery.query("worker"), {0: "192.168.0.1:1002"})

  def test_retry(self):
    with mock.patch(
        "monolith.native_training.service_discovery._RETRY_MAX_BACKOFF_SECS",
        0):
      with mock.patch(_CONSUL_CLIENT) as MockClient:
        mock_client = mock.MagicMock()

        def raise_timeout(*args, **kwargs):
          raise socket.timeout()

        mock_client.register.side_effect = raise_timeout
        MockClient.return_value = mock_client
        discovery = service_discovery.ConsulServiceDiscovery("unique_id")

        self.assertRaises(socket.timeout, discovery.register, "ps", 0,
                          "192.168.0.1:1001")

  def test_registeration_failed(self):
    with mock.patch(
        "monolith.native_training.service_discovery._RETRY_MAX_BACKOFF_SECS",
        0), mock.patch(_CONSUL_CLIENT) as MockClient:
      MockClient.return_value = FakeConsul(blacklist=["192.168.0.1"])
      discovery = service_discovery.ConsulServiceDiscovery("unique_id")
      with self.assertRaises(OSError):
        discovery.register("ps", 0, "192.168.0.1:1001")


class TfConfigServiceDiscoveryTest(unittest.TestCase):

  def test_tf_conf_sd(self):
    cluster = {
        'chief': ['host0:2222'],
        'ps': ['host1:2222', 'host2:2222'],
        'worker': ['host3:2222', 'host4:2222', 'host5:2222']
    }
    task = {'type': 'worker', 'index': 1}
    tf_conf = {'cluster': cluster, 'task': task}

    ps_list = cluster['ps']
    discovery = service_discovery.TfConfigServiceDiscovery(tf_conf)
    self.assertEqual(discovery.query('ps'),
                     {i: addr for i, addr in enumerate(ps_list)},
                     "['host1:2222', 'host2:2222']")

    worker_list = cluster['chief'] + cluster['worker']
    self.assertEqual(discovery.query('worker'),
                     {i: addr for i, addr in enumerate(worker_list)},
                     "[host0:2222, 'host1:2222', 'host2:2222']")

    self.assertEqual(discovery.addr, 'host4:2222', 'host4:2222')
    self.assertEqual(discovery.server_type, 'worker', 'worker')
    self.assertEqual(discovery.index, 2, 2)


class ZKServiceDiscoveryTest(unittest.TestCase):

  def test_basic(self):
    with mock.patch(_ZK_CLIENT) as MockClient:
      MockClient.return_value = FakeKazooClient("test_model")

      discovery = service_discovery.ZKServiceDiscovery("test_model", "fask")
      discovery.register("server", 0, "192.168.0.1:1001")
      discovery.register("server", 1, "192.168.0.2:1002")
      self.assertDictEqual(discovery.query("server"), {
          0: "192.168.0.1:1001",
          1: "192.168.0.2:1002"
      })
      discovery.deregister("server", 0, "192.168.0.1:1001")
      discovery.deregister("server", 1, "192.168.0.2:1002")
      self.assertDictEqual(discovery.query("server"), {})
      discovery.close()

  def test_duplicate_registration(self):
    with mock.patch(_ZK_CLIENT) as MockClient:
      MockClient.return_value = FakeKazooClient("test_model")
      discovery = service_discovery.ZKServiceDiscovery("test_model", "fask")
      discovery.register("server", 0, "192.168.0.1:1001")
      discovery.register("server", 0, "192.168.0.1:1002")
      self.assertDictEqual(discovery.query("server"), {0: "192.168.0.1:1002"})
      discovery.close()

  def test_multi_names(self):
    with mock.patch(_ZK_CLIENT) as MockClient:
      MockClient.return_value = FakeKazooClient("test_model")
      discovery = service_discovery.ZKServiceDiscovery("test_model", "fask")
      discovery.register("ps", 0, "192.168.0.1:1001")
      discovery.register("worker", 0, "192.168.0.1:1002")
      self.assertDictEqual(discovery.query("worker"), {0: "192.168.0.1:1002"})
      del discovery

  @mock.patch(
      "monolith.native_training.service_discovery._ZK_REGISTRATION_PERIOD",
      0.01)
  def test_periodic_registration(self):
    with mock.patch(_ZK_CLIENT) as MockClient:
      c = FakeKazooClient("test_model")
      MockClient.return_value = c
      discovery = service_discovery.ZKServiceDiscovery("test_model")
      discovery.register("ps", 0, "192.168.0.1:1001")
      c.set("/monolith/test_model/ps.0", "wrongdata".encode())
      time.sleep(1)
      # Periodic registration should register it again
      self.assertDictEqual(discovery.query("ps"), {0: "192.168.0.1:1001"})

  def test_listener(self):
    with mock.patch(_ZK_CLIENT) as MockClient:
      c = FakeKazooClient("test_model")
      MockClient.return_value = c
      discovery = service_discovery.ZKServiceDiscovery("test_model")
      discovery.register("ps", 0, "192.168.0.1:1001")
      listener = c.listeners[0]
      listener(KazooState.LOST)
      listener(KazooState.CONNECTED)
      self.assertDictEqual(discovery.query("ps"), {0: "192.168.0.1:1001"})


class UtilsTest(unittest.TestCase):

  def test_deregister_all(self):
    with mock.patch(_CONSUL_CLIENT) as MockClient:
      MockClient.return_value = FakeConsul()
      discovery = service_discovery.ConsulServiceDiscovery("unique_id")
      discovery.register("server", 0, "192.168.0.1:1001")
      service_discovery.deregister_all("unique_id")
      self.assertDictEqual(discovery.query("server"), {})


if __name__ == "__main__":
  unittest.main()
