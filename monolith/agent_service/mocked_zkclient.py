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
from functools import partial
from kazoo.protocol.states import ZnodeStat, WatchedEvent, EventType, KeeperState
from kazoo.exceptions import NoNodeError, NodeExistsError, NotEmptyError, CancelledError
import os
import time
from threading import Lock
from typing import List, Dict, Union, Callable, Optional


class ChildrenWatch:

  def __init__(self,
               client,
               path: str,
               func: Union[Callable[[List[str]], None],
                           Callable[[List[str], WatchedEvent], None]],
               send_event=False):
    self.path = path
    self.send_event = send_event
    self._stopped = False
    self._func = func
    catalog = client._catalog
    catalog.add_children_watch(self)

  def __call__(self, children: List[str], event: WatchedEvent):
    if self.send_event:
      self._func(children, event)
    else:
      self._func(children)


class DataWatch:

  def __init__(self, client, path: str,
               func: Union[Callable[[bytes, ZnodeStat, WatchedEvent], None],
                           Callable[[bytes, ZnodeStat], None]]):
    self.path = path
    self._func = func
    catalog = client._catalog
    catalog.add_data_watch(self)

  def __call__(self, data: bytes, state: ZnodeStat, event: WatchedEvent):
    try:
      self._func(data, state, event)
    except TypeError:
      self._func(data, state)


class Election(object):

  def __init__(self, client, path, identifier=None):
    self.lock = Lock()

  def run(self, func, *args, **kwargs):
    if not callable(func):
      raise ValueError("leader function is not callable")

    try:
      with self.lock:
        func(*args, **kwargs)
    except CancelledError:
      pass

  def cancel(self):
    self.lock.cancel()


class Node:

  def __init__(self,
               path: str,
               value: bytes = b'',
               ephemeral: bool = False,
               data_watch: DataWatch = None,
               children_watch: ChildrenWatch = None):
    self.path: str = path
    self.value: bytes = value
    self.ephemeral: bool = ephemeral
    self.children: Dict[str, Node] = {}

    self._ctime = int(time.time())
    self._mtime = int(time.time())
    self._version = 0

    self._data_watch = data_watch
    self._children_watch = children_watch

    event = WatchedEvent(type=EventType.CREATED,
                         state=KeeperState.CONNECTED,
                         path=self.path)
    if self._data_watch is not None:
      self._data_watch(self.value, self.state, event)
    if self._children_watch is not None:
      self._children_watch([], event)

  @property
  def state(self):
    return ZnodeStat(czxid=0,
                     mzxid=0,
                     ctime=self._ctime,
                     mtime=self._mtime,
                     version=self._version,
                     cversion=0,
                     aversion=0,
                     ephemeralOwner=0,
                     dataLength=len(self.value),
                     numChildren=len(self.children),
                     pzxid=0)

  @property
  def basename(self):
    return os.path.basename(self.path)

  def set(self, value: bytes):
    self._mtime = int(time.time())
    self._version += 1
    self.value = value

    if self._data_watch is not None:
      event = WatchedEvent(type=EventType.CHANGED,
                           state=KeeperState.CONNECTED,
                           path=self.path)
      self._data_watch(self.value, self.state, event)

  def get(self):
    return self.value

  def set_data_watch(self, watch: DataWatch):
    self._data_watch = watch
    self._data_watch(self.value, self.state, None)

  def set_children_watch(self, watch: ChildrenWatch):
    self._children_watch = watch
    self._children_watch(list(self.children.keys()), None)

  def create_child(self,
                   path: str,
                   value: bytes = b'',
                   ephemeral: bool = False,
                   data_watch=None,
                   children_watch=None):
    basename = os.path.basename(path)
    self._mtime = int(time.time())
    if self.path == os.path.sep:  # root
      child_path = f'{os.path.sep}{basename}'
    else:
      child_path = f'{self.path}{os.path.sep}{basename}'
    node = Node(child_path, value, ephemeral, data_watch, children_watch)
    self.children[basename] = node

    if self._children_watch is not None:
      event = WatchedEvent(type=EventType.CHILD,
                           state=KeeperState.CONNECTED,
                           path=self.path)
      self._children_watch(list(self.children.keys()), event)

    return node

  def get_or_create_child(self, path):
    name = os.path.basename(path)
    if name in self.children:
      return self.children[name]
    else:
      return self.create_child(path)

  def get_child(self, path):
    return self.children.get(os.path.basename(path), None)

  def has_child(self, path=None):
    if path is None:
      return len(self.children) > 0
    else:
      return os.path.basename(path) in self.children

  def remove_child(self, path, recursive: bool = False):
    if self.has_child(path):
      self._mtime = int(time.time())
      node = self.children[os.path.basename(path)]
      if not recursive and node.has_child():
        raise NotEmptyError(f'{path} is not empty!')
      del self.children[os.path.basename(path)]

      if self._children_watch is not None:
        event = WatchedEvent(type=EventType.CHILD,
                             state=KeeperState.CONNECTED,
                             path=self.path)
        self._children_watch(list(self.children.keys()), event=event)
    else:
      raise NoNodeError(f'{path} is not exists!')

  def __del__(self):
    event = WatchedEvent(type=EventType.DELETED,
                         state=KeeperState.CONNECTED,
                         path=self.path)
    if self._data_watch is not None:
      self._data_watch(self.value, self.state, event)
      self._data_watch = None

    for child in list(self.children.keys()):
      del self.children[child]

    if self._children_watch is not None:
      self._children_watch(list(self.children.keys()), event=event)
      self._children_watch = None

    del self.path, self.value, self.ephemeral, self._ctime, self._mtime, self._version
    del self._data_watch, self._children_watch, self.children


class Catalog:

  def __init__(self):
    self.root = Node(os.path.sep)
    self._data_watches = {}
    self._children_watches = {}
    self._sequence_paths = {}

  def add_data_watch(self, watch: DataWatch):
    self._data_watches[watch.path] = watch
    try:
      node = self.get(watch.path)
      node.set_data_watch(watch)
    except Exception:
      pass

  def add_children_watch(self, watch: ChildrenWatch):
    self._children_watches[watch.path] = watch
    try:
      node = self.get(watch.path)
      node.set_children_watch(watch)
    except Exception as e:
      pass

  def ensure_path(self, path: str) -> Node:
    items = [item for item in path.split(os.path.sep) if len(item) > 0]
    cpath, cnode = '', self.root
    for item in items:
      cpath = f'{cpath}{os.path.sep}{item}'
      cnode = cnode.get_or_create_child(cpath)
      if cnode.path in self._data_watches and cnode._data_watch is None:
        cnode._data_watch = self._data_watches[cnode.path]
      if cnode.path in self._children_watches and cnode._children_watch is None:
        cnode._children_watch = self._children_watches[cnode.path]

    return cnode

  def create(self,
             path: str,
             value: bytes = b'',
             ephemeral: bool = False,
             makepath: bool = False,
             sequence: bool = False):
    if sequence:
      if path in self._sequence_paths:
        self._sequence_paths[path] += 1
        path = f'{path}{self._sequence_paths[path]:010d}'
      else:
        self._sequence_paths[path] = 0
        path = f'{path}{0:010d}'

    dirname = os.path.dirname(path)
    if makepath:
      pnode = self.ensure_path(dirname)
    else:
      pnode = self.get(dirname)

    if pnode.has_child(path):
      raise NodeExistsError(f'{path} Exists!')
    else:
      data_watch = self._data_watches.get(path, None)
      children_watch = self._children_watches.get(path, None)
      return pnode.create_child(path, value, ephemeral, data_watch,
                                children_watch)

  def delete(self, path: str, recursive: bool = False):
    dirname = os.path.dirname(path)
    pnode = self.get(dirname)
    pnode.remove_child(path, recursive)

  def set(self, path: str, value: bytes):
    self.get(path).set(value)

  def get(self, path: str) -> Node:
    items = [item for item in path.split(os.path.sep) if len(item) > 0]
    cpath, cnode = '', self.root
    for item in items:
      cpath = f'{cpath}{os.path.sep}{item}'
      cnode = cnode.get_child(cpath)
      if cnode is None:
        raise NoNodeError(f'{path} is not exists!')

    return cnode


class FakeKazooClient:

  def __init__(self, zk_server: str = None):
    self._zk_server = zk_server
    self._catalog: Optional[Catalog] = None

    self.DataWatch = partial(DataWatch, self)
    self.ChildrenWatch = partial(ChildrenWatch, self)
    self.Election = partial(Election, self)

  def ensure_path(self, path: str):
    self._catalog.ensure_path(path)

  def start(self):
    self._catalog = Catalog()

  def create(self,
             path: str,
             value: bytes = b'',
             acl=None,
             ephemeral: bool = False,
             makepath: bool = False,
             include_data: bool = False,
             sequence: bool = False):
    node = self._catalog.create(path, value, ephemeral, makepath, sequence)

    if include_data:
      return node.path, node.state
    else:
      return node.path

  def delete(self, path: str, recursive: bool = True):
    self._catalog.delete(path, recursive)

  def set(self, path: str, value: bytes):
    self._catalog.set(path, value)

  def get(self, path: str):
    node = self._catalog.get(path)
    return node.value, node.state

  def exists(self, path: str):
    try:
      node = self._catalog.get(path)
      return True
    except NoNodeError as e:
      return False

  def get_children(self, path: str, include_data=False):
    node = self._catalog.get(path)
    children = list(node.children.keys())
    if include_data:
      return children, node.state
    else:
      return children

  def retry(self, func, *args, **kwargs):
    return func(*args, **kwargs)

  def stop(self):
    self._catalog = None

  def close(self):
    if self._catalog is not None:
      self.stop()

  def add_listener(self, listener):
    pass
