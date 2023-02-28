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

import importlib
import os
import threading


class _Lib:
  """A lib that will delay import when used."""

  def __init__(self):
    self._lib = None
    self._lock = threading.Lock()

  @property
  def lib(self):
    with self._lock:
      if self._lib is None:
        if self.enable_bps:
          self._lib = importlib.import_module("byteps.tensorflow")
        else:
          self._lib = importlib.import_module("horovod.tensorflow")
    return self._lib

  @property
  def enable_bps(self):
    return int(os.getenv("MONOLITH_WITH_BYTEPS", "0"))

  def init(self):
    return self.lib.init()

  def rank(self):
    return self.lib.rank()

  def size(self):
    return self.lib.size()

  def allgather(self, *args, **kwargs):
    return self.lib.allgather(*args, **kwargs)

  def broadcast(self, *args, **kwargs):
    return self.lib.broadcast(*args, **kwargs)

  def BroadcastGlobalVariablesHook(self, *args, **kwargs):
    return self.lib.BroadcastGlobalVariablesHook(*args, **kwargs)


_lib = _Lib()


def __getattr__(name):
  """Export all method in _Lib class"""
  return getattr(_lib, name)
