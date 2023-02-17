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
from queue import Queue, Empty
import socket
import threading
from typing import Dict, List
import ipaddress

import netifaces


class NodeAliveChecker:

  def __init__(self, addrs: List, timeout: int = 1, num_thread: int = 10):
    self._addrs = addrs
    self._timeout = timeout
    self._num_thread = num_thread

    self._lock = threading.Lock()
    self._alive = set()
    self._dead = set()

    self._q = Queue()
    for addr in self._addrs:
      self._q.put(addr)
    self._start()

  def _ping(self, addr):
    skt = None
    try:
      ip, port = addr.rsplit(':', 1)
      ip = ip.strip('[]')
      is_ipv6 = is_ipv6_address(ip)
      skt = socket.socket(socket.AF_INET6 if is_ipv6 else socket.AF_INET,
                          socket.SOCK_STREAM)
      skt.settimeout(self._timeout)

      skt.connect((ip, int(port)))
      with self._lock:
        self._alive.add(addr)
    except Exception as err:
      print("cannot connect to {}, because {}".format(addr, err))
      with self._lock:
        self._dead.add(addr)
    finally:
      if skt:
        skt.close()

  def _check_open(self):
    try:
      while True:
        addr = self._q.get_nowait()
        self._ping(addr)
    except Empty as err:
      pass

  def _start(self):
    threads = []
    for i in range(self._num_thread):
      t = threading.Thread(target=self._check_open)
      t.start()
      threads.append(t)

    for t in threads:
      t.join()

  def all_nodes_alive(self):
    with self._lock:
      return len(self._dead) == 0

  def get_dead_nodes(self):
    with self._lock:
      return list(self._dead)

  def get_alive_nodes(self):
    with self._lock:
      return list(self._alive)

  def get_addrs(self):
    with self._lock:
      return self._addrs


def is_ipv6_address(ip: str):
  try:
    ip_obj = ipaddress.ip_address(ip)
  except ValueError:
    return False
  return ip_obj.version == 6


def concat_ip_and_port(ip: str, port: int):
  if not is_ipv6_address(ip):
    return f"{ip}:{port}"
  else:
    return f"[{ip}]:{port}"


def _get_eth_netinterfaces():
  return [i for i in netifaces.interfaces() if i.startswith("eth")]


def is_ipv4_supported():
  l = _get_eth_netinterfaces()
  for i in l:
    addrs = netifaces.ifaddresses(i)
    if netifaces.AFINET in addrs:
      return True
  return False


def get_local_server_addr(port: int):
  """Given a port. Returns an addr.
  In the machine that supports IPv4, it is equivalent to gethostbyname(gethostname()).
  """
  l = _get_eth_netinterfaces()
  for i in l:
    addrs = netifaces.ifaddresses(i)
    if netifaces.AF_INET in addrs:
      return concat_ip_and_port(addrs[netifaces.AF_INET][0]["addr"], port)
    elif netifaces.AF_INET6 in addrs:
      return concat_ip_and_port(addrs[netifaces.AF_INET6][0]["addr"], port)
  return None


class AddressFamily(object):
  IPV4 = 'ipv4'
  IPV6 = 'ipv6'
