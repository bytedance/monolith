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
import unittest
from unittest import mock
import random
import time

from monolith.native_training import net_utils

_SOCKET = 'monolith.native_training.net_utils.socket.socket'
_FAILED_TIME = 0
_DEAD_SET = set()


class socket:
  AF_INET = -1
  SOCK_STREAM = -1

  def __init__(self, family=-1, stype=-1):
    self._family = family
    self._stype = stype
    self._timeout = 1

  def settimeout(self, timeout):
    self._timeout = timeout

  def connect(self, addr):
    ip, port = addr
    sleep = random.uniform(0, 2 * self._timeout)
    time.sleep(sleep)
    print('sleep {}, connect to {}:{}'.format(sleep, ip, port))
    if sleep > self._timeout:
      global _FAILED_TIME
      global _DEAD_SET
      _FAILED_TIME += 1
      tmp_add = ':'.join([ip, str(port)])
      _DEAD_SET.add(tmp_add)
      raise RuntimeError('{}:{} connect error'.format(ip, port))

  def close(self):
    pass

  @classmethod
  def socket(cls, family, stype):
    return socket(family, stype)


class NetUtilsTest(unittest.TestCase):

  def test_basic(self):
    with mock.patch(_SOCKET) as tmp_socket:
      tmp_socket.return_value = socket()

      addrs = [
          'localhost:1233', 'localhost:1234', 'localhost:1235',
          'localhost:1236', 'localhost:1238'
      ]
      alive_checker = net_utils.NodeAliveChecker(addrs)
      self.assertEqual(set(alive_checker.get_addrs()), set(addrs))

      self.assertEqual(len(alive_checker.get_alive_nodes()), 5 - _FAILED_TIME)
      self.assertEqual(len(alive_checker.get_dead_nodes()), _FAILED_TIME)
      self.assertEqual(set(alive_checker.get_alive_nodes()),
                       set(addrs) - _DEAD_SET)
      self.assertEqual(set(alive_checker.get_dead_nodes()), _DEAD_SET)

      self.assertEqual(alive_checker.all_nodes_alive(), _FAILED_TIME == 0)

  def test_get_another_nic_ip(self):
    net_utils.get_another_nic_ip()


if __name__ == "__main__":
  unittest.main()
