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
import unittest
from unittest import mock

import numpy as np
from absl import logging
from six.moves.http_client import OK

from monolith.native_training import consul

_HTTP_CONNECTION_TARGET = "monolith.native_training.consul.HTTPConnection"


class ConsulTest(unittest.TestCase):

  def test_lookup(self):
    with mock.patch(_HTTP_CONNECTION_TARGET) as MockHttpConnection:
      resp = mock.MagicMock()
      resp.status = OK
      data = [{"Port": 1234, "Host": "192.168.0.1", "Tags": {"index": "0"}}]
      resp.read.return_value = json.dumps(data).encode("utf-8")
      MockHttpConnection.return_value.getresponse.return_value = resp
      client = consul.Client()
      result = client.lookup("test_name")
      self.assertEqual(result, data)

  def test_register(self):
    with mock.patch(_HTTP_CONNECTION_TARGET) as MockHttpConnection:
      resp = mock.MagicMock()
      resp.status = OK
      MockHttpConnection.return_value.getresponse.return_value = resp
      client = consul.Client()
      client = client.register("test_name", 12345)

  def test_deregister(self):
    with mock.patch(_HTTP_CONNECTION_TARGET) as MockHttpConnection:
      resp = mock.MagicMock()
      resp.status = OK
      MockHttpConnection.return_value.getresponse.return_value = resp
      client = consul.Client()
      client = client.deregister("test_name", 12345)


if __name__ == "__main__":
  unittest.main()
