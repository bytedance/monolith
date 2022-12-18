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
from kazoo.exceptions import NoNodeError, NodeExistsError

import unittest

from monolith.agent_service.mocked_zkclient import FakeKazooClient


class MockedZKClientTest(unittest.TestCase):
  client = None

  @classmethod
  def setUpClass(cls) -> None:
    cls.client = FakeKazooClient()
    cls.client.start()

  @classmethod
  def tearDownClass(cls) -> None:
    cls.client.stop()

  def test_create(self):
    path = '/monolith/zk/data'
    try:
      real_path = self.client.create(path, makepath=True)
      self.assertEqual(real_path, path)
    except NoNodeError as e:
      logging.info(f'{e}')
    except NodeExistsError as e:
      logging.info(f'{e}')

  def test_set_get(self):
    path = '/monolith/zk/data'
    data = b'hi, I am Fitz!'
    try:
      real_path, state = self.client.create(path,
                                            makepath=True,
                                            include_data=True)
      self.assertEqual(real_path, path)
    except NoNodeError as e:
      logging.info(f'{e}')
    except NodeExistsError as e:
      logging.info(f'{e}')

    not_exists_path = f"{path}/error"
    try:
      self.client.set(not_exists_path, data)
    except NoNodeError as e:
      logging.error(f'{e}')

    self.client.set(path, b'hi, I am Fitz!')

    try:
      gdata, _ = self.client.get(not_exists_path)
      self.assertEqual(gdata, data)
    except NoNodeError as e:
      logging.error(f'{e}')

    gdata, state = self.client.get(path)
    self.assertEqual(gdata, data)

  def test_delete(self):
    path = '/monolith/zk/data'
    try:
      real_path = self.client.create(path, makepath=True)
      self.assertEqual(real_path, path)
    except NoNodeError as e:
      logging.info(f'{e}')
    except NodeExistsError as e:
      logging.info(f'{e}')

    self.client.delete(path)
    self.client.delete('/monolith')

  def test_data_watch(self):
    path = '/monolith/zk/data'
    try:
      real_path = self.client.create(path, makepath=True)
      self.assertEqual(real_path, path)
    except NoNodeError as e:
      logging.info(f'{e}')
    except NodeExistsError as e:
      logging.info(f'{e}')

    def data_watch(data, state, event):
      print('data_watch', data, state, event)

    self.client.DataWatch(path=path, func=data_watch)

  def test_children_watch(self):
    path = '/monolith/zk/data'

    def children_watch(children, event):
      print('children_watch', children, event)

    self.client.ChildrenWatch(path='/monolith/zk',
                              func=children_watch,
                              send_event=True)

    try:
      real_path = self.client.create(path, makepath=True)
      self.assertEqual(real_path, path)
    except NoNodeError as e:
      logging.info(f'{e}')
    except NodeExistsError as e:
      logging.info(f'{e}')

    def data_watch(data, state, event):
      print('data_watch', data, state, event)

    self.client.DataWatch(path=path, func=data_watch)

    self.client.create('/monolith/zk/test', b'123')


if __name__ == "__main__":
  unittest.main()
