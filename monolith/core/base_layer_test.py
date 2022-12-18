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

from monolith.core import base_layer
from monolith.core import hyperparams
import unittest


class BaseLayerTest(unittest.TestCase):

  def test_create_child(self):
    layer_p = base_layer.BaseLayer.params()
    layer_p.name = 'test'
    layer = layer_p.instantiate()
    layer._disable_create_child = False  # pylint: disable=protected-access
    layer.create_child(name='a', params=layer_p)
    self.assertTrue('a' in layer.children)

  def test_create_children(self):
    layer_p = base_layer.BaseLayer.params()
    layer_p.name = 'test'
    layer = layer_p.instantiate()
    layer._disable_create_child = False  # pylint: disable=protected-access
    layer.create_children(name='a', params=[layer_p, layer_p])
    self.assertTrue('a' in layer.children)
    self.assertEqual(len(layer.a), 2)


if __name__ == '__main__':
  unittest.main()
