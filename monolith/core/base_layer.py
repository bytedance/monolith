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

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Base class for all layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from monolith.core.hyperparams import InstantiableParams
from monolith.core.py_utils import NestedMap
from collections import defaultdict

_layer_loss = defaultdict(dict)  # _layer_loss[graph][name]
_name_inuse = defaultdict(int)


class BaseLayer(object):

  @classmethod
  def params(cls):
    """Returns the layer params."""
    p = InstantiableParams(cls)
    p.define('name', get_uname(cls.__name__), 'Name of this layer object.')
    return p

  def __init__(self, params):
    """Layer constructor.
        Args:
            params: A params used to construct this layer.
        """
    assert params.name, ('Layer params for %s must have a "name"' %
                         self.__class__.__name__)

    # Child layers created by this layer through CreateChild/CreateChildren.
    self._private_children = NestedMap()

  @property
  def children(self):
    """Returns children layers of this layer in a `.NestedMap`."""
    return self._private_children

  def __getattr__(self, name):
    """Returns the child layer of the given name."""
    if name == '_private_children':
      raise AttributeError(
          'pre-mature access to __getattr__ before _private_children '
          'is created.')
    if name in self._private_children:
      return self._private_children[name]
    elif (hasattr(type(self), name) and
          isinstance(getattr(type(self), name), property)):
      # There was an AttributeError raised by a property getter.
      # Call property getter again directly to raise the same error.
      return getattr(type(self), name).fget(self)
    else:
      raise AttributeError('%s is not a sub-layer of %s.' % (name, self))

  def __call__(self, *args, **kwargs):
    """Forwards call to FProp."""
    return self.fprop(*args, **kwargs)

  def fprop(self, *args, **kwargs):
    """Forward propagation.
        The central interface that subclasses should implement. The caller
        calls `FProp`.
        Args:
            *args: List args.
            **kwargs: Keyward args.
        """
    del args
    del kwargs
    raise NotImplementedError('Abstract method of %s' % self)

  def create_child(self, name, params):
    """Create a sub layer.
            The created sub layer can be accessed by `name`. E.g.::
                self.create_child('foo', ...)
                self.foo.fprop...
        or::
                self.children['foo'].fprop...
                self.children.foo.fprop...
        Args:
            name: Sub layer name used as key to access it as attribute
            params: `Hyperparams` object to instantiate a layer.
        """
    # self._check_name(name)
    if not params.name:
      params.name = self.p.name
    # p = copy_params_to(self.p, params.copy())
    # params = copy_params_to(self.p, params.copy())
    child = params.instantiate()
    self._private_children[name] = child

  def create_children(self, name, params):
    """Create a list or dict of sub layers.

        The created sub layer list can be accessed by `name`. E.g.:
            self.create_children('foo', ...)
            self.foo[10].FProp...
        or:
            self.children['foo'][10].Fprop...
            self.children.foo[10].Fprop...

        Args:
            name: The name for the sub layers, which is used as the key into
                vars/theta.
            params: a list of `Hyperparams` objects to create.
        """
    self._private_children[name] = []
    for index, param in enumerate(params):
      if not param.name:
        param.name = '%s_%d' % (name, index)
      child = param.instantiate()
      self._private_children[name].append(child)


def get_uname(name):
  if name in _name_inuse:
    _name_inuse[name] += 1
    return "{name}_{idx}".format(name=name, idx=_name_inuse[name])
  else:
    return name


def add_layer_loss(name, loss):
  graph_layer_loss = _layer_loss[tf.compat.v1.get_default_graph()]
  if name in graph_layer_loss:
    graph_layer_loss[name] += loss
  else:
    graph_layer_loss[name] = loss


def get_layer_loss():
  return _layer_loss[tf.compat.v1.get_default_graph()]
