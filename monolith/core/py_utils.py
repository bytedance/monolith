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
"""Common utilities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import re
import six

_NAME_PATTERN = re.compile('[A-Za-z_][A-Za-z0-9_]*')


class NestedMap(dict):
  """A simple helper to maintain a dict.
  It is a sub-class of dict with the following extensions/restrictions:
    - It supports attr access to its members (see examples below).
    - Member keys have to be valid identifiers.
  E.g.::
      >>> foo = NestedMap()
      >>> foo['x'] = 10
      >>> foo.y = 20
      >>> assert foo.x * 2 == foo.y
  """

  # Disable pytype attribute checking.
  _HAS_DYNAMIC_ATTRIBUTES = True
  # keys in this list are not allowed in a NestedMap.
  _RESERVED_KEYS = set(dir(dict))
  # sentinel value for deleting keys used in Filter.
  _DELETE = object()

  def __init__(self, *args, **kwargs):
    super(NestedMap, self).__init__(*args, **kwargs)
    for key in self.keys():
      assert isinstance(key, six.string_types), (
          'Key in a NestedMap has to be a six.string_types. Currently type: %s,'
          ' value: %s' % (str(type(key)), str(key)))
      NestedMap.CheckKey(key)
      assert key not in NestedMap._RESERVED_KEYS, ('%s is a reserved key' % key)

  def __setitem__(self, key, value):
    # Make sure key is a valid expression and is not one of the reserved
    # attributes.
    assert isinstance(key, six.string_types), (
        'Key in a NestedMap has to be a six.string_types. Currently type: %s, '
        'value: %s' % (str(type(key)), str(key)))
    NestedMap.CheckKey(key)
    assert key not in NestedMap._RESERVED_KEYS, ('%s is a reserved key' % key)
    super(NestedMap, self).__setitem__(key, value)

  def __setattr__(self, name, value):
    self.__setitem__(name, value)

  def __getattr__(self, name):
    try:
      return self[name]
    except KeyError as e:
      raise AttributeError('%s; available attributes: %s' %
                           (e, sorted(list(self.keys()))))

  def __delattr__(self, name):
    try:
      del self[name]
    except KeyError as e:
      raise AttributeError('%s; available attributes: %s' %
                           (e, sorted(list(self.keys()))))

  def copy(self):  # Don't delegate w/ super: dict.copy() -> dict.
    return NestedMap(self)

  def __deepcopy__(self, unused_memo):
    """Deep-copies the structure but not the leaf objects."""
    return self.DeepCopy()

  def DeepCopy(self):
    """Deep-copies the structure but not the leaf objects."""
    return self.Pack(self.Flatten())

  @staticmethod
  def FromNestedDict(x):
    """Converts every dict in nested structure 'x' to a NestedMap."""
    if isinstance(x, dict):
      res = NestedMap()
      for k, v in six.iteritems(x):
        res[k] = NestedMap.FromNestedDict(v)
      return res
    elif isinstance(x, (list, tuple)):
      return type(x)(NestedMap.FromNestedDict(v) for v in x)
    else:
      return x

  @staticmethod
  def CheckKey(key):
    """Asserts that key is valid NestedMap key."""
    if not (isinstance(key, six.string_types) and _NAME_PATTERN.match(key)):
      raise ValueError('Invalid NestedMap key \'{}\''.format(key))

  def GetItem(self, key):
    """Gets the value for the nested `key`.
    Note that indexing lists is not supported, names with underscores will be
    considered as one key.
    Args:
      key: str of the form
        `([A-Za-z_][A-Za-z0-9_]*)(.[A-Za-z_][A-Za-z0-9_]*)*.`.
    Returns:
      The value for the given nested key.
    Raises:
      KeyError if a key is not present.
    """
    current = self
    # Note: This can't support lists. List keys are ambiguous as underscore is
    # not reserved for list indexing but also allowed to be used in keys.
    # E.g., this is a valid nested map where the key 'a_0' is not well defined
    # {'a_0': 3, 'a': [4]}.
    for k in key.split('.'):
      current = current[k]
    return current

  def Get(self, key, default=None):
    """Gets the value for nested `key`, returns `default` if key does not exist.
    Note that indexing lists is not supported, names with underscores will be
    considered as one key.
    Args:
      key: str of the form
        `([A-Za-z_][A-Za-z0-9_]*)(.[A-Za-z_][A-Za-z0-9_]*)*.`.
      default: Optional default value, defaults to None.
    Returns:
      The value for the given nested key or `default` if the key does not exist.
    """
    try:
      return self.GetItem(key)
    # TypeError is raised when an intermediate item is a list and we try to
    # access an element of it with a string.
    except (KeyError, TypeError):
      return default

  def Set(self, key, value):
    """Sets the value for a nested key.
    Note that indexing lists is not supported, names with underscores will be
    considered as one key.
    Args:
      key: str of the form
        `([A-Za-z_][A-Za-z0-9_]*)(.[A-Za-z_][A-Za-z0-9_]*)*.`.
      value: The value to insert.
    Raises:
      ValueError if a sub key is not a NestedMap or dict.
    """
    current = self
    sub_keys = key.split('.')
    for i, k in enumerate(sub_keys):
      self.CheckKey(k)
      # We have reached the terminal node, set the value.
      if i == (len(sub_keys) - 1):
        current[k] = value
      else:
        if k not in current:
          current[k] = NestedMap()
        if not isinstance(current[k], (dict, NestedMap)):
          raise ValueError('Error while setting key {}. Sub key "{}" is of type'
                           ' {} but must be a dict or NestedMap.'
                           ''.format(key, k, type(current[k])))
        current = current[k]

  def _RecursiveMap(self, fn, flatten=False):
    """Traverse recursively into lists and NestedMaps applying `fn`.
    Args:
      fn: The function to apply to each item (leaf node).
      flatten: If true, the result should be a single flat list. Otherwise the
        result will have the same structure as this NestedMap.
    Returns:
      The result of applying fn.
    """

    def Recurse(v, key=''):
      """Helper function for _RecursiveMap."""
      if isinstance(v, NestedMap):
        ret = [] if flatten else NestedMap()
        deleted = False
        for k in sorted(v.keys()):
          res = Recurse(v[k], key + '.' + k if key else k)
          if res is self._DELETE:
            deleted = True
            continue
          elif flatten:
            ret += res
          else:
            ret[k] = res
        if not ret and deleted:
          return self._DELETE
        return ret
      elif isinstance(v, list):
        ret = []
        deleted = False
        for i, x in enumerate(v):
          res = Recurse(x, '%s[%d]' % (key, i))
          if res is self._DELETE:
            deleted = True
            continue
          elif flatten:
            ret += res
          else:
            ret.append(res)
        if not ret and deleted:
          return self._DELETE
        return ret
      else:
        ret = fn(key, v)
        if flatten:
          ret = [ret]
        return ret

    res = Recurse(self)
    if res is self._DELETE:
      return [] if flatten else NestedMap()
    return res

  def Flatten(self):
    """Returns a list containing the flattened values in the `.NestedMap`.
    Unlike py_utils.Flatten(), this will only descend into lists and NestedMaps
    and not dicts, tuples, or namedtuples.
    """
    return self._RecursiveMap(lambda _, v: v, flatten=True)

  def FlattenItems(self):
    """Flatten the `.NestedMap` and returns <key, value> pairs in a list.
    Returns:
      A list of <key, value> pairs, where keys for nested entries will be
      represented in the form of `foo.bar[10].baz`.
    """
    return self._RecursiveMap(lambda k, v: (k, v), flatten=True)

  def Pack(self, lst):
    """Returns a copy of this with each value replaced by a value in lst."""
    assert len(self.FlattenItems()) == len(lst)
    v_iter = iter(lst)
    return self._RecursiveMap(lambda unused_k, unused_v: next(v_iter))

  def Transform(self, fn):
    """Returns a copy of this `.NestedMap` with fn applied on each value."""
    return self._RecursiveMap(lambda _, v: fn(v))

  def IsCompatible(self, other):
    """Returns true if self and other are compatible.
    If x and y are two compatible `.NestedMap`, `x.Pack(y.Flatten())` produces y
    and vice versa.
    Args:
      other: Another `.NestedMap`.
    """
    items = self._RecursiveMap(lambda k, _: k, flatten=True)
    other_items = other._RecursiveMap(lambda k, _: k, flatten=True)  # pylint: disable=protected-access
    return items == other_items

  def Filter(self, fn):
    """Returns a copy with entries where fn(entry) is True."""
    return self.FilterKeyVal(lambda _, v: fn(v))

  def FilterKeyVal(self, fn):
    """Returns a copy of this `.NestedMap` filtered by fn.
    If fn(key, entry) is True, the entry is copied into the returned NestedMap.
    Otherwise, it is not copied.
    Args:
      fn: a callable of (string, entry)->boolean.
    Returns:
      A `.NestedMap` contains copied entries from this `'.NestedMap`.
    """
    return self._RecursiveMap(lambda k, v: v if fn(k, v) else self._DELETE)

  def _ToStrings(self):
    """Returns debug strings in a list for this `.NestedMap`."""
    kv = self.FlattenItems()
    maxlen = max([len(k) for k, _ in kv]) if kv else 0
    return sorted([k + ' ' * (4 + maxlen - len(k)) + str(v) for k, v in kv])

  def DebugString(self):
    """Returns a debug string for this `.NestedMap`."""
    return '\n'.join(self._ToStrings())

  def VLog(self, level=None, prefix=None):
    """Logs the debug string at the level."""
    if level is None:
      level = 0
    if prefix is None:
      prefix = 'nmap: '
    for l in self._ToStrings():
      tf.logging.vlog(level, '%s %s', prefix, l)
