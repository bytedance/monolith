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
"""Defines Params base class, used for defining class/function parameters."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import re
import six
from inspect import Parameter, signature

import tensorflow as tf


def _is_named_tuple(x):
  """Returns whether an object is an instance of a collections.namedtuple.
    Examples::
      _is_named_tuple((42, 'hi')) ==> False
      Foo = collections.namedtuple('Foo', ['a', 'b'])
      _is_named_tuple(Foo(a=42, b='hi')) ==> True
    Args:
    x: The object to check.
    """
  return isinstance(x, tuple) and hasattr(x, '_fields')


class _SortedDict(dict):
  """A dict with a __repr__ that is always sorted by key."""

  def __repr__(self):
    return '{' + ', '.join(
        '%r: %r' % item for item in sorted(self.items())) + '}'


class _Param(object):
  """Stores data for a single parameter."""

  def __init__(self, name, default_value, description):
    self._name = name
    self._value = default_value
    self._description = description

  def __eq__(self, other):
    # pylint: disable=protected-access
    return self._name == other._name and self._value == other._value

  # Deep copy the value only if it is supported.
  def __deepcopy__(self, memo):
    if isinstance(self._value, (tf.Tensor)):
      # In case self._value is a tensor, let's just make a reference.
      value = self._value
    else:
      value = copy.deepcopy(self._value, memo)
    p = _Param(self._name, value, self._description)
    # Q(yonghui): Is this the right use of memo.
    memo[id(self)] = p
    return p

  def to_string(self, nested_depth):
    """Prints the parameter as a string."""

    def GetRepr(val):
      """Get the representation of `val`."""
      if isinstance(val, Params):
        return _SortedDict({k: GetRepr(v) for k, v in val.iter_params()})
      if isinstance(val, dict):
        return _SortedDict({k: GetRepr(v) for k, v in six.iteritems(val)})
      if isinstance(val, (list, tuple)) and not _is_named_tuple(val):
        # NB: this constructor signature works for tuples, but not namedtuples.
        return type(val)([GetRepr(v) for v in val])
      # NOTE(markmurphy): I introduced Repr() because it's impossible (afaik) to
      # overwrite the __str__ or __repr__ method of a types.FunctionType object.
      if hasattr(val, 'Repr'):
        return val.Repr()
      return val

    nested_indent = '  ' * nested_depth
    if isinstance(self._value, Params):
      # pylint: disable=protected-access
      value_str = self._value._to_string(nested_depth)
    elif isinstance(self._value, six.string_types):
      return '%s%s: "%s"' % (nested_indent, self._name, self._value)
    else:
      value_str = str(GetRepr(self._value))
    return '%s%s: %s' % (nested_indent, self._name, value_str)

  def set(self, value):
    # Note that we don't make a copy of Params objects.
    # TODO(sadovsky): Maybe add safeguard to ensure that Params object is not
    # owned by other Params objects.
    self._value = value

  def get(self):
    return self._value


def copy_params_to(from_p, to_p, skip=None):
  """Copy from one Params to another, with optional skipped params.

  Args:
    from_p: Source params to copy from.
    to_p: Destination params to copy to.
    skip: If not None, a list of strings of param names to skip.

  Returns:
    None
  """
  for n, p in from_p.iter_params():
    if skip and n in skip:
      continue
    if isinstance(p, Params):
      to_p.set(**{n: p.copy()})
    else:
      to_p.set(**{n: p})
  return to_p


class Params(object):
  """Stores data for a set of parameters.
    Provides attribute-based API, e.g. "params.foo = 5".
    Uses internal {'name': Params} dict for storing parameter data.
    """

  def __init__(self):
    self.__dict__['_immutable'] = False
    self._params = {}  # name => _Param

  def __setattr__(self, name, value):
    if self._immutable:
      raise TypeError('This Params instance is immutable.')
    if name == '_params' or name == '_immutable':
      self.__dict__[name] = value
    else:
      try:
        self._params[name].set(value)
      except KeyError:
        raise AttributeError(self._key_error_string(name))

  def __getattr__(self, name):
    if name == '_params' or name == '_immutable':
      return self.__dict__[name]
    try:
      return self._params[name].get()
    except KeyError:
      # cPickle expects __getattr__ to raise AttributeError, not KeyError.
      raise AttributeError(self._key_error_string(name))

  def __setitem__(self, name, value):
    self.__setattr__(name, value)

  def __getitem__(self, key):
    return self.__getattr__(key)

  def __dir__(self):
    return sorted(self._params.keys())

  def __contains__(self, name):
    return name in self._params

  def __len__(self):
    return len(self._params)

  # Note: This gets called by Params.__eq__() on nested Params objects.
  def __eq__(self, other):
    return isinstance(other, Params) and self._params == other._params  # pylint: disable=protected-access

  def __ne__(self, other):
    return not self == other

  def __str__(self):
    return self._to_string(0)

  def _to_string(self, nested_depth):
    # Note: We use iteritems() below so as to sort by name.
    sorted_param_strs = [
        v.to_string(nested_depth + 1)
        for (_, v) in sorted(six.iteritems(self._params))
    ]
    nested_indent = '  ' * nested_depth
    return '{\n%s\n%s}' % ('\n'.join(sorted_param_strs), nested_indent)

  # Override __deepcopy__ so that copy.deepcopy(self._params) properly
  # deep-copies nested Params objects.
  # TODO(sadovsky): Is it okay not to touch memo?
  def __deepcopy__(self, unused_memo):
    return self.copy()

  def _similar_keys(self, name):
    """Return a list of params keys that are similar to name."""

    def _overlaps(name, key):
      """The fraction of 3-char substrings in <name> that appear in key."""
      matches = 0
      trials = 0
      for i in range(len(name) - 3):
        trials += 1
        if name[i:i + 3] in key:
          matches += 1
      if trials:
        return float(matches) / trials
      return 0

    if '_params' in self.__dict__:
      return [key for key in self._params if _overlaps(name, key) > 0.5]
    return []

  def _key_error_string(self, name):
    similar = self._similar_keys(name)
    if similar:
      return name + ' (did you mean: [%s])' % (','.join(sorted(similar)))
    return name

  def copy(self):
    return self._copy_to(type(self)())

  def _copy_to(self, res):
    # pylint: disable=protected-access
    res._params = copy.deepcopy(self._params)
    res._immutable = self._immutable
    # pylint: enable=protected-access
    return res

  # TODO(sadovsky):
  # - Maybe let users specify whether this parameter is allowed to have
  #   value=None, and if not, assert on get(), like required proto field.
  # - Maybe enforce that value is one of
  #     {number, string, bool, list, dict, Params}.
  def define(self, name, default_value, description):
    """Defines a parameter.
        Args:
          name: The parameter name. Must only contain lowercase letters, numbers,
              and underscores. Must start with lowercase letter.
          default_value: Default value for this parameter. May be None.
          description: String description of this parameter.
        Raises:
          AttributeError: If parameter 'name' is already defined.
        """
    if self._immutable:
      raise TypeError('This Params instance is immutable.')
    assert name is not None and isinstance(
        name, six.string_types) and (re.match('^[a-z][a-z0-9_]*$', name)
                                     is not None)
    if name in self._params:
      raise AttributeError('Parameter %s is already defined' % name)
    self._params[name] = _Param(name, default_value, description)

  def contain(self, name):
    return name in self._params

  def freeze(self):
    """Marks this Params as immutable."""
    self._immutable = True

  def is_immutable(self):
    """Return whether this Params is immutable."""
    return self._immutable

  def _get_nested(self, name):
    """Returns nested param by its name."""
    parts = name.split('.')
    curr = self
    for i, part in enumerate(parts[:-1]):
      # get the value (nested Params object) associated with name 'part'.
      try:
        is_list = re.match(r'^(.+)\[(.+)\]$', part)
        if is_list:
          part = is_list.group(1)
          list_index = int(is_list.group(2))
        # pylint: disable=protected-access
        curr = curr._params[part].get()
        if is_list:
          curr = curr[list_index]
      except KeyError:
        raise AttributeError('.'.join(parts[:i + 1]))
      assert isinstance(curr, Params), ('Cannot introspect %s for %s' %
                                        (type(curr), '.'.join(parts[:i + 1])))
    return curr, parts[-1]

  def set(self, **kwargs):
    """Sets multiple parameters.
        Dots in names indicate navigation into nested Params objects. We do not
        allow navigation into lists or dicts, and may ban these types altogether in
        favor of string representations.
        Args:
          **kwargs: Name-value pairs to set.
        Returns:
          self
        """
    if self._immutable:
      raise TypeError('This Params instance is immutable: %s' % self)
    for name, value in six.iteritems(kwargs):
      # get nested param.
      param, key = self._get_nested(name)
      # Update the value associated with key.
      try:
        # pylint: disable=protected-access
        param._params[key].set(value)
      except KeyError:
        raise AttributeError(self._key_error_string(name))
    return self

  def get(self, name):
    """get parameter.
        Dots in names indicate navigation into nested Params objects. We do not
        allow navigation into lists or dicts, and may ban these types altogether in
        favor of string representations.
        Args:
          name: (str) Name.
        Returns:
          value.
        Raises:
          AttributeError: if parameter is not found
        """
    param, key = self._get_nested(name)
    # get the value associated with key.
    try:
      # pylint: disable=protected-access
      return param._params[key].get()
    except KeyError:
      raise AttributeError(self._key_error_string(name))

  def delete(self, *args):
    """Deletes multiple parameters.
        Dots in names indicate navigation into nested Params objects. We do not
        allow navigation into lists or dicts, and may ban these types altogether in
        favor of string representations.
        Args:
          *args: List of names.
        Returns:
          self
        """
    if self._immutable:
      raise TypeError('This Params instance is immutable.')
    for name in args:
      # get nested param.
      param, key = self._get_nested(name)
      # delete the key.
      try:
        # pylint: disable=protected-access
        del param._params[key]
      except KeyError:
        raise AttributeError(self._key_error_string(name))
    return self

  def iter_params(self):
    """Pythonic dict-like iteration."""
    for name, param in six.iteritems(self._params):
      yield (name, param.get())


allowed_kwargs = {
    'input_dim', 'input_shape', 'batch_input_shape', 'weights',
    'activity_regularizer', 'autocast', 'implementation', 'name'
}


def _inverted_index(ips: 'InstantiableParams', idx):
  for name, item in ips.iter_params():
    if isinstance(item, Params):
      _inverted_index(item, idx)
    else:
      idx[name] = item


class InstantiableParams(Params):
  """Params which can be instantiated.
    When using InstantiableParams, callers must provide a class which supports
    initialization using a Params instance.
    This covers a common use case of Params to hold a configuration for a given
    class.
    """

  def __init__(self, cls=None):
    super(InstantiableParams, self).__init__()
    self.define('cls', cls, 'Cls that this param object is associated with.')

  def instantiate(self):
    """instantiate an instance that this Params is configured for."""
    assert self.cls is not None
    # The class initializer is expected to support initialization using Params.

    parameters = signature(self.cls.__init__).parameters
    if len(parameters) == 2 and hasattr(self.cls,
                                        'params') and 'params' in parameters:
      return self.cls(self)
    else:
      index, args = {}, {}
      _inverted_index(self, index)
      for name, p in parameters.items():
        if p.kind in {Parameter.VAR_KEYWORD, Parameter.VAR_POSITIONAL}:
          continue

        if name not in {'self', 'cls'} and name in index:
          args[name] = index[name]

      for key in allowed_kwargs:
        if key in index and index[key] is not None:
          args[key] = index[key]

      return self.cls(**args)

  def copy(self):
    return self._copy_to(type(self)(self.cls))


def update_params(ips: Params, args):
  for key, value in ips.iter_params():
    if isinstance(value, Params):
      update_params(value, args)
    else:
      if key in args:
        ips[key] = args.pop(key)
