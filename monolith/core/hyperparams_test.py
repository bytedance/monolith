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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import enum
import functools
import re
import sys

from absl import app as absl_app
from absl import flags
from absl import logging
import tensorflow as tf
import unittest

import monolith.core.hyperparams as _params


class TestEnum(enum.Enum):
  """Test enum class."""
  A = 1
  B = 2


class ParamsTest(unittest.TestCase):

  def test_equals(self):
    params1 = _params.Params()
    params2 = _params.Params()
    self.assertTrue(params1 == params2)
    params1.define('first', 'firstvalue', '')
    self.assertFalse(params1 == params2)
    params2.define('first', 'firstvalue', '')
    self.assertTrue(params1 == params2)
    some_object = object()
    other_object = object()
    params1.define('second', some_object, '')
    params2.define('second', other_object, '')
    self.assertFalse(params1 == params2)
    params2.second = some_object
    self.assertTrue(params1 == params2)
    params1.define('third', _params.Params(), '')
    params2.define('third', _params.Params(), '')
    self.assertTrue(params1 == params2)
    params1.third.define('fourth', 'x', '')
    params2.third.define('fourth', 'y', '')
    self.assertFalse(params1 == params2)
    params2.third.fourth = 'x'
    self.assertTrue(params1 == params2)
    # Comparing params to non-param instances.
    self.assertFalse(params1 == 3)
    self.assertFalse(3 == params1)

  def test_deep_copy(self):
    inner = _params.Params()
    inner.define('alpha', 2, '')
    inner.define('tensor', tf.constant(0), '')
    outer = _params.Params()
    outer.define('beta', 1, '')
    outer.define('inner', inner, '')
    outer_copy = outer.copy()
    self.assertIsNot(outer, outer_copy)
    self.assertEqual(outer, outer_copy)
    self.assertIsNot(outer.inner, outer_copy.inner)
    self.assertEqual(outer.inner, outer_copy.inner)
    self.assertEqual(outer.inner.alpha, outer_copy.inner.alpha)
    self.assertIs(outer.inner.tensor, outer_copy.inner.tensor)

  def test_copy_params_to(self):
    source = _params.Params()
    dest = _params.Params()
    source.define('a', 'a', '')
    source.define('b', 'b', '')
    source.define('c', 'c', '')
    dest.define('a', '', '')
    _params.copy_params_to(source, dest, skip=['b', 'c'])
    self.assertEqual(source.a, dest.a)
    self.assertNotIn('b', dest)
    self.assertNotIn('c', dest)

  def test_define_existing(self):
    p = _params.Params()
    p.define('foo', 1, '')
    self.assertRaisesRegex(AttributeError, 'already defined',
                           lambda: p.define('foo', 1, ''))

  def test_legal_param_names(self):
    p = _params.Params()
    self.assertRaises(AssertionError, lambda: p.define(None, 1, ''))
    self.assertRaises(AssertionError, lambda: p.define('', 1, ''))
    self.assertRaises(AssertionError, lambda: p.define('_foo', 1, ''))
    self.assertRaises(AssertionError, lambda: p.define('Foo', 1, ''))
    self.assertRaises(AssertionError, lambda: p.define('1foo', 1, ''))
    self.assertRaises(AssertionError, lambda: p.define('foo$', 1, ''))
    p.define('foo_bar', 1, '')
    p.define('foo9', 1, '')

  def test_set_and_get(self):
    p = _params.Params()
    self.assertRaisesRegex(AttributeError, 'foo', lambda: p.set(foo=4))
    # We use setattr() because lambda cannot contain explicit assignment.
    self.assertRaisesRegex(AttributeError, 'foo', lambda: setattr(p, 'foo', 4))
    p.define('foo', 1, '')
    self.assertEqual(p.foo, 1)
    self.assertEqual(p.get('foo'), 1)
    self.assertIn('foo', p)
    self.assertNotIn('bar', p)
    p.set(foo=2)
    self.assertEqual(p.foo, 2)
    self.assertEqual(p.get('foo'), 2)
    p.foo = 3
    self.assertEqual(p.foo, 3)
    self.assertEqual(p.get('foo'), 3)
    p.delete('foo')
    self.assertNotIn('foo', p)
    self.assertNotIn('bar', p)
    self.assertRaisesRegex(AttributeError, 'foo', lambda: p.foo)
    self.assertRaisesRegex(AttributeError, 'foo', p.get, 'foo')

  def test_set_and_get_nested_param(self):
    innermost = _params.Params()
    innermost.define('delta', 22, '')
    innermost.define('zeta', 5, '')

    inner = _params.Params()
    inner.define('alpha', 2, '')
    inner.define('innermost', innermost, '')

    outer = _params.Params()
    outer.define('beta', 1, '')
    outer.define('inner', inner, '')
    outer.define('d', dict(foo='bar'), '')

    self.assertEqual(inner.alpha, 2)
    self.assertEqual(outer.beta, 1)
    self.assertEqual(outer.d['foo'], 'bar')
    self.assertEqual(outer.inner.alpha, 2)
    self.assertEqual(outer.inner.innermost.delta, 22)
    self.assertEqual(outer.inner.innermost.zeta, 5)

    self.assertEqual(inner.get('alpha'), 2)
    self.assertEqual(outer.get('beta'), 1)
    self.assertEqual(outer.get('d')['foo'], 'bar')
    self.assertEqual(outer.get('inner.alpha'), 2)
    self.assertEqual(outer.get('inner.innermost.delta'), 22)
    self.assertEqual(outer.get('inner.innermost.zeta'), 5)

    outer.set(**{'inner.alpha': 3})
    outer.set(d=dict(foo='baq'))
    outer.delete('beta')
    outer.delete('inner.innermost.zeta')

    self.assertEqual(inner.alpha, 3)
    self.assertRaisesRegex(AttributeError, 'beta', lambda: outer.beta)
    self.assertEqual(outer.d['foo'], 'baq')
    self.assertEqual(outer.inner.alpha, 3)
    self.assertEqual(outer.inner.innermost.delta, 22)
    self.assertRaisesRegex(AttributeError, 'zeta',
                           lambda: outer.inner.innermost.zeta)

    self.assertEqual(inner.get('alpha'), 3)
    self.assertRaisesRegex(AttributeError, 'beta', outer.get, 'beta')
    self.assertEqual(outer.get('d')['foo'], 'baq')
    self.assertEqual(outer.get('inner.alpha'), 3)
    self.assertEqual(outer.get('inner.innermost.delta'), 22)
    self.assertRaisesRegex(AttributeError, 'inner.innermost.zeta', outer.get,
                           'inner.innermost.zeta')

    # NOTE(igushev): Finding nested Param object is shared between Get, Set and
    # Delete, so we test only Set.
    self.assertRaisesRegex(AttributeError, r'inner\.gamma',
                           lambda: outer.set(**{'inner.gamma': 5}))
    self.assertRaisesRegex(AttributeError, r'inner\.innermost\.bad',
                           lambda: outer.set(**{'inner.innermost.bad': 5}))
    self.assertRaisesRegex(AssertionError, '^Cannot introspect',
                           lambda: outer.set(**{'d.foo': 'baz'}))

  def test_freeze(self):
    p = _params.Params()
    self.assertRaises(AssertionError, lambda: p.define('_immutable', 1, ''))
    self.assertRaisesRegex(AttributeError, 'foo', lambda: p.set(foo=4))
    # We use setattr() because lambda cannot contain explicit assignment.
    self.assertRaisesRegex(AttributeError, 'foo', lambda: setattr(p, 'foo', 4))
    p.define('foo', 1, '')
    p.define('nested', p.copy(), '')
    self.assertEqual(p.foo, 1)
    self.assertEqual(p.get('foo'), 1)
    self.assertEqual(p.nested.foo, 1)
    p.freeze()

    self.assertRaises(TypeError, lambda: p.set(foo=2))
    self.assertEqual(p.get('foo'), 1)
    self.assertRaises(TypeError, lambda: setattr(p, 'foo', 3))
    self.assertEqual(p.foo, 1)
    self.assertRaises(TypeError, lambda: p.delete('foo'))
    self.assertEqual(p.foo, 1)
    self.assertRaises(TypeError, lambda: p.define('bar', 1, ''))
    self.assertRaisesRegex(AttributeError, 'bar', p.get, 'bar')

    p.nested.foo = 2
    self.assertEqual(p.foo, 1)
    self.assertEqual(p.nested.foo, 2)

    self.assertRaises(TypeError, lambda: setattr(p, '_immutable', False))

    # Copies are still immutable.
    q = p.copy()
    self.assertRaises(TypeError, lambda: q.set(foo=2))

  def test_to_string(self):
    outer = _params.Params()
    outer.define('foo', 1, '')
    inner = _params.Params()
    inner.define('bar', 2, '')
    outer.define('inner', inner, '')
    outer.define('list', [1, inner, 2], '')
    outer.define('dict', {'a': 1, 'b': inner}, '')
    outer.define('enum', TestEnum.B, '')
    self.assertEqual(
        '\n' + str(outer), """
{
  dict: {'a': 1, 'b': {'bar': 2}}
  enum: TestEnum.B
  foo: 1
  inner: {
    bar: 2
  }
  list: [1, {'bar': 2}, 2]
}""")

  def test_iter_params(self):
    keys, values = ['a', 'b', 'c', 'd', 'e'], [True, None, 'zippidy', 78.5, 5]
    p = _params.Params()
    for k, v in zip(keys, values):
      p.define(k, v, 'description of %s' % k)

    k_set, v_set = set(keys), set(values)
    number_of_params = 0
    for k, v in p.iter_params():
      self.assertIn(k, k_set)
      self.assertIn(v, v_set)
      number_of_params += 1
    self.assertEqual(number_of_params, len(keys))

  def test_similar_keys(self):
    p = _params.Params()
    p.define('activation', 'RELU', 'Can be a string or a list of strings.')
    p.define('activations', 'RELU', 'Many activations.')
    p.define('cheesecake', None, 'dessert')
    p.define('tofu', None, 'not dessert')

    def set_param():
      p.actuvation = 1

    self.assertRaisesRegexp(
        AttributeError,
        re.escape('actuvation (did you mean: [activation,activations])'),
        set_param)


if __name__ == "__main__":
  unittest.main(verbosity=2)
