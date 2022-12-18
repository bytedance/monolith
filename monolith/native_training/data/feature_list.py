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

from absl import logging, flags
from dataclasses import dataclass
import inspect
import threading
from typing import List, Dict, Optional, Set, Tuple, get_type_hints
from monolith.native_training.data.utils import get_slot_feature_name, get_slot_from_feature_name
from monolith.native_training.utils import add_to_collections

_BOOL_FLAGS = {'true', 'yes', 't', 'y', '1'}
_cache = {}

FLAGS = flags.FLAGS


def new_instance(cls, args):
  signature = inspect.signature(cls.__init__)
  valid_args = {}
  for key, param in signature.parameters.items():
    if key not in {'cls', 'self'}:
      if param.name in args:
        valid_args[param.name] = args[param.name]

  return cls(**valid_args)


@dataclass
class Feed:
  feed_name: str = None
  shared: bool = None
  feature_id: int = None

  def __post_init__(self):
    if self.shared is not None:
      self.shared = self.shared.lower() in _BOOL_FLAGS
    else:
      self.shared = False
    if isinstance(self.feature_id, str):
      self.feature_id = int(self.feature_id)

  @property
  def name(self):
    return self.feed_name


@dataclass
class Cache:
  cache_column: str = None
  cache_name: str = None
  capacity: int = None
  timeout: int = None
  cache_type: str = None
  cache_key_class: str = None

  def __post_init__(self):
    if isinstance(self.capacity, str):
      self.capacity = int(self.capacity)
    if isinstance(self.timeout, str):
      self.timeout = int(self.timeout)

  @property
  def name(self):
    if self.cache_name is not None:
      return self.cache_name
    elif self.cache_key_class is not None:
      return self.cache_key_class
    elif self.cache_column is not None:
      return 'cache_column'
    else:
      raise Exception('no name for cache')


@dataclass
class Feature:
  feature_name: str = None
  depend: List[str] = None
  method: str = None
  slot: int = None
  args: List[str] = None
  feature_version: int = None
  shared: bool = False
  cache_keys: List[str] = None
  need_raw: bool = False
  feature_id: int = None
  input_optional: List[bool] = None
  feature_group: List[str] = None

  def __post_init__(self):
    if isinstance(self.feature_group, str):
      self.feature_group = [
          item.strip().replace('"', '').replace("'", '')
          for item in self.feature_group.strip().split(',')
      ]

    if isinstance(self.depend, str):
      self.depend = [
          item.strip().replace('"', '').replace("'", '')
          for item in self.depend.strip().split(',')
      ]

    if isinstance(self.input_optional, str):
      self.input_optional = [
          item.strip().replace('"', '').replace("'", '') == 'true'
          for item in self.input_optional.strip().split(',')
      ]

    if isinstance(self.args, str):
      self.args = [
          item.strip().replace('"', '').replace("'", '')
          for item in self.args.strip().split(',')
      ]

    if isinstance(self.cache_keys, str):
      self.cache_keys = [
          item.strip().replace('"', '').replace("'", '')
          for item in self.cache_keys.strip().split(',')
      ]

    if isinstance(self.slot, str):
      self.slot = int(self.slot)

    if isinstance(self.shared, str):
      self.shared = self.shared.lower() in _BOOL_FLAGS

    if isinstance(self.need_raw, str):
      self.need_raw = self.need_raw.lower() in _BOOL_FLAGS

    if isinstance(self.feature_id, str):
      self.feature_id = int(self.feature_id)

  def __str__(self):
    terms = []
    for name, clz in get_type_hints(Feature).items():
      value = getattr(self, name)
      if value is not None:
        if clz == str:
          terms.append("{}={}".format(name, value))
        elif clz == int:
          terms.append("{}={}".format(name, value))
        elif clz == bool:
          if value:
            terms.append("{}=true".format(name))
        elif clz._name == 'List' and len(clz.__args__) == 1:
          if clz.__args__[0] == str:
            terms.append("{}={}".format(name, ','.join(value)))
          elif clz.__args__[0] == bool:
            format_value = [str(b).lower() for b in value]
            terms.append("{}={}".format(name, ','.join(format_value)))
        else:
          raise ValueError("Type Error")
    return ';'.join(terms)

  @property
  def name(self):
    term_list = []
    for term in self.feature_name.split('-'):
      if term.startswith('fc_'):
        term = term[3:]
      elif self.feature_name.startswith('f_'):
        term = term[2:]
      term_list.append(term)

    return '-'.join(term_list).lower()

  @property
  def depend_strip_prefix(self):
    depend = []
    for dep in self.depend:
      term_list = []
      for term in dep.split('-'):
        if term.startswith('fc_'):
          term = term[3:]
        elif term.startswith('f_'):
          term = term[2:]
        term_list.append(term)

      depend.append('-'.join(term_list).lower())
    return depend


class FeatureList(object):
  _lock = threading.Lock()

  def __init__(self, column_name: Optional[Set[str]], feeds: Dict[str, Feed],
               caches: Dict[str, Cache], features: Dict[str, Feature]):
    self.column_name = column_name
    self.feeds = feeds
    self.caches = caches
    self.features = features
    self.__slots = {feat.slot: feat for feat in features.values()}
    add_to_collections('feature_list', self)

  def __getitem__(self, item) -> Feature:
    if isinstance(item, int):
      return self.__slots[item]
    else:
      assert isinstance(item, str)
      item = item.strip()
      if item in self.features:
        return self.features[item]
      elif f'f_{item}' in self.features:
        return self.features[f'f_{item}']
      elif f'fc_{item}' in self.features:
        return self.features[f'fc_{item}']
      else:
        if '-' in item:
          new_item = '-'.join([f'fc_{term}' for term in item.split('-')])
          if new_item in self.features:
            return self.features[new_item]

          new_item = '-'.join([f'f_{term}' for term in item.split('-')])
          if new_item in self.features:
            return self.features[new_item]

        raise Exception('there is no feature {}'.format(item))

  def get(self, item, default=None):
    try:
      return self.__getitem__(item)
    except:
      return default

  def __len__(self):
    return len(self.features)

  def __contains__(self, item):
    return item in self.features or f'f_{item}' in self.features or f'fc_{item}' in self.features or item in self.__slots

  def __iter__(self):
    return iter(self.features.values())

  @classmethod
  def parse(cls, fname: str = None, use_old_name: bool = True) -> 'FeatureList':
    fname = fname or FLAGS.feature_list
    assert fname is not None
    with cls._lock:
      if fname in _cache:
        return _cache[fname]
      column_name = None
      feeds, caches, features = {}, {}, {}
      with open(fname) as stream:
        for line in stream:
          line = line.strip()
          if len(line) == 0 or line.startswith("#"):
            continue

          if line.startswith('column_name'):
            start = len('column_name:')
            column_name = {item.strip() for item in line[start:].split(',')}
            continue

          if line.startswith('cache_column'):
            cache = Cache(cache_column=line[len('cache_column:'):].strip())
            caches[cache.name] = cache
            continue

          params = {}
          items = line.split('=')
          for i in range(len(items) - 1):
            if i == 0:
              key = items[i].strip()
            else:
              start = items[i].rindex(" ")
              key = items[i][start:].strip()

            if i == len(items) - 2:
              value = items[i + 1]
            else:
              end = items[i + 1].rindex(" ")
              value = items[i + 1][0:end]

            params[key] = value.strip().rstrip(',').rstrip(';').rstrip()

          try:
            if line.startswith('feed'):
              feed = new_instance(Feed, params)
              feeds[feed.name] = feed
            elif line.startswith('cache'):
              cache = new_instance(Cache, params)
              caches[cache.name] = cache
            else:
              feat = new_instance(Feature, params)
              if use_old_name:
                features[feat.feature_name] = feat
              else:
                features[feat.name] = feat
          except Exception as e:
            print(line)
            raise e

      feat_list = cls(column_name, feeds, caches, features)
      _cache[fname] = feat_list

      return feat_list


def get_feature_name_and_slot(item) -> Tuple[str, Optional[int]]:
  if isinstance(item, int):
    try:
      feature_list = FeatureList.parse()
      return feature_list.get(item).feature_name, item
    except:
      return get_slot_feature_name(item), item
  elif isinstance(item, str):
    try:
      feature_list = FeatureList.parse()
      assert item in feature_list
      return item, feature_list[item].slot
    except:
      return item, get_slot_from_feature_name(item)
  else:
    # for FeatureColumn
    assert hasattr(item, 'feature_name') and hasattr(item, 'feature_slot')
    return item.feature_name, item.feature_slot
