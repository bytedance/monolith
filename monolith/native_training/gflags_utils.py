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

from absl.flags import FlagValues
from absl import logging, flags
import dataclasses
from dataclasses import Field
from enum import Enum
import re
import sys
from typing import get_type_hints, Iterable, Tuple

FLAGS = flags.FLAGS

_SPACE = re.compile(r"\s+")
_PARAM = re.compile(r"^:param\s+([a-zA-Z0-9._-]+)\s*:\s*(.*)")


class Status(Enum):
  Init = 1
  Open = 2
  Extend = 3
  Closed = 4


def _extract_help_info(cls, help_info, is_nested):
  if is_nested:
    for base in cls.__bases__:
      assert _extract_help_info(base, help_info, is_nested) == Status.Closed

  doc = [
      " ".join(re.split(_SPACE, line.strip()))
      for line in cls.__doc__.split('\n')
      if len(line.strip()) > 0
  ]

  key_stack = []
  status = Status.Init
  for i, line in enumerate(doc):
    matched = _PARAM.match(line)
    if matched:
      new_key, info = matched.groups()
      if status == Status.Init:
        help_info[new_key] = [info]
        key_stack.append(new_key)
      elif status == Status.Open or status == Status.Extend:
        old_key = key_stack.pop()
        assert old_key != new_key
        help_info[new_key] = [info]
        key_stack.append(new_key)
      else:
        assert status == Status.Closed
        break

      # trans status
      status = Status.Open
    else:
      if status == Status.Init:
        pass
      elif status == Status.Open:
        key = key_stack[-1]
        help_info[key].append(line)
        status = Status.Extend
      elif status == Status.Extend:
        key = key_stack[-1]
        help_info[key].append(line)
      else:
        assert status == Status.Closed
        break

    if i + 1 == len(doc):
      status = Status.Closed

  return status


def extract_help_info(cls, is_nested=True):
  help_info = {}
  status = _extract_help_info(cls, help_info, is_nested)
  assert status == Status.Closed

  return {key: " ".join(value) for key, value in help_info.items()}


def extract_flags_decorator(remove_flags=None, is_nested=True):

  def decorator(cls):
    extract_flags(flags, cls, is_nested, remove_flags)
    return cls

  return decorator


def extract_flags(gflags, dcls, is_nested=True, skip_flags=None) -> FlagValues:
  FLAGS = gflags.FLAGS
  help_info = extract_help_info(dcls, is_nested)
  skip_flags = set() if skip_flags is None else set(skip_flags)
  for key, dtype in get_type_hints(dcls).items():
    if key not in help_info.keys() or key in skip_flags:
      continue
    default = getattr(dcls, key)
    help_str = "default={}, {}".format(default, help_info.get(key, ""))
    try:
      if dtype == int:
        gflags.DEFINE_integer(key, default, "{}, {}".format('int', help_str))
      elif dtype == bool:
        gflags.DEFINE_bool(key, default, "{}, {}".format('bool', help_str))
      elif dtype == str:
        gflags.DEFINE_string(key, default, "{}, {}".format('string', help_str))
      elif dtype == float:
        gflags.DEFINE_float(key, default, "{}, {}".format('float', help_str))
      elif issubclass(dtype, Enum):
        default_value = default.name.lower()
        enum_values = [name.lower() for name in dtype._member_names_]
        gflags.DEFINE_enum(key, default_value, enum_values,
                           "{}, {}".format('enum', help_str))
      else:
        raise ValueError("only <int|bool|str|float|enum> is support!")
    except:
      pass

  return FLAGS


def get_flags_parser(flags, FLAGS):

  def flags_parser(args):
    try:
      return FLAGS(args)
    except flags.Error as error:
      logging.error('FATAL Flags parsing error: {}\n{}'.format(
          error, FLAGS.get_help(include_special_flags=False)))
      logging.error('Pass --helpshort or --helpfull to see help on flags.\n')
      sys.exit(1)

  return flags_parser


def update(config):
  """
    update config's attr value using flags.FLAGS
    if config's attr value is default value and FLAGS' attr value is not default

    config: any type of Config like CpuTraingingConfig, DistributedCpuTrainingConfig
    example: see gflags_utils_test.py test_update()
  """
  FLAGS = flags.FLAGS
  cls = config.__class__
  for key, dtype in get_type_hints(cls).items():
    tmp = getattr(cls, key)
    if isinstance(tmp, Field):
      field = tmp
      default = field.default if field.default is not None else field.default_factory(
      )
    else:
      default = tmp

    from_code = config.__dict__.get(key, default)

    try:
      if not hasattr(FLAGS, key):
        continue
    except:
      continue

    if issubclass(dtype, Enum):
      from_cmd = dtype[getattr(FLAGS, key).upper()]
    else:
      from_cmd = getattr(FLAGS, key)

    if from_code == default and from_cmd != default:
      # user has not set this field, it should not overwrite by cmd
      config.__dict__[key] = from_cmd
    else:
      continue

  return config


class LinkDataclassToFlags:
  """Links a field's default value to a flag. Example:
  
  flags.DEFINE_int("c_value", 0, "")

  @LinkDataclassToFlags(linked_map={"v": "v_value"})
  @dataclass.dataclasses
  class C:
    v: int = 0

  When we instantiate C as c, if v_value is not 0, and c.v is the default value (which is 0 in this case),
  c.v = FLAGS.v_value
  """

  def __init__(self, linked_list=None, linked_map=None):
    """Elements `e` in linked_list is equivalent to `e:e` in the linekd_map """
    self._m = {}
    self._m.update(linked_map or {})
    linked_list = linked_list or []
    for name in linked_list:
      self._m[name] = name

  def __call__(self, cls):

    assert dataclasses.is_dataclass(
        cls), "LinkDataclassToFlag should be used on dataclasses"
    fields = dataclasses.fields(cls)
    named_fields = {field.name: field for field in fields}
    for name, flag in self._m.items():
      if name not in named_fields:
        raise ValueError(f"{name} is not a valid attribute of {type(cls)}")
      if flag not in FLAGS:
        raise ValueError(f"{flag} is not defined in gflags")

    orig_init = cls.__init__

    def __init__(cls_self, *args, **kwargs):
      orig_init(cls_self, *args, **kwargs)
      for name, flag in self._m.items():
        if getattr(cls_self, name) == named_fields[name].default and getattr(
            FLAGS, flag) != FLAGS[flag].default:
          setattr(cls_self, name, getattr(FLAGS, flag))

    cls.__init__ = __init__

    return cls
