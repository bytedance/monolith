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

import abc
from typing import List, Union

from monolith.native_training.data.transform import transform_config_pb2
from idl.matrix.proto.line_id_pb2 import LineId


class Transform(abc.ABC):

  @abc.abstractmethod
  def as_proto(self) -> transform_config_pb2.TransformConfig():
    pass

  @abc.abstractmethod
  def _is_leaf_node(self) -> bool:
    pass


class Compose(Transform):
  """Composes several transforms together.

  Args:
      transforms (list of ``Transform`` objects): list of transforms to compose.

  Example:
      >>> transforms.Compose([
      >>>     transforms.FilterByFid(has_fids=[1]),
      >>>     transforms.FilterByLabel(thresholds=[-100]),
      >>> ])
  """

  def __init__(self, transforms: List[Transform]):
    assert all(isinstance(t, Transform) for t in transforms)
    self.transforms = transforms

  def as_proto(self) -> transform_config_pb2.TransformConfig():
    config = transform_config_pb2.TransformConfig()
    for t in self.transforms:
      config.MergeFrom(t.as_proto())
    return config

  def _is_leaf_node(self) -> bool:
    return False


class FilterByFid(Transform):

  def __init__(self,
               has_fids: List[int] = None,
               filter_fids: List[int] = None,
               select_fids: List[int] = None):
    self.has_fids = has_fids
    self.filter_fids = filter_fids
    self.select_fids = select_fids

  def as_proto(self) -> transform_config_pb2.TransformConfig():
    config = transform_config_pb2.TransformConfig()
    transform = config.configs.add()
    transform.basic_config.filter_by_fid.has_fids.extend(self.has_fids)
    transform.basic_config.filter_by_fid.filter_fids.extend(self.filter_fids)
    transform.basic_config.filter_by_fid.select_fids.extend(self.select_fids)
    return config

  def _is_leaf_node(self) -> bool:
    return True


class FilterByAction(Transform):

  def __init__(self, has_actions: List[int] = None):
    self.has_actions = has_actions

  def as_proto(self) -> transform_config_pb2.TransformConfig():
    config = transform_config_pb2.TransformConfig()
    transform = config.configs.add()
    transform.basic_config.filter_by_action.has_actions.extend(self.has_actions)
    return config

  def _is_leaf_node(self) -> bool:
    return True


class FilterByLabel(Transform):

  def __init__(self, thresholds=List[float]):
    self.thresholds = thresholds

  def as_proto(self) -> transform_config_pb2.TransformConfig():
    config = transform_config_pb2.TransformConfig()
    transform = config.configs.add()
    transform.basic_config.filter_by_label.thresholds.extend(self.thresholds)
    return config

  def _is_leaf_node(self) -> bool:
    return True


class FilterByValue(Transform):

  def __init__(
      self,
      field_name: str,
      op: str,
      operand: Union[float, int, str, List[float], List[int], List[str]],
      keep_empty: bool = False,
  ):
    assert op in {
        'gt', 'ge', 'eq', 'lt', 'le', 'neq', 'between', 'in', 'not-in', 'all',
        'any', 'diff', 'startswith', 'endswith'
    }
    fields = LineId.DESCRIPTOR.fields_by_name
    assert field_name in fields
    assert operand is not None

    field = fields[field_name]
    string_operand = []

    if field.has_options:
      assert op in {'all', 'any', 'diff'}
      assert field.cpp_type in {
          field.CPPTYPE_INT32, field.CPPTYPE_INT64, field.CPPTYPE_UINT32,
          field.CPPTYPE_UINT64
      }
      if not isinstance(operand, (list, tuple)):
        assert isinstance(operand, int)
        int_operand, float_operand = [operand], []
      else:
        assert all(isinstance(o, int) for o in operand)
        int_operand, float_operand = list(operand), []
    elif field.cpp_type in {field.CPPTYPE_DOUBLE, field.CPPTYPE_FLOAT}:
      if op == 'between':
        assert all(isinstance(o, (int, float)) for o in operand)
        int_operand, float_operand = [], [float(o) for o in operand]
      else:
        int_operand, float_operand = [], [float(operand)]
    elif field.cpp_type in {
        field.CPPTYPE_INT32, field.CPPTYPE_INT64, field.CPPTYPE_UINT32,
        field.CPPTYPE_UINT64
    }:
      if op in {'in', 'not-in', 'between'}:
        assert all(isinstance(o, int) for o in operand)
        int_operand, float_operand = list(operand), []
      else:
        int_operand, float_operand = [int(operand)], []
    elif field.cpp_type == field.CPPTYPE_STRING:
      int_operand, float_operand = [], []
      if isinstance(operand, str):
        string_operand.append(operand)
      elif isinstance(operand, (list, tuple)):
        assert all(isinstance(o, str) for o in operand)
        string_operand.extend(operand)
      else:
        raise RuntimeError("params error!")
    else:
      raise RuntimeError("params error!")

    self.field_name = field_name
    self.op = op
    self.float_operand = float_operand
    self.int_operand = int_operand
    self.string_operand = string_operand
    self.keep_empty = keep_empty

  def as_proto(self) -> transform_config_pb2.TransformConfig():
    config = transform_config_pb2.TransformConfig()
    transform = config.configs.add()
    transform.basic_config.filter_by_value.field_name = self.field_name
    transform.basic_config.filter_by_value.op = self.op
    transform.basic_config.filter_by_value.float_operand.extend(
        self.float_operand)
    transform.basic_config.filter_by_value.int_operand.extend(self.int_operand)
    transform.basic_config.filter_by_value.string_operand.extend(
        self.string_operand)
    transform.basic_config.filter_by_value.keep_empty = self.keep_empty

    return config

  def _is_leaf_node(self) -> bool:
    return True


class AddLabel(Transform):

  def __init__(self, config: str, negative_value: float,
               new_sample_rate: float):
    self.config = config
    self.negative_value = negative_value
    self.new_sample_rate = new_sample_rate

  def as_proto(self) -> transform_config_pb2.TransformConfig():
    config = transform_config_pb2.TransformConfig()
    transform = config.configs.add()
    transform.basic_config.add_label.negative_value = self.negative_value
    transform.basic_config.add_label.new_sample_rate = self.new_sample_rate

    for task in self.config.split(';'):
      # skip empty parts, e.g. config = '1,2:3:1.0;'
      if len(task) == 0:
        continue

      task_label_config = transform.basic_config.add_label.task_label_configs.add(
      )
      pos_actions, neg_actions, sample_rate = task.split(':')
      pos_actions_list = [
          int(pos) for pos in pos_actions.split(',') if len(pos) > 0
      ]
      neg_actions_list = [
          int(neg) for neg in neg_actions.split(',') if len(neg) > 0
      ]
      task_label_config.pos_actions.extend(pos_actions_list)
      task_label_config.neg_actions.extend(neg_actions_list)
      task_label_config.sample_rate = float(sample_rate)
    return config

  def _is_leaf_node(self) -> bool:
    return True


class LogicalOr(Transform):

  def __init__(self, x: Transform, y: Transform):
    self.x = x
    self.y = y
    assert x._is_leaf_node() and y._is_leaf_node()

  def as_proto(self) -> transform_config_pb2.TransformConfig():
    config = transform_config_pb2.TransformConfig()
    transform = config.configs.add()
    transform.logical_or_config.x.CopyFrom(
        self.x.as_proto().configs[0].basic_config)
    transform.logical_or_config.y.CopyFrom(
        self.y.as_proto().configs[0].basic_config)
    return config

  def _is_leaf_node(self) -> bool:
    return False
