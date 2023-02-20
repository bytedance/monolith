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
from typing import List

from monolith.native_training.data.transform import transform_config_pb2


class Transform(abc.ABC):

  @abc.abstractmethod
  def as_proto(self) -> transform_config_pb2.TransformConfig():
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
    transform.filter_by_fid.has_fids.extend(self.has_fids)
    transform.filter_by_fid.filter_fids.extend(self.filter_fids)
    transform.filter_by_fid.select_fids.extend(self.select_fids)
    return config


class FilterByAction(Transform):

  def __init__(self, has_actions: List[int] = None):
    self.has_actions = has_actions

  def as_proto(self) -> transform_config_pb2.TransformConfig():
    config = transform_config_pb2.TransformConfig()
    transform = config.configs.add()
    transform.filter_by_action.has_actions.extend(self.has_actions)
    return config


class FilterByLabel(Transform):

  def __init__(self, thresholds=List[float]):
    self.thresholds = thresholds

  def as_proto(self) -> transform_config_pb2.TransformConfig():
    config = transform_config_pb2.TransformConfig()
    transform = config.configs.add()
    transform.filter_by_label.thresholds.extend(self.thresholds)
    return config


class AddLabel(Transform):

  def __init__(self, config: str, negative_value: float,
               new_sample_rate: float):
    self.config = config
    self.negative_value = negative_value
    self.new_sample_rate = new_sample_rate

  def as_proto(self) -> transform_config_pb2.TransformConfig():
    config = transform_config_pb2.TransformConfig()
    transform = config.configs.add()
    transform.add_label.negative_value = self.negative_value
    transform.add_label.new_sample_rate = self.new_sample_rate

    for task in self.config.split(';'):
      # skip empty parts, e.g. config = '1,2:3:1.0;'
      if len(task) == 0:
        continue

      task_label_config = transform.add_label.task_label_configs.add()
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
