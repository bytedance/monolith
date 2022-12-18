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

import copy
import itertools
from typing import List

import tensorflow as tf


def _iterate(nested, action):
  """Iterate nested structures. `action` should take element and returns a element."""
  if nested is None:
    pass
  elif isinstance(nested, (list, tuple)):
    r = []
    for v in nested:
      r.append(_iterate(v, action))
    if isinstance(nested, tuple):
      r = tuple(r)
    nested = r
  elif isinstance(nested, dict):
    for k, v in nested.items():
      nested[k] = _iterate(nested[k], action)
  else:
    nested = action(nested)

  return nested


class NestedTensors:

  def __init__(self, nested):
    self._nested = nested
    self._id_mapping = {}
    self._ragged_tensors = []
    self._tensors = []
    self._other_objs = []
    self._nested = _iterate(self._nested, self._add_tensor)

  def _add_tensor(self, tensor):
    obj_id = id(tensor)
    if not obj_id in self._id_mapping:
      if isinstance(tensor, tf.Tensor):
        self._id_mapping[obj_id] = (0, len(self._tensors))
        self._tensors.append(tensor)
      elif isinstance(tensor, tf.RaggedTensor):
        if tensor.ragged_rank != 1:
          raise ValueError("Nested tensor doesn't support nested RaggedTensor.")
        self._id_mapping[obj_id] = (1, len(self._ragged_tensors))
        self._ragged_tensors.append(tensor)
      elif isinstance(tensor, (bool, int, str, tf.Variable, None)):
        # There are some cases we want to keep it as it is.
        self._id_mapping[obj_id] = (2, len(self._other_objs))
        self._other_objs.append(tensor)
      else:
        raise ValueError("Tensor is not supported. {}".format(tensor))

    return obj_id

  def get_tensors(self) -> List[tf.Tensor]:
    flatten_ragged_tensors = self._ragged_to_flatten(self._ragged_tensors)
    return self._tensors + flatten_ragged_tensors

  def get_nested_result(self, tensors: List[tf.Tensor]):
    flatten_ragged_tensors = tensors[len(self._tensors):]
    tensors = tensors[:len(self._tensors)]
    assert len(flatten_ragged_tensors) == len(self._ragged_tensors) * 2
    ragged_tensors = self._flatten_to_ragged(flatten_ragged_tensors)

    tensor_tuple = (tensors, ragged_tensors, self._other_objs)
    result = copy.deepcopy(self._nested)

    def action(obj_id):
      idx = self._id_mapping[obj_id]
      return tensor_tuple[idx[0]][idx[1]]

    return _iterate(result, action)

  @staticmethod
  def _convert_ragged_to_tensors(ragged):
    return ragged.values, ragged.row_splits

  @staticmethod
  def _convert_tensors_to_ragged(values, row_splits):
    return tf.RaggedTensor.from_row_splits(values, row_splits, validate=False)

  def _ragged_to_flatten(self, ragged_tensors):
    return list(
        itertools.chain.from_iterable((self._convert_ragged_to_tensors(ragged)
                                       for ragged in ragged_tensors)))

  def _flatten_to_ragged(self, tensors):
    ragged_values = tensors[::2]
    ragged_row_splits = tensors[1::2]
    return [
        self._convert_tensors_to_ragged(*combined)
        for combined in zip(ragged_values, ragged_row_splits)
    ]