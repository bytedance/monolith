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

import tensorflow as tf
from typing import Callable, DefaultDict, Dict, Iterable, List, Tuple, Optional
from monolith.native_training.static_reshape_op import static_reshape, StaticReshapeNBuilder


def maybe_squeeze_3d_tensor(x: tf.RaggedTensor):
  """Expected to return a raggedtensor which shape is [None/batch_size, None (RaggedRank)]
  Supports tensor type:
  [None/batch_size, None],
  [None/batch_size, 1, None]
  """
  if not isinstance(x, tf.RaggedTensor):
    raise ValueError("input must be RaggedTensor")
  if len(x.shape) == 2:
    return x
  elif len(x.shape) == 3:
    return tf.squeeze(x, axis=1)
  else:
    raise ValueError("Unknown shape of RaggedTensor. ", x)


def pack_tensors(
    keyed_tensors: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
  """Compact multiple tensors into 1 tensor."""
  builder = StaticReshapeNBuilder()
  for key in sorted(keyed_tensors):
    builder.add(keyed_tensors[key], (None,))
  outputs, sizes = builder.build()
  return tf.concat(outputs, 0), sizes


def get_keyed_shape(
    keyed_tensors: Dict[str, tf.Tensor]) -> Dict[str, List[int]]:
  return {key: val.shape.as_list() for key, val in keyed_tensors.items()}


def unpack_tensors(keyed_shape: Dict[str, List[int]],
                   packed: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
  """The reverse method of _pack_tensors."""
  m = {}
  tensor, length = packed[0], packed[1]
  flat_tensors = tf.split(tensor, length, num=len(keyed_shape))
  builder = StaticReshapeNBuilder()
  for i, key in enumerate(sorted(keyed_shape)):
    builder.add(flat_tensors[i], keyed_shape[key])
  outputs, _ = builder.build()
  for i, key in enumerate(sorted(keyed_shape)):
    m[key] = outputs[i]
  return m


def _get_flat_tensor_and_size(input_tensor):
  reshaped = tf.reshape(input_tensor, [-1])
  return reshaped, tf.size(reshaped)


def split_tensors_with_type(
    keyed_tensors: Dict[str, tf.Tensor]) -> List[Dict[str, tf.Tensor]]:
  type_dict_dict = {}
  type_set = set()
  for key in sorted(keyed_tensors):
    tensor = keyed_tensors[key]
    if (str(tensor.dtype) not in type_set):
      type_set.add(str(tensor.dtype))
      type_dict_dict[str(tensor.dtype)] = {}
    type_dict_dict[str(tensor.dtype)][key] = tensor

  convert_list = []
  for key in sorted(type_dict_dict):
    convert_list.append(type_dict_dict[key])

  return convert_list


def merge_dicts(
    tensor_dict_list: List[Dict[str, tf.Tensor]]) -> Dict[str, tf.Tensor]:
  res_d = {}
  for d in tensor_dict_list:
    for key in d.keys():
      res_d[key] = d[key]

  return res_d


def pack_typed_keyed_tensors(
    list_keyed_tensors: List[Dict[str, tf.Tensor]]) -> List[tf.Tensor]:
  builder = StaticReshapeNBuilder()

  def flatten(tensor):
    return builder.add(tensor, (None,))

  list_keyed_id = tf.nest.map_structure(flatten, list_keyed_tensors)
  outputs, sizes = builder.build()

  packed_tensors = []
  packed_size_size_list = []

  for d in list_keyed_id:
    tensors = [outputs[d[key]] for key in sorted(d.keys())]
    packed_tensors.append(tf.concat(tensors, 0))
    packed_size_size_list.append(len(d.keys()))

  packed_size_size = tf.constant(packed_size_size_list, dtype=tf.int64)
  concat_offset_size = tf.concat([packed_size_size, sizes], 0)
  packed_tensors.append(concat_offset_size)
  return packed_tensors


def get_typed_keyed_shape(
    list_keyed_tensors: List[Dict[str,
                                  tf.Tensor]]) -> List[Dict[str, List[int]]]:
  list_keyed_shape = []
  for d in list_keyed_tensors:
    list_keyed_shape.append(get_keyed_shape(d))
  return list_keyed_shape


def unpack_packed_tensors(
    list_keyed_shape: List[Dict[str, List[int]]],
    packed_list: List[tf.Tensor]) -> List[Dict[str, tf.Tensor]]:
  length = len(packed_list)
  if (length < 2):
    raise ValueError("Wrong packed_list length")
  concat_offset_size = packed_list[-1]
  packed_size_size = tf.slice(concat_offset_size, [0], [length - 1])
  packed_size = tf.slice(concat_offset_size, [length - 1], [-1])
  packed_size = tf.split(packed_size,
                         packed_size_size,
                         num=len(list_keyed_shape))

  builder = StaticReshapeNBuilder()
  unpack_list = []
  for i, d in enumerate(list_keyed_shape):
    d_size = packed_size[i]
    d_tensor = tf.split(packed_list[i], d_size, num=len(list_keyed_shape[i]))
    for j, key in enumerate(sorted(d.keys())):
      builder.add(d_tensor[j], list_keyed_shape[i][key])

  outputs, _ = builder.build()
  idx = 0
  for d in list_keyed_shape:
    unpack_d = {}
    for key in sorted(d.keys()):
      unpack_d[key] = outputs[idx]
      idx += 1
    unpack_list.append(unpack_d)

  return unpack_list
