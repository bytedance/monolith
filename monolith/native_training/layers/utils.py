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
from tensorflow.python.ops import array_ops

from monolith.native_training.monolith_export import monolith_export


@monolith_export
class MergeType:
  CONCAT = 'concat'
  STACK = 'stack'
  NONE = None


@monolith_export
class DCNType:
  Vector = "vector"
  Matrix = "matrix"
  Mixed = "mixed"


def check_dim(dim):
  if dim is None:
    return -1
  elif isinstance(dim, int):
    return dim
  elif isinstance(dim, tf.compat.v1.Dimension):
    return dim.value
  else:
    raise Exception(f'dim {dim} is error')


def dim_size(inputs, axis: int):
  shape = inputs.get_shape().as_list()
  assert len(shape) > axis
  dim = check_dim(shape[axis])
  if dim == -1:
    return array_ops.shape(inputs)[axis]
  else:
    return dim


@monolith_export
def merge_tensor_list(tensor_list,
                      merge_type: str = 'concat',
                      num_feature: int = None,
                      axis: int = 1,
                      keep_list: bool = False):
  """将Tensor列表合并
  
  Args:
    tensor_list (:obj:`List[tf.Tensor]`): 输入的Tensor列表
    merge_type (:obj:`str`): 合并类型, 支持stack/concat两种, 如果设为None, 则不做任何处理
    num_feature (:obj:`int`): 特征个数
    axis (:obj:`int`): merge延哪个轴进行
    keep_list (:obj:`bool`): 输出结果是否保持list
  
  """

  if isinstance(tensor_list, tf.Tensor):
    tensor_list = [tensor_list]
  assert merge_type in {'stack', 'concat', None}

  if len(tensor_list) == 1:
    shapes = [check_dim(dim) for dim in tensor_list[0].get_shape().as_list()]

    if len(shapes) == 3:
      (batch_size, num_feat, emb_size) = shapes
      if merge_type == MergeType.STACK:
        output = tensor_list if keep_list else tensor_list[0]
      elif merge_type == MergeType.CONCAT:
        tensor_list[0] = tf.reshape(tensor_list[0],
                                    shape=(batch_size, num_feat * emb_size))
        output = tensor_list if keep_list else tensor_list[0]
      else:
        output = tf.unstack(tensor_list[0], axis=axis)
    elif len(shapes) == 2 and num_feature is not None and num_feature > 1:
      (batch_size, emb_size) = shapes
      emb_size = int(emb_size / num_feature)
      if merge_type == MergeType.STACK:
        tensor_list[0] = tf.reshape(tensor_list[0],
                                    shape=(batch_size, num_feature, emb_size))
        output = tensor_list if keep_list else tensor_list[0]
      elif merge_type == MergeType.CONCAT:
        output = tensor_list if keep_list else tensor_list[0]
      else:
        tensor_list[0] = tf.reshape(tensor_list[0],
                                    shape=(batch_size, num_feature, emb_size))
        output = tf.unstack(tensor_list[0], axis=axis)
    elif len(shapes) == 2:
      output = tensor_list if keep_list else tensor_list[0]
    else:
      raise Exception("shape error: ({})".format(', '.join(map(str, shapes))))
  elif merge_type == 'stack':
    stacked = tf.stack(tensor_list, axis=axis)
    output = [stacked] if keep_list else stacked
  elif merge_type == 'concat':
    concated = tf.concat(tensor_list, axis=axis)
    output = [concated] if keep_list else concated
  else:
    output = tensor_list

  return output
