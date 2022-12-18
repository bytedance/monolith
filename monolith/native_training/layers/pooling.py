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

from tensorflow.keras.layers import Layer
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops

from monolith.native_training.utils import with_params, check_list
from monolith.native_training.monolith_export import monolith_export


@monolith_export
class Pooling(Layer):
  """Pooling基类

  Args:
    kwargs (:obj:`dict`): 其它位置参数, 详情请参考 `TF Layer`_
    
  .. _TF Layer: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer
    
  """

  def __init__(self, **kwargs):
    super(Pooling, self).__init__(**kwargs)

  def pool(self, vec_list):
    raise NotImplementedError

  def call(self, vec_list, **kwargs):
    check_list(vec_list, lambda x: x > 0)
    if len(vec_list) == 1:
      return vec_list[0]
    return self.pool(vec_list)


@monolith_export
@with_params
class SumPooling(Pooling):
  """Sum pooling, 加法池化

  Args:
    kwargs (:obj:`dict`): 其它位置参数, 详情请参考 `TF Layer`_
    
  .. _TF Layer: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer
  
  """

  def __init__(self, **kwargs):
    super(SumPooling, self).__init__(**kwargs)

  def pool(self, vec_list):
    return math_ops.add_n(vec_list)


@monolith_export
@with_params
class AvgPooling(Pooling):
  """Average pooling, 平匀池化

  Args:
    kwargs (:obj:`dict`): 其它位置参数, 详情请参考 `TF Layer`_
    
  .. _TF Layer: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer
  
  """

  def __init__(self, **kwargs):
    super(AvgPooling, self).__init__(**kwargs)

  def pool(self, vec_list):
    return math_ops.add_n(vec_list) / len(vec_list)


@monolith_export
@with_params
class MaxPooling(Pooling):
  """Max pooling, 最大池化

  Args:
    kwargs (:obj:`dict`): 其它位置参数, 详情请参考 `TF Layer`_
    
  .. _TF Layer: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer
  
  """

  def __init__(self, **kwargs):
    super(MaxPooling, self).__init__(**kwargs)

  def pool(self, vec_list):
    return math_ops.reduce_max(array_ops.stack(vec_list), axis=0)
