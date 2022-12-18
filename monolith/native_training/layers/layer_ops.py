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

from monolith.native_training.runtime.ops import gen_monolith_ops

layer_ops_lib = gen_monolith_ops


def ffm(left: tf.Tensor,
        right: tf.Tensor,
        dim_size: int,
        int_type: str = 'multiply') -> tf.Tensor:
  output = layer_ops_lib.FFM(left=left,
                             right=right,
                             dim_size=dim_size,
                             int_type=int_type)
  return output


@tf.RegisterGradient('FFM')
def _ffm_grad(op, grad: tf.Tensor) -> tf.Tensor:
  left, right = op.inputs[0], op.inputs[1]
  dim_size = op.get_attr('dim_size')
  int_type = op.get_attr('int_type')

  (left_grad, right_grad) = layer_ops_lib.FFMGrad(grad=grad,
                                                  left=left,
                                                  right=right,
                                                  dim_size=dim_size,
                                                  int_type=int_type)
  return left_grad, right_grad


