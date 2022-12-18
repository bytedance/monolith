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

from absl import logging
from typing import List, Tuple

import tensorflow as tf

from monolith.native_training.runtime.ops import gen_monolith_ops

reshape_ops = gen_monolith_ops


def static_reshape(
    inputs: List[tf.Tensor],
    shapes: List[Tuple[int]],
    enable_parallelism: bool = True) -> Tuple[List[tf.Tensor], tf.Tensor]:
  """
  Arguments:
    inputs: List of input tensors.
    shapes: List of target shapes for input tensors.
  
  Returns:
    outputs: List of reshaped tensors.
    sizes: A Tensor containing the size of output tensors.
  """
  return reshape_ops.monolith_static_reshape_n(
      inputs=inputs, shapes=shapes, enable_parallelism=enable_parallelism)


class StaticReshapeNBuilder:

  def __init__(self, enable_parallelism: bool = True):
    self.inputs = []
    self.shapes = []
    self.enable_parallelism = enable_parallelism

  def add(self, input: tf.Tensor, shape: Tuple[int]) -> int:
    """Returns index of input added."""
    self.inputs.append(input)
    self.shapes.append(shape)
    return len(self.inputs) - 1

  def build(self):
    return static_reshape(inputs=self.inputs,
                          shapes=self.shapes,
                          enable_parallelism=self.enable_parallelism)
