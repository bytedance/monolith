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

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

from typing import Union, Callable
import tensorflow as tf

import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.util.tf_export import keras_export
from tensorflow.keras.initializers import Constant

from monolith.native_training.runtime.ops import gen_monolith_ops

training_ops = gen_monolith_ops


class RmspropOptimizer(tf.compat.v1.train.Optimizer):
  """http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf"""

  def __init__(self,
               learning_rate=5e-6,
               beta1: float = 0.99,
               beta2: float = 0.999,
               epsilon: float = 1e-8,
               weight_decay: float = 0.0,
               use_locking: bool = False,
               use_v2: bool = False,
               name="Rmsprop"):
    super().__init__(use_locking, name)
    self._learning_rate = learning_rate
    self._beta1 = beta1
    self._beta2 = beta2
    self._epsilon = epsilon
    self._weight_decay = weight_decay
    self._use_v2 = use_v2
    # Created in Initialize.
    self._learning_rate_tensor = None

  def _create_slots(self, var_list):
    # Create slots for the first and second moments.
    for v in var_list:
      self._zeros_slot(v, "m", self._name + "/m")
      self._zeros_slot(v, "v", self._name + "/v")

  def _prepare(self):
    learning_rate = self._call_if_callable(self._learning_rate)
    self._learning_rate_tensor = tf.convert_to_tensor(learning_rate,
                                                      name="learning_rate")

  def _apply_dense(self, grad, var):
    raise NotImplementedError(
        "Please use tf.compat.v1.disable_eager_execution() instead of tf.compat.v1.disable_v2_behavior()"
    )

  def _resource_apply_dense(self, grad, var):
    m = self.get_slot(var, "m")
    v = self.get_slot(var, "v")
    return training_ops.resource_apply_rmsprop(var.handle,
                                               m.handle,
                                               v.handle,
                                               tf.cast(
                                                   self._learning_rate_tensor,
                                                   grad.dtype.base_dtype),
                                               self._beta1,
                                               self._beta2,
                                               self._epsilon,
                                               self._weight_decay,
                                               grad,
                                               use_locking=self._use_locking,
                                               use_v2=self._use_v2)
