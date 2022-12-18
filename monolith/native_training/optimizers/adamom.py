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

training_ops = gen_monolith_ops


class AdamomOptimizer(tf.compat.v1.train.Optimizer):
  def __init__(self,
               learning_rate=5e-6,
               ada_decay: float = 0.9999,
               mom_decay: float = 0.99,
               epsilon: float = 1e-6,
               weight_decay: float = 0.0,
               use_locking: bool = False,
               name="Adamom"):
    super().__init__(use_locking, name)
    self._learning_rate = learning_rate
    self._ada_decay = ada_decay
    self._mom_decay = mom_decay
    self._epsilon = epsilon
    self._weight_decay = weight_decay
    # Created in Initialize.
    self._learning_rate_tensor = None

  def _create_slots(self, var_list):
    # Create slots for the first and second moments.
    for v in var_list:
      self._zeros_slot(v, "m", self._name + "/m")
      self._zeros_slot(v, "v", self._name + "/v")
      self._zeros_slot(v, "c", self._name + "/c")

  def _prepare(self):
    learning_rate = self._call_if_callable(self._learning_rate)
    self._learning_rate_tensor = tf.convert_to_tensor(learning_rate,
                                                      name="learning_rate")

  def _resource_apply_dense(self, grad, var):
    m = self.get_slot(var, "m")
    v = self.get_slot(var, "v")
    c = self.get_slot(var, "c")
    return training_ops.resource_apply_adamom(var.handle,
                                              m.handle,
                                              v.handle,
                                              c.handle,
                                              tf.cast(
                                                  self._learning_rate_tensor,
                                                  grad.dtype.base_dtype),
                                              self._ada_decay,
                                              self._mom_decay,
                                              self._epsilon,
                                              self._weight_decay,
                                              grad,
                                              use_locking=self._use_locking)
