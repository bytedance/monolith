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

import abc
import tensorflow as tf


class LearningRateFunction():
  """The learning rate function base class.

  You can use a learning rate function to modulate how the learning rate
  of your optimizer changes over time.

  A `LearningRateFunction` instance can be passed in as the `learning_rate`
  argument of any dense optimizer or as `learning_rate_fn` argument for adding
  feature slice of embedding table.

  To implement your own function object, you should implement the `__call__`
  method.

  """

  @abc.abstractmethod
  def __call__(self):
    raise NotImplementedError("Learning rate function must override __call__")

  # Used to check whether two LearningRateFunctions have the same feature.
  def __str__(self):
    return "LearningRateFunction(\"%s\",Params:%s)" % (
        self.__class__.__name__, ",".join([
            "%s=%s" % (key, self.__dict__[key]) for key in sorted(self.__dict__)
        ]))


class PolynomialDecay(LearningRateFunction):
  """A LearningRateFunction that uses an polynomial decay schedule.

  This function applies a polynomial decay function to a provided
  `initial_learning_rate` to reach an `end_learning_rate` in the given `decay_steps`.

  """

  def __init__(self,
               initial_learning_rate,
               decay_steps,
               end_learning_rate=0.0001,
               power=1.0,
               cycle=False,
               name=None):
    """Applies polynomial decay to the learning rate.

    Args:
      initial_learning_rate: A scalar `float32` or `float64` `Tensor` or a Python
        number. The initial learning rate.
      decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number. Must
        be positive.  See the decay computation above.
      end_learning_rate: A scalar `float32` or `float64` `Tensor` or a Python
        number.  The minimal end learning rate.
      power: A scalar `float32` or `float64` `Tensor` or a Python number.  The
        power of the polynomial. Defaults to linear, 1.0.
      cycle: A boolean, whether or not it should cycle beyond decay_steps.
      name: String.  Optional name of the operation. Defaults to
        'PolynomialDecay'.

    Returns:
      A scalar `Tensor` of the same type as `initial_learning_rate`.  The decayed
      learning rate.

    """
    super(PolynomialDecay, self).__init__()
    self.initial_learning_rate = initial_learning_rate
    self.decay_steps = decay_steps
    self.end_learning_rate = end_learning_rate
    self.power = power
    self.cycle = cycle
    self.name = name

  def __call__(self):
    global_step = tf.compat.v1.train.get_or_create_global_step()
    return tf.compat.v1.train.polynomial_decay(
        learning_rate=self.initial_learning_rate,
        global_step=global_step,
        decay_steps=self.decay_steps,
        end_learning_rate=self.end_learning_rate,
        power=self.power,
        cycle=self.cycle)
