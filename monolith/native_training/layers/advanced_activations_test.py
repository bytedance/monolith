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
import tensorflow.keras.activations as acts
import tensorflow.keras.layers as lyacts

from monolith.native_training.layers.advanced_activations import get, serialize


def serde(act):
  _act = get(act)
  sered_act = serialize(_act)
  get(sered_act)


all_acts = [
    'relu', 'leakyrelu', 'elu', 'softmax', 'thresholdedrelu', 'prelu',
    'exponential', 'gelu', 'hardsigmoid', 'linear', 'selu', 'sigmoid',
    'softplus', 'softsign', 'swish', 'tanh'
]

raw_acts = [
    acts.tanh, acts.sigmoid, acts.softsign, acts.softplus, acts.softmax,
    acts.exponential, acts.elu, acts.gelu, acts.hard_sigmoid, acts.selu,
    acts.swish, acts.relu, acts.linear
]

lay_acts = [
    lyacts.ReLU(),
    lyacts.PReLU(),
    lyacts.ThresholdedReLU(),
    lyacts.ELU(),
    lyacts.Softmax(),
    lyacts.LeakyReLU()
]


class ActivationsTest(tf.test.TestCase):

  def test_get_from_str(self):
    for act in all_acts:
      serde(act)

  def test_get_from_layers(self):
    for act in lay_acts:
      serde(act)

  def test_get_from_func(self):
    for act in lay_acts:
      serde(act)

  def test_params(self):
    for act in all_acts:
      cls = get(act).__class__
      p = cls.params()
      # print(p.new_instance())

  def test_call(self):
    inp = tf.random.uniform(shape=(100, 200))
    out = []
    for act in all_acts:
      out.append(get(act)(inp))

    sum_out = tf.reduce_sum(tf.add_n(out))
    with self.session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      print(sess.run(sum_out))


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
