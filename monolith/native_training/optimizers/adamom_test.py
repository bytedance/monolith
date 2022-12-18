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
from tensorflow.python.framework.ops import name_from_scope_name

from monolith.native_training.optimizers import adamom


class AdamomTest(tf.test.TestCase):

  def testBasic(self):
    v = tf.Variable([0.1], name="var")
    loss = 0.12 * v
    opt = adamom.AdamomOptimizer(learning_rate=0.1,
                                 weight_decay=0.01,
                                 ada_decay=0.99,
                                 mom_decay=0.9)
    update = opt.minimize(loss)
    with self.session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      sess.run(update)
      all_vars = tf.compat.v1.all_variables()
      vars_map = sess.run({var.name: var for var in all_vars})
    eps = 1e-8
    found_count = 0
    for name, val in vars_map.items():
      if name.find("/m") >= 0:
        found_count += 1
        self.assertNear(val, 0.0121, eps)
      elif name.find("/c") >= 0:
        found_count += 1
        self.assertNear(val, 1.0, eps)
      elif name.find("/v") >= 0:
        found_count += 1
        self.assertNear(val, 0.014641, eps)
      else:
        found_count += 1
        # Must be variable
        self.assertNear(val, 0.090000336, eps)
    self.assertEqual(found_count, 4)


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()