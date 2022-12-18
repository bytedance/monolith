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

from monolith.native_training import variables
from monolith.native_training import test_utils


class CachedVariableTest(tf.test.TestCase):

  def testBasic(self):
    servers, config = test_utils.create_test_ps_cluster(2)
    with tf.compat.v1.Session(target=servers[0].target, config=config) as sess:
      with tf.variable_creator_scope(variables.cached_variable_creator):
        with tf.device("/job:ps/task:1"):
          var = tf.Variable(5.0)
      sess.run([
          tf.compat.v1.global_variables_initializer(),
          tf.compat.v1.local_variables_initializer()
      ])
      # We use var * 1.0 since direct run var will use var.ref()
      # which is original value of var.
      self.assertAllEqual(5.0, self.evaluate(var * 1.0))

      update_op = var.assign_add(2.0)
      sess.run(update_op)
      # update op won't take effect until fetch happened.
      self.assertAllEqual(5.0, self.evaluate(var * 1.0))
      # But the original value should be updated.
      self.assertAllEqual(7.0, self.evaluate(var))

      sess.run(variables.fetch_all_cached_variables())
      sess.run(variables.assign_all_cached_variables())
      # update takes effect.
      self.assertAllEqual(7.0, self.evaluate(var * 1.0))

  def testHook(self):
    servers, config = test_utils.create_test_ps_cluster(2)
    with tf.variable_creator_scope(variables.cached_variable_creator):
      with tf.device("/job:ps/task:1"):
        var = tf.Variable(5.0)
    var_cached = var * 1.0
    sub_op = tf.compat.v1.assign_sub(var, 1.0)
    with tf.compat.v1.train.SingularMonitoredSession(
        master=servers[0].target,
        config=config,
        hooks=[variables.FetchAllCachedVariablesHook()]) as sess:
      var_cached_value = sess.run(var_cached)
      self.assertAllEqual(5.0, var_cached_value)
      sess.run(sub_op)
      # At most twice, local var will be finally updated.
      var_cached_value = sess.run(var_cached)
      var_cached_value = sess.run(var_cached)
      self.assertAllEqual(4.0, var_cached_value)

  def testGradient(self):
    servers, config = test_utils.create_test_ps_cluster(2)
    with tf.variable_creator_scope(variables.cached_variable_creator):
      with tf.device("/job:ps/task:1"):
        var = tf.Variable(5.0)
    loss = var
    opt = tf.compat.v1.train.GradientDescentOptimizer(1.0)
    op = opt.minimize(loss)
    with tf.compat.v1.Session(target=servers[0].target, config=config) as sess:
      sess.run([
          tf.compat.v1.global_variables_initializer(),
          tf.compat.v1.local_variables_initializer()
      ])
      sess.run(op)
      self.assertAllEqual(4.0, sess.run(var))
      # The result should not be fetched yet.
      self.assertAllEqual(5.0, sess.run(var * 1.0))


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
