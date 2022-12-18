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

from monolith.native_training.hooks import session_hooks


class SessionHooksTest(tf.test.TestCase):

  def testBasic(self):
    self.assertTrue(session_hooks.get_current_session() is None)
    with tf.compat.v1.train.MonitoredSession(
        hooks=[session_hooks.SetCurrentSessionHook()]) as sess:
      sess.run([])
      session_hooks.get_current_session()
      self.assertTrue(session_hooks.get_current_session() is not None)


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
