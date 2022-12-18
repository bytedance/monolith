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

from monolith.native_training.hooks import hook_utils


class HookUtilsTest(tf.test.TestCase):

  def testBeforeAfterSaverListener(self):
    # This is mainly for testing compiling
    base_l = tf.estimator.CheckpointSaverListener()
    l1 = hook_utils.BeforeSaveListener(base_l)
    l2 = hook_utils.AfterSaveListener(base_l)
    with self.session() as sess:
      l1.before_save(sess, 0)
      l1.after_save(sess, 0)
      l2.before_save(sess, 0)
      l2.after_save(sess, 0)


if __name__ == "__main__":
  tf.test.main()
