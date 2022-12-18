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

import os

import tensorflow as tf

from monolith.native_training import file_ops


class WritableFileTest(tf.test.TestCase):

  def test_basic(self):
    filename = os.environ["TEST_TMPDIR"] + "/test_basic/test_name"

    times = 1000

    @tf.function
    def write():
      f = file_ops.WritableFile(filename)
      for i in tf.range(times):
        f.append("1234")
      f.close()

    self.evaluate(write())

    with tf.io.gfile.GFile(filename) as f:
      self.assertAllEqual(f.read(), "1234" * times)

  def test_hook(self):
    filename = os.environ["TEST_TMPDIR"] + "/test_hook/test_name"
    f = file_ops.WritableFile(filename)
    write_op = f.append("1234")

    with tf.compat.v1.train.MonitoredSession(
        hooks=[file_ops.FileCloseHook([f])]) as sess:
      sess.run(write_op)

    with tf.io.gfile.GFile(filename) as f:
      self.assertAllEqual(f.read(), "1234")


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
