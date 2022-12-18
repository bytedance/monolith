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

from monolith.native_training.model_export import export_utils
from monolith.native_training.model_export import export_context


class ExportUtilsTest(tf.test.TestCase):

  def testBasic(self):
    # Currently we only test gramar until we can figure out a way
    # to compile tensorflow serving here.
    with export_context.enter_export_mode(
        export_context.EXPORT_MODE.STANDALONE):

      def remote_func(d):
        return d["a"] * 3 + d["b"] * 4

      helper = export_utils.RemotePredictHelper("test_func", {
          "a": tf.constant(1),
          "b": tf.constant(2)
      }, remote_func)

      result = helper.call_remote_predict("model_name")
      self.assertIsInstance(result, tf.Tensor)


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
