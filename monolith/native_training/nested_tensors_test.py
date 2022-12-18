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

from monolith.native_training import nested_tensors


class NestedTensorTest(tf.test.TestCase):

  def testBasic(self):
    n = nested_tensors.NestedTensors({
        "a": tf.ones([]),
        "b": (tf.ones([]), tf.ones([])),
    })
    tensors = n.get_tensors()
    replaced = [tf.zeros_like(tensor) for tensor in tensors]
    result = n.get_nested_result(replaced)
    result = self.evaluate(result)
    self.assertDictEqual(result, {"a": 0, "b": (0, 0)})

  def testConstant(self):
    n = nested_tensors.NestedTensors({"a": {"b": 2}})
    tensors = n.get_tensors()
    self.assertLen(tensors, 0)
    result = n.get_nested_result([])
    self.assertDictEqual(result, {"a": {"b": 2}})

  def testRaggedTensor(self):

    n = nested_tensors.NestedTensors(tf.ragged.constant([[], [1], [2, 3]]))
    tensors = n.get_tensors()
    result = n.get_nested_result(tensors)
    self.assertAllEqual(result, [[], [1], [2, 3]])

  def testRaggedTensorWithPlaceHolder(self):

    n = nested_tensors.NestedTensors(tf.ragged.constant([[], [1], [2, 3]]))
    tensors = n.get_tensors()
    phs = [tf.compat.v1.placeholder(dtype=t.dtype) for t in tensors]
    result = n.get_nested_result(tensors)


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
