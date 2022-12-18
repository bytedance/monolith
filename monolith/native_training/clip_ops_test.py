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

import numpy as np

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util

from monolith.native_training import clip_ops


class ClipOpsTest(tf.test.TestCase):

  def _test_clip_by_global_norm(self, inputs, clip_norm, expected=None):
    with tf.compat.v1.Session() as sess, test_util.use_gpu():
      t_list = [ops.convert_to_tensor(t, dtype=tf.float32) for t in inputs]
      clipped = clip_ops.clip_by_global_norm(t_list, clip_norm)
      r, second_branch_check_input_soundness = sess.run([clipped, t_list])
      result, _ = r
      if expected is None:
        expected, _ = sess.run(tf.clip_by_global_norm(t_list, clip_norm))
      self.assertAllClose(result, expected)
      # second_branch_check_input_soundness will break allclose,
      #   if input mem (t_list) gets modified inplace (clipped).
      self.assertAllClose(second_branch_check_input_soundness, inputs)

  def test_clip_by_global_norm(self):
    # Simple example
    self._test_clip_by_global_norm([[-3.0, 0.0, 0.0], [4.0, 0.0, 0.0]], 4.0,
                                   [[-2.4, 0.0, 0.0], [3.2, 0.0, 0.0]])
    # Uneven shape example
    self._test_clip_by_global_norm([[-3.0, 0.0, 0.0], [0.0, 0.0, 4.0, 0.0]],
                                   4.0,
                                   [[-2.4, 0.0, 0.0], [0.0, 0.0, 3.2, 0.0]])
    # No clipping.
    self._test_clip_by_global_norm([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 4.0,
                                   [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    # Zero norm.
    self._test_clip_by_global_norm([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], 4.0,
                                   [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    # Exploded grad.
    nan_arr = np.empty((2, 3))
    nan_arr[:] = np.nan
    self._test_clip_by_global_norm(
        [[float('inf'), float('inf'), float('inf')],
         [float('inf'), float('inf'), float('inf')]], 4.0, nan_arr)
    # Large grad.
    DENSE_SHAPES = [(328, 128), (128,), (128,), (128, 64), (64,), (64,), (1,),
                    (256, 256), (256,), (256,), (256, 128), (128,), (128,),
                    (128, 1), (1,), (1,), (2488, 256), (256,), (256,),
                    (3184, 256), (256,), (256,), (96, 128), (128,), (128,),
                    (128, 64), (64,), (64,), (1,), (64, 16), (16,), (16,),
                    (1609, 2048), (2048,), (2048,), (2048, 1024), (1024,),
                    (1024,), (1024, 512), (512,), (512,), (512, 256), (256,),
                    (256,), (256, 1), (1,), (1,), (96, 64), (64,), (64,),
                    (64, 1), (1,), (1,)]
    grads = [np.random.uniform(size=s) for s in DENSE_SHAPES]
    self._test_clip_by_global_norm(grads, 1.0)


class NormOpsTest(tf.test.TestCase):

  def _test_global_norm(self, inputs, expected):
    with tf.compat.v1.Session() as sess, test_util.use_gpu():
      inputs = [ops.convert_to_tensor(t, dtype=tf.float32) for t in inputs]
      g = sess.run(clip_ops._global_norm(inputs))
      self.assertAllClose(g, expected)

  @test_util.run_gpu_only
  def test_it(self):
    self._test_global_norm(
        [[float('inf'), float('inf'), float('inf')],
         [float('inf'), float('inf'), float('inf')]], float('inf'))
    self._test_global_norm([[-3.0, 0.0, 0.0], [4.0, 0.0, 0.0]], 5.0)
    self._test_global_norm([[-3.0, 0.0, 0.0], [0.0, 0.0, 4.0, 0.0]], 5.0)


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
