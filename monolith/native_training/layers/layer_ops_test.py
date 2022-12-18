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
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import test_util

from monolith.native_training.layers.layer_ops import ffm
from monolith.native_training.layers import layer_ops

tf.random.set_seed(0)


class LayerOpsTest(tf.test.TestCase):

  def test_ffm_mul(self):
    with test_util.use_gpu():
      left = tf.random.uniform(shape=(8, 10 * 4), minval=0, maxval=10)
      right = tf.random.uniform(shape=(8, 12 * 4), minval=0, maxval=10)
      output_maybe_on_gpu = ffm(left=left, right=right, dim_size=4)
      if tf.test.is_gpu_available():
        self.assertEqual(output_maybe_on_gpu.device,
                         '/job:localhost/replica:0/task:0/device:GPU:0')
      with tf.device("/device:CPU:0"):
        output_on_cpu = ffm(left=left, right=right, dim_size=4)
        self.assertEqual(output_on_cpu.device,
                         '/job:localhost/replica:0/task:0/device:CPU:0')
      self.assertTrue(output_maybe_on_gpu.shape == (8, 480))
      self.assertAllEqual(output_maybe_on_gpu, output_on_cpu)

  def test_ffm_mul_grad(self):
    with test_util.use_gpu():
      left = tf.random.uniform(shape=(8, 10 * 4), minval=0, maxval=10)
      right = tf.random.uniform(shape=(8, 12 * 4), minval=0, maxval=10)
      with tf.GradientTape() as g:
        g.watch(left)
        g.watch(right)
        out = ffm(left=left, right=right, dim_size=4)
        loss = tf.reduce_sum(out)
        left_grad_maybe_on_gpu, right_grad_maybe_on_gpu = g.gradient(
            loss, [left, right])
        self.assertTrue(left_grad_maybe_on_gpu.shape == (8, 40))
        self.assertTrue(right_grad_maybe_on_gpu.shape == (8, 48))

      with tf.device("/device:CPU:0"), tf.GradientTape() as g:
        g.watch(left)
        g.watch(right)
        out = ffm(left=left, right=right, dim_size=4)
        loss = tf.reduce_sum(out)
        left_grad_on_cpu, right_grad_on_cpu = g.gradient(loss, [left, right])
        self.assertEqual(left_grad_on_cpu.device,
                         '/job:localhost/replica:0/task:0/device:CPU:0')
        self.assertEqual(right_grad_on_cpu.device,
                         '/job:localhost/replica:0/task:0/device:CPU:0')
        self.assertAllEqual(left_grad_maybe_on_gpu, left_grad_on_cpu)
        self.assertAllEqual(right_grad_maybe_on_gpu, right_grad_on_cpu)

  def test_ffm_dot(self):
    with test_util.use_gpu():
      left = tf.random.uniform(shape=(8, 10 * 4), minval=0, maxval=10)
      right = tf.random.uniform(shape=(8, 12 * 4), minval=0, maxval=10)
      output_maybe_on_gpu = ffm(left=left,
                                right=right,
                                dim_size=4,
                                int_type='dot')
      if tf.test.is_gpu_available():
        self.assertEqual(output_maybe_on_gpu.device,
                         '/job:localhost/replica:0/task:0/device:GPU:0')
      with tf.device("/device:CPU:0"):
        output_on_cpu = ffm(left=left, right=right, dim_size=4, int_type='dot')
        self.assertEqual(output_on_cpu.device,
                         '/job:localhost/replica:0/task:0/device:CPU:0')
      self.assertTrue(output_maybe_on_gpu.shape == (8, 120))
      self.assertAllEqual(output_maybe_on_gpu, output_on_cpu)

  def test_ffm_dot_grad(self):
    with test_util.use_gpu():
      left = tf.random.uniform(shape=(8, 10 * 4), minval=0, maxval=10)
      right = tf.random.uniform(shape=(8, 12 * 4), minval=0, maxval=10)
      with tf.GradientTape() as g:
        g.watch(left)
        g.watch(right)
        out = ffm(left=left, right=right, dim_size=4, int_type='dot')
        loss = tf.reduce_sum(out)
        left_grad_maybe_on_gpu, right_grad_maybe_on_gpu = g.gradient(
            loss, [left, right])

        self.assertTrue(left_grad_maybe_on_gpu.shape == (8, 40))
        self.assertTrue(right_grad_maybe_on_gpu.shape == (8, 48))

      with tf.device("/device:CPU:0"), tf.GradientTape() as g:
        g.watch(left)
        g.watch(right)
        out = ffm(left=left, right=right, dim_size=4, int_type='dot')
        loss = tf.reduce_sum(out)
        left_grad_on_cpu, right_grad_on_cpu = g.gradient(loss, [left, right])
        self.assertEqual(left_grad_on_cpu.device,
                         '/job:localhost/replica:0/task:0/device:CPU:0')
        self.assertEqual(right_grad_on_cpu.device,
                         '/job:localhost/replica:0/task:0/device:CPU:0')
        self.assertAllEqual(left_grad_maybe_on_gpu, left_grad_on_cpu)
        self.assertAllEqual(right_grad_maybe_on_gpu, right_grad_on_cpu)



if __name__ == '__main__':
  tf.test.main()
