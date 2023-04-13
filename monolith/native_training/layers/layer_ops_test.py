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

  def test_feature_insight(self):
    segment_sizes = [3, 2, 4]
    input_embedding = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                       1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                       2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9]
    input_embedding_tensor = tf.constant(value=input_embedding, shape=(3, 9), dtype=tf.float32)
    weight = [0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1,
              0.9, 0.8, 0.7, 0.6, 0.5, 0.6, 0.7, 0.8, 0.9]
    weight_tensor = tf.constant(value=weight, shape=(9, 2), dtype=tf.float32)
    
    input_embedding_splits = tf.split(input_embedding_tensor, num_or_size_splits=segment_sizes, axis=1)
    weight_splits = tf.split(weight_tensor, num_or_size_splits=segment_sizes, axis=0)
    concatenated = tf.concat([tf.matmul(ip, w) for ip, w in zip(input_embedding_splits, weight_splits)], axis=1)
    k, num_feature = 2, 3
    segment_ids = []
    for i in range(num_feature):
      segment_ids.extend([i] * k)
    segment_ids_tensor = tf.constant(value=segment_ids, shape=(k * num_feature,), dtype=tf.int32)
    res_exp = tf.transpose(tf.math.segment_sum(tf.transpose(concatenated * concatenated), segment_ids=segment_ids_tensor))
    out = layer_ops.feature_insight(input_embedding_tensor, weight_tensor, segment_sizes, aggregate=True)
    self.assertAllClose(out, res_exp)

  def test_feature_insight_grad(self):
    segment_sizes = [3, 2, 4]
    input_embedding = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                       1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                       2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9]
    input_embedding_tensor = tf.constant(value=input_embedding, shape=(3, 9), dtype=tf.float32)
    weight = [0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1,
              0.9, 0.8, 0.7, 0.6, 0.5, 0.6, 0.7, 0.8, 0.9]
    weight_tensor = tf.constant(value=weight, shape=(9, 2), dtype=tf.float32)
    
    with tf.GradientTape(persistent=True) as g:
      g.watch(input_embedding_tensor)
      g.watch(weight_tensor)
      input_embedding_splits = tf.split(input_embedding_tensor, num_or_size_splits=segment_sizes, axis=1)
      weight_splits = tf.split(weight_tensor, num_or_size_splits=segment_sizes, axis=0)
      res_exp = tf.concat([tf.matmul(ip, w) for ip, w in zip(input_embedding_splits, weight_splits)], axis=1)
      out = layer_ops.feature_insight(input_embedding_tensor, weight_tensor, segment_sizes)

    input_embedding_grad_exp = g.gradient(res_exp, input_embedding_tensor)
    weight_grad_exp = g.gradient(res_exp, weight_tensor)
    input_embedding_grad = g.gradient(out, input_embedding_tensor)
    weight_grad = g.gradient(out, weight_tensor)
    
    self.assertAllClose(out, res_exp)
    self.assertAllClose(input_embedding_grad, input_embedding_grad_exp)
    self.assertAllClose(weight_grad, weight_grad_exp)



if __name__ == '__main__':
  tf.test.main()
