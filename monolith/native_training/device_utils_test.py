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

from absl import logging
import tensorflow as tf

from monolith.native_training import device_utils


class DeviceUtilsTest(tf.test.TestCase):

  def test_basic(self):
    with tf.Graph().as_default() as g, g.device(device_utils.default_device_fn):
      a = tf.constant(1)
      self.assertEqual(a.device, "/device:CPU:0")

  def test_cpu_only(self):
    device_utils.disable_gpu_training()
    with tf.Graph().as_default() as g, g.device(device_utils.default_device_fn):
      with tf.device("/device:GPU:0"):
        a = tf.constant(1)
      self.assertEqual(a.device, "/device:CPU:0")

  def test_str_context(self):
    device_utils.enable_gpu_training()
    with tf.Graph().as_default() as g, g.device(device_utils.default_device_fn):
      a = tf.constant(1)
      with tf.device("GPU:0"):
        b = tf.constant(1)
      c = tf.constant(1)
      self.assertEqual(a.device, "/device:CPU:0")
      self.assertEqual(b.device, "/device:GPU:0")
      self.assertEqual(c.device, "/device:CPU:0")

  def test_str_nested_contexts(self):
    device_utils.enable_gpu_training()
    with tf.Graph().as_default() as g, g.device(device_utils.default_device_fn):
      a = tf.constant(1)
      with tf.device("CPU:0"):
        b = tf.constant(1)
        with tf.device("GPU:0"):
          c = tf.constant(1)
          with tf.device("GPU:1"):
            d = tf.constant(1)
      self.assertEqual(a.device, "/device:CPU:0")
      self.assertEqual(b.device, "/device:CPU:0")
      self.assertEqual(c.device, "/device:GPU:0")
      self.assertEqual(d.device, "/device:GPU:1")

  def test_cpu_device_merge(self):
    # For example, in async training case, we have device job and task string.
    device_utils.disable_gpu_training()
    with tf.Graph().as_default() as g, g.device(device_utils.default_device_fn):
      with tf.device("/job:my_ps/task:0"):
        a = tf.constant(1)
        with tf.device("GPU:0"):
          assert not device_utils.within_placement_context_of("GPU")
          assert device_utils.within_placement_context_of("CPU")
          b = tf.constant(1)
    self.assertEqual(a.device, "/job:my_ps/task:0/device:CPU:0")
    self.assertEqual(b.device, "/job:my_ps/task:0/device:CPU:0")

  def test_gpu_device_merge(self):
    device_utils.enable_gpu_training()
    with tf.Graph().as_default() as g, g.device(device_utils.default_device_fn):
      with tf.device("/job:worker/task:0"):
        with tf.device("/job:ps"):
          a = tf.constant(1)
        with tf.device("GPU:0"):
          b = tf.constant(1)
        with device_utils.maybe_device_if_allowed("GPU:1"):
          assert device_utils.within_placement_context_of("GPU")
          assert not device_utils.within_placement_context_of("CPU")
          c = tf.constant(1)
    self.assertEqual(a.device, "/job:ps/task:0/device:CPU:0")
    self.assertEqual(b.device, "/job:worker/task:0/device:GPU:0")
    self.assertEqual(c.device, "/job:worker/task:0/device:GPU:1")

  def test_process_gpu_map(self):
    self.assertEqual(
        device_utils.get_visible_gpus(local_rank=2, processes_per_gpu=1), "2")
    self.assertEqual(
        device_utils.get_visible_gpus(local_rank=1, processes_per_gpu=2), "0")
    self.assertEqual(
        device_utils.get_visible_gpus(local_rank=2, processes_per_gpu=2), "1")
    self.assertEqual(
        device_utils.get_visible_gpus(local_rank=3, processes_per_gpu=2), "1")


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
