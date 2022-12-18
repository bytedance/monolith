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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import enum
import functools
import re
import sys

from absl import app as absl_app
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf
import unittest

import monolith.core.base_embedding_host_call as base_embedding_host_call

tf.disable_eager_execution()


class BaseEmbeddingHostCallTest(unittest.TestCase):

  def test_compute_new_value(self):
    global_step = tf.train.get_or_create_global_step()
    params = {
        "enable_host_call": False,
        "context": None,
        "cpu_test": False,
        "host_call_every_n_steps": 100
    }
    host_call = base_embedding_host_call.BaseEmbeddingHostCall(
        "", False, False, False, False, 10, params)

    base_value = tf.zeros([10], dtype=tf.int32)
    delta_value = tf.ones([2], dtype=tf.int32)
    offset = tf.constant(1, dtype=tf.int32)
    base_value = host_call._compute_new_value(base_value, delta_value, offset)
    expected_value = tf.constant([0, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=tf.int32)
    ret = tf.reduce_all(tf.math.equal(base_value, expected_value))
    with tf.Session() as sess:
      ret = sess.run(ret)
      self.assertTrue(ret)

    offset = tf.constant(5)
    base_value = host_call._compute_new_value(base_value, delta_value, offset)
    expected_value = tf.constant([0, 1, 1, 0, 0, 1, 1, 0, 0, 0], dtype=tf.int32)
    ret = tf.reduce_all(tf.math.equal(base_value, expected_value))
    with tf.Session() as sess:
      ret = sess.run(ret)
      self.assertTrue(ret)

    offset = tf.constant(6)
    base_value = host_call._compute_new_value(base_value, delta_value, offset)
    expected_value = tf.constant([0, 1, 1, 0, 0, 1, 2, 1, 0, 0], dtype=tf.int32)
    ret = tf.reduce_all(tf.math.equal(base_value, expected_value))
    with tf.Session() as sess:
      ret = sess.run(ret)
      self.assertTrue(ret)


if __name__ == "__main__":
  unittest.main(verbosity=2)
