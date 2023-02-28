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

from monolith.native_training import native_task
from monolith.native_training import hvd_lib
from monolith.native_training import sync_training_hooks


class EofAwareTaskTest(tf.test.TestCase):

  def test_basic(self):

    class TestTask(native_task.NativeTask):

      def create_input_fn(self, mode):

        def input_fn():
          return tf.data.Dataset.from_tensor_slices(
              tf.constant([1, 2, 3], dtype=tf.int64))

        return input_fn

      def create_model_fn(self):

        def model_fn(features, mode, config):
          gs = tf.compat.v1.train.get_or_create_global_step()
          train_op = tf.compat.v1.assign_add(gs, features)
          return tf.estimator.EstimatorSpec(mode,
                                            train_op=train_op,
                                            loss=tf.constant(0.0))

        return model_fn

    hvd_lib.init()
    p = TestTask.params()
    p.name = "test"
    t = TestTask(p)
    t = sync_training_hooks.EofAwareTask(t)
    est = tf.estimator.Estimator(t.create_model_fn())
    est.train(t.create_input_fn(tf.estimator.ModeKeys.TRAIN))
    self.assertEqual(est.get_variable_value("global_step"), 6)

  def test_dict(self):

    class TestTask(native_task.NativeTask):

      def create_input_fn(self, mode):

        def input_fn():
          ds = tf.data.Dataset.from_tensor_slices(
              tf.constant([1, 2, 3], dtype=tf.int64))
          ds = ds.map(lambda x: {"1": x})
          return ds

        return input_fn

      def create_model_fn(self):

        def model_fn(features, mode, config):
          gs = tf.compat.v1.train.get_or_create_global_step()
          train_op = tf.compat.v1.assign_add(gs, features["1"])
          return tf.estimator.EstimatorSpec(mode,
                                            train_op=train_op,
                                            loss=tf.constant(0.0))

        return model_fn

    hvd_lib.init()
    p = TestTask.params()
    p.name = "test"
    t = TestTask(p)
    t = sync_training_hooks.EofAwareTask(t)
    est = tf.estimator.Estimator(t.create_model_fn())
    est.train(t.create_input_fn(tf.estimator.ModeKeys.TRAIN))
    self.assertEqual(est.get_variable_value("global_step"), 6)


if __name__ == "__main__":
  tf.test.main()
