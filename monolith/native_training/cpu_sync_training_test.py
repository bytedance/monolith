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
import time

import numpy as np
import tensorflow as tf

# Note: this needs to be set before monolith.native_training, to ensure the
# imports work correctly.
os.environ["MONOLITH_WITH_HOROVOD"] = "True"

from monolith.native_training import cpu_training, embedding_combiners, feature, device_utils
from monolith.native_training.native_task import NativeTask
from monolith.native_training.data.training_instance.python.parser_utils import advanced_parse

import horovod.tensorflow as hvd

test_folder = os.environ["TEST_TMPDIR"]


class FeatureTask(NativeTask):
  """A test task that will collect some information in model_fn."""

  def create_input_fn(self, _):

    def input_fn():
      return tf.data.Dataset.from_tensors({
          "feature": tf.ragged.constant([[1, 2, 3, 4]], dtype=np.int64)
      }).map(advanced_parse).repeat(5)

    return input_fn

  def create_model_fn(self):

    def model_fn(mode, **kwargs):
      slot = self.ctx.feature_factory.create_feature_slot(
          feature.FeatureSlotConfig(name="slot"))
      s = slot.add_feature_slice(5)
      fc = feature.FeatureColumnV1(slot, "feature")
      embedding = fc.embedding_lookup(s)
      if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=tf.constant(0))
      all_embeddings = [fc.get_all_embeddings_concat()]
      loss = tf.reduce_sum(embedding)
      grads = tf.gradients(loss, all_embeddings)
      print1 = tf.print("embedding: ", embedding)
      print2 = tf.print("all_embeddings: ", all_embeddings)
      with tf.control_dependencies([print1, print2]):
        train_op = tf.group(
            self._ctx.feature_factory.apply_gradients(zip(
                grads, all_embeddings)))
      return tf.estimator.EstimatorSpec(mode,
                                        train_op=train_op,
                                        loss=loss,
                                        predictions=tf.constant(0))

    return model_fn


class FloatFeatureTask(NativeTask):
  """A test task that will use float feature in model_fn."""

  def create_input_fn(self, _):

    def input_fn():
      return tf.data.Dataset.from_tensors({
          "ragged_feature": tf.ragged.constant([[0, 0]], dtype=np.int64),
          "float_feature": tf.constant([[1.]], dtype=tf.float32)
      }).map(advanced_parse)

    return input_fn

  def create_model_fn(self):

    def model_fn(features, mode, **kwargs):
      slot = self.ctx.feature_factory.create_feature_slot(
          feature.FeatureSlotConfig(name="slot"))
      s = slot.add_feature_slice(5)
      fc = feature.FeatureColumnV1(slot, "ragged_feature")
      embedding = fc.embedding_lookup(s)
      float_feature = features["float_feature"]
      predictions = tf.reduce_sum(float_feature, axis=-1)
      if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
      all_embeddings = [fc.get_all_embeddings_concat()]
      loss = tf.reduce_sum(embedding)
      grads = tf.gradients(loss, all_embeddings)
      train_op = tf.group(
          self._ctx.feature_factory.apply_gradients(zip(grads, all_embeddings)))
      return tf.estimator.EstimatorSpec(mode,
                                        train_op=train_op,
                                        loss=loss,
                                        predictions=predictions)

    return model_fn


class NonFeatureTask(NativeTask):

  def create_input_fn(self, _):

    def input_fn():
      return tf.data.Dataset.from_tensors([1])

    return input_fn

  def create_model_fn(self):

    def model_fn(features, mode, config):
      return tf.estimator.EstimatorSpec(mode,
                                        train_op=tf.group(features),
                                        loss=tf.constant(0.0),
                                        predictions=tf.constant(0))

    return model_fn


class SequenceFeatureTask(NativeTask):
  """A test task that will use float feature in model_fn."""

  def create_input_fn(self, mode):
    del mode

    def input_fn():
      return tf.data.Dataset.from_tensors({
          "sequence_feature":
              tf.ragged.constant([[1, 2], [], [3, 4, 5]], dtype=np.int64),
      }).map(advanced_parse)

    return input_fn

  def create_model_fn(self):

    def model_fn(features, mode, **kwargs):
      slot = self.ctx.feature_factory.create_feature_slot(
          feature.FeatureSlotConfig(name="slot"))
      s = slot.add_feature_slice(5)
      fc = feature.FeatureColumnV1(slot,
                                   "sequence_feature",
                                   combiner=embedding_combiners.FirstN(2))
      embedding = fc.embedding_lookup(s)
      sequence_feature = features["sequence_feature"]
      predictions = tf.reduce_sum(sequence_feature, axis=-1)
      if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
      all_embeddings = [fc.get_all_embeddings_concat()]
      loss = tf.reduce_sum(all_embeddings)
      grads = tf.gradients(loss, all_embeddings)
      train_op = tf.group(
          self._ctx.feature_factory.apply_gradients(zip(grads, all_embeddings)))
      return tf.estimator.EstimatorSpec(mode,
                                        train_op=train_op,
                                        loss=loss,
                                        predictions=predictions)

    return model_fn


class CpuSyncTrainTest(tf.test.TestCase):

  def test_cpu_training_feature(self):
    hvd.init()
    p = FeatureTask.params()
    p.name = "feature_task"
    task = FeatureTask(p)
    training = cpu_training.CpuTraining(
        cpu_training.CpuTrainingConfig(num_workers=hvd.size(),
                                       num_ps=0,
                                       reorder_fids_in_data_pipeline=True,
                                       embedding_prefetch_capacity=1,
                                       enable_sync_training=True), task)
    run_config = tf.estimator.RunConfig(
        model_dir=os.path.join(test_folder, "test_cpu_sync_training_feature"),
        device_fn=device_utils.default_device_fn)
    est = tf.estimator.Estimator(training.create_model_fn(), config=run_config)
    est.train(training.create_input_fn(tf.estimator.ModeKeys.TRAIN), steps=2)

  def test_cpu_training_float_feature(self):
    hvd.init()
    p = FloatFeatureTask.params()
    p.name = "float_feature_task"
    task = FloatFeatureTask(p)
    training = cpu_training.CpuTraining(
        cpu_training.CpuTrainingConfig(num_workers=hvd.size(),
                                       num_ps=0,
                                       reorder_fids_in_data_pipeline=True,
                                       embedding_prefetch_capacity=1,
                                       enable_sync_training=True), task)
    run_config = tf.estimator.RunConfig(
        model_dir=os.path.join(test_folder,
                               "test_cpu_sync_training_float_feature"),
        device_fn=device_utils.default_device_fn)
    est = tf.estimator.Estimator(training.create_model_fn(), config=run_config)
    est.train(training.create_input_fn(tf.estimator.ModeKeys.TRAIN), steps=2)

  def test_cpu_training_sequence_feature(self):
    hvd.init()
    p = SequenceFeatureTask.params()
    p.name = "sequence_feature_task"
    task = SequenceFeatureTask(p)
    training = cpu_training.CpuTraining(
        cpu_training.CpuTrainingConfig(num_workers=hvd.size(),
                                       num_ps=0,
                                       reorder_fids_in_data_pipeline=True,
                                       embedding_prefetch_capacity=1,
                                       enable_sync_training=True,
                                       hashtable_init_capacity=100000,
                                       enable_embedding_postpush=True), task)
    run_config = tf.estimator.RunConfig(
        model_dir=os.path.join(test_folder,
                               "test_cpu_training_sequence_feature"),
        device_fn=device_utils.default_device_fn)
    est = tf.estimator.Estimator(training.create_model_fn(), config=run_config)
    est.train(training.create_input_fn(tf.estimator.ModeKeys.TRAIN), steps=2)

  def test_cpu_training_non_feature(self):
    hvd.init()
    p = NonFeatureTask.params()
    p.name = "non_feature_task"
    task = NonFeatureTask(p)
    training = cpu_training.CpuTraining(
        cpu_training.CpuTrainingConfig(num_workers=hvd.size(),
                                       num_ps=0,
                                       embedding_prefetch_capacity=1,
                                       hashtable_init_capacity=100000,
                                       enable_sync_training=True), task)
    run_config = tf.estimator.RunConfig(
        model_dir=os.path.join(test_folder,
                               "test_cpu_sync_training_non_feature"),
        device_fn=device_utils.default_device_fn)
    est = tf.estimator.Estimator(training.create_model_fn(), config=run_config)
    est.train(training.create_input_fn(tf.estimator.ModeKeys.TRAIN), steps=2)


class DistributedSyncTrainTest(tf.test.TestCase):

  def test_basic(self):
    hvd.init()
    model_dir = os.path.join(test_folder, "sync_training_basic")
    params = FeatureTask.params()
    params.name = "test_task"
    params.train.max_steps = 2
    # TODO(zouxuan): async push breaks the test, needs further triage.
    cpu_training.distributed_sync_train(
        cpu_training.DistributedCpuTrainingConfig(
            model_dir=model_dir,
            enable_sync_training=True,
            reorder_fids_in_data_pipeline=True,
            embedding_prefetch_capacity=1,
            hashtable_init_capacity=100000,
            enable_embedding_postpush=False), params)

  def test_sparse_pipelining(self):
    hvd.init()
    model_dir = os.path.join(test_folder, "sync_training_pipelined")
    params = FeatureTask.params()
    params.name = "test_task"
    params.train.max_steps = 4
    cpu_training.distributed_sync_train(
        cpu_training.DistributedCpuTrainingConfig(
            model_dir=model_dir,
            enable_sync_training=True,
            reorder_fids_in_data_pipeline=True,
            embedding_prefetch_capacity=1,
            enable_pipelined_bwda2a=True,
            enable_pipelined_fwda2a=True,
            hashtable_init_capacity=100000,
            enable_embedding_postpush=False), params)


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
