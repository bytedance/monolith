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

from collections import defaultdict
import copy
import os
import subprocess
import threading
import time
from google.protobuf import text_format
from typing import Dict, List
from unittest import mock

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io import file_io

from monolith.native_training import cpu_training
from monolith.native_training import entry
from monolith.native_training import feature
from monolith.native_training import utils
from monolith.native_training.debugging import debugging_server
from monolith.native_training.model_export import saved_model_exporters
from monolith.native_training.model_export.export_context import ExportMode
from monolith.native_training.native_task import NativeTask
from monolith.native_training.proto import debugging_info_pb2
from monolith.native_training.runtime.hash_table import \
    embedding_hash_table_pb2
from monolith.native_training.service_discovery import ServiceDiscovery

FLAGS = flags.FLAGS
# TODO(leqi.zou): Finally remove this or rework with a better gflag util.
flags.DEFINE_bool("use_native_multi_hash_table", False,
                  "The test flag to control if use multi hash table.")


def inc_global_step_op() -> tf.Operation:
  global_step = tf.compat.v1.train.get_or_create_global_step()
  global_step = tf.compat.v1.assign_add(global_step, 1)
  return tf.group(global_step)


class FeatureTask(NativeTask):
  """A test task that will collect some information in model_fn."""

  def create_input_fn(self, mode):
    del mode

    def input_fn():
      tensor = tf.ragged.constant([[0, 0]], dtype=tf.int64)
      return tf.data.Dataset.from_tensors({"feature": tensor})

    return input_fn

  def create_model_fn(self):

    def model_fn(mode, features, config):
      slot = self.ctx.feature_factory.create_feature_slot(
          feature.FeatureSlotConfig(name="slot"))
      s = slot.add_feature_slice(5)
      fc = feature.FeatureColumnV1(slot, "feature")
      embedding = fc.embedding_lookup(s)
      all_embeddings = [fc.get_all_embeddings_concat()]
      if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode, predictions=tf.math.reduce_sum(embedding))
      grads = tf.gradients(-embedding, all_embeddings)
      train_op = tf.group(
          inc_global_step_op(),
          self._ctx.feature_factory.apply_gradients(zip(grads, all_embeddings)),
          features["feature"])
      return tf.estimator.EstimatorSpec(mode,
                                        train_op=train_op,
                                        loss=tf.constant(0.0),
                                        predictions=tf.constant(0))

    return model_fn

  def create_serving_input_receiver_fn(self):

    def serving_input_receiver_fn():
      return tf.estimator.export.ServingInputReceiver(
          {"feature": tf.ragged.constant([[0, 0]], dtype=tf.int64)},
          tf.compat.v1.placeholder(tf.string))

    return serving_input_receiver_fn


class FloatFeatureTask(NativeTask):
  """A test task that will use float feature in model_fn."""

  def create_input_fn(self, mode):
    del mode

    def input_fn():
      return tf.data.Dataset.from_tensors({
          "ragged_feature": tf.ragged.constant([[0, 0]], dtype=np.int64),
          "float_feature": tf.constant([[1.]], dtype=tf.float32)
      })

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
      all_embeddings = [fc.get_all_embeddings_concat()]
      grads = tf.gradients(-embedding, all_embeddings)
      train_op = tf.group(
          inc_global_step_op(),
          self._ctx.feature_factory.apply_gradients(zip(grads, all_embeddings)))
      return tf.estimator.EstimatorSpec(mode,
                                        train_op=train_op,
                                        loss=tf.constant(0.0),
                                        predictions=predictions)

    return model_fn


class SequenceFeatureTask(NativeTask):
  """A test task that will use float feature in model_fn."""

  def create_input_fn(self, mode):
    del mode

    def input_fn():
      return tf.data.Dataset.from_tensors({
          "sequence_feature":
              tf.ragged.constant([[1, 2], [], [3, 4, 5]], dtype=np.int64),
      })

    return input_fn

  def create_model_fn(self):

    def model_fn(features, mode, **kwargs):
      slot = self.ctx.feature_factory.create_feature_slot(
          feature.FeatureSlotConfig(name="slot"))
      s = slot.add_feature_slice(5)
      fc = feature.FeatureColumnV1(slot,
                                   "sequence_feature",
                                   combiner=feature.FeatureColumnV1.first_n(2))
      embedding = fc.embedding_lookup(s)
      sequence_feature = features["sequence_feature"]
      predictions = tf.reduce_sum(sequence_feature, axis=-1)
      all_embeddings = [fc.get_all_embeddings_concat()]
      grads = tf.gradients(-embedding, all_embeddings)
      train_op = tf.group(
          inc_global_step_op(),
          self._ctx.feature_factory.apply_gradients(zip(grads, all_embeddings)))
      return tf.estimator.EstimatorSpec(mode,
                                        train_op=train_op,
                                        loss=tf.constant(0.0),
                                        predictions=predictions)

    return model_fn


class FeatureWithSlotOccurrenceThresholdTask(NativeTask):
  """A test task that will collect some information in model_fn."""

  def create_input_fn(self, mode):
    del mode

    def input_fn():
      return tf.data.Dataset.from_tensors(
          {"feature": tf.ragged.constant([[0, 0]], dtype=np.int64)})

    return input_fn

  def create_model_fn(self):

    def model_fn(mode, **kwargs):

      slot = self.ctx.feature_factory.create_feature_slot(
          feature.FeatureSlotConfig(name="slot",
                                    slot_id=2021,
                                    occurrence_threshold=3))
      s = slot.add_feature_slice(5)
      fc = feature.FeatureColumnV1(slot, "feature")
      embedding = fc.embedding_lookup(s)
      all_embeddings = [fc.get_all_embeddings_concat()]
      grads = tf.gradients(-embedding, all_embeddings)
      train_op = tf.group(
          inc_global_step_op(),
          self._ctx.feature_factory.apply_gradients(zip(grads, all_embeddings)))
      return tf.estimator.EstimatorSpec(mode,
                                        train_op=train_op,
                                        loss=tf.constant(0.0),
                                        predictions=tf.constant(0))

    return model_fn


class FeatureWithExpireTimeTask(NativeTask):
  """A test task that will collect some information in model_fn."""

  def create_input_fn(self, mode):
    del mode

    def input_fn():
      return tf.data.Dataset.from_tensors({
          "feature_1":
              tf.ragged.constant([[1 << 48, (1 << 48) + 1]], dtype=np.int64),
          "feature_2":
              tf.ragged.constant([[2 << 48, (2 << 48) + 1]], dtype=np.int64),
          "req_time":
              tf.constant([[100]], dtype=tf.int64),
      })

    return input_fn

  def create_model_fn(self):

    def model_fn(mode, features, **kwargs):
      slot_1 = self.ctx.feature_factory.create_feature_slot(
          feature.FeatureSlotConfig(
              name="slot_1",
              slot_id=1,
              expire_time=0,
              default_vec_initializer=entry.ZerosInitializer()))
      s_1 = slot_1.add_feature_slice(5)
      fc_1 = feature.FeatureColumnV1(slot_1, "feature_1")
      embedding_1 = fc_1.embedding_lookup(s_1)

      slot_2 = self.ctx.feature_factory.create_feature_slot(
          feature.FeatureSlotConfig(
              name="slot_2",
              slot_id=2,
              expire_time=1,
              default_vec_initializer=entry.ZerosInitializer()))
      s_2 = slot_2.add_feature_slice(5)
      fc_2 = feature.FeatureColumnV1(slot_2, "feature_2")
      embedding_2 = fc_2.embedding_lookup(s_2)

      predictions = tf.concat([embedding_1, embedding_2], axis=0)
      all_embeddings = [
          fc_1.get_all_embeddings_concat(),
          fc_2.get_all_embeddings_concat(),
      ]
      grads = tf.gradients([embedding_1, embedding_2], all_embeddings)

      train_op = tf.group(
          inc_global_step_op(),
          self._ctx.feature_factory.apply_gradients(zip(grads, all_embeddings)))
      return tf.estimator.EstimatorSpec(mode,
                                        train_op=train_op,
                                        loss=tf.constant(0.0),
                                        predictions=predictions)

    return model_fn


class NonFeatureTask(NativeTask):

  def create_input_fn(self, mode):
    del mode

    def input_fn():
      return tf.data.Dataset.from_tensors([1])

    return input_fn

  def create_model_fn(self):

    def model_fn(features, mode, config):
      return tf.estimator.EstimatorSpec(mode,
                                        train_op=tf.group(
                                            inc_global_step_op(), features),
                                        loss=tf.constant(0.0),
                                        predictions=tf.constant(0))

    return model_fn


class CpuTrainTest(tf.test.TestCase):

  def test_cpu_training_feature(self):
    p = FeatureTask.params()
    p.name = "feature_task"
    task = FeatureTask(p)
    training = cpu_training.CpuTraining(cpu_training.CpuTrainingConfig(), task)
    est = tf.estimator.Estimator(
        training.create_model_fn(),
        os.path.join(os.environ["TEST_TMPDIR"], "test_cpu_training_feature"))
    est.train(training.create_input_fn(tf.estimator.ModeKeys.TRAIN))

  def test_with_misc_features(self):
    p = FeatureTask.params()
    p.name = "misc_features"
    task = FeatureTask(p)
    training = cpu_training.CpuTraining(
        cpu_training.CpuTrainingConfig(feature_eviction_on_save=True), task)
    est = tf.estimator.Estimator(
        training.create_model_fn(),
        os.path.join(os.environ["TEST_TMPDIR"], "test_with_misc_features"))
    est.train(training.create_input_fn(tf.estimator.ModeKeys.TRAIN))

  def test_with_export_when_saving(self):
    p = FeatureTask.params()
    p.serving.export_when_saving = True
    task = FeatureTask(p)
    training = cpu_training.CpuTraining(cpu_training.CpuTrainingConfig(), task)
    est = tf.estimator.Estimator(
        training.create_model_fn(),
        os.path.join(os.environ["TEST_TMPDIR"], "test_with_export_when_saving"))
    est.train(training.create_input_fn(tf.estimator.ModeKeys.TRAIN))

  def test_dense_only_export(self):
    p = FeatureTask.params()
    p.serving.export_when_saving = True
    p.serving.export_mode = ExportMode.DISTRIBUTED
    task = FeatureTask(p)
    training = cpu_training.CpuTraining(
        cpu_training.CpuTrainingConfig(dense_only_save_checkpoints_steps=10),
        task)
    est = tf.estimator.Estimator(
        training.create_model_fn(),
        os.path.join(os.environ["TEST_TMPDIR"], "test_dense_only_export"))
    est.train(training.create_input_fn(tf.estimator.ModeKeys.TRAIN))

  def test_with_prefetch_postpush(self):
    p = FeatureTask.params()
    p.name = "feature_task"
    task = FeatureTask(p)
    training = cpu_training.CpuTraining(
        cpu_training.CpuTrainingConfig(enable_variable_prefetch=True,
                                       enable_variable_postpush=True,
                                       enable_embedding_postpush=True,
                                       embedding_prefetch_capacity=1), task)
    est = tf.estimator.Estimator(
        training.create_model_fn(),
        os.path.join(os.environ["TEST_TMPDIR"], "test_with_prefetch_postpush"))
    est.train(training.create_input_fn(tf.estimator.ModeKeys.TRAIN))

  def test_cpu_training_float_feature(self):
    p = FloatFeatureTask.params()
    p.name = "float_feature_task"
    task = FloatFeatureTask(p)
    training = cpu_training.CpuTraining(cpu_training.CpuTrainingConfig(), task)
    est = tf.estimator.Estimator(
        training.create_model_fn(),
        os.path.join(os.environ["TEST_TMPDIR"],
                     "test_cpu_training_float_feature"))
    est.train(training.create_input_fn(tf.estimator.ModeKeys.TRAIN))

  def test_cpu_training_sequence_feature(self):
    p = SequenceFeatureTask.params()
    p.name = "sequence_feature_task"
    task = SequenceFeatureTask(p)
    training = cpu_training.CpuTraining(cpu_training.CpuTrainingConfig(), task)
    est = tf.estimator.Estimator(
        training.create_model_fn(),
        os.path.join(os.environ["TEST_TMPDIR"],
                     "test_cpu_training_sequence_feature"))
    est.train(training.create_input_fn(tf.estimator.ModeKeys.TRAIN))

  def test_cpu_training_with_slot_occurrence_threshold(self):
    p = FeatureWithSlotOccurrenceThresholdTask.params()
    p.name = "feature_with_slot_occurrence_task"
    task = FeatureWithSlotOccurrenceThresholdTask(p)
    training = cpu_training.CpuTraining(cpu_training.CpuTrainingConfig(), task)
    est = tf.estimator.Estimator(
        training.create_model_fn(),
        os.path.join(os.environ["TEST_TMPDIR"],
                     "test_cpu_training_with_slot_occurrence_threshold"))
    est.train(training.create_input_fn(tf.estimator.ModeKeys.TRAIN))
    slot_to_occurrence_threshold = training._slot_to_occurrence_threshold
    self.assertEqual(len(slot_to_occurrence_threshold), 1)
    self.assertTrue(2021 in slot_to_occurrence_threshold)
    self.assertEqual(slot_to_occurrence_threshold[2021], 3)

  def test_cpu_training_with_expire_time(self):
    p = FeatureWithExpireTimeTask.params()
    p.name = "feature_with_expire_time_task"
    task = FeatureWithExpireTimeTask(p)
    training = cpu_training.CpuTraining(cpu_training.CpuTrainingConfig(), task)
    base_name = os.path.join(os.environ["TEST_TMPDIR"],
                             "test_cpu_training_with_expire_time")
    # train
    est = tf.estimator.Estimator(training.create_model_fn(), base_name)
    est = est.train(training.create_input_fn(tf.estimator.ModeKeys.TRAIN))
    slot_to_expire_time = training._slot_to_expire_time
    self.assertEqual(len(slot_to_expire_time), 2)
    self.assertTrue(1 in slot_to_expire_time)
    self.assertTrue(2 in slot_to_expire_time)
    self.assertEqual(slot_to_expire_time[1], 0)
    self.assertEqual(slot_to_expire_time[2], 1)

    #predict
    result = est.predict(training.create_input_fn(
        tf.estimator.ModeKeys.PREDICT))

    result = list(result)
    expected = [[0, 0, 0, 0, 0],
                [-0.001414, -0.001414, -0.001414, -0.001414, -0.001414]]
    self.assertAllClose(result, expected)

  def test_cpu_training_non_feature(self):
    p = NonFeatureTask.params()
    p.name = "non_feature_task"
    task = NonFeatureTask(p)
    training = cpu_training.CpuTraining(cpu_training.CpuTrainingConfig(), task)
    est = tf.estimator.Estimator(
        training.create_model_fn(),
        os.path.join(os.environ["TEST_TMPDIR"],
                     "test_cpu_training_non_feature"))
    est.train(training.create_input_fn(tf.estimator.ModeKeys.TRAIN))

  def test_gpu_export(self):
    p = FeatureTask.params()
    p.name = "gpu_export"
    task = FeatureTask(p)
    training = cpu_training.CpuTraining(cpu_training.CpuTrainingConfig(), task)
    model_dir = os.path.join(os.environ["TEST_TMPDIR"], "test_gpu_export")
    est = tf.estimator.Estimator(training.create_model_fn(), model_dir)
    est.train(training.create_input_fn(tf.estimator.ModeKeys.TRAIN))
    export_dir_base = os.path.join(model_dir, "saved_models")
    exporter = saved_model_exporters.DistributedExporter(
        training.create_model_fn(),
        model_dir,
        export_dir_base,
        with_remote_gpu=True)
    exporter.export_saved_model(training.create_serving_input_receiver_fn())


_DISTRIBUTED_TRAIN_BINARY = "monolith/native_training/cpu_training_distributed_test_binary"


class DistributedTrainTest(tf.test.TestCase):

  def _run_process(self, args_tmpl: List, num_ps: int, num_workers: int):
    processes = []
    for i in range(num_ps):
      args = copy.copy(args_tmpl)
      args.append("--server_type=ps")
      args.append("--index={}".format(i))

      process = subprocess.Popen(args)
      processes.append(process)

    for i in range(num_workers):
      args = copy.copy(args_tmpl)
      args.append("--server_type=worker")
      args.append("--index={}".format(i))
      process = subprocess.Popen(args)
      processes.append(process)
      if i == 0:
        # this is best effort waiting, otherwise test may take 30 secs to finish.
        # The goal here is to wait for chief to initialize global variables.
        time.sleep(1)
    processes.reverse()
    return processes

  def _run_test(self, args_tmpl: List, num_ps: int, num_workers: int):
    processes = self._run_process(args_tmpl, num_ps, num_workers)
    print(" ".join(args_tmpl), num_ps, num_workers)
    for process in processes:
      # We give 70 secs to timeout because of 30 secs querying interval.
      self.assertEqual(process.wait(timeout=150), 0)

  def _test_dir(self):
    return os.path.join(os.environ["TEST_TMPDIR"], "DistributedTrainTest",
                        self._testMethodName)

  def _test_args(self, num_ps, num_workers, case=0):
    args = [
        _DISTRIBUTED_TRAIN_BINARY, "--test_case={}".format(case),
        "--test_dir={}".format(self._test_dir()), "--num_ps={}".format(num_ps),
        "--num_workers={}".format(num_workers),
        "--uuid={}".format(self._testMethodName),
        f"--use_native_multi_hash_table={FLAGS.use_native_multi_hash_table}"
    ]
    return args

  # TODO(leqi.zou): Currently, this test mocks too much, should find a way to elegantly solve
  # the shutdown problem both in test and training
  # This test may takes 30 secs to be finished because variable initialization problem.
  def test0_basic(self):
    num_ps = 4
    # We have 2 workers and 1 chief
    num_workers = 3
    args_tmpl = self._test_args(num_ps, num_workers)
    self._run_test(args_tmpl, num_ps, num_workers)

  def test0_with_extra_ps(self):
    num_ps = 2
    num_workers = 1
    num_extra_ps = 2
    args_tmpl = self._test_args(num_ps, num_workers)
    args_tmpl.append("--num_extra_ps={}".format(num_extra_ps))
    self._run_test(args_tmpl, num_ps + num_extra_ps, num_workers)

  def test0_with_redundant_ps(self):
    num_ps = 4
    num_workers = 2
    num_redundant_ps = 2
    args_tmpl = self._test_args(num_ps, num_workers)
    args_tmpl.append("--num_redundant_ps={}".format(num_redundant_ps))
    self._run_test(args_tmpl, num_ps + num_redundant_ps, num_workers)

  def test1_with_debugging_server(self):
    if FLAGS.use_native_multi_hash_table:
      # Debugging server doesnt support multi hash table.
      return
    num_ps = 2
    num_workers = 1
    args_tmpl = self._test_args(num_ps, num_workers, case=1)
    processes = self._run_process(args_tmpl, num_ps, num_workers)

    model_dir = os.path.join(self._test_dir(),
                             "test1_with_debugging_server/model")
    while True:
      ckpt_state = tf.train.get_checkpoint_state(model_dir)
      if ckpt_state:
        break
      time.sleep(1)

    debugging_info_str = file_io.read_file_to_string(
        utils.get_debugging_info_file_name(model_dir), binary_mode=True)
    debugging_info = debugging_info_pb2.DebuggingInfo()
    debugging_info.ParseFromString(debugging_info_str)
    self.assertEqual(debugging_info.num_workers, num_workers)
    self.assertLen(debugging_info.cluster.ps_addrs, num_ps)
    self.assertLen(debugging_info.feature_name_configs, 1)
    self.assertEqual(debugging_info.feature_name_configs[0].feature_name,
                     "feature")

    worker = debugging_server.DebuggingWorker(model_dir)
    self.assertEqual(worker.fetch_variables(["global_step:0", "test"]),
                     {'global_step:0': '1'})
    fids = ["0", "1", "2", "0"]
    result = worker.fetch_features(["feature"] * 3 + ["test"], fids)
    for idx in range(2):
      fid = fids[idx]
      entry_dump = embedding_hash_table_pb2.EntryDump()
      text_format.Parse(result["feature"][fid], entry_dump)
      self.assertLen(entry_dump.num, 5)
    self.assertNotIn("2", result["feature"])
    self.assertNotIn("test", result)

    for process in processes:
      process.kill()

  def test2_temporary_error(self):
    num_ps = 1
    num_workers = 1
    args_tmpl = self._test_args(num_ps, num_workers, case=2)
    self._run_test(args_tmpl, num_ps, num_workers)


class LocalTrainTest(tf.test.TestCase):

  def testBasic(self):
    print(tf.compat.v1.get_default_graph().as_graph_def())
    p = FeatureTask.params()
    p.name = "feature_task"
    p.train.max_steps = 1
    cpu_training.local_train(p,
                             model_dir=os.path.join(os.environ["TEST_TMPDIR"],
                                                    "local_train_basic"),
                             profiling=False)

  def testWithPs(self):
    print(tf.compat.v1.get_default_graph().as_graph_def())
    p = FeatureTask.params()
    p.name = "feature_task"
    p.train.max_steps = 1
    cpu_training.local_train(p,
                             num_ps=2,
                             model_dir=os.path.join(os.environ["TEST_TMPDIR"],
                                                    "local_train_with_ps"),
                             profiling=False)


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  app.run(tf.test.main)
