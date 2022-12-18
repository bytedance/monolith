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

import io
import os
import time

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from monolith.native_training import cluster_manager
from monolith.native_training import cpu_training
from monolith.native_training import feature
from monolith.native_training import native_task
from monolith.native_training import service_discovery
from monolith.native_training import utils

flags.DEFINE_integer("test_case", None, "The number of test case.")
flags.DEFINE_string("test_dir", None, "The test folder.")

flags.DEFINE_string("server_type", None,
                    "The type of this process. Can be 'ps' or 'worker'")
flags.DEFINE_integer("index", None,
                     "The index of the current process in servers.")
flags.DEFINE_integer("num_ps", None, "The number of ps")
flags.DEFINE_integer("num_workers", None, "The number of worker")
flags.DEFINE_integer("num_extra_ps", 0, "The number of extra ps.")
flags.DEFINE_integer("num_redundant_ps", 0, "The number of redundant ps.")
flags.DEFINE_string("uuid", "", "uuid")
flags.DEFINE_bool("use_native_multi_hash_table", False,
                  "Use native MultiHashTable.")

FLAGS = flags.FLAGS


def _sleep_short():
  time.sleep(0.1)


# In the test, we want query as fast as possible.
cluster_manager._cluster_query_failure_handler = _sleep_short
cpu_training._EXTRA_PS_BENCHMARK_SECS = 0.5


class SyncHook(tf.estimator.SessionRunHook):

  def __init__(self, num_workers, index):
    self._num_workers = num_workers
    self._index = index
    self._var = None
    self._assign_op = None

  def begin(self):
    collections = [tf.compat.v1.GraphKeys.LOCAL_VARIABLES
                  ] if self._index == 0 else [tf.compat.v1.GraphKeys.VARIABLES]
    self._var = tf.compat.v1.get_variable(
        "TEST_SYNC_VAR",
        initializer=[False] * self._num_workers,
        dtype=tf.bool,
        trainable=False,
        collections=collections,
    )
    self._assign_op = self._var[self._index].assign(True)

  def after_create_session(self, session, coord):
    session.run(self._assign_op)
    if self._index == 0:
      # To prevent chief finishing before other workers start
      while True:
        if sum(session.run(self._var)) == self._num_workers:
          break
        time.sleep(0.5)


class FeatureTask(native_task.NativeTask):
  """A test task that will collect some information in model_fn."""

  @classmethod
  def params(cls):
    p = super().params()
    p.define("training_hooks", [], "Training hooks")
    return p

  def create_input_fn(self, mode):
    del mode

    def input_fn():
      return tf.data.Dataset.from_tensors(
          {"feature": tf.ragged.constant([[0, 1]], dtype=tf.int64)})

    return input_fn

  def create_model_fn(self):

    def model_fn(mode, features, config):
      slot = self.ctx.feature_factory.create_feature_slot(
          feature.FeatureSlotConfig(name="slot"))
      s = slot.add_feature_slice(5)
      fc = feature.FeatureColumnV1(slot, "feature")
      embedding = fc.embedding_lookup(s)
      if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=tf.constant(0))
      all_embeddings = [fc.get_all_embeddings_concat()]
      grads = tf.gradients(-embedding, all_embeddings)
      global_step = tf.compat.v1.train.get_or_create_global_step()
      train_op = tf.group(
          global_step.assign_add(1),
          self._ctx.feature_factory.apply_gradients(zip(grads, all_embeddings)),
          features["feature"])
      return tf.estimator.EstimatorSpec(
          mode,
          train_op=train_op,
          loss=tf.constant(0.0),
          training_hooks=[SyncHook(FLAGS.num_workers, FLAGS.index)] +
          self.p.training_hooks)

    return model_fn


class HostServiceDiscovery(service_discovery.ServiceDiscovery):

  def __init__(self, base_path: str):
    self._base_path = base_path

  def register(self, name: str, index: int, addr: str):
    os.makedirs(self._named_path(name), exist_ok=True)
    with io.open(os.path.join(self._named_path(name), str(index)),
                 "w") as writer:
      writer.write(addr)

  def deregister(self, name: str, index: int, addr: str):
    pass

  def query(self, name: str):
    basepath = self._named_path(name)
    if not os.path.exists(basepath):
      return {}
    indexes = os.listdir(basepath)
    result = {}
    for index in indexes:
      f = os.path.join(basepath, index)
      with io.open(f, "r") as reader:
        addr = reader.read()
      result[int(index)] = addr
    return result

  def _named_path(self, name: str):
    return os.path.join(self._base_path, name)


def test_run(params):
  model_dir = os.path.join(FLAGS.test_dir, f"{FLAGS.uuid}/model")
  config = cpu_training.DistributedCpuTrainingConfig(
      server_type=FLAGS.server_type,
      index=FLAGS.index,
      num_ps=FLAGS.num_ps,
      num_extra_ps=FLAGS.num_extra_ps,
      num_redundant_ps=FLAGS.num_redundant_ps,
      num_workers=FLAGS.num_workers,
      model_dir=model_dir,
      uuid=FLAGS.uuid,
      enable_model_ckpt_info=True,
      use_native_multi_hash_table=FLAGS.use_native_multi_hash_table)
  # It is not easy to prevent worker doing things too fast
  params.train.max_pending_seconds_for_barrier = 2
  discovery = HostServiceDiscovery(
      os.path.join(FLAGS.test_dir, f"{FLAGS.uuid}/service_discovery"))
  cpu_training.distributed_train(config, discovery, params)


def test0():
  params = FeatureTask.params()
  params.name = "test_task"
  test_run(params)


def test1():

  def no_shutdown(*args, **kwargs):
    while True:
      time.sleep(1)

  cpu_training._shutdown_ps = no_shutdown
  test0()


class RaiseErrorHook(tf.estimator.SessionRunHook):

  def __init__(self, first):
    self._first = first

  def before_run(self, run_context):
    if self._first:
      self._first = False
      raise tf.errors.DeadlineExceededError(None, None,
                                            "test ddl exceeded error")


def test2():
  params = FeatureTask.params()
  params.name = "test_task"
  first = True
  params.training_hooks = [RaiseErrorHook(first)]
  test_run(params)


def main(_):
  test_cases = [test0, test1, test2]
  test_cases[FLAGS.test_case]()


if __name__ == "__main__":
  app.run(main)
