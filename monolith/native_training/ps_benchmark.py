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
import dataclasses
import time
from typing import Dict, List

from absl import logging
import tensorflow as tf

from monolith.native_training.optimizers.adamom import AdamomOptimizer
from monolith.native_training import logging_ops
from monolith.native_training import native_task
from monolith.native_training import service_discovery
from monolith.native_training import utils

# We need a scope name unique enough to prevent name confliction.
_SCOPE_NAME = "machine_benchmark_I_AM_PECULIAR"


@dataclasses.dataclass
class BenchmarkConfig:
  ps_list: List
  num_ps_required: int
  num_workers: int
  index: int
  benchmark_secs: float = 60.0
  # If non-empty, it will skip the benchmark and
  # use `ps_str_overridden` here. Separated by `,`
  #
  # Example: `127.0.0.1:1,127.0.0.1:2`
  ps_str_overridden: str = ""


class _BenchmarkWorkerHook(tf.estimator.SessionRunHook):

  def __init__(self, config: BenchmarkConfig, throughput_tensor: tf.Tensor):
    with tf.name_scope(_SCOPE_NAME):
      self._config = config
      self._throughput_tensor = throughput_tensor

      self._result = tf.Variable("", trainable=False)
      self._result_placeholder = tf.compat.v1.placeholder(tf.string, [])
      self._result_assign = self._result.assign(self._result_placeholder)
      self._ready = tf.Variable([False] * self._config.num_workers,
                                trainable=False)
      self._make_ready = self._ready[self._config.index].assign(True)
      self._done = tf.Variable([False] * self._config.num_workers,
                               trainable=False)
      self._make_done = self._done[self._config.index].assign(True)
      self._start_time = None

  def after_create_session(self, sess, coord):
    sess.run(self._make_ready)
    self._wait(lambda: sum(sess.run(self._ready)) >= int(self._config.
                                                         num_workers * 0.9))
    # Before we start we wait for another 1 secs to make sure everyone
    # got the result.
    time.sleep(1)
    logging.info("Benchmark started.")
    self._start_time = time.time()

  def before_run(self, run_context):
    if self._config.ps_str_overridden:
      run_context.session.run(
          self._result_assign,
          feed_dict={self._result_placeholder: self._config.ps_str_overridden})
    result_value = run_context.session.run(self._result)
    if result_value:
      raise tf.errors.OutOfRangeError(None, None, "Benchmark is done already.")

  def after_run(self, run_context, run_values):
    duration = time.time() - self._start_time
    logging.info("Benchmarking {} seconds".format(duration))
    run_context.request_stop()

  def end(self, sess):
    sess.run(self._make_done)
    self._wait(lambda: sum(sess.run(self._done)) == self._config.num_workers,
               timeout=10)

    if self._config.index == 0 and not self._config.ps_str_overridden:
      # OK now we know how ps should look like.
      throughput_value = sess.run(self._throughput_tensor)
      reversed_sorted_throughput_and_ps = sorted(
          [[throughput, i, self._config.ps_list[i].split(":")[0]]
           for i, throughput in enumerate(throughput_value)])
      sorted_throughput_and_ps = [
          item[:] for item in reversed_sorted_throughput_and_ps
      ]

      logging.info("Measure result (throughput, ps): {}".format([
          "ps_{}({}):{}".format(ps, ip, throughput)
          for throughput, ps, ip in reversed(sorted_throughput_and_ps)
      ]))

      for i in range(len(reversed_sorted_throughput_and_ps) - 1):
        for j in range(i + 1, len(reversed_sorted_throughput_and_ps)):
          if reversed_sorted_throughput_and_ps[i][
              2] == reversed_sorted_throughput_and_ps[j][2]:
            sorted_throughput_and_ps[j][0] += reversed_sorted_throughput_and_ps[
                i][0]

      sorted_throughput_and_ps = sorted(sorted_throughput_and_ps, reverse=True)

      logging.info(
          "Measure result (throughput, ps) (ps with the same ip addresses had their throughput adjusted): {}"
          .format([
              "ps_{}({}):{}".format(ps, ip, throughput)
              for throughput, ps, ip in sorted_throughput_and_ps
          ]))

      selected_ps = [
          self._config.ps_list[i] for throughput, i, _ in
          sorted_throughput_and_ps[:self._config.num_ps_required]
      ]
      ps_str = ",".join(selected_ps)
      sess.run(self._result_assign,
               feed_dict={self._result_placeholder: ps_str})

    ps_str = ""

    def ps_ready():
      nonlocal ps_str
      ps_str = sess.run(self._result)
      return bool(ps_str)

    self._wait(ps_ready)
    self._config.ps_list.clear()
    selected_ps = ps_str.decode().split(",")
    for i in range(self._config.num_ps_required):
      self._config.ps_list.append(selected_ps[i])

  def _wait(self, cond, timeout=3600):
    start_time = time.time()
    while time.time() - start_time < timeout:
      if cond():
        break
      time.sleep(0.5)


class _DummyCheckpointSaverHook(tf.estimator.CheckpointSaverHook):
  """A saver hook which won't perform the first save (which happpend on after_create_session)."""

  def __init__(self, checkpoint_dir=None, save_steps=10240, **kwargs):
    if not checkpoint_dir:
      checkpoint_dir = os.path.join(os.environ.get('HOME', "/"), 'tmp')
    super(_DummyCheckpointSaverHook, self).__init__(checkpoint_dir, save_steps)
    logging.info("Create DummyCheckpointSaverHook.")

  def begin(self):
    return

  def after_create_session(self, session, coord):
    return

  def before_run(self, run_context):
    return None

  def after_run(self, run_context, run_values):
    return

  def end(self, session):
    return

  def _save(self, session, step: int) -> bool:
    return False


class PsBenchMarkTask(native_task.NativeTask):

  @classmethod
  def params(cls):
    p = super().params()
    p.define("bm_config", None, "The BenchmarkConfig.")
    return p

  def create_input_fn(self, mode):
    del mode

    def input_fn():
      with tf.name_scope(_SCOPE_NAME):
        return tf.data.Dataset.from_tensor_slices([[
            tf.constant(0.12),
            tf.constant(0.23),
            tf.constant(0.34),
            tf.constant(0.45)
        ]]).repeat().prefetch(2)

    return input_fn

  def create_model_fn(self):

    def model_fn(features, mode, config):
      logging.info("Running model_fn of the ps benchmark")
      del config
      bm_config: BenchmarkConfig = self.p.bm_config
      global_step = tf.compat.v1.train.get_or_create_global_step()
      with tf.name_scope(_SCOPE_NAME):
        throughputs = []
        for ps_i in range(len(bm_config.ps_list)):
          with tf.device(utils.ps_device(ps_i)):
            var = tf.Variable(initial_value=[[0.0] * 256] * 256, trainable=True)
            with tf.control_dependencies([features]):
              ts_before = tf.timestamp()

            i = tf.constant(0)

          grad = tf.reshape(tf.tile(features, [16384]), [256, 256])

          def while_body(i):
            nonlocal var
            nonlocal grad
            with tf.control_dependencies([i]):
              new_grads = tf.split(grad, [64, 64, 64, 64], axis=1)
              output_grads = []
              for ii in range(4):
                sum_grads = []
                for jj in range(10):
                  a, b, c, d = tf.split(new_grads[ii] +
                                        tf.cast(jj / 10, dtype=tf.float32),
                                        [16, 16, 16, 16],
                                        axis=1)
                  for _ in range(10):
                    sum_grads.append(tf.math.sqrt(tf.math.sqrt(a * b) * c + d))
                output_grads.append(tf.math.add_n(sum_grads))
              concat_grads = tf.concat(output_grads, -1)
              var_fetched = tf.identity(var)
            with tf.control_dependencies([var_fetched, concat_grads]):
              return i + 1

          def cond(i):
            nonlocal ts_before
            with tf.control_dependencies([i]):
              ts_now = tf.timestamp()
            return ts_now - ts_before <= bm_config.benchmark_secs

          with tf.device(utils.ps_device(ps_i)):
            (i,) = tf.while_loop(cond, while_body, [i])
            j = tf.identity(i)
          with tf.control_dependencies([j]):
            ts_now = tf.timestamp()
            throughput = tf.cast(j, tf.float32) / tf.cast(
                ts_now - ts_before, tf.float32)
          throughputs.append(throughput)
        mean_throughput, update_op = tf.compat.v1.metrics.mean_tensor(
            tf.stack(throughputs))
        hook = _BenchmarkWorkerHook(bm_config, mean_throughput)
        saver_hook = _DummyCheckpointSaverHook()
        inc_global_step = global_step.assign_add(1)
        if mode == tf.estimator.ModeKeys.PREDICT:
          return tf.estimator.EstimatorSpec(mode=mode,
                                            predictions=tf.constant(0.0))
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=tf.constant(0.0),
                                          train_op=tf.group(
                                              update_op, inc_global_step),
                                          training_hooks=[hook],
                                          training_chief_hooks=[saver_hook])

    return model_fn
