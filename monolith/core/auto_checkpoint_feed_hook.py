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

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

from __future__ import division
from __future__ import print_function

import threading
import time
import os
from six.moves import queue as Queue  # pylint: disable=redefined-builtin
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow.compat.v1 as tf
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf.tpu import compilation_result_pb2 as tpu_compilation_result
from tensorflow.python.ops import summary_ops_v2 as contrib_summary

_USER_PROVIDED_SIGNAL_NAME = "_user_provided_signal_name"
_TPU_ESTIMATOR = 'tpu_estimator'
_ITERATIONS_PER_LOOP_VAR = 'iterations_per_loop'


class PeriodicLogger(object):

  def __init__(self, seconds):
    self._log_every_n_seconds = seconds
    self._last_log_time = 0

  def log(self, msg, *args, **kw):
    if time.time() - self._last_log_time > self._log_every_n_seconds:
      self._last_log_time = time.time()
      tf.compat.v1.logging.info(msg, *args, **kw)


class _SIGNAL(object):
  """Signal used to control the thread of infeed/outfeed.

  All preserved signals must be negative numbers. Positive numbers are used to
  indicate the number of iterations for next training/evaluation loop.
  """
  NEXT_BATCH = -1
  STOP = -2


class _OpQueueContext(object):
  """Manages work queue and thread for a infeed/outfeed thread."""

  def __init__(self, name, target, args):
    self._name = name
    self._queue = Queue.Queue()
    args = (self,) + args
    self._thread = threading.Thread(name=name, target=target, args=args)
    self._thread.daemon = True
    self._thread.start()

  def stop(self):
    self._queue.put(_SIGNAL.STOP)

  def send_next_batch_signal(self, iterations):
    self._queue.put(iterations)

  def read_iteration_counts(self):
    while True:
      iterations = self._queue.get(block=True)
      tf.compat.v1.logging.debug('%s read iterations %s', self._name,
                                 iterations)
      if iterations == _SIGNAL.STOP:
        tf.compat.v1.logging.info('%s received shutdown signal, stopping.',
                                  self._name)
        return
      yield iterations

  def join(self):
    tf.compat.v1.logging.info('Shutting down %s thread.', self._name)
    self.stop()
    self._thread.join()


class _OpSignalOnceQueueContext(_OpQueueContext):
  """Manages work queue and thread for a infeed/outfeed thread.

  This subclass only signals once.
  """

  def __init__(self, name, target, args):
    super(_OpSignalOnceQueueContext, self).__init__(name, target, args)
    self._has_signaled = False

  def send_next_batch_signal(self, iterations):
    if not self._has_signaled:
      self._queue.put(iterations)
      self._has_signaled = True


class TPUInfeedOutfeedSessionWithEndOfStreamHandlingHook(
    tf.estimator.SessionRunHook):
  """A Session hook setting up the TPU initialization, infeed, and outfeed.

  This hook does two major things:
  1. initialize and shutdown TPU system.
  2. launch and join the threads for infeed enqueue and (optional) outfeed
     dequeue.
  """

  def __init__(self,
               ctx,
               enqueue_ops,
               dequeue_ops,
               tpu_compile_op,
               run_infeed_loop_on_coordinator=True,
               rendezvous=None,
               master=None,
               session_config=None,
               tpu_init_ops=None,
               outfeed_every_n_steps=1):
    self._master_job = ctx.master_job
    self._enqueue_ops = enqueue_ops
    self._dequeue_ops = dequeue_ops
    self._rendezvous = rendezvous
    self._master = master
    self._session_config = session_config
    self._init_ops = list(tpu_init_ops or [])
    if ctx.embedding_config is None:
      self._embedding_layer_config = None
    else:
      self._embedding_layer_config = (
          ctx.embedding_config.tpu_embedding.config_proto)
    self._run_infeed_loop_on_coordinator = run_infeed_loop_on_coordinator
    self._initial_infeed_sleep_secs = (
        ctx.config.tpu_config.initial_infeed_sleep_secs)
    self._tpu_compile_op = tpu_compile_op

    # When using model parallelism, the TPU is pre-initialized at startup to
    # fetch mesh information. We skip re-initializing it here for
    # MeshTensorFlow since it places variables on TPU directly. Reinitialize tpu
    # is causing the variable corruption since the previous allocated memory
    # might be overwritten for other purpose.
    if (ctx.model_parallelism_enabled and
        (ctx.config.tpu_config.per_host_input_for_training is
         tpu_config.InputPipelineConfig.BROADCAST)):
      self._should_initialize_tpu = False
    else:
      self._should_initialize_tpu = True
    self._outfeed_every_n_steps = outfeed_every_n_steps

    self.stopping_signal = False

  def _create_or_get_iterations_per_loop(self):
    """Creates or gets the iterations_per_loop variable.

    In TPUEstimator, the user provided computation, the model_fn, is wrapped
    inside a tf.while_loop for peak performance. The iterations of the loop are
    specified by this variable, which adjusts its value on the CPU after each TPU
    program execution and before the next TPU execution.

    The purpose of using a variable, rather then a constant, is to allow
    TPUEstimator adapt the TPU training iterations according to the final steps
    specified by users. For example, if the user sets the iterations_per_loop as 4
    in TPUConfig and steps as 10 in TPUEstimator.train(), the iterations_per_loop
    variable will have the following value before each TPU training.

        - 1-th TPU execution: iterations_per_loop = 4
        - 2-th TPU execution: iterations_per_loop = 4
        - 3-th TPU execution: iterations_per_loop = 2

    As model_fn increases the global step once per train_op invocation, the global
    step is 10 after all TPU executions, matching the steps=10 inputs passed in by
    users.

    Returns:
      A TF non-trainable resource variable.

    Raises:
      RuntimeError: If multi iterations_per_loop variables were found.
    """
    graph = tf.compat.v1.get_default_graph()
    collection_name = '{}_{}'.format(_TPU_ESTIMATOR, _ITERATIONS_PER_LOOP_VAR)
    iter_vars = graph.get_collection(collection_name)
    if len(iter_vars) == 1:
      return iter_vars[0]
    elif len(iter_vars) > 1:
      raise RuntimeError('Multiple iterations_per_loop_var in collection.')

    with ops.colocate_with(tf.compat.v1.train.get_global_step()):
      with tf.compat.v1.variable_scope(_TPU_ESTIMATOR,
                                       reuse=tf.compat.v1.AUTO_REUSE):
        return tf.compat.v1.get_variable(
            _ITERATIONS_PER_LOOP_VAR,
            initializer=tf.compat.v1.initializers.zeros(),
            shape=[],
            dtype=tf.dtypes.int32,
            trainable=False,
            collections=[
                collection_name, tf.compat.v1.GraphKeys.LOCAL_VARIABLES
            ],
            use_resource=True)

  def begin(self):
    tf.compat.v1.logging.info('TPU job name %s', self._master_job)
    self._iterations_per_loop_var = self._create_or_get_iterations_per_loop()
    if self._should_initialize_tpu:
      self._finalize_ops = [
          tf.compat.v1.tpu.shutdown_system(job=self._master_job)
      ]
    else:
      self._finalize_ops = []

    summary_writer_init_ops = contrib_summary.summary_writer_initializer_op()
    self._init_ops.extend(summary_writer_init_ops)
    # Get all the writer resources from the initializer, so we know what to
    # flush.
    for op in summary_writer_init_ops:
      self._finalize_ops.append(contrib_summary.flush(writer=op.inputs[0]))

  def _run_infeed(self, queue_ctx, session):
    tf.compat.v1.logging.info('Starting infeed thread controller.')
    if self._initial_infeed_sleep_secs:
      tf.compat.v1.logging.info('Infeed thread sleeping for %d seconds.',
                                self._initial_infeed_sleep_secs)
      time.sleep(self._initial_infeed_sleep_secs)
      tf.compat.v1.logging.info('Infeed thread starting after sleep')

    with self._rendezvous.catch_errors(source='infeed', session=session):
      if self._run_infeed_loop_on_coordinator:
        for count, steps in enumerate(queue_ctx.read_iteration_counts()):
          for i in xrange(steps):
            tf.compat.v1.logging.debug('Infeed enqueue for iteration (%d, %d)',
                                       count, i)
            session.run(self._enqueue_ops)
      else:
        for _ in queue_ctx.read_iteration_counts():
          session.run(self._enqueue_ops)
      tf.compat.v1.logging.info('Infeed thread finished, shutting down.')

  def _run_outfeed(self, queue_ctx, session):
    tf.compat.v1.logging.info('Starting outfeed thread controller.')
    status_logger = PeriodicLogger(seconds=60)
    with self._rendezvous.catch_errors(source='outfeed', session=session):
      stopping_signals = False
      for count, steps in enumerate(queue_ctx.read_iteration_counts()):
        step_counter = 0
        for i in xrange(steps):
          tf.compat.v1.logging.debug('Outfeed dequeue for iteration (%d, %d)',
                                     count, i)
          if step_counter % self._outfeed_every_n_steps == 0:
            ret = session.run(self._dequeue_ops)
            if _USER_PROVIDED_SIGNAL_NAME in ret:
              if 'stopping' not in ret[_USER_PROVIDED_SIGNAL_NAME]:
                raise RuntimeError('ret[{}] must contain key \'stopping\'.'
                                  ).format(_USER_PROVIDED_SIGNAL_NAME)
              if ret[_USER_PROVIDED_SIGNAL_NAME]['stopping'][0] == True \
                and stopping_signals == False:
                stopping_signals = True
                tf.compat.v1.logging.info(
                    'Encountered stop signal at iteration (%d, %d).', count, i)
          step_counter += 1
          status_logger.log('Outfeed finished for iteration (%d, %d)', count, i)
        if stopping_signals == True:
          tf.compat.v1.logging.info(
              'Set shared stop signal at iteration (%d, %d).', count, i)
          self.stopping_signal = True
      tf.compat.v1.logging.info('Outfeed thread finished, shutting down.')

  def _create_infeed_controller(self, name, target, args):
    return _OpQueueContext(name=name, target=target, args=args)

  def _assertCompilationSucceeded(self, result, coord):
    proto = tpu_compilation_result.CompilationResultProto()
    proto.ParseFromString(result)
    if proto.status_error_message:
      tf.compat.v1.logging.error('Compilation failed: {}'.format(
          proto.status_error_message))
      coord.request_stop()
    else:
      tf.compat.v1.logging.info('Compilation succeeded')

  def after_create_session(self, session, coord):
    if self._should_initialize_tpu:
      tf.compat.v1.logging.info('Init TPU system')
      start = time.time()
      with tf.Graph().as_default():
        with tf.compat.v1.Session(self._master,
                                  config=self._session_config) as sess:
          sess.run(
              tf.compat.v1.tpu.initialize_system(
                  job=self._master_job,
                  embedding_config=self._embedding_layer_config))
      tf.compat.v1.logging.info('Initialized TPU in %d seconds',
                                time.time() - start)

    session.run(self._init_ops,
                options=config_pb2.RunOptions(timeout_in_ms=30 * 60 * 1000))

    if os.environ.get('TPU_SPLIT_COMPILE_AND_EXECUTE', '') == '1':
      tf.compat.v1.logging.info(
          'Compiling user program: this may take a while...')
      self._assertCompilationSucceeded(session.run(self._tpu_compile_op), coord)

    self._infeed_controller = self._create_infeed_controller(
        name='InfeedController', target=self._run_infeed, args=(session,))

    self._outfeed_controller = _OpQueueContext(name='OutfeedController',
                                               target=self._run_outfeed,
                                               args=(session,))

    # Enable the worker watchdog to terminate workers on coordinator exit.
    watchdog_timeout = int(os.environ.get('TF_TPU_WATCHDOG_TIMEOUT', '0'))
    if watchdog_timeout > 0:
      session_support.start_worker_watchdog(session,
                                            shutdown_timeout=watchdog_timeout)

  def before_run(self, run_context):
    if self.stopping_signal == True:
      tf.compat.v1.logging.info(
          'Throw OutOfRangeError error due to encountering stopping signal in before_run.'
      )
      raise tf.errors.OutOfRangeError(None, None, 'Stopped by stopping signal.')

    iterations = run_context.session.run(self._iterations_per_loop_var)

    tf.compat.v1.logging.info('Enqueue next (%d) batch(es) of data to infeed.',
                              iterations)
    self._infeed_controller.send_next_batch_signal(iterations)

    tf.compat.v1.logging.info(
        'Dequeue next (%d) batch(es) of data from outfeed.', iterations)
    self._outfeed_controller.send_next_batch_signal(iterations)

  def end(self, session):
    tf.compat.v1.logging.info('Stop infeed thread controller')
    self._infeed_controller.join()
    self._rendezvous.record_done('infeed')

    tf.compat.v1.logging.info('Stop output thread controller')
    self._outfeed_controller.join()
    self._rendezvous.record_done('outfeed')

    tf.compat.v1.logging.info('Shutdown TPU system.')
    session.run(self._finalize_ops)

  @staticmethod
  def get_stopping_signals_and_name(features):
    stopping_signals = None
    if _USER_PROVIDED_SIGNAL_NAME in features:
      tf.compat.v1.logging.info("Get stopping signals and name.")
      sum_stopping_signals = tf.compat.v1.tpu.cross_replica_sum(
          tf.cast(features[_USER_PROVIDED_SIGNAL_NAME], tf.int32))
      stopping_signals = {'stopping': sum_stopping_signals > 0}

    return stopping_signals, _USER_PROVIDED_SIGNAL_NAME
