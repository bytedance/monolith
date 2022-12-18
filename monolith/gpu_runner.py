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

"""GPU Runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import sys
import os
import time

import tensorflow as tf
import horovod.tensorflow as hvd
from mpi4py import MPI

from monolith.base_runner import BaseRunner
from monolith.core import model_registry

FLAGS = flags.FLAGS

flags.DEFINE_string("task", default=None, help="Name of the task class to run.")

flags.DEFINE_string(
    "model_dir",
    default=None,
    help=("The directory where the model and summaries are stored."))

flags.DEFINE_integer(
    "save_checkpoints_steps",
    default=None,
    help=
    ("Save checkpoint every save_checkpoints_steps. If None, no checkpoint saved."
    ))

flags.DEFINE_enum("mode", "train", ["train_and_eval", "train", "eval"],
                  "Job mode.")


class GPURunner(BaseRunner):

  def __init__(self, task_param, *args, **kwargs):
    super(GPURunner, self).__init__(*args, **kwargs)
    # TODO(youlong.cheng): all the parse logic should genearte a hyperparam class.
    self._model_dir = FLAGS.model_dir
    self._save_checkpoints_steps = FLAGS.save_checkpoints_steps
    #TODO(hemang.jangle) Allow subclass task_params to override tpu_runner params
    self._task_param = task_param
    self._mode = FLAGS.mode

  def create_estimator(self, model_fn):
    """Creates the Estimator."""
    if self._task_param.accelerator == "horovod":
      # Horovod: save checkpoints only on worker 0 to prevent other workers from
      # corrupting them. @Hao.sheng: However, we still need to use the same
      # model_dir so each worker where to load the checkpoint in the train_and_eval
      # mode.
      model_dir = self._model_dir  #if hvd.rank() == 0 else None
      save_checkpoints_steps = self._save_checkpoints_steps if hvd.rank(
      ) == 0 else None
      config = tf.compat.v1.ConfigProto()
      config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1
      config.gpu_options.allow_growth = True
      config.gpu_options.visible_device_list = str(hvd.local_rank())
      config = tf.estimator.RunConfig(
          model_dir=model_dir,
          save_checkpoints_steps=save_checkpoints_steps,
          session_config=config)
      num_gpus = hvd.size()
    else:
      num_gpus = 1
      config = tf.compat.v1.estimator.RunConfig(
          model_dir=self._model_dir,
          save_checkpoints_steps=self._save_checkpoints_steps)

    return tf.compat.v1.estimator.Estimator(
        model_fn=model_fn,
        params={
            "train_batch_size":
                self._task_param.train.per_replica_batch_size,
            "eval_batch_size":
                self._task_param.eval.per_replica_batch_size,
            "accelerator":
                self._task_param.accelerator,
            "num_replicas":
                num_gpus,
            "hvd_rank":
                hvd.rank() if self._task_param.accelerator == "horovod" else 0
        },
        config=config)

  def run(self):
    try:
      current_step = tf.train.load_variable(self._model_dir,
                                            tf.compat.v1.GraphKeys.GLOBAL_STEP)
    except (TypeError, ValueError, tf.errors.NotFoundError):
      current_step = 0
    logging.info("Current step :{}".format(current_step))

    task = self._task_param.instantiate()
    input_fn_train = task.create_input_fn(tf.estimator.ModeKeys.TRAIN)
    input_fn_eval = task.create_input_fn(tf.estimator.ModeKeys.EVAL)
    model_fn = task.create_model_fn()
    if self._task_param.accelerator == "horovod":
      # Horovod: initialize Horovod.
      hvd.init()

      # Horovod: pin GPU to be used to process local rank (one GPU per process)
      gpus = tf.config.experimental.list_physical_devices('GPU')
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
          tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()],
                                                     'GPU')

    est = self.create_estimator(model_fn)
    start_timestamp = time.time()  # This time will include compilation time

    if self._mode == 'train':
      if self._task_param.accelerator == "horovod":
        # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states from
        # rank 0 to all other processes. This is necessary to ensure consistent
        # initialization of all workers when training is started with random weights or
        # restored from a checkpoint.
        bcast_hook = hvd.BroadcastGlobalVariablesHook(0)
        est.train(input_fn_train,
                  max_steps=self._task_param.train.max_steps,
                  hooks=[bcast_hook])
      else:
        est.train(input_fn_train, max_steps=self._task_param.train.max_steps)
    elif self._mode == 'eval':
      eval_output_dir = os.path.join(self._model_dir, 'eval')
      tf.io.gfile.makedirs(eval_output_dir)
      total_examples = self._task_param.input.eval_examples
      eval_batch_size = self._task_param.eval.per_replica_batch_size
      num_steps = total_examples // eval_batch_size
      logging.info(
          "Evaluation: total_examples:{} eval_batch_size:{} num_steps: {}".
          format(total_examples, eval_batch_size, num_steps))
      eval_results = est.evaluate(input_fn_eval, steps=num_steps)
      logging.info("Eval results: {}".format(eval_results))
      # Summary writer writes out eval metrics.
      summary_writer = tf.compat.v1.summary.FileWriter(eval_output_dir)
      self.write_summary(eval_results, summary_writer, current_step)
      summary_writer.close()
    else:  # train_and_eval
      steps_per_eval = self._task_param.eval.steps_per_eval
      max_steps = self._task_param.train.max_steps
      eval_output_dir = os.path.join(self._model_dir, 'eval')
      tf.io.gfile.makedirs(eval_output_dir)
      while current_step < self._task_param.train.max_steps:
        # Train for up to steps_per_eval number of steps.
        # At the end of training, a checkpoint will be written to --model_dir.

        next_checkpoint = min(current_step + steps_per_eval, max_steps)
        if self._task_param.accelerator == "horovod":
          bcast_hook = hvd.BroadcastGlobalVariablesHook(0)
          est.train(input_fn_train,
                    max_steps=next_checkpoint,
                    hooks=[bcast_hook])
        else:
          est.train(input_fn_train, max_steps=next_checkpoint)
        current_step = next_checkpoint

        logging.info(
            "Finished training up to step {}. Elapsed seconds {}.".format(
                next_checkpoint,
                time.time() - start_timestamp))
        total_examples = self._task_param.input.eval_examples
        eval_batch_size = self._task_param.eval.per_replica_batch_size

        num_steps = total_examples // eval_batch_size

        if self._task_param.accelerator != "horovod" or hvd.rank() == 0:

          logging.info("Starting to evaluate.")
          time.sleep(10)

          #eval_results = hvd.allreduce(eval_results)
          eval_results = est.evaluate(input_fn_eval, steps=num_steps)
          logging.info("Eval results at step {}: {}".format(
              next_checkpoint, eval_results))
          # Summary writer writes out eval metrics.
          summary_writer = tf.compat.v1.summary.FileWriter(eval_output_dir)
          self.write_summary(eval_results, summary_writer, current_step)
          summary_writer.close()

        # Hovorod: Make sure all workers are synced at the end of one round
        # https://github.com/horovod/horovod/issues/159
        # https://github.com/horovod/horovod/issues/1380
        if self._task_param.accelerator == "horovod":
          MPI.COMM_WORLD.barrier()
      elapsed_time = int(time.time() - start_timestamp)
      logging.info(
          "Finished training up to step {}. Elapsed seconds {}.".format(
              max_steps, elapsed_time))


def main(unused_argv):
  task_name = FLAGS.task
  task_param = model_registry.GetParams(task_name)

  logging.info("task_param: {}".format(str(task_param)))
  runner = GPURunner(task_param)
  runner.run()


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.compat.v1.disable_v2_behavior()
  app.run(main)
