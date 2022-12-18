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
"""Base task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import os
import sys
import time

import tensorflow.compat.v1 as tf
from cloud_tpu_client import client

from monolith.base_runner import BaseRunner
from monolith.core import model_registry
from monolith.core.auto_checkpoint_feed_hook import TPUInfeedOutfeedSessionWithEndOfStreamHandlingHook
from monolith.core.base_embedding_task import BaseEmbeddingTask

FLAGS = flags.FLAGS

flags.DEFINE_string("tf_version", default="nightly", help="TensorFlow version")

flags.DEFINE_string(
    "tpu",
    default=None,
    help="The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string(
    "gcp_project",
    default=None,
    help="Project name for the Cloud TPU-enabled project. If not specified, "
    "we will attempt to automatically detect the GCE project from metadata.")

flags.DEFINE_string(
    "tpu_zone",
    default=None,
    help="GCE zone where the Cloud TPU is located in. If not specified, we "
    "will attempt to automatically detect the zone from metadata.")

flags.DEFINE_string("task", default=None, help="Name of the task class to run.")

flags.DEFINE_string(
    "model_dir",
    default=None,
    help=("The directory where the model and summaries are stored."))

flags.DEFINE_enum("mode", "train", ["train_and_eval", "train", "eval"],
                  "Job mode.")

flags.DEFINE_integer(
    "save_checkpoints_steps",
    default=None,
    help=
    ("Save checkpoint every save_checkpoints_steps. If None, no checkpoint saved."
    ))

flags.DEFINE_integer("iterations_per_loop",
                     default=10000,
                     help=("This is the number of train steps running "
                           "in TPU system before returning to CPU host ."))

# TPU Embedding flags.
flags.DEFINE_bool(
    "pipeline_execution",
    default=False,
    help=("If True, speed up training by overlaping embedding lookups with "
          "dense layer computations. Embedding lookups will be one step old."))

flags.DEFINE_bool("enable_tpu_version_config",
                  default=True,
                  help=("Whether enable tpu configuration or not."))

flags.DEFINE_integer("host_call_every_n_steps",
                     default=500,
                     help=("Host call every n steps."))

# Whether enable handling end of stream and auto checkpointing. If this is False, then
# not handle end of stream. If this is True, enable end of stream handling
# and save a checkpoint before training job end.
flags.DEFINE_bool(
    "enable_stopping_signals",
    default=False,
    help=("Whether enable stopping signals and auto checkpointing."))

# This is only set to True when use CPU to do some simple test. Note that the internal
# embedding update logic is not implemented yet. So do not use this mode to do actually
# training.
flags.DEFINE_bool("cpu_test",
                  default=False,
                  help=("Wheter use CPU in TPU estimator."))

# Allowed value are "div" and "mod". "div" is the default partition_strategy.
# Use 'mod' which runs faster than 'div' given our id distribution especially
# with the incremental generated data. Incremental generated data are more likely
# to have processing ids distributed in some small ranges of vocab table rather
# than randomly distributed across vocab whole table. So 'div' will make
# those some cores more busy with processing those id ranges. 'mod' here will
# help distribute ids more evenly across more cores.
flags.DEFINE_string("partition_strategy",
                    default="mod",
                    help=("Partition strategy of embedding table."))

# This will override end_date if provided not empty value.
flags.DEFINE_string("overwrite_end_date",
                    default="",
                    help=("End date of input data."))


class TPURunner(BaseRunner):

  def __init__(self, task_param, *args, **kwargs):
    super(TPURunner, self).__init__(*args, **kwargs)
    # TODO(youlong.cheng): all the parse logic should genearte a hyperparam class.
    self._tpu = FLAGS.tpu
    self._tpu_zone = FLAGS.tpu_zone
    self._gcp_project = FLAGS.gcp_project
    self._num_replicas_per_host = 8
    self._model_dir = FLAGS.model_dir
    self._pipeline_execution = FLAGS.pipeline_execution
    self.iterations_per_loop = FLAGS.iterations_per_loop
    self._enable_tpu_version_config = FLAGS.enable_tpu_version_config
    self._host_call_every_n_steps = FLAGS.host_call_every_n_steps
    self._enable_stopping_signals = FLAGS.enable_stopping_signals
    self._cpu_test = FLAGS.cpu_test
    self._partition_strategy = FLAGS.partition_strategy

    if task_param.train.save_checkpoints_steps is not None:
      self._save_checkpoints_steps = task_param.train.save_checkpoints_steps
      logging.info(
          "Overwrite save_checkpoints_steps by task_param.train: {}".format(
              self._save_checkpoints_steps))
    else:
      self._save_checkpoints_steps = FLAGS.save_checkpoints_steps
      logging.info("Use save_checkpoints_steps by FLAGS: {}".format(
          self._save_checkpoints_steps))
    #TODO(hemang.jangle) Allow subclass task_params to override tpu_runner params
    self._task_param = task_param
    self._task_param.accelerator = "tpu"
    if FLAGS.overwrite_end_date is not None and FLAGS.overwrite_end_date != "" and self._task_param.train.contain(
        "end_date"):
      self._task_param.train.end_date = FLAGS.overwrite_end_date
      logging.info(
          "Use flag end_date {} to replace parameter train.end_date.".format(
              self._task_param.train.end_date))
    self._mode = FLAGS.mode
    self._task = None

  def _experimental_gradient_multiplier_fn(self, global_step):
    return self._task_param.gradient_multiplier

  def _create_params(self, total_replicas):
    # TODO(youlong.cheng): this is a little bit Adhoc solution, consider
    # abstract HostCall class with hyper_parameter.

    params = {
        "model_dir": self._model_dir,
        "enable_host_call": self._host_call_every_n_steps > 0,
        "num_replicas": total_replicas,
        "accelerator": self._task_param.accelerator,
        "host_call_every_n_steps": self._host_call_every_n_steps,
        "enable_stopping_signals": self._enable_stopping_signals,
        "cpu_test": self._cpu_test,
    }
    logging.info("params: {}".format(params))
    return params

  def create_tpu_estimator(self, model_fn, feature_config, table_config):
    """Creates the TPU Estimator, with accelerated lookups for embedding tables."""
    if self._enable_tpu_version_config == True:
      logging.info(
          "Enable tpu version config, reset remote tpu version with {}".format(
              tf.__version__))
      # This is to let the cloud TPU always restart in case last round operation is still not finished.
      tpu_client = client.Client(tpu=self._tpu,
                                 zone=self._tpu_zone,
                                 project=self._gcp_project)
      tpu_client.configure_tpu_version(version=tf.__version__)
      tpu_client.wait_for_healthy()

    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        self._tpu, zone=self._tpu_zone, project=self._gcp_project)
    num_hosts = tpu_cluster_resolver.cluster_spec().num_tasks("worker")
    total_replicas = self._num_replicas_per_host * num_hosts
    train_global_batch_size = total_replicas * self._task_param.train.per_replica_batch_size
    logging.info(
        "num_hosts: {} total_replicas: {} train_global_batch_size: {}".format(
            num_hosts, total_replicas, train_global_batch_size))

    # experimental_host_call_every_n_steps can't be 0. If _host_call_every_n_steps is not specified,
    # then experimental_host_call_every_n_steps will use 100.
    if self._host_call_every_n_steps == 0:
      _experimental_host_call_every_n_steps = 100
    else:
      _experimental_host_call_every_n_steps = self._host_call_every_n_steps

    if self._enable_stopping_signals is True:
      experimental_feed_hook = TPUInfeedOutfeedSessionWithEndOfStreamHandlingHook
    else:
      experimental_feed_hook = None

    config = tf.compat.v1.estimator.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=self._model_dir,
        save_checkpoints_steps=self._save_checkpoints_steps,
        tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
            iterations_per_loop=self.iterations_per_loop,
            experimental_host_call_every_n_steps=
            _experimental_host_call_every_n_steps,
            per_host_input_for_training=tf.compat.v1.estimator.tpu.
            InputPipelineConfig.PER_HOST_V2,
            experimental_allow_per_host_v2_parallel_get_next=True,
            experimental_feed_hook=experimental_feed_hook,
        ))

    # Disable meta_optimizer which is not needed and takes long time to run.
    config.session_config.graph_options.rewrite_options.disable_meta_optimizer = True

    if feature_config and table_config:
      embedding_config_spec = tf.compat.v1.estimator.tpu.experimental.EmbeddingConfigSpec(
          feature_to_config_dict=feature_config,
          table_to_config_dict=table_config,
          partition_strategy=self._partition_strategy,
          pipeline_execution_with_tensor_core=self._pipeline_execution,
          experimental_gradient_multiplier_fn=self.
          _experimental_gradient_multiplier_fn,
          optimization_parameters=tf.compat.v1.tpu.experimental.
          AdagradParameters(learning_rate=1.0))
    else:
      embedding_config_spec = None

    params = self._create_params(total_replicas)

    if self._task_param.eval.per_replica_batch_size is not None:
      eval_batch_size = self._task_param.eval.per_replica_batch_size * total_replicas
    else:
      eval_batch_size = train_global_batch_size

    return tf.compat.v1.estimator.tpu.TPUEstimator(
        use_tpu=True,
        model_fn=model_fn,
        config=config,
        train_batch_size=train_global_batch_size,
        eval_batch_size=eval_batch_size,
        params=params,
        embedding_config_spec=embedding_config_spec), total_replicas

  def create_tpu_estimator_on_cpu(self, model_fn, feature_config, table_config):
    if self._host_call_every_n_steps == 0:
      _experimental_host_call_every_n_steps = 100
    else:
      _experimental_host_call_every_n_steps = self._host_call_every_n_steps

    config = tf.compat.v1.estimator.tpu.RunConfig(
        cluster=None,
        model_dir=None,
        save_checkpoints_steps=self._save_checkpoints_steps,
        tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
            iterations_per_loop=self.iterations_per_loop,
            experimental_host_call_every_n_steps=
            _experimental_host_call_every_n_steps,
            per_host_input_for_training=tf.compat.v1.estimator.tpu.
            InputPipelineConfig.PER_HOST_V2,
            experimental_allow_per_host_v2_parallel_get_next=True))

    if feature_config and table_config:
      embedding_config_spec = tf.compat.v1.estimator.tpu.experimental.EmbeddingConfigSpec(
          feature_to_config_dict=feature_config,
          table_to_config_dict=table_config,
          pipeline_execution_with_tensor_core=self._pipeline_execution,
          experimental_gradient_multiplier_fn=self.
          _experimental_gradient_multiplier_fn,
          optimization_parameters=tf.compat.v1.tpu.experimental.
          AdagradParameters(learning_rate=1.0))
    else:
      embedding_config_spec = None

    total_replicas = 1
    params = self._create_params(total_replicas)

    return tf.compat.v1.estimator.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=config,
        train_batch_size=128,
        params=params,
        embedding_config_spec=embedding_config_spec), total_replicas

  def run(self):
    try:
      current_step = tf.train.load_variable(self._model_dir,
                                            tf.compat.v1.GraphKeys.GLOBAL_STEP)
    except (TypeError, ValueError, tf.errors.NotFoundError):
      current_step = 0
    logging.info("Current step :{}".format(current_step))

    task = self._task_param.instantiate()
    self._task = task
    feature_config, table_config = None, None
    if isinstance(task, BaseEmbeddingTask):
      task.init_slot_to_env()
      feature_config, table_config = task.create_feature_and_table_config_dict()
    input_fn_train = task.create_input_fn(tf.estimator.ModeKeys.TRAIN)

    model_fn = task.create_model_fn()

    assert self._cpu_test is False or self._mode == 'train', \
      "Cpu test can only work with train mode."

    if self._cpu_test:
      # If running CPU test, wrap model a little bit to pre-process features.
      def model_fn_test_wrapper(features, mode, params):
        features = task.process_features_for_cpu_test(features)
        return model_fn(features, mode, params)

      est, total_replicas = self.create_tpu_estimator_on_cpu(
          model_fn_test_wrapper, feature_config, table_config)
    else:
      est, total_replicas = self.create_tpu_estimator(model_fn, feature_config,
                                                      table_config)

    start_timestamp = time.time()  # This time will include compilation time

    if self._mode == 'train':
      est.train(input_fn=input_fn_train,
                max_steps=self._task_param.train.max_steps)
    elif self._mode == 'eval':
      input_fn_eval = task.create_input_fn(tf.estimator.ModeKeys.EVAL)
      total_examples = self._task_param.input.eval_examples
      eval_batch_size = self._task_param.eval.per_replica_batch_size * total_replicas

      eval_steps = total_examples // eval_batch_size
      logging.info(
          "Evaluation: total_examples:{} eval_batch_size:{} num_eval_steps: {}".
          format(total_examples, eval_batch_size, eval_steps))
      output_dir = os.path.join(self._model_dir, 'eval')
      tf.io.gfile.makedirs(output_dir)

      # Run evaluation when there's a new checkpoint
      for ckpt in tf.train.checkpoints_iterator(self._model_dir,
                                                timeout=60 * 60 * 5):
        # Terminate eval job when final checkpoint is reached
        current_step = int(os.path.basename(ckpt).split('-')[1])
        try:
          current_step = int(os.path.basename(ckpt).split('-')[1])
          logging.info("Starting to evaluate step: {}".format(current_step))
        except:
          logging.warning("Could not find current step value")

        try:
          start_timestamp = time.time(
          )  # This time will include compilation time
          eval_results = est.evaluate(input_fn=input_fn_eval,
                                      steps=eval_steps,
                                      checkpoint_path=ckpt)
          elapsed_time = int(time.time() - start_timestamp)
          logging.info("Eval results: {}. Elapsed seconds: {}".format(
              eval_results, elapsed_time))

          # Summary writer writes out eval metrics.
          summary_writer = tf.compat.v1.summary.FileWriter(output_dir)
          self.write_summary(eval_results, summary_writer, current_step)
          summary_writer.close()

          if current_step >= self._task_param.train.max_steps:
            logging.info("Evaluation finished after training step {}".format(
                current_step))
            break

        except tf.errors.NotFoundError:
          # Since the coordinator is on a different job than the TPU worker,
          # sometimes the TPU worker does not finish initializing until long after
          # the CPU job tells it to start evaluating. In this case, the checkpoint
          # file could have been deleted already.
          logging.info(
              "Checkpoint {} no longer exists, skipping checkpoint".format(
                  ckpt))
    else:  # train_and_eval
      raise TypeError("{} has not been supported.".format(self._mode))


def main(unused_argv):
  task_name = FLAGS.task
  task_param = model_registry.GetParams(task_name)

  logging.info("FLAGS:")
  for key, value in FLAGS.__flags.items():
    logging.info("{}: {}".format(key, value.value))
  logging.info("task_param: {}".format(str(task_param)))
  runner = TPURunner(task_param)
  runner.run()


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.disable_v2_behavior()
  app.run(main)
