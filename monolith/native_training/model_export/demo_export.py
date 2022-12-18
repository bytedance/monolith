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

import getpass

from absl import app
from absl import flags

import tensorflow as tf

from monolith.native_training.data.training_instance.python.parse_instance_ops import parse_instances
from monolith.native_training import model
from monolith.native_training import cpu_training
from monolith.native_training.model import TestFFMModel
from monolith.native_training.model_export.export_context import ExportMode, enter_export_mode
from monolith.native_training.model_export.saved_model_exporters import StandaloneExporter, DistributedExporter

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "num_ps",
    default=5,
    help=("Number of parameter servers. Must align with training.")),

flags.DEFINE_string(
    "model_dir",
    default="/tmp/{}/monolith/native_training/demo/ckpt".format(
        getpass.getuser()),
    help=("Model dir containing training ckpts."),
)

flags.DEFINE_string(
    "export_base",
    default="/tmp/{}/monolith/native_training/demo/saved_model".format(
        getpass.getuser()),
    help=("The path to saved exported saved model."),
)

flags.DEFINE_enum_class("export_mode",
                        default=ExportMode.STANDALONE,
                        enum_class=ExportMode,
                        help="standalone or distributed")


def export_saved_model(model_dir, export_base, num_ps, export_mode):
  tf.compat.v1.disable_eager_execution()
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  params = TestFFMModel.params()
  params.name = "demo_export"
  params.train.per_replica_batch_size = 64
  task = params.instantiate()
  cpu_training_task = cpu_training.CpuTraining(
      cpu_training.CpuTrainingConfig(num_ps=num_ps), task)
  if export_mode == ExportMode.STANDALONE:
    exporter = StandaloneExporter(cpu_training_task.create_model_fn(),
                                  model_dir=model_dir,
                                  export_dir_base=export_base)
  elif export_mode == ExportMode.DISTRIBUTED:
    exporter = DistributedExporter(cpu_training_task.create_model_fn(),
                                   model_dir=model_dir,
                                   export_dir_base=export_base,
                                   shared_embedding=False)

  def serving_input_receiver_fn():
    receiver_tensors = {}
    features = {}
    instances_placeholder = tf.compat.v1.placeholder(dtype=tf.string,
                                                     shape=(None,))
    receiver_tensors["instances"] = instances_placeholder
    parsed_results = parse_instances(
        instances_placeholder,
        fidv1_features=[i for i in range(model._NUM_SLOTS)],
        fidv2_features=None,
        misc_float_features=[],
        misc_int64_features=[])
    for i in range(model._NUM_SLOTS):
      features["feature_{}".format(i)] = parsed_results["slot_{}".format(i)]
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

  exporter.export_saved_model(serving_input_receiver_fn)


def main(_):
  export_saved_model(FLAGS.model_dir, FLAGS.export_base, FLAGS.num_ps,
                     FLAGS.export_mode)


if __name__ == "__main__":
  app.run(main)
