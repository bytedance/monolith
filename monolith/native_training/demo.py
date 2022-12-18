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

"""bazel run -c opt monolith/native_training:demo -- --num_ps=0"""

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf

from monolith.native_training import cpu_training
from monolith.native_training.model import TestFFMModel
from monolith.native_training.model_export.export_context import ExportMode

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "num_ps",
    default=0,
    help=(
        "Number of parameter servers. 0 means no parameter server. Everything "
        "runs on the single local server."))
flags.DEFINE_string(
    "model_dir",
    default=None,
    help="Directory where model parameters, graph, etc are saved.")


def main(_):
  params = TestFFMModel.params()
  params.name = "test_ffm_model"
  params.train.per_replica_batch_size = 64
  params.serving.export_when_saving = True
  params.serving.export_mode = ExportMode.DISTRIBUTED
  cpu_training.local_train(params,
                           num_ps=FLAGS.num_ps,
                           model_dir=FLAGS.model_dir,
                           steps=100,
                           save_checkpoints_steps=50)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.compat.v1.disable_eager_execution()
  app.run(main)
