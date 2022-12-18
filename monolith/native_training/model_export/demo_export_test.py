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

import tensorflow as tf

from monolith.native_training.model_export.export_context import ExportMode
from monolith.native_training.model_export import demo_export
from monolith.native_training import cpu_training
from monolith.native_training.model import TestFFMModel

tf.compat.v1.disable_eager_execution()


class DemoExportTest(tf.test.TestCase):

  def test_demo_export(self):
    model_dir = os.path.join(os.environ["TEST_TMPDIR"], "test_ffm_model")
    params = TestFFMModel.params()
    params.name = "test_ffm_model"
    params.train.per_replica_batch_size = 64
    cpu_training.local_train(params, num_ps=5, model_dir=model_dir)

    demo_export.export_saved_model(
        model_dir,
        os.path.join(os.environ["TEST_TMPDIR"], "standalone_saved_model"), 5,
        ExportMode.STANDALONE)

    demo_export.export_saved_model(
        model_dir,
        os.path.join(os.environ["TEST_TMPDIR"], "distributed_saved_model"), 5,
        ExportMode.DISTRIBUTED)


if __name__ == "__main__":
  tf.test.main()
