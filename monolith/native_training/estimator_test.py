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
import unittest

import tensorflow as tf

from monolith.native_training.runner_utils import RunnerConfig
from monolith.native_training.input import generate_ffm_example
from monolith.native_training.model import TestFFMModel,\
  _VOCAB_SIZES, _NUM_EXAMPLES
from monolith.native_training.estimator import Estimator, import_saved_model
from monolith.native_training.utils import get_test_tmp_dir

model_name = 'estimator_test'

model_dir = "{}/{}/ckpt".format(get_test_tmp_dir(), model_name)
export_base = "{}/{}/ckpt/exported_models".format(get_test_tmp_dir(),
                                                  model_name)
conf = RunnerConfig(is_local=True,
                    num_ps=0,
                    model_dir=model_dir,
                    use_native_multi_hash_table=False)


def get_saved_model_path(export_base):
  try:
    candidates = []
    for f in os.listdir(export_base):
      if not (f.startswith('temp') or f.startswith('tmp')):
        fname = os.path.join(export_base, f)
        if os.path.isdir(fname):
          candidates.append(fname)
    candidates.sort()
    return candidates[-1]
  except:
    return ""


class EstimatorTrainTest(unittest.TestCase):
  """The tests here are for testing complilation."""
  params = None

  @classmethod
  def setUpClass(cls) -> None:
    if tf.io.gfile.exists(model_dir):
      tf.io.gfile.rmtree(model_dir)

    params = TestFFMModel.params()
    params.metrics.enable_deep_insight = False
    params.train.per_replica_batch_size = 64
    params.serving.export_dir_base = export_base
    params.serving.shared_embedding = True

    cls.params = params

  def train(self):
    estimator = Estimator(self.params, conf)
    estimator.train(steps=10)

  def eval(self):
    estimator = Estimator(self.params, conf)
    estimator.evaluate(steps=10)

  def predict(self):
    estimator = Estimator(self.params, conf)
    estimator.predict()

  def export_saved_model(self):
    estimator = Estimator(self.params, conf)
    estimator.export_saved_model()

  def import_saved_model(self):
    saved_model_path = get_saved_model_path(export_base)
    print('saved_model_path', saved_model_path, flush=True)
    with import_saved_model(saved_model_path=saved_model_path) as infer:
      # There are some bugs here since functions to restore tables are not called. Will
      # resolve this by using resource concept in the future.
      start = time.time()
      num_ins = 0
      for i in range(10):
        features = [
            generate_ffm_example(_VOCAB_SIZES) for _ in range(_NUM_EXAMPLES)
        ]
        num_ins += len(features)
        infer(features)
      end = time.time()
      print(start, end, num_ins, 1000 * (end - start) / 10)

  def test_local(self):
    self.train()
    self.eval()
    self.predict()
    self.export_saved_model()
    self.import_saved_model()


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  unittest.main()
