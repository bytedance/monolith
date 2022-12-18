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

from absl import app

from monolith.native_training import ps_benchmark
from monolith.native_training import cpu_training
from monolith.native_training import utils


class PsBenchmarkTest(tf.test.TestCase):

  def testBasic(self):
    p = ps_benchmark.PsBenchMarkTask.params()
    p.bm_config = ps_benchmark.BenchmarkConfig(ps_list=["ps0", "ps1"],
                                               num_ps_required=1,
                                               num_workers=1,
                                               index=0,
                                               benchmark_secs=1.0)
    cpu_training.local_train(p,
                             num_ps=2,
                             model_dir=utils.get_test_tmp_dir() + "/basic")
    self.assertEqual(len(p.bm_config.ps_list), 1)

  def testSkipBenchmark(self):
    p = ps_benchmark.PsBenchMarkTask.params()
    p.bm_config = ps_benchmark.BenchmarkConfig(ps_list=["ps0", "ps1"],
                                               num_ps_required=1,
                                               num_workers=1,
                                               index=0,
                                               benchmark_secs=1.0,
                                               ps_str_overridden="overridden")
    cpu_training.local_train(p,
                             num_ps=2,
                             model_dir=utils.get_test_tmp_dir() +
                             "/skip_benchmark")
    self.assertEqual(p.bm_config.ps_list[0], "overridden")


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  app.run(tf.test.main)
