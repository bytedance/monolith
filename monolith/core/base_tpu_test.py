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

import tensorflow as tf

from monolith.core import model_registry
from monolith.tpu_runner import TPURunner


class BaseTPUTest(tf.test.TestCase):
  """Base class for tpu test."""

  def runWithCPU(self, task_name):
    task_param = model_registry.GetParams(task_name)
    runner = TPURunner(task_param)
    runner._cpu_test = True
    runner._host_call_every_n_steps = 0
    runner.run()

  def runMergeVectorTestOnCPU(self, task_name):
    task_param = model_registry.GetParams(task_name)
    task_param.merge_vector = True
    runner = TPURunner(task_param)
    runner._cpu_test = True
    runner._host_call_every_n_steps = 0
    runner.run()

    env = runner._task._env
    # Verify slot number should be same before and after merged vectors.
    self.assertEqual(len(env._slot_to_dims.keys()),
                     len(env._slot_to_merged_dims.keys()))

    # Verify all slots are same, merged dims are expected.
    for slot_id, original_dims in env._slot_to_dims.items():
      #Verify slot_id are all the same before and after merged vecotrs.
      self.assertIn(slot_id, env._slot_to_merged_dims)
      merged_dims = env._slot_to_merged_dims[slot_id]

      if original_dims[0] == 1:
        # Veirfy bias dim is same.
        self.assertEqual(merged_dims[0], 1)
        # Verify merged vector dim is same.
        if len(original_dims) > 1:
          expect_merged_dim = sum(original_dims[1:])
          self.assertEqual(len(merged_dims), 2)
          self.assertEqual(expect_merged_dim, merged_dims[1])
        else:
          self.assertEqual(len(merged_dims), 1)
      else:
        # Verify merged vector dim is same.
        expect_merged_dim = sum(original_dims)
        self.assertEqual(len(merged_dims), 1)
        self.assertEqual(expect_merged_dim, merged_dims[0])

    # Verify all split features are as expected.
    for name, embedding in env._tpu_features.items():
      if "slot_" in name:
        slot_id = int(name.split("_")[1])
        index = int(name.split("_")[2])
        expect_dim = env._slot_to_dims[slot_id][index]
        actual_dim = embedding.get_shape().as_list()[1]
        self.assertEqual(actual_dim, expect_dim)
