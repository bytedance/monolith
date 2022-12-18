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
import unittest

from monolith.native_training.model_export import export_state_utils
from monolith.native_training.model_export import export_pb2


class ExportStateUtilsTest(unittest.TestCase):

  def test_basic(self):
    state = export_pb2.ServingModelState()
    entry = state.entries.add()
    entry.export_dir = "a"
    entry.global_step = 1
    dir = os.path.join(os.environ["TEST_TMPDIR"], "basic")
    export_state_utils.overwrite_export_saver_listener_state(dir, state)
    new_state = export_state_utils.get_export_saver_listener_state(dir)
    self.assertEquals(new_state, state)


if __name__ == "__main__":
  unittest.main()
