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

from monolith.native_training import cluster_manager


class ClusterManagerTest(unittest.TestCase):

  def testBasic(self):
    ps_addrs = ["0.0.0.0:{}".format(i) for i in range(3)]
    file_name = cluster_manager._get_ps_cluster_file_name(
        model_dir=os.path.join(os.environ["TEST_TMPDIR"], "ClusterManagerTest",
                               self._testMethodName),
        uuid=self._testMethodName)
    cluster_manager._save_ps_cluster_to_file(file_name, ps_addrs)
    new_ps_addrs = cluster_manager._fetch_ps_cluster_from_file(file_name)
    self.assertEqual(ps_addrs, new_ps_addrs)


if __name__ == "__main__":
  unittest.main()
