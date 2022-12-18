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

from absl import logging
import unittest
from monolith.agent_service.resource_utils import cal_available_memory_v2, total_memory_v2, \
  cal_cpu_usage_v2, cal_model_info_v2, cal_available_memory


class UtilTest(unittest.TestCase):

  def test_cal_avaiable_memory_v2(self):
    total = total_memory_v2()
    available = cal_available_memory_v2()
    logging.info(f'the total memory is {total}, and {available} is available')
    self.assertTrue(0 < available < total)

  def test_cal_cpu_usage_v2(self):
    usage = cal_cpu_usage_v2()
    logging.info(f'the cpu usage is {usage}')
    self.assertTrue(0 <= usage <= 100)


if __name__ == '__main__':
  unittest.main()
