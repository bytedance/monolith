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

import unittest
from monolith.native_training.summary.utils import \
  prepare_head, SummaryType, get_nas_weight_json


class UtilsTest(unittest.TestCase):

  def test_read_head_gating(self):
    segment_names = ['f1', 'f2', 'f3']
    segment_sizes = [3, 5, 9]
    group_info = [['f1', 'f2'], ['f3', 'f4'], ['f5', 'f6']]
    data, nas_type = prepare_head(segment_names, segment_sizes, group_info)
    data_exp = b'{"tag_type": "gating", "segment_names": ["f1", "f2", "f3"], "segment_sizes": [3, 5, 9], "group_index": [0, 0, 1]}'

    self.assertEqual(nas_type, SummaryType.GATING)
    self.assertEqual(data.numpy(), data_exp)

  def test_read_head_selecting(self):
    segment_names = ['f1', 'f2', 'f3']
    segment_sizes = [[3, 6], [5, 10], [4, 8, 16]]
    data, nas_type = prepare_head(segment_names, segment_sizes)
    data_exp = b'{"tag_type": "selecting", "segment_names": ["f1", "f2", "f3"], "segment_sizes": [[3, 6], [5, 10], [4, 8, 16]]}'
    
    self.assertEqual(nas_type, SummaryType.SELECTING)
    self.assertEqual(data.numpy(), data_exp)


if __name__ == "__main__":
  unittest.main()
