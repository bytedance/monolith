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

import sys
import unittest

from monolith.core.hyperparams_test import ParamsTest
from monolith.core.base_layer_test import BaseLayerTest
from monolith.core.base_embedding_host_call_test import BaseEmbeddingHostCallTest
from monolith.core.util_test import UtilTest


def suite():
  suite = unittest.TestSuite()
  suite.addTest(unittest.makeSuite(ParamsTest))
  suite.addTest(unittest.makeSuite(BaseLayerTest))
  suite.addTest(unittest.makeSuite(BaseEmbeddingHostCallTest))
  suite.addTest(unittest.makeSuite(UtilTest))
  return suite


if __name__ == '__main__':
  runner = unittest.TextTestRunner(verbosity=2)
  runner.run(suite())
