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

import signal
import unittest

from monolith.native_training import signal_utils


class SignalUtilsTest(unittest.TestCase):

  def testBasic(self):
    # Add twice to test two handlers case.
    signal_utils.add_siguser1_handler()
    signal.raise_signal(signal.SIGUSR1)


if __name__ == "__main__":
  unittest.main()
