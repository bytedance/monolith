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

import threading
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from absl import flags
from absl import app
from google.protobuf import text_format

from monolith.native_training.alert import alert_pb2
from monolith.native_training.alert import alert_manager

FLAGS = flags.FLAGS


if __name__ == "__main__":
  absltest.main()
