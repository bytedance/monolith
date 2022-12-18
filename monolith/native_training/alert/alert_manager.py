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

import copy
import threading
import time
import traceback
from typing import List, NamedTuple

from absl import flags
from absl import logging
from google.protobuf import text_format


FLAGS = flags.FLAGS

flags.DEFINE_string("monolith_alert_proto", "",
                    "The text format of alert proto.")
def get_default_alert_manager():
  return None
