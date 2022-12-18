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
from absl import logging
import importlib

from tensorflow.python.tools import module_util as _module_util

from monolith.native_training import data
from monolith.native_training import layers
from monolith.native_training import model_export
from monolith.native_training import entry
from monolith.native_training import native_model as base_model
from monolith.native_training import estimator
from monolith.utils import enable_monkey_patch


def add_module(module):
  try:
    if isinstance(module, str):
      name = module.split('.')[-1]
      module = importlib.import_module(module)
    else:
      name = module.__name__.split('.')[-1]

    if name == 'native_model':
      name = 'base_model'
  except ImportError as e:
    raise e
  sys.modules[f'{__name__}.{name}'] = module


add_module(data)
add_module(layers)
add_module(model_export)
add_module(entry)
add_module(base_model)
add_module(estimator)

try:
  enable_monkey_patch()
except:
  logging.error('enable_monkey_patch failed')
