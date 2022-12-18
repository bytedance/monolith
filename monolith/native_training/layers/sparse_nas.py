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

from glob import glob
import os
from absl import logging, flags
import numpy as np
from typing import List

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine.input_spec import InputSpec

from monolith.native_training.utils import with_params
from monolith.native_training.monolith_export import monolith_export
from monolith.native_training.layers.layer_ops import bernoulli_gate, discrete_gate, discrete_truncated_gate
from monolith.native_training.layers.utils import check_dim, dim_size

FLAGS = flags.FLAGS

