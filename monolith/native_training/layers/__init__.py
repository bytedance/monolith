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

import types
import tensorflow as tf
from tensorflow.keras.layers import *

from monolith.native_training.layers.mlp import MLP
from monolith.native_training.layers.feature_cross import *
from monolith.native_training.layers.feature_trans import *
from monolith.native_training.layers.feature_seq import *
from monolith.native_training.layers.advanced_activations import *
from monolith.native_training.layers.add_bias import AddBias
from monolith.native_training.layers.lhuc import LHUCTower
from monolith.native_training.layers.logit_correction import LogitCorrection
from monolith.native_training.layers.norms import LayerNorm, GradNorm
from monolith.native_training.layers.pooling import SumPooling, AvgPooling, MaxPooling
from monolith.native_training.layers.utils import MergeType, DCNType
from monolith.native_training.layers.multi_task import MMoE, SNR
from monolith.native_training.utils import params as _params

del globals()['Dense']
from monolith.native_training.layers.dense import Dense

keras_layers = {}
for name in dir(tf.keras.layers):
  if name.startswith("_") or name == "Layer":
    continue
  cls = getattr(tf.keras.layers, name)
  try:
    if issubclass(cls, Layer) and not hasattr(cls, 'params'):
      cls.params = types.MethodType(_params, cls)
      keras_layers[name] = cls
  except:
    pass
