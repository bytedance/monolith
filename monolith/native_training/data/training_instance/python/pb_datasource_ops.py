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
import os
from typing import Dict, List, Iterable, Callable

import tensorflow as tf
from tensorflow.python.framework import load_library

from monolith.native_training.runtime.ops import gen_monolith_ops

pb_datasource_ops = gen_monolith_ops


def filter_by_fids(variant: tf.Tensor,
                   filter_fids: List[int] = None,
                   has_fids: List[int] = None,
                   select_fids: List[int] = None,
                   has_actions: List[int] = None):
  return pb_datasource_ops.set_filter(variant, filter_fids or [], has_fids or
                                      [], select_fids or [], has_actions or [])


def filter_by_value(variant: tf.Tensor, field_name: str, op: str,
                    operand: float):
  return pb_datasource_ops.value_filter(variant, field_name, op, operand)


def negative_sample(variant: tf.Tensor, drop_rate: float, label_index: int,
                    threshold: float):
  return pb_datasource_ops.negative_sample(variant, drop_rate, label_index,
                                           threshold)


def variant_dummy(variant: tf.Tensor):
  return pb_datasource_ops.variant_dummy(variant)
