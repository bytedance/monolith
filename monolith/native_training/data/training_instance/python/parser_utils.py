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

import tensorflow as tf
from collections import deque
from typing import Callable

from monolith.native_training import ragged_utils

_extra_parse_steps = deque([])


def add_extra_parse_step(parse_fn: Callable):
  _extra_parse_steps.append(parse_fn)


class RaggedEncodingHelper:
  """Helper methods to precompute ragged encodings in input_fn, as a workaround
  
  Fundamentally, we should modify TensorFlow Dataset structure handler to compute
  provided encoding tensor in RowParition of a RaggedTensor.
  """

  @staticmethod
  def expand(name_to_ragged_ids,
             with_precomputed_nrows=True,
             with_precomputed_value_rowids=False):
    """Expand the RaggedTensor format in dict to precompute encodings within data iterator."""
    d = {}
    for k, v in name_to_ragged_ids.items():
      if isinstance(v, tf.RaggedTensor):
        d[k] = {
            # Basics
            "values":
                v.values,
            "row_splits":
                v.row_splits,
            "nrows":
                v.nrows() if with_precomputed_nrows else None,
            "value_rowids":
                ragged_utils.fused_value_rowids(v)
                if with_precomputed_value_rowids else None
        }
      else:
        d[k] = v
    return d

  @staticmethod
  def contract(name_to_ragged_ids):
    """Contract to recover RaggedTensor-only dict after computed."""
    d = {}
    for k, v in name_to_ragged_ids.items():
      if isinstance(v, dict) and ("values" in v) and ("row_splits" in v):
        t = tf.RaggedTensor.from_row_splits(v["values"],
                                            v["row_splits"],
                                            validate=False)
        if "nrows" in v:
          assert t._row_partition._nrows is None, "Shouldn't override the exisiting nrows."
          t._row_partition._nrows = v["nrows"]
        if "value_rowids" in v:
          assert t._row_partition._value_rowids is None, "Shouldn't override the exisiting tensor."
          t._row_partition._value_rowids = v["value_rowids"]
        d[k] = t
      else:
        d[k] = v
    return d


def advanced_parse(features):
  while _extra_parse_steps:
    fn = _extra_parse_steps.popleft()
    features = fn(features)

  return features
