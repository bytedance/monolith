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

from monolith.native_training.runtime.ops import gen_monolith_ops

ops = gen_monolith_ops


def fused_value_rowids(rt: tf.RaggedTensor):
  """Equivalent to rt.value_rowids(), but with much less ops."""
  if not isinstance(rt, tf.RaggedTensor):
    raise ValueError("rt must be RaggedTensor")
  if not hasattr(rt, "monolith_fused_value_rowids"):
    rt.monolith_fused_value_rowids = ops.monolith_fused_value_rowids(
        rt.row_splits)
  return rt.monolith_fused_value_rowids
