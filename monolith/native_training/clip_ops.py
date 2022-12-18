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

import os
from typing import List, Tuple

import tensorflow as tf
from monolith.native_training import device_utils
from monolith.native_training.runtime.ops import gen_monolith_ops

gen_clip_ops = gen_monolith_ops


def _global_norm(t_list: List[tf.Tensor]) -> tf.Tensor:
  """Computes the global norm of multiple tensors."""
  l2_sum = gen_clip_ops.global_l2_reduce(t_list)
  with tf.device("/device:CPU:0"):
    # We use CPU here because it's only a scalar, and the downstream clip needs it on HostMemory at first.
    return tf.sqrt(l2_sum)


def clip_by_global_norm(t_list: List[tf.Tensor],
                        clip_norm: tf.Tensor,
                        use_norm=None) -> Tuple[List[tf.Tensor], tf.Tensor]:
  """Clips values of multiple tensors by the ratio of the sum of their norms.

  Given a tuple or list of tensors `t_list`, and a clipping ratio `clip_norm`,
  this operation returns a list of clipped tensors `list_clipped`
  and the global norm (`global_norm`) of all tensors in `t_list`. Optionally,
  if you've already computed the global norm for `t_list`, you can specify
  the global norm with `use_norm`.

  To perform the clipping, the values `t_list[i]` are set to:
      t_list[i] * clip_norm / max(global_norm, clip_norm)
  where:
      global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))

  If `clip_norm > global_norm` then the entries in `t_list` remain as they are,
  otherwise they're all shrunk by the global ratio.
  If `global_norm == infinity` then the entries in `t_list` are all set to `NaN`
  to signal that an error occurred.

  Args:
    t_list: A list of mixed `Tensors`.
    clip_norm: A 0-D (scalar) `Tensor` > 0. The clipping ratio.
    use_norm: A 0-D (scalar) `Tensor` of type `float` (optional). The global
      norm to use. If not provided, TensorFlow `global_norm()` is used to compute the norm.
  Returns:
    list_clipped: A list of `Tensors` of the same type as `list_t`.
    global_norm: A 0-D (scalar) `Tensor` representing the global norm.
  Raises:
    TypeError: If `t_list` is not a sequence.
  """
  with tf.name_scope('clip_by_global_norm'):
    if not isinstance(t_list, list):
      raise TypeError("t_list should be a list")
    norm_fn = _global_norm if device_utils.within_placement_context_of(
        "GPU") else tf.linalg.global_norm
    global_norm = norm_fn(t_list) if use_norm is None else use_norm
    list_clipped = gen_clip_ops.monolith_clip_by_global_norm(
        t_list, global_norm, clip_norm)
  return list_clipped, global_norm
