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

from typing import Dict, List, Callable, Tuple

import tensorflow as tf

from absl import flags

from monolith.native_training.runtime.ops import gen_monolith_ops

flags.DEFINE_integer(
    "monolith_default_machine_info_mem_limit", 1 << 62,
    "The default value for mem_limit in machine info. (Bytes)")
FLAGS = flags.FLAGS

logging_ops = gen_monolith_ops


def tensors_timestamp(
    tensors: List[tf.Tensor]) -> Tuple[List[tf.Tensor], tf.Tensor]:
  """Gets the timestamp when the tensors are ready."""
  return logging_ops.monolith_tensors_timestamp(tensors)


def emit_timer(key: str,
               value: tf.Tensor,
               tags: Dict[str, str] = None) -> tf.Operation:
  tags = tags or {}
  tag_str = "|".join([f"{k}={v}" for k, v in tags.items()])
  return logging_ops.monolith_metric_v2(value, key=key, tags=tag_str)


def machine_info(mem_limit=None, shared_name=None) -> tf.Tensor:
  """Returns a MachineInfo tensor which contains a MachineInfo resource."""
  if mem_limit is None:
    mem_limit = FLAGS.monolith_default_machine_info_mem_limit
  return logging_ops.monolith_machine_info(mem_limit=mem_limit,
                                           name=shared_name,
                                           shared_name=shared_name)


def check_machine_health(machine_info_tensor: tf.Tensor) -> tf.Tensor:
  """Returns a scalar string tensor, which is serialized version of MachineHealthResult."""
  return logging_ops.monolith_check_machine_health(machine_info_tensor)
