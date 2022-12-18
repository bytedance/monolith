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

# 64 MB
TOUCHED_KEY_SET_CAPACITY = 64 * 1024 * 1024 // (8 * 4)
TOUCHED_KEY_SET_CONCURRENCY_LEVEL = 1024

touched_key_set_ops = gen_monolith_ops


def create_touched_key_set(capacity: int,
                           concurrency_level: int,
                           name_suffix: str = "") -> tf.Tensor:
  """Creates a touched key set"""
  return touched_key_set_ops.MonolithTouchedKeySet(
      capacity=capacity,
      concurrency_level=concurrency_level,
      shared_name="MonolithTouchedKeySet" + name_suffix)


class TouchedKeySet(object):

  def __init__(self,
               capacity: int = TOUCHED_KEY_SET_CAPACITY,
               concurrency_level: int = TOUCHED_KEY_SET_CONCURRENCY_LEVEL,
               name_suffix: str = ""):
    self._set = create_touched_key_set(capacity, concurrency_level)
    self._capacity = capacity
    self._concurrency_level = concurrency_level

  def insert(self, ids: tf.Tensor) -> int:
    return touched_key_set_ops.monolith_touched_key_set_insert(self._set, ids)

  def steal(self) -> int:
    return touched_key_set_ops.monolith_touched_key_set_steal(self._set)

  @property
  def capacity(self) -> int:
    return self._capacity

  @property
  def concurrency_level(self) -> int:
    return self._concurrency_level

  @property
  def handle(self) -> tf.Tensor:
    return self._set
