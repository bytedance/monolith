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

from typing import Callable

import tensorflow as tf
from monolith.native_training.runtime.hash_table import embedding_hash_table_pb2


@tf.function
def iterate_table_and_apply(table: "HashTable",
                            apply_fn: Callable[[tf.Tensor], None],
                            limit=1000,
                            nshards=4,
                            name="IterateTable"):
  """Iterate the hash table, and call apply_fn for each slice.
  Args:
    apply_fn - a fn that accepts a 1-D tf string which is serialized EntryDump.
    limit - the maximum number of strings that will be fed into apply_fn (to save the memory usage).
    nshards - the parallelism of calling apply_fn.
  """
  for i in tf.range(nshards):
    offset = tf.constant(0, dtype=tf.int64)
    dump = tf.constant([], dtype=tf.string)
    while tf.math.equal(tf.size(dump), limit) or tf.math.equal(offset, 0):
      tf.autograph.experimental.set_loop_options(
          parallel_iterations=1,
          shape_invariants=[(dump, tf.TensorShape([None])),
                            (offset, tf.TensorShape([]))])
      offset, dump = table.save_as_tensor(i, nshards, limit, offset)
      apply_fn(dump)


def infer_dim_size(
    config: embedding_hash_table_pb2.EmbeddingHashTableConfig) -> int:
  dim_size = 0
  for segment in config.entry_config.segments:
    dim_size += segment.dim_size
  return dim_size
