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
import shutil
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

from monolith.native_training import distribution_ops


def run_fused_reorder_by_indicies(suffices=None):

  # Generate num_slots of lists of int64, where number of unique ids is around num_ids
  num_ids = int(1e6)
  num_slots = 30
  num_of_shards = 256
  int64 = np.iinfo(np.int64)
  ids = list(
      set(np.random.randint(low=int64.min, high=int64.max + 1, size=num_ids)))
  split_indicies = [0] + sorted(np.random.choice(num_ids, num_slots))
  ids_list = []
  for i in range(1, len(split_indicies)):
    slot_ids = ids[split_indicies[i - 1]:split_indicies[i]]
    slot_ids = np.concatenate([slot_ids, slot_ids])  # force dups
    np.random.shuffle(slot_ids)
    ids_list.append(slot_ids)

  # input: ids_list
  session_config = tf.compat.v1.ConfigProto()
  session_config.graph_options.rewrite_options.disable_meta_optimizer = False
  session_config.graph_options.rewrite_options.memory_optimization = 1
  session_config.intra_op_parallelism_threads = 4
  with tf.compat.v1.Session(config=session_config) as sess:
    ids_list = [ops.convert_to_tensor(ids, dtype=tf.int64) for ids in ids_list]
    reorder_op = distribution_ops.fused_reorder_by_indices(
        ids_list, num_of_shards=num_of_shards)
    start = time.time()
    _ = sess.run(reorder_op)
    return time.time() - start


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  # np.random.seed(1234)
  print('> Sess.run Wall Time:',
        np.average([run_fused_reorder_by_indicies() for _ in range(5)]))
