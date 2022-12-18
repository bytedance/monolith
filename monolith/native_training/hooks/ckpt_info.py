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
from typing import DefaultDict

import numpy as np
import tensorflow as tf

from monolith.native_training import hash_table_ops
from monolith.native_training import hash_table_utils
from monolith.native_training import multi_hash_table_ops
from monolith.native_training.proto import ckpt_info_pb2

_MAX_SLOT = 102400


class FidSlotCountSaverListener(tf.estimator.CheckpointSaverListener):

  def __init__(self, model_dir: str):
    self._model_dir = model_dir
    all_tables = tf.compat.v1.get_collection(
        hash_table_ops._HASH_TABLE_GRAPH_KEY)
    self.all_multi_hash_tables = tf.compat.v1.get_collection(
        multi_hash_table_ops._MULTI_HASH_TABLE_GRAPH_KEY)
    if not all_tables and not self.all_multi_hash_tables:
      # MultiHashTable info is collected in a different way.
      # This usually means the listener is created before hash table is created
      # Throws an error here
      raise ValueError(
          ("Unable to find hash tables. "
           "It may be caused by creating the listener before calling model_fn"))

    device_to_tables = DefaultDict(list)
    for table in all_tables:
      device_to_tables[table.table.device].append(table)

    self._count_vars = {}
    count_ops = []

    for device, tables in device_to_tables.items():
      with tf.device(device):
        device_unique_str = str(device).replace(":", "_")
        count_var = tf.compat.v1.get_variable(
            f"monolith_fid_slot_count/{device_unique_str}",
            shape=[_MAX_SLOT],
            dtype=tf.int64,
            initializer=tf.compat.v1.zeros_initializer(tf.int64),
            collections=[])

        self._count_vars[device] = count_var

        def apply_fn(entry):
          slot = hash_table_ops.extract_slot_from_entry(entry)
          slot = tf.math.minimum(slot, _MAX_SLOT - 1)
          update = tf.ones_like(slot, dtype=tf.int64)
          index = tf.reshape(slot, [-1, 1])
          scattered = tf.scatter_nd(index, update, [_MAX_SLOT])
          count_var.assign_add(scattered, use_locking=True)

        for table in tables:
          count_ops.append(
              hash_table_utils.iterate_table_and_apply(table, apply_fn))

    self._count_op = tf.group(count_ops)

    init_ops = []
    for count_var in self._count_vars.values():
      init_ops.append(count_var.initializer)
    self._init_op = tf.group(init_ops)

  def before_save(self, session, global_step_value):
    if self.all_multi_hash_tables:
      return
    session.run(self._init_op)
    session.run(self._count_op)
    counts = session.run(list(self._count_vars.values()))
    counts = np.sum(counts, axis=0)
    info = ckpt_info_pb2.CkptInfo()
    for slot, count in enumerate(counts):
      if count:
        info.slot_counts[slot] = count
    tf.io.gfile.makedirs(self._model_dir)
    with tf.io.gfile.GFile(
        os.path.join(self._model_dir, f"ckpt.info-{global_step_value}"),
        "w") as f:
      f.write(str(info))
