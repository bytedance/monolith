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


def add_batch_norm_into_update_ops():
  ops = tf.compat.v1.get_default_graph().get_operations()
  update_ops = [
      op for op in ops
      if 'AssignMovingAvg' in op.name and op.type == "AssignSubVariableOp"
  ]

  for update_op in update_ops:
    tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, update_op)
