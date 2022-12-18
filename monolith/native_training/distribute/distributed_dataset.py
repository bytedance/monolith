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

from collections import defaultdict
import dataclasses
from typing import List
import traceback

from absl import logging
import tensorflow as tf
from tensorflow.python.data.ops import dataset_ops

from monolith.native_training import utils
from monolith.native_training import native_task_context
from monolith.native_training.distribute import str_queue
from monolith.native_training.hooks import session_hooks


def create_dynamic_sharding_dataset(
    glob_patterns: List[str],
    name="dynamic_sharding_dataset") -> tf.data.Dataset:
  """The idea here is create 2 queues to create the filename database
  shared: glob_patterns_queue (element is like /some/path/*)
  shared: filenames_queue (element is like /some/path/data0)

  The reason why we have two shared queues is the list of filename is too long
  and can't fit into the memory. So we need expand on demand.
  """
  with tf.name_scope(name):
    device = utils.ps_device(0) if native_task_context.get().num_ps > 0 else ""

    # Queues on ps 0 or host if no ps.
    with tf.device(device):
      pattern_queue = str_queue.StrQueue(initial_elements=glob_patterns,
                                         name="glob_patterns_queue")

      @tf.function
      def glob_pattern():
        # We are in critical section already.
        pattern, out_of_range = pattern_queue._raw_dequeue()
        if not out_of_range:
          filenames = tf.io.matching_files(pattern)
        else:
          filenames = tf.constant([""])
        return filenames, out_of_range

      filenames_queue = str_queue.StrQueue(
          critical_section=pattern_queue.critical_section,
          auto_enqueue_fn=glob_pattern,
          name="filenames_queue")

      dequeued_filename = filenames_queue.dequeue()

    def filename_generator():
      filename_bytes, out_of_range = session_hooks.get_current_session().run(
          dequeued_filename)
      if out_of_range:
        raise StopIteration()
      return filename_bytes.decode()

    dummy_dataset = tf.data.Dataset.from_tensors(0).repeat()

    # Instead of map, we directly instantiate the MapDataset
    # because we don't want to keep preserve_cardinality.
    filename_dataset = dataset_ops.MapDataset(
        dummy_dataset,
        lambda _: tf.py_function(
            func=filename_generator, inp=[], Tout=tf.string),
        preserve_cardinality=False)
    return filename_dataset
