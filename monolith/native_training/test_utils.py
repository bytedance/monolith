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

from monolith.native_training import entry
from monolith.native_training import utils
from monolith.native_training.runtime.hash_table import \
    embedding_hash_table_pb2


def generate_test_hash_table_config(dim: int = 2,
                                    use_float16: float = False,
                                    learning_rate: float = 1.0):
  """Creates a valid hash table config."""
  table_config = embedding_hash_table_pb2.EmbeddingHashTableConfig()
  table_config.cuckoo.SetInParent()
  segment = table_config.entry_config.segments.add()
  segment.dim_size = dim
  segment.opt_config.sgd.SetInParent()
  segment.opt_config.stochastic_rounding_float16 = use_float16
  segment.init_config.zeros.SetInParent()
  segment.comp_config.fp32.SetInParent()
  return entry.HashTableConfigInstance(table_config, [learning_rate])


def create_test_ps_cluster(num_ps: int):
  """Generates a config based on servers"""
  servers = []
  for i in range(num_ps):
    servers.append(tf.distribute.Server.create_local_server())
  cluster_def = tf.train.ClusterDef()
  job = cluster_def.job.add()
  job.name = utils.PS_JOB_NAME
  for i, server in enumerate(servers):
    job.tasks[i] = server.target[len('grpc://'):]
  return servers, tf.compat.v1.ConfigProto(cluster_def=cluster_def)


def profile_it(fn):
  """Decorator for testcase to profile locally."""

  def wrapped_fn(*args, **kwargs):
    options = tf.profiler.experimental.ProfilerOptions(host_tracer_level=2,
                                                       python_tracer_level=1,
                                                       device_tracer_level=1)
    tf.profiler.experimental.start("/tmp/tests_profile", options)
    res = fn(*args, **kwargs)
    tf.profiler.experimental.stop()
    time.sleep(
        1)  # ensure distinct profile dir names defined by timestamp on sec
    return res

  return wrapped_fn
