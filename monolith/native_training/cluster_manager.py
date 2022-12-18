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
import time
from typing import Dict, List, Tuple

from absl import logging
import tensorflow as tf

from monolith.native_training.metric import cli
from monolith.native_training.service_discovery import ServiceDiscovery

_MCLI = cli.get_cli(prefix="monolith.containers")


def emit_store(name, value, tagkv=None):
  _MCLI.emit_store(name, value, tagkv)


def generate_session_config(cluster_and_task=None):
  if cluster_and_task is None:
    session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
  else:
    cluster = cluster_and_task[0]
    task = cluster_and_task[1]
    spec = tf.train.ClusterSpec(cluster)
    device_filters = ["/job:ps", "/job:chief"]
    if task["type"] != "chief":
      device_filters += ["/job:{}/task:{}".format(task["type"], task["index"])]
    session_config = tf.compat.v1.ConfigProto(cluster_def=spec.as_cluster_def(),
                                              allow_soft_placement=True,
                                              device_filters=device_filters)
  session_config.share_cluster_devices_in_session = True
  session_config.experimental.share_session_state_in_clusterspec_propagation = True
  # grappler doesn't really understand RaggedTensor.
  session_config.graph_options.rewrite_options.disable_meta_optimizer = True
  return session_config


def get_training_cluster(
    discovery: ServiceDiscovery,
    worker_addr: str,
    index: int,
    num_redundant_ps: int,
    num_required_ps: int,
    num_workers: int,
    model_dir: str,
    uuid: str,
    model_name: str = None,
    cluster_type: str = "stable") -> Tuple[Dict[str, List], Dict]:
  if index == 0:
    if num_redundant_ps:
      file_name = _get_ps_cluster_file_name(model_dir, uuid)
      # In the case of chief restart, first obtain the ps cluster from the file.
      ps_addrs = _fetch_ps_cluster_from_file(file_name, timeout=0)
      if len(ps_addrs) != num_required_ps:
        # The ps cluster cannot be obtained from the file, so it is queried
        # through service discovery. Then assign the ps cluster to the file.
        ps_addrs = _query_ps_cluster(discovery, num_required_ps, model_name,
                                     cluster_type)
        _save_ps_cluster_to_file(file_name, ps_addrs)
    else:
      # By default, the ps cluster is queried by discovery.
      ps_addrs = _query_ps_cluster(discovery, num_required_ps, model_name,
                                   cluster_type)

    fake_worker_list = ["0.0.0.0:{}".format(i) for i in range(1, num_workers)]
    cluster = {
        "chief": [worker_addr],
        "worker": fake_worker_list,
        "ps": ps_addrs,
    }
    task = {"type": "chief", "index": 0}

  else:
    chief_addr = _query_chief_addr(discovery)
    # Due to current TF limitation (TF_CONFIG doesn't support dict),
    # we need to provide a fake worker list in cluster
    worker_index = index - 1
    fake_worker_list = ["0.0.0.0:{}".format(i) for i in range(1, num_workers)]
    fake_worker_list[worker_index] = worker_addr

    if num_redundant_ps:
      file_name = _get_ps_cluster_file_name(model_dir, uuid)
      # Get the ps cluster from the file.
      ps_addrs = _fetch_ps_cluster_from_file(file_name)
    else:
      # By default, the ps cluster is queried by discovery.
      ps_addrs = _query_ps_cluster(discovery, num_required_ps)

    cluster = {
        "chief": [chief_addr],
        "worker": fake_worker_list,
        "ps": ps_addrs,
    }
    task = {"type": "worker", "index": worker_index}

  assert len(cluster["ps"]) == num_required_ps
  return cluster, task


def _cluster_query_failure_handler():
  time.sleep(5)


def _query_chief_addr(discovery: ServiceDiscovery):
  worker_addr_dict = None
  while True:
    worker_addr_dict = discovery.query("worker")
    if 0 in worker_addr_dict:
      break
    _cluster_query_failure_handler()

  return worker_addr_dict[0]


def _query_ps_cluster(discovery: ServiceDiscovery,
                      num_required_ps: int,
                      model_name: str = None,
                      cluster_type: str = "stable"):
  start = time.time()
  ps_addr_dict = None
  while True:
    ps_addr_dict = discovery.query("ps")
    num_left_ps = max(0, num_required_ps - len(ps_addr_dict))
    logging.info("Got {} ps, {} left!".format(len(ps_addr_dict), num_left_ps))
    if model_name:
      tags = {
          "model_name": model_name,
          "cluster_type": cluster_type,
      }
      emit_store("num_left_ps", num_left_ps, tags)
      emit_store("job_waiting", 1, tags)
    if len(ps_addr_dict) >= num_required_ps:
      break
    _cluster_query_failure_handler()

  ps_addrs = [addr for index, addr in sorted(ps_addr_dict.items())
             ][:num_required_ps]
  return ps_addrs


def _save_ps_cluster_to_file(file_name: str, ps_addrs: List[str]):
  ps_str = ",".join(ps_addrs)
  tf.io.gfile.makedirs(os.path.dirname(file_name))
  tmp_name = file_name + "-tmp"
  with tf.io.gfile.GFile(tmp_name, mode="w") as f:
    f.write(ps_str)
  tf.io.gfile.rename(tmp_name, file_name, overwrite=True)


def _fetch_ps_cluster_from_file(file_name: str, timeout=1800):
  ps_str = ""
  start_time = time.time()

  while True:
    try:
      with tf.io.gfile.GFile(file_name) as f:
        ps_str = f.read()
    except tf.errors.NotFoundError:
      pass

    if bool(ps_str) or time.time() - start_time > timeout:
      break
    _cluster_query_failure_handler()

  ps_addrs = ps_str.split(",") if ps_str else []
  return ps_addrs


def _get_ps_cluster_file_name(model_dir: str, uuid: str):
  return os.path.join(model_dir, "ps_cluster_dir", uuid or "ps_info")
