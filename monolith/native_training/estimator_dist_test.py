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
import socket
import time
import unittest
from multiprocessing import Process

import tensorflow as tf

from monolith.native_training.runner_utils import RunnerConfig
from monolith.native_training.model import TestFFMModel
from monolith.native_training.service_discovery import TfConfigServiceDiscovery
from monolith.native_training.estimator import Estimator
from monolith.native_training.utils import get_test_tmp_dir

model_name = 'dist_test'
model_dir = "{}/{}/ckpt".format(get_test_tmp_dir(), model_name)
export_base = "{}/{}/saved_model".format(get_test_tmp_dir(), model_name)
_EXIT_SUCCESS = 0


def get_free_port():
  """TODO(fitzwang) this function is not safe in preemption env"""
  sock = socket.socket()
  sock.bind(('', 0))
  ip, port = sock.getsockname()
  sock.close()
  return port


def get_cluster(ps_num, worker_num):
  cluster = {
      'ps': ['localhost:{}'.format(get_free_port()) for _ in range(ps_num)],
      'worker': [
          'localhost:{}'.format(get_free_port()) for _ in range(worker_num - 1)
      ],
      'chief': ['localhost:{}'.format(get_free_port())]
  }

  return cluster


def get_saved_model_path(export_base):
  try:
    candidates = []
    for f in os.listdir(export_base):
      if not (f.startswith('temp') or f.startswith('tmp')):
        fname = os.path.join(export_base, f)
        if os.path.isdir(fname):
          candidates.append(fname)
    candidates.sort()
    return candidates[-1]
  except:
    return ""


class EstimatorTrainTest(unittest.TestCase):
  """The tests here are for testing complilation."""
  params = None

  @classmethod
  def setUpClass(cls) -> None:
    if tf.io.gfile.exists(model_dir):
      tf.io.gfile.rmtree(model_dir)

    params = TestFFMModel.params()
    params.metrics.enable_deep_insight = False
    params.train.per_replica_batch_size = 64

    cls.params = params

  def train(self):
    ps_num, worker_num = 2, 3
    cluster = get_cluster(ps_num, worker_num)

    def start_server(server_type, index):
      task = {'type': server_type, 'index': index}
      tf_confg = {'cluster': cluster, 'task': task}
      discovery = TfConfigServiceDiscovery(tf_confg)

      dct_config = RunnerConfig(index=discovery.index,
                                model_dir=model_dir,
                                ps_num=ps_num,
                                worker_num=worker_num,
                                server_type=discovery.server_type)
      estimator = Estimator(self.params, dct_config, discovery)
      estimator.train(steps=10)

    threads = []
    for i in range(ps_num):
      thread = Process(target=start_server, args=('ps', i))
      thread.start()
      threads.append(thread)

    for i in range(worker_num):
      if i == 0:
        thread = Process(target=start_server, args=('chief', i))
      else:
        thread = Process(target=start_server, args=('worker', i - 1))
      thread.start()
      threads.append(thread)
      if i == 0:
        time.sleep(1)

    for thread in threads:
      thread.join()
      assert thread.exitcode == _EXIT_SUCCESS

  def eval(self):
    ps_num, worker_num = 2, 3
    cluster = get_cluster(ps_num, worker_num)

    def start_server(server_type, index):
      task = {'type': server_type, 'index': index}
      tf_confg = {'cluster': cluster, 'task': task}
      discovery = TfConfigServiceDiscovery(tf_confg)

      dct_config = RunnerConfig(index=discovery.index,
                                model_dir=model_dir,
                                ps_num=ps_num,
                                worker_num=worker_num,
                                server_type=discovery.server_type)

      estimator = Estimator(self.params, dct_config, discovery)
      estimator.evaluate(steps=10)

    threads = []
    for i in range(ps_num):
      thread = Process(target=start_server, args=('ps', i))
      thread.start()
      threads.append(thread)

    for i in range(worker_num):
      if i == 0:
        thread = Process(target=start_server, args=('chief', i))
      else:
        thread = Process(target=start_server, args=('worker', i - 1))
      thread.start()
      threads.append(thread)
      if i == 0:
        time.sleep(1)

    for thread in threads:
      thread.join()
      assert thread.exitcode == _EXIT_SUCCESS

  def test_dist(self):
    self.train()
    self.eval()


if __name__ == "__main__":
  unittest.main()
