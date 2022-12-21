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

import subprocess
from typing import List
import time
from absl import app
from absl import flags
import os
from monolith.native_training import yarn_runtime
from socket import socket
import json

flags.DEFINE_enum('training_type', 'batch', ['batch', 'stream'], "type of training to launch")
FLAGS = flags.FLAGS


occupied_ports = set()

def get_rand_port():
  # this function returns a unique unused port
  while True:
    with socket() as s:
      s.bind(('',0))
      port = s.getsockname()[1]
      if port not in occupied_ports:
        occupied_ports.add(port)
        return port


def launch_workers(num_ps: int, num_workers: int):
  args = [
    "markdown/demo/demo_model",
    f"--training_type={FLAGS.training_type}",
    "--model_dir=/tmp/movie_lens_tutorial",
    "--model_name=movie_lens_tutorial"
  ]
  assert num_workers > 1, "must have more than 1 workers"
  ip = yarn_runtime.get_local_host()
  ps_addrs = [f'{ip}:{get_rand_port()}' for i in range(num_ps)]
  worker_addrs = [f'{ip}:{get_rand_port()}' for i in range(num_workers)]
  
  env = os.environ.copy()
  tf_config = {
    "cluster": {
      "worker": worker_addrs,
      "ps": ps_addrs,
    }
  }
  
  processes = []
  for i in range(num_ps):
    tf_config['task'] = {"type": "ps", "index": i}
    env['TF_CONFIG'] = json.dumps(tf_config)
    processes.append(subprocess.Popen(args, env=env))

  for i in range(num_workers):
    tf_config['task'] = {"type": "worker", "index": i}
    env['TF_CONFIG'] = json.dumps(tf_config)
    processes.append(subprocess.Popen(args, env=env))
    if i == 0:
      time.sleep(2)
  return processes


def main(_):
  num_ps = 2
  num_workers = 2
  processes = launch_workers(
    num_ps,
    num_workers
  )
  try:
    for p in processes:
      p.wait()
  finally:
    for p in processes:
      p.kill()

if __name__ == "__main__":
  app.run(main)
