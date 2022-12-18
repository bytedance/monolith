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

from absl import app, flags, logging
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from kazoo.client import KazooClient
import os
import copy
import subprocess
import signal
from subprocess import CalledProcessError
import threading
import time
from typing import List

from monolith.agent_service.replica_manager import ReplicaManager
from monolith.agent_service.agent_service import AgentService
from monolith.agent_service.utils import AgentConfig, DeployType, check_port_open
from monolith.native_training.zk_utils import MonolithKazooClient
from monolith.native_training import env_utils
from monolith.agent_service.agent_v1 import AgentV1
from monolith.agent_service.agent_v3 import AgentV3
from monolith.agent_service.model_manager import ModelManager

FLAGS = flags.FLAGS
flags.DEFINE_string('tfs_log', '/var/log/tfs.std.log',
                    'The tfs log file path')
def main(_):
  if FLAGS.conf is None:
    print(FLAGS.get_help())
    return

  config = AgentConfig.from_file(FLAGS.conf)
  conf_path = os.path.dirname(FLAGS.conf)
  if config.agent_version == 1:
    agent = AgentV1(config, conf_path, FLAGS.tfs_log)
  elif config.agent_version == 2:
    raise Exception('agent_version v2 is not support')
  elif config.agent_version == 3:
    agent = AgentV3(config, conf_path, FLAGS.tfs_log)
  else:
    raise Exception(f"agent_version error {config.agent_version}")

  # start model manager for rough sort model
  model_manager = ModelManager(config.rough_sort_model_name,
                               config.rough_sort_model_p2p_path,
                               config.rough_sort_model_local_path, True)
  ret = model_manager.start()
  if not ret:
    logging.error('model_manager start failed, kill self')
    os.kill(os.getpid(), signal.SIGKILL)

  agent.start()
  agent.wait_for_termination()


if __name__ == '__main__':
  try:
    env_utils.setup_hdfs_env()
  except Exception as e:
    logging.error('setup_hdfs_env fail {}!'.format(e))
  logging.info(f'environ is : {os.environ!r}')
  app.run(main)
