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
from abc import ABCMeta, abstractmethod
from monolith.agent_service.utils import AgentConfig, TFS_HOME, TFSServerType

TFS_BINARY = f'{TFS_HOME}/bin/tensorflow_model_server'
PROXY_BINARY = f'{TFS_HOME}/bin/server'


def get_cmd_path():
  path = os.path.abspath(__file__)
  return path


def get_cmd_and_port(config: AgentConfig,
                     conf_path: str = None,
                     server_type: str = None,
                     config_file: str = None,
                     tfs_binary: str = TFS_BINARY,
                     proxy_binary: str = PROXY_BINARY):
  if server_type == TFSServerType.PS:
    return config.get_cmd_and_port(tfs_binary,
                                   server_type=TFSServerType.PS,
                                   config_file=config_file)
  elif server_type == TFSServerType.ENTRY:
    return config.get_cmd_and_port(tfs_binary,
                                   server_type=TFSServerType.ENTRY,
                                   config_file=config_file)
  elif server_type == TFSServerType.DENSE:
    return config.get_cmd_and_port(tfs_binary,
                                   server_type=TFSServerType.DENSE,
                                   config_file=config_file)
  else:
    proxy_conf = os.path.join(conf_path, 'proxy.conf')
    if os.path.exists(proxy_conf):
      cmd = f'{proxy_binary} --port={config.proxy_port} ' \
            f'--grpc_target=localhost:{config.tfs_entry_port} --conf_file={proxy_conf} &'
    else:
      cmd = f'{proxy_binary} --port={config.proxy_port} ' \
            f'--grpc_target=localhost:{config.tfs_entry_port} &'
    return cmd, config.proxy_port


class ServingLog(object):

  def __init__(self, log_prefix: str, tfs_log: str):
    self._log_prefix = log_prefix
    self._tfs_log = tfs_log
    self._cwd = None
    self._log = None

  def __enter__(self):
    dirname = os.path.dirname(self._tfs_log)
    basename = os.path.basename(self._tfs_log)
    log_filename = os.path.join(dirname, f"{self._log_prefix}_{basename}")
    self._cwd = os.getcwd()
    os.chdir(f'{TFS_HOME}/bin')
    return open(log_filename, 'a')

  def __exit__(self, exc_type, exc_val, exc_tb):
    os.chdir(self._cwd)


class AgentBase(metaclass=ABCMeta):

  def __init__(self, conf: AgentConfig):
    self.config = conf

  @abstractmethod
  def start(self):
    raise NotImplementedError("start is not implemented")

  @abstractmethod
  def wait_for_termination(self):
    raise NotImplementedError("wait_for_termination is not implemented")
