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
import subprocess
import signal
from subprocess import CalledProcessError
import threading
import time
from typing import List

from monolith.agent_service.replica_manager import ReplicaManager
from monolith.agent_service.agent_service import AgentService
from monolith.agent_service.utils import AgentConfig, TFSServerType, DeployType, check_port_open
from monolith.agent_service.agent_base import AgentBase, ServingLog, TFS_HOME, get_cmd_and_port, TFS_BINARY
from monolith.native_training.zk_utils import MonolithKazooClient


class ProcessType(Enum):
  PS = 1
  ENTRY = 2
  PROXY = 3
  UNKONWN = 4
  DENSE = 5


class ProcessNode(object):

  def __init__(self,
               config: AgentConfig,
               replica_mgr: ReplicaManager,
               proc_type: ProcessType,
               is_tce_main: bool = False,
               conf_path: str = None,
               tfs_log: str = None,
               tfs_binary: str = TFS_BINARY):
    assert proc_type != ProcessType.UNKONWN

    self._config = config
    self._replica_mgr = replica_mgr
    self._shell = False
    self._stderr = subprocess.STDOUT
    self._env = os.environ
    self.proc_type = proc_type
    self.is_tce_main = is_tce_main
    self._tfs_log = tfs_log
    self._is_failover = False
    self._port = 0

    if proc_type == ProcessType.PS:
      self._cmd, self._port = get_cmd_and_port(config,
                                               conf_path,
                                               TFSServerType.PS,
                                               tfs_binary=tfs_binary)
    elif proc_type == ProcessType.ENTRY:
      self._env = os.environ.copy()
      self._env["PORT2"] = str(self._config.agent_port)
      self._cmd, self._port = get_cmd_and_port(config,
                                               conf_path,
                                               TFSServerType.ENTRY,
                                               tfs_binary=tfs_binary)
    elif proc_type == ProcessType.DENSE:
      self._cmd, self._port = get_cmd_and_port(config,
                                               conf_path,
                                               TFSServerType.DENSE,
                                               tfs_binary=tfs_binary)
    else:
      self._cmd, self._port = get_cmd_and_port(config,
                                               conf_path,
                                               tfs_binary=tfs_binary)

    self._popen = None
    self._sub_procs = {}

  @property
  def sub_procs(self):
    return self._sub_procs

  def add_subproc(self, pn: 'ProcessNode'):
    if pn.proc_type in self._sub_procs:
      logging.warning(f'process {pn.proc_type} exists!')
    else:
      self._sub_procs[pn.proc_type] = pn

  @property
  def returncode(self):
    if self._is_failover:
      return None
    else:
      return None if self._popen is None else self._popen.returncode

  def poll(self):
    if self._is_failover:
      return None
    else:
      return None if self._popen is None else self._popen.poll()

  def kill(self):
    # kill subprocess
    for proc in self._sub_procs.values():
      if proc is None:
        continue
      cnt, max_cnt = 0, 3
      proc.poll()
      while proc.returncode is None and cnt < max_cnt:
        logging.info(f"kill proc {proc}")
        proc.kill()
        time.sleep(1)
        proc.poll()
        cnt += 1

    # kill self
    if self._popen is not None:
      cnt, max_cnt = 0, 3
      self._popen.poll()
      while self._popen.returncode is None and cnt < max_cnt:
        logging.info(f"kill proc {self._popen}")
        self._popen.kill()
        time.sleep(1)
        self._popen.poll()
        cnt += 1

  def run(self):
    waiting_sec, max_waiting_sec = 0, 3600
    if self.proc_type == ProcessType.ENTRY:
      # waiting for PS status change
      time.sleep(self._config.update_model_status_interval * 2)
      waiting_sec += self._config.update_model_status_interval * 2

      # check at least one replica of all PSs are stared
      while not self._replica_mgr.is_ps_set_started(
      ) and waiting_sec < max_waiting_sec:
        time.sleep(self._config.update_model_status_interval * 2)
        waiting_sec += self._config.update_model_status_interval * 2

      # check at least one replica of Dense are stared
      if self._config.dense_alone:
        while not self._replica_mgr.is_dense_set_started(
        ) and waiting_sec < max_waiting_sec:
          time.sleep(self._config.update_model_status_interval * 2)
          waiting_sec += self._config.update_model_status_interval * 2

    if waiting_sec >= max_waiting_sec:
      if proc_type == ProcessType.PS:
        logging.error("found PS timeout")
      if proc_type == ProcessType.DENSE:
        logging.error("found Dense timeout")
      return False

    # strat self
    with ServingLog(self.proc_type.name.lower(), self._tfs_log) as log_stdout:
      # Popen will return
      self._popen = subprocess.Popen(self._cmd.split(),
                                     shell=self._shell,
                                     stderr=self._stderr,
                                     stdout=log_stdout,
                                     env=self._env)
      logging.info(f'pid of <{self._cmd}> is {self._popen.pid}')

    if not self.wait_for_started():
      logging.error(f"start {self.proc_type} failed")
      return False

      # start subprocess
    for proc in self._sub_procs.values():
      if not proc.run():
        logging.error(f"start {proc} failed")
        return False

    return True

  def failover(self):
    self._is_failover = True
    if not self.is_tce_main and (self.proc_type == ProcessType.PS or
                                 self.proc_type == ProcessType.DENSE):
      logging.info(f"failover {self.proc_type}, run")
      self.run()
    else:
      logging.info(f"failover {self.proc_type}, kill")
      self.kill()
    self._is_failover = False

  def wait_for_started(self):
    if self._port == 0:
      return True
    waiting_sec, max_waiting_sec = 0, 3600
    while waiting_sec <= max_waiting_sec:
      ret = check_port_open(self._port)
      if ret:
        logging.info(f"proc {self.proc_type} opened!")
        return True
      logging.info(f"proc {self.proc_type} not open!")
      time.sleep(10)
      waiting_sec += 10
    logging.info(f"proc {self.proc_type} start failed!")
    return False


def get_proc(node: ProcessNode, res: List[ProcessNode]):
  res.append(node)
  for proc in node.sub_procs.values():
    if proc is not None:
      get_proc(proc, res)


class ProcessMgr(object):
  _is_killed = False
  _lock = threading.RLock()

  def __init__(self):
    self.pid = os.getpid()
    self._sub_procs: List[ProcessNode] = []

    signal.signal(signal.SIGTERM, self.signal_handler)
    signal.signal(signal.SIGINT, self.signal_handler)

    self._thread_stopped = False
    self._thread = threading.Thread(target=self._poll)
    self._pool = ThreadPoolExecutor(max_workers=2)

  def add_subproc(self, proc: ProcessNode):
    self._sub_procs.append(proc)

  def signal_handler(self, signum, frame):

    def target():
      with self._lock:
        if not ProcessMgr._is_killed:
          self._thread_stopped = True
          ProcessMgr._is_killed = True
          if signum in {signal.SIGINT, signal.SIGTERM}:
            logging.info(f"catch signal {signum}, kill all")
            self.kill_all()
            return True
          else:
            raise RuntimeError(f"unknown signal {signum} at {frame}")
        else:
          return True

    future = self._pool.submit(target)
    future.result()

  def _poll(self):
    procs = []
    for proc in self._sub_procs:
      get_proc(proc, procs)

    logging.info(f"the number of procs is {len(procs)} ")
    while not self._thread_stopped and not ProcessMgr._is_killed:
      time.sleep(1)
      for proc in procs:
        proc.poll()
        if ProcessMgr._is_killed or self._thread_stopped:
          break
        if proc.returncode is not None:
          logging.info(
              f"{proc.proc_type} {proc.returncode} shutdown, kill all proc...")
          #proc.failover()
          #先简化管理, 有进程挂掉的话就整体挂了
          self.kill_all()

  def start(self):
    for proc in self._sub_procs:
      if not proc.run():
        logging.error(f"start {proc} failed, kill all proc")
        self.kill_all()

    logging.info('start poll thread')
    self._thread.start()

  def kill_all(self, include_self=True):
    with self._lock:
      ProcessMgr._is_killed = True
      for proc in self._sub_procs:
        logging.info(f"kill proc {proc.proc_type}")
        # [todo] add graceful shutdown later
        proc.kill()
      if include_self:
        logging.info("kill self")
        os.kill(os.getpid(), signal.SIGKILL)


class AgentV1(AgentBase):

  def __init__(self,
               config: AgentConfig,
               conf_path: str,
               tfs_log: str,
               tfs_binary: str = TFS_BINARY):
    super(AgentV1, self).__init__(config)
    self.zk = MonolithKazooClient(hosts=config.zk_servers)
    self.replica_mgr = ReplicaManager(self.zk, config)
    self.agent_service = AgentService(self.replica_mgr.watcher,
                                      port=config.agent_port)

    pm = ProcessMgr()
    if config.deploy_type == DeployType.MIXED:
      ps_proc = ProcessNode(config,
                            self.replica_mgr,
                            proc_type=ProcessType.PS,
                            conf_path=conf_path,
                            tfs_log=tfs_log,
                            tfs_binary=tfs_binary)
      pm.add_subproc(ps_proc)

      if config.dense_alone:
        dense_proc = ProcessNode(config,
                                 self.replica_mgr,
                                 proc_type=ProcessType.DENSE,
                                 conf_path=conf_path,
                                 tfs_log=tfs_log,
                                 tfs_binary=tfs_binary)
        pm.add_subproc(dense_proc)

      entry_proc = ProcessNode(config,
                               self.replica_mgr,
                               proc_type=ProcessType.ENTRY,
                               is_tce_main=True,
                               conf_path=conf_path,
                               tfs_log=tfs_log,
                               tfs_binary=tfs_binary)
      pm.add_subproc(entry_proc)
    elif config.deploy_type == DeployType.ENTRY:
      proxy_proc = ProcessNode(config,
                               self.replica_mgr,
                               proc_type=ProcessType.ENTRY,
                               is_tce_main=True,
                               conf_path=conf_path,
                               tfs_log=tfs_log,
                               tfs_binary=tfs_binary)
      pm.add_subproc(proxy_proc)
    elif config.deploy_type == DeployType.PS:
      ps_proc = ProcessNode(config,
                            self.replica_mgr,
                            proc_type=ProcessType.PS,
                            is_tce_main=True,
                            conf_path=conf_path,
                            tfs_log=tfs_log,
                            tfs_binary=tfs_binary)
      pm.add_subproc(ps_proc)
    else:
      dense_proc = ProcessNode(config,
                               self.replica_mgr,
                               proc_type=ProcessType.DENSE,
                               is_tce_main=True,
                               conf_path=conf_path,
                               tfs_log=tfs_log,
                               tfs_binary=tfs_binary)
      pm.add_subproc(dense_proc)

    self.process_mgr = pm

  def start(self):
    self.zk.start()
    logging.info(f'start kazoo finished!')
    self.replica_mgr.start()
    logging.info(f'start replica_mgr finished!')
    self.agent_service.start()
    logging.info(f'start agent service at localhost:{self.config.agent_port}')
    self.process_mgr.start()
    logging.info(f'start ProcessMgr finished!')

  def wait_for_termination(self):
    self.agent_service.wait_for_termination()

  def stop(self):
    self.process_mgr.kill_all(include_self=False)
    logging.info(f'close ProcessMgr finished!')
    self.agent_service.stop()
    logging.info(f'close agent service at localhost:{self.config.agent_port}')
    self.replica_mgr.stop()
    logging.info(f'close replica_mgr finished!')
    self.zk.stop()
    logging.info(f'close kazoo finished!')
