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
import sys
import time
import socket
from typing import List
from absl import logging, flags

import tensorflow as tf
import tensorflow.python.data.experimental.service as dsvc
from tensorflow_estimator.python.estimator.util import _DatasetInitializerHook


FLAGS = flags.FLAGS


def check_port(host: str, port: int, timeout: float = 1) -> bool:
  is_ipv6 = ':' in host.strip('[]')
  skt = socket.socket(socket.AF_INET6 if is_ipv6 else socket.AF_INET,
                      socket.SOCK_STREAM)
  skt.settimeout(timeout)
  try:
    skt.connect((host, int(port)))
    return True
  except socket.error as e:
    return False


def enable_sync_training():
  enable_hvd = eval(os.getenv("MONOLITH_WITH_HOROVOD", "0").strip('"').strip("'"))
  enable_bps = eval(os.getenv("MONOLITH_WITH_BYTEPS", "0").strip('"').strip("'"))
  enable_fid_g2g = os.environ.get('MONOLITH_WITH_HOROVOD_FID_G2G')
  try:
    return FLAGS.enable_sync_training or enable_hvd or enable_bps or enable_fid_g2g
  except:
    return enable_hvd or enable_bps or enable_fid_g2g


class MLPEnv(object):
  def __init__(self):
    self._mlp_env = {k: v for k, v in os.environ.items() if k.startswith('MLP_')}
    self.framework = self._get('MLP_FRAMEWORK')
    self.ssh_port = self._get('MLP_SSH_PORT')
    self.log_path = self._get('MLP_LOG_PATH')
    self.debug_port = self._get('MLP_DEBUG_PORT')
    self.entrypoint_dir = self._get('MLP_ENTRYPOINT_DIR')
    self.task_cmd = self._get('MLP_TASK_CMD')
    self.role = self._get('MLP_ROLE', "").upper()
    self.index = int(self._get('MLP_ROLE_INDEX', 0))
    self.all_roles = {k.split('_')[1]: int(self._get(k, 0)) for k in self._mlp_env
                      if k.endswith('_NUM') and len(k.split('_')) == 3}

    if len(self._mlp_env) > 0 and len(self.all_roles) > 0:
      self.avaiable = True
    else:
      self.avaiable = False

    self.cpu = int(self._get('MLP_CPU', 0))
    self.gpu = int(self._get('MLP_GPU', 0))
    self.gpu_type = self._get('MLP_GPU_TYPE', "")
    self.mem = int(self._get('MLP_MEM', 0))

    self.host = self._get('MLP_HOST')
    self.port = int(self._get('MLP_PORT', 0))
    self._has_started_profiler = False

  def _get(self, name: str, default=None):
    value = self._mlp_env.get(name)
    if value:
      return value.strip().strip('"').strip("'")
    else:
      return default

  def num_replicas(self, role: str = None):
    role = (role or self.role).upper()
    key = f'MLP_{role}_NUM'
    logging.info(f"{key}, mlp_env: {self._mlp_env}")
    return int(self._get(key, 0))

  def get_all_host(self, role: str = None, is_primary: bool = True) -> List[str]:
    role = (role or self.role).upper()
    if is_primary:
      key = f'MLP_{role}_ALL_PRIMARY_HOSTS'
    else:
      key = f'MLP_{role}_ALL_HOSTS'
    return self._get(key)

  def get_all_addrs(self, role: str = None, is_primary: bool = True) -> List[str]:
    role = (role or self.role).upper()
    if is_primary:
      key = f'MLP_{role}_ALL_PRIMARY_ADDRS'
    else:
      key = f'MLP_{role}_ALL_ADDRS'
    addrs = self._get(key)
    if addrs:
      return addrs.split(',')
    else:
      return []

  def get_host(self, role: str = None, index: int = None, is_primary: bool = True) -> str:
    role = (role or self.role).upper()
    if role == self.role:
      index = self.index if index is None else index
    else:
      index = 0 if index is None else index
    if is_primary:
      key = f'MLP_{role}_{index}_PRIMARY_HOST'
    else:
      key = f'MLP_{role}_{index}_HOST'

    return self._get(key)

  def get_addr(self, role: str = None, index: int = None, is_primary: bool = True) -> str:
    role = (role or self.role).upper()
    if role == self.role:
      index = self.index if index is None else index
    else:
      index = 0 if index is None else index
    host = self.get_host(role, index, is_primary)
    key = f'MLP_{role}_{index}_PORT'
    port = self._get(key)
    if host and port:
      return f'{host}:{port}'
    else:
      return None

  def dispatcher_target(self, role: str = None) -> str:
    addr = self.dispatcher_addr(role)
    return f'grpc://{addr}'

  def dispatcher_addr(self, role: str = None) -> str:
    role = (role or 'dispatcher').upper()
    return self.get_addr(role)

  def wait(self, role: str = None, index: int = 0, timeout: int = -1, use_ssh: bool = True):
    host = self.get_host(role, index, True)
    port = self.ssh_port if use_ssh else self.port
    if host:
      current = 0
      while True:
        if timeout > 0 and current >= timeout:
          logging.info(f'wait {host}:{port} timeout!')
          break
        if check_port(host, port):
          return
        else:
          time.sleep(5)
          current += 5
    else:
      logging.info('host is None')

  def join(self, role: str = 'worker', index: int = 0, use_ssh: bool = True):
    self.wait(role, index, use_ssh=use_ssh)
    host = self.get_host(role, index, True)
    port = self.ssh_port if use_ssh else self.port
    if host:
      while True:
        if not check_port(host, port, timeout=60):
          return
        else:
          time.sleep(10)
    else:
      logging.info('host is None')
    if self._has_started_profiler:
      tf.profiler.experimental.stop()
      logging.info('profiler stopped!')
    os._exit(0)

  @property
  def queue_device(self) -> str:
    if 'PS' in self.all_roles:
      return "/job:ps/task:0/device:CPU:0"
    elif 'WORKER' in self.all_roles:
      return "/job:worker/task:0/device:CPU:0"
    else:
      return "/device:CPU:0"

  def start_profiler(self, port=6666):
    logging.info(f'start_profiler at {self.host}:{port}')
    tf.profiler.experimental.server.start(port)
    self._has_started_profiler = True

  def profiler_trace(self,
                     role: str = 'dsworker',
                     index: int = -1,
                     host_tracer_level=2,
                     python_tracer_level=0,
                     device_tracer_level=1,
                     delay_ms=10000):
    logdir = self._get("TENSORBOARD_LOG_PATH", "/tensorboard_logs/")
    options = tf.profiler.experimental.ProfilerOptions(
      host_tracer_level=host_tracer_level,
      python_tracer_level=python_tracer_level,
      device_tracer_level=device_tracer_level,
      delay_ms=delay_ms)
    if index < 0:
      all_addrs = self.get_all_addrs(role)
      service_addr=','.join(map(lambda addr: f'grpc://{addr}', all_addrs))
    else:
      service_addr=f'grpc://{self.get_addr(role, index)}'
    tf.profiler.experimental.client.trace(
      service_addr=service_addr,
      logdir=logdir,
      duration_ms=delay_ms,
      options=options
    )


def mlp_pass(dispatcher_role: str = 'dispatcher',
             dsworker_role: str = 'dsworker',
             worker_role: str = 'worker'):
  dispatcher_role = None if dispatcher_role is None else dispatcher_role.upper()
  dsworker_role = None if dsworker_role is None else dsworker_role.upper()
  worker_role = None if worker_role is None else worker_role.upper()

  mlp_env = MLPEnv()
  if mlp_env.avaiable:
    logging.info('MLP is available')
    if mlp_env.role == dispatcher_role:
      if dispatcher_role:
        dispatcher = dsvc.DispatchServer(
            dsvc.DispatcherConfig(port=mlp_env.port))
        logging.info('Dispatcher started...')
        mlp_env.join()
    elif mlp_env.role == dsworker_role:
      if dsworker_role:
        logging.info('Waiting for dispatcher start...')
        assert dispatcher_role is not None
        mlp_env.wait(dispatcher_role, use_ssh=False)
        logging.info('Dispatcher started, dsworker begin to start...')
        dispatcher_address = mlp_env.dispatcher_addr(role=dispatcher_role)
        worker = dsvc.WorkerServer(
          dsvc.WorkerConfig(dispatcher_address=dispatcher_address,
                            worker_address=f'{mlp_env.host}:{mlp_env.port}',
                            port=mlp_env.port))
        logging.info('Dsworker started....')
        mlp_env.start_profiler()
        mlp_env.join()
    elif mlp_env.role == worker_role:
      if dispatcher_role:
        logging.info("wait dispatcher start ...")
        mlp_env.wait(dispatcher_role, use_ssh=False)
      if dsworker_role:
        logging.info("dispatcher started, wait ds worker start ...")
        for idx in range(mlp_env.num_replicas(role=dsworker_role)):
          mlp_env.wait(dsworker_role, index=idx, use_ssh=False)
          logging.info(f'dsworker {idx} started! ')
      logging.info(f"worker {mlp_env.index} start at {mlp_env.host}:{mlp_env.port}")
      mlp_env.start_profiler()


def begin(self):
  self._initializer = self._iterator.initializer
  self._broadcast_dataset_id = None
  self._rank = -1
  graph = tf.compat.v1.get_default_graph()
  if enable_sync_training() and not hasattr(graph, 'dry_run'):
    try:
      enable_bps = int(os.getenv("MONOLITH_WITH_BYTEPS", "0"))
      if enable_bps:
        import byteps.tensorflow as hvd
      else:
        import horovod.tensorflow as hvd

      dataset_ids = tf.compat.v1.get_collection(key='registed_dataset_id')
      if dataset_ids is not None and len(dataset_ids) > 0:
        dataset_id = dataset_ids[0]
        if dataset_id is not None:
          self._rank = hvd.rank()
          self._broadcast_dataset_id = [dataset_id, 
                                        hvd.broadcast(tensor=dataset_id,
                                                      root_rank=0,
                                                      name="broadcast_dataset_id")]
          graph.clear_collection(name='registed_dataset_id')
    except (ImportError,  tf.errors.NotFoundError) as e:
      logging.info(f'import byteps/horovod error: {e}')


def after_create_session(self, session, coord):
  del coord
  if self._broadcast_dataset_id is not None:
    dataset_id, bc_dataset_id = session.run(self._broadcast_dataset_id)
    logging.info(f'dataset_id is {dataset_id}, bc_dataset_id is {bc_dataset_id}, rank {self._rank}')
    self._broadcast_dataset_id = None
  session.run(self._initializer)


_DatasetInitializerHook.begin = begin
_DatasetInitializerHook.after_create_session = after_create_session

