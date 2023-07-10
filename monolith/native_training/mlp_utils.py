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
import signal
from subprocess import Popen, PIPE

import tensorflow as tf
import tensorflow.python.data.experimental.service as dsvc
from tensorflow_estimator.python.estimator.util import _DatasetInitializerHook
from monolith.native_training.distribution_utils import get_mpi_rank, \
  get_mpi_size, get_mpi_local_size, enable_sync_training, get_device_str

FLAGS = flags.FLAGS


from monolith.native_training import yarn_runtime


def kill_by_port(port: int):
  process = Popen(["lsof", "-i", ":{0}".format(port)], stdout=PIPE, stderr=PIPE)
  stdout, stderr = process.communicate()
  try:
    pid = None
    for process in str(stdout.decode("utf-8")).split("\n")[1:]:       
      data = [x for x in process.split(" ") if x]
      if data and len(data) > 1:
        pid = int(data[1])
        break
    print('pid is', pid)
  except:
    pass
  if pid is not None:
    os.kill(pid, signal.SIGKILL)


def check_port(host: str, port: int, timeout: float = 1) -> bool:
  is_ipv6 = ':' in host.strip('[]')
  skt = socket.socket(socket.AF_INET6 if is_ipv6 else socket.AF_INET,
                      socket.SOCK_STREAM)
  start = time.time()
  skt.settimeout(timeout)
  while True:
    try:
      skt.connect((host, int(port)))
      return True
    except socket.timeout as e:
      return False
    except socket.error as e:
      now = time.time()
      remaining = timeout - int(now - start)
      if remaining > 0:
        skt.settimeout(remaining)
        continue
      else:
        return False


class MLPEnv(object):
  def __init__(self):
    self._mlp_env = {k: v for k, v in os.environ.items()
                     if k.startswith('MLP_') or k.startswith('MPI_')}
    self.framework = self._get('MLP_FRAMEWORK')
    self.ssh_port = self._get('MLP_SSH_PORT')
    self.log_path = self._get('MLP_LOG_PATH')
    self.debug_port = self._get('MLP_DEBUG_PORT')
    self.entrypoint_dir = self._get('MLP_ENTRYPOINT_DIR')
    self.task_cmd = self._get('MLP_TASK_CMD')
    self.role = self._get('MLP_ROLE', "").upper()
    self.all_roles = {k.split('_')[1]: int(self._get(k, 0)) for k in self._mlp_env
                      if k.endswith('_NUM') and len(k.split('_')) == 3}
    if self.enable_mpi:
      self.index = get_mpi_rank()
      self.all_roles['WORKER'] = get_mpi_size()
      self.port = int(self._get('MLP_PORT', 0)) + self.index
      logging.info(f'total process is {get_mpi_size()}, this is {get_mpi_rank()}, port is {self.port}')
    else:
      self.index = int(self._get('MLP_ROLE_INDEX', 0))
      self.port = int(self._get('MLP_PORT', 0))
      logging.info(f'enable_mpi is False, index {self.index}, port {self.port}')

    if len(self._mlp_env) > 0 and len(self.all_roles) > 0:
      self.avaiable = True
    else:
      self.avaiable = False

    self.cpu = int(self._get('MLP_CPU', 0))
    self.gpu = int(self._get('MLP_GPU', 0))
    self.gpu_type = self._get('MLP_GPU_TYPE', "")
    self.mem = int(self._get('MLP_MEM', 0))

    self.host = yarn_runtime.get_local_host()#self._get('MLP_HOST')

    self._has_started_profiler = False

  @property
  def enable_mpi(self):
    return 'OMPI_COMM_WORLD_RANK' in os.environ and self.role == "WORKER"

  def _get(self, name: str, default=None):
    value = self._mlp_env.get(name)
    if value:
      return value.strip().strip('"').strip("'")
    else:
      return default

  def num_replicas(self, role: str = None):
    role = (role or self.role).upper()
    if self.enable_mpi and role == 'WORKER':
      return get_mpi_size()
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
    if self.enable_mpi and role == 'WORKER':
      index = (self.index if index is None else index) // get_mpi_local_size()
    elif role == self.role:
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

    if self.enable_mpi and role == 'WORKER':
      key = f'MLP_{role}_0_PORT'
      port = self._get(key)
      if port is not None:
        port = str(int(port) + index)
    else:
      key = f'MLP_{role}_{index}_PORT'
      port = self._get(key)
    if host and port:
      return f'{host}:{port}'
    else:
      return None

  def get_port(self, role: str = None, index: int = None) -> int:
    role = (role or self.role).upper()
    if self.enable_mpi and role == 'WORKER':
      index = self.index if index is None else index
      key = f'MLP_{role}_0_PORT'
      return self._get(key, 2222) + index
    else:
      index = 0 if index is None else index
      key = f'MLP_{role}_{index}_PORT'
      return self._get(key, 2222)

  def dispatcher_target(self, role: str = None) -> str:
    addr = self.dispatcher_addr(role)
    if addr:
      return f'grpc://{addr}'
    else:
      return 'grpc://localhost:5050'

  def dispatcher_addr(self, role: str = None) -> str:
    role = (role or 'dispatcher').upper()
    return self.get_addr(role)

  def wait(self, role: str = None, index: int = 0, timeout: int = -1, use_ssh: bool = True):
    host = self.get_host(role, index, True)
    port = self.ssh_port if use_ssh else self.get_port(role, index)
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
    port = self.ssh_port if use_ssh else self.get_port(role, index)
    if host:
      while True:
        if not check_port(host, port, timeout=60):
          break #return
        else:
          time.sleep(10)
    else:
      logging.info('host is None')
    if self._has_started_profiler:
      try:
        tf.profiler.experimental.stop()
      except Exception as e:
        logging.info(f'experimental stop error: {e}')
      logging.info('profiler stopped!')
    logging.info(f'current role: {self.role}:{self.index} exit')
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
    if self.enable_mpi:
      port += self.index
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


def add_mpi_exception_hook():
  if 'OMPI_COMM_WORLD_RANK' not in os.environ:
    return
  logging.info("add_mpi_exception_hook")
  # Global error handler
  def global_except_hook(exctype, value, traceback):
    try:
      sys.stderr.write(
          "\n*****************************************************\n")
      sys.stderr.write("Uncaught exception was detected on rank {}. \n".format(
          int(os.environ.get('OMPI_COMM_WORLD_RANK', -1))))
      from traceback import print_exception
      print_exception(exctype, value, traceback)
      sys.stderr.write(
          "*****************************************************\n\n\n")
      sys.stderr.write("\n")
      sys.stderr.write("Calling MPI_Abort() to shut down MPI processes...\n")
      sys.stderr.flush()
    finally:
      try:
        import mpi4py.MPI
        mpi4py.MPI.COMM_WORLD.Abort(1)
      except Exception as e:
        sys.stderr.write(
            "*****************************************************\n")
        sys.stderr.write(
            "Sorry, we failed to stop MPI, this process will hang.\n")
        sys.stderr.write(
            "*****************************************************\n")
        sys.stderr.flush()
        raise e
  sys.excepthook = global_except_hook

def mlp_pass(dispatcher_role: str = 'dispatcher',
             dsworker_role: str = 'dsworker',
             worker_role: str = 'worker',
             ps_role: str = 'ps'):
  dispatcher_role = None if dispatcher_role is None else dispatcher_role.upper()
  dsworker_role = None if dsworker_role is None else dsworker_role.upper()
  worker_role = None if worker_role is None else worker_role.upper()
  pa_role = None if ps_role is None else ps_role.upper()


  if FLAGS.dataset_use_dataservice:
    _DatasetInitializerHook.begin = begin
    _DatasetInitializerHook.after_create_session = after_create_session
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
      if FLAGS.dataset_use_dataservice:
        if dispatcher_role:
          logging.info("wait dispatcher start ...")
          mlp_env.wait(dispatcher_role, use_ssh=False)
          FLAGS.data_service_dispatcher = mlp_env.dispatcher_target()
        if dsworker_role:
          logging.info("dispatcher started, wait ds worker start ...")
          for idx in range(mlp_env.num_replicas(role=dsworker_role)):
            mlp_env.wait(dsworker_role, index=idx, use_ssh=False)
            logging.info(f'dsworker {idx} started! ')
      logging.info(f"worker {mlp_env.index} start at {mlp_env.host}:{mlp_env.port}")
      logging.info(f'{mlp_env.all_roles}')
      # if ps_role is None or ps_role not in mlp_env.all_roles:
      #   mlp_env.start_profiler()


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
          #with tf.device(None), tf.device(get_device_str(True)):
          self._broadcast_dataset_id = [dataset_id,
                                        hvd.broadcast(tensor=dataset_id,
                                                      root_rank=0,
                                                      name="broadcast_dataset_id")]
          graph.clear_collection(name='registed_dataset_id')
    except Exception as e:
      logging.info(f'import byteps/horovod error: {e}')


def after_create_session(self, session, coord):
  del coord
  if self._broadcast_dataset_id is not None:
    dataset_id, bc_dataset_id = session.run(self._broadcast_dataset_id)
    logging.info(f'dataset_id is {dataset_id}, bc_dataset_id is {bc_dataset_id}, rank {self._rank}')
    self._broadcast_dataset_id = None
  session.run(self._initializer)
