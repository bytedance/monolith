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
import socket
import unittest
import copy
import subprocess
from absl import flags, logging
from typing import Dict, List

import tensorflow as tf
from tensorflow.python.framework import test_util

from monolith.native_training.runner_utils import RunnerConfig
from monolith.native_training.estimator import Estimator, import_saved_model, RunConfig
from monolith.native_training.utils import get_test_tmp_dir
from monolith.native_training.tasks.sparse_dense_gpu.model_test import gen_input_file, MultiHeadModel

FLAGS = flags.FLAGS


#copy from monolith/native_training/cpu_training_test.py
class DistributedTrainTest(tf.test.TestCase):
  _DISTRIBUTED_TRAIN_BINARY = "monolith/native_training/tasks/sparse_dense_gpu/model"

  @classmethod
  def setUpClass(cls) -> None:
    FLAGS.dataset_input_patterns = f"/tmp/estimator_mode_test_eb.pb"
    gen_input_file(FLAGS.dataset_input_patterns)

    # maybe file is not enough to train
    def link_some_file(suffix):
      if not os.path.exists(FLAGS.dataset_input_patterns + suffix):
        os.symlink(FLAGS.dataset_input_patterns,
                   FLAGS.dataset_input_patterns + suffix)

    for i in range(10):
      link_some_file(str(i))
    FLAGS.dataset_input_patterns += "{INT(0,99)}"

  def find_free_port(self, count):
    port_list = []
    while len(port_list) < count:
      sock = socket.socket()
      sock.bind(('', 0))
      port = sock.getsockname()[1]
      if port not in port_list:
        port_list.append(port)
    return port_list

  def _run_test(self,
                task_name: str,
                other_args: List,
                num_ps: int,
                num_workers: int,
                num_dsworkers: int,
                other_env: Dict = {},
                worker_args: List = [],
                use_mpi_run=False):
    cur_modir = "{}/{}/ckpt".format(get_test_tmp_dir(), task_name)
    os.makedirs(cur_modir)
    logging.info(f"show cur_modir: {cur_modir}")
    args_tmpl = [
        self._DISTRIBUTED_TRAIN_BINARY,
        f"--model_dir={cur_modir}",
        f"--num_ps={num_ps}",
        f"--num_workers={num_workers}",
        f"--uuid={self._testMethodName}",
        f"--dataset_input_patterns={FLAGS.dataset_input_patterns}",
        f"--dataset_input_use_snappy=False",
        "--lagrangex_header=True",
        "--sort_id=False",
        "--kafka_dump=False",
        "--kafka_dump_prefix=False",
        "--data_type=ExampleBatch",
        "--discovery_type=mlp",
        "--operation_timeout_in_ms=10000",
        "--disable_native_metrics=True",
    ] + other_args
    my_env = os.environ.copy()
    my_env.update(other_env)

    def fill_host_env(role_name, num_role, cur_port_list):
      if num_role <= 0:
        return
      all_host = []
      all_addr = []
      role_name = role_name.upper()
      for i in range(num_role):
        all_host.append("localhost")
        cur_port = cur_port_list[i]
        all_addr.append(f"localhost:{cur_port}")
        my_env[f"MLP_{role_name}_{i}_PORT"] = f"{cur_port}"
        my_env[f"MLP_{role_name}_{i}_HOST"] = f"localhost"
        my_env[f"MLP_{role_name}_{i}_PRIMARY_HOST"] = f"localhost"
      my_env[f"MLP_{role_name}_NUM"] = f"{num_role}"
      my_env[f"MLP_{role_name}_ALL_HOSTS"] = f"{','.join(all_host)}"
      my_env[f"MLP_{role_name}_ALL_PRIMARY_HOSTS"] = my_env[
          f"MLP_{role_name}_ALL_HOSTS"]
      my_env[f"MLP_{role_name}_ALL_ADDRS"] = f"{','.join(all_addr)}"
      my_env[f"MLP_{role_name}_ALL_PRIMARY_ADDRS"] = my_env[
          f"MLP_{role_name}_ALL_ADDRS"]

    #data_service_dispachter
    num_dispatcher = 0
    if num_dsworkers:
      num_dispatcher = 1
    all_port = self.find_free_port(num_ps + num_workers + num_dsworkers +
                                   num_dispatcher)
    ps_port = all_port[:num_ps]
    worker_port = all_port[num_ps:num_ps + num_workers]
    dsworker_port = all_port[num_ps + num_workers:-num_dispatcher]
    dispatcher_port = all_port[-num_dispatcher:]

    fill_host_env('ps', num_ps, ps_port)
    fill_host_env('worker', num_workers, worker_port)
    fill_host_env('dispatcher', num_dispatcher, dispatcher_port)
    fill_host_env('dsworker', num_dsworkers, dsworker_port)

    processes = {}
    log_files = []

    def start_process(role_name, num_role, cur_port_list, use_mpi_run=False):
      if use_mpi_run:
        hostfile = f"{cur_modir}/../hostfile"
        f = open(hostfile, "w")
        f.write(f"localhost slots={num_role}")
        f.close()

        args = copy.copy(args_tmpl)
        args.append(f"--server_type={role_name}")
        if role_name == "worker":
          args += worker_args
        cur_env = copy.deepcopy(my_env)
        cur_env["MLP_ROLE"] = role_name
        cur_env["MLP_PORT"] = f"{cur_port_list[0]}"
        cur_env["MLP_SSH_PORT"] = f"{worker_port[0]}"
        cur_env["MONOLITH_WITH_HOROVOD"] = f"1"
        cur_env["MONOLITH_WITH_HOROVOD_FID_G2G"] = f"1"
        cur_env["MONOLITH_WITH_ALLREDUCE_FUSION"] = f"one"
        #cur_env["MONOLITH_GPU_FEATURE_FACTORY_FUSION_LEVEL"] = f"1"
        #cur_env["HOROVOD_MPI_THREADS_DISABLE"] = f"1"
        #cur_env["GPU_AFFINITY_NIC_ADDRESS"] = f"1"
        #cur_env["NCCL_SOCKET_IFNAME"] = f"eth0"
        #cur_env["NCCL_P2P_LEVEL"] = f"1"
        mpi_run_args = [
            "mpirun",
            "--map-by",
            f"ppr:{num_role}:node",
            "-np",
            f"{num_role}",
            "--hostfile",
            hostfile,
            "--allow-run-as-root",
            "-oversubscribe",
            "--tag-output",
            "--report-bindings",
            #"--mca", "btl_tcp_if_include", "eth0", "--mca", "oob_tcp_if_include", "eth0"
        ]

        for k, v in cur_env.items():
          mpi_run_args.append("-x")
          mpi_run_args.append(f"{k}={v}")
        args = mpi_run_args + args
        process = subprocess.Popen(args)
        logging.info(f"start a process for {role_name}:{range(num_role)}")
        processes[f"{role_name}:{0}"] = process
        for i in range(1, num_role):
          processes[f"{role_name}:{i}"] = None
      else:
        for i in reversed(range(num_role)):
          log_file = open(cur_modir + f"/../{role_name}_{i}.log", 'w')
          log_files.append(log_file)
          args = copy.copy(args_tmpl)
          args.append(f"--server_type={role_name}")
          args.append("--index={}".format(i))
          cur_env = copy.deepcopy(my_env)
          cur_env["MLP_ROLE"] = role_name
          cur_env["MLP_ROLE_INDEX"] = f"{i}"
          cur_env["MLP_PORT"] = f"{cur_port_list[i]}"
          cur_env["MLP_SSH_PORT"] = f"{worker_port[0]}"
          #if i == 0 and role_name == "worker":
          #  time.sleep(5)
          process = subprocess.Popen(args, env=cur_env)
          logging.info(f"start a process for {role_name}:{i}")
          processes[f"{role_name}:{i}"] = process

    start_process('dispatcher', num_dispatcher, dispatcher_port)
    start_process('dsworker', num_dsworkers, dsworker_port)
    start_process('ps', num_ps, ps_port)
    start_process('worker', num_workers, worker_port, use_mpi_run=use_mpi_run)

    print(" ".join(args_tmpl), num_ps, num_workers, num_dsworkers)

    def wait_for_process(role_name, num_role, timeout=10, ignore_timeout=False):
      for i in range(num_role):
        role = f"{role_name}:{i}"
        if role not in processes:
          continue
        process = processes[role]
        if process is None:
          continue
        if not ignore_timeout:
          self.assertEqual(process.wait(timeout=timeout), 0)
        else:
          try:
            self.assertEqual(process.wait(timeout=timeout), 0)
          except subprocess.TimeoutExpired as e:
            logging.warning(f"exit process for {role} timeout")
            process.terminate()
        processes.pop(role)
        logging.info(f"exit process for {role}")

    wait_for_process('worker', 1, 250)
    wait_for_process('worker', num_workers, timeout=10, ignore_timeout=True)
    wait_for_process('ps', num_ps, timeout=1)
    wait_for_process('dsworker', num_dsworkers, timeout=1,
                     ignore_timeout=True)  #maybe chief port not free
    wait_for_process('dispatcher',
                     num_dispatcher,
                     timeout=1,
                     ignore_timeout=True)  #maybe chief port not free

    for log_file in log_files:
      log_file.flush()
      log_file.close()
    tf.io.gfile.rmtree(cur_modir)

  def run_cpu(self, name, other_args):
    # TODO cpu mode run gpu have error
    if test_util.is_gpu_available(cuda_only=True):
      return
    args = [
        "--enable_gpu_training=False",
        "--enable_sync_training=False",
        "--embedding_prefetch_capacity=1",
        "--enable_embedding_postpush=True",
        "--chief_timeout_secs=20",
    ] + other_args
    num_ps = 2
    num_workers = 2
    num_dsworkers = 0
    self._run_test(f"full_cpu_{name}", args, num_ps, num_workers, num_dsworkers)

  def test_cpu0(self):
    args = [
        "--enable_fused_layout=False",
        "--use_native_multi_hash_table=False",
    ]
    self.run_cpu('0', args)

  def test_cpu1(self):
    args = [
        "--enable_fused_layout=False",
        "--use_native_multi_hash_table=True",
    ]
    self.run_cpu('1', args)

  def test_cpu2(self):
    args = [
        "--enable_fused_layout=True",
        "--use_native_multi_hash_table=True",
    ]
    self.run_cpu('2', args)

  def test_cpu3(self):
    args = [
        "--enable_fused_layout=True",
        "--use_native_multi_hash_table=False",
    ]
    self.run_cpu('3', args)

  def sparse_dense_run(self, name, other_args):
    if not test_util.is_gpu_available(cuda_only=True):
      return
    gpus = tf.config.list_physical_devices('GPU')
    args = [
        "--enable_gpu_training=True",
        "--enable_sync_training=True",
        "--enable_partial_sync_training=True",
        "--embedding_prefetch_capacity=1",
        "--enable_embedding_postpush=True",
        '--params_override={"train.max_steps": 10}',
    ] + other_args
    worker_args = []
    num_ps = 2
    num_workers = min(2, len(gpus))
    num_dsworkers = 1
    other_env = {}
    self._run_test(f"sparse_dense_{name}",
                   args,
                   num_ps,
                   num_workers,
                   num_dsworkers,
                   other_env=other_env,
                   worker_args=worker_args,
                   use_mpi_run=True)

  def test_sparse_dense0(self):
    args = [
        "--enable_fused_layout=True",
        "--use_native_multi_hash_table=False",
    ]
    self.sparse_dense_run('0', args)

  def test_sparse_dense1(self):
    args = [
        "--enable_fused_layout=True",
        "--use_native_multi_hash_table=True",
    ]
    self.sparse_dense_run('1', args)

  def test_sparse_dense2(self):
    args = [
        "--enable_fused_layout=False",
        "--use_native_multi_hash_table=False",
    ]
    self.sparse_dense_run('2', args)

  def test_sparse_dense3(self):
    args = [
        "--enable_fused_layout=False",
        "--use_native_multi_hash_table=True",
    ]
    self.sparse_dense_run('3', args)

  def full_gpu_run(self, name, other_args):
    if not test_util.is_gpu_available(cuda_only=True):
      return
    gpus = tf.config.list_physical_devices('GPU')
    args = [
        "--enable_gpu_training=True",
        "--enable_sync_training=True",
        "--reorder_fids_in_data_pipeline=True",
        "--filter_type=probabilistic_filter",
        "--embedding_prefetch_capacity=1",
        "--enable_async_optimize=False",
        '--params_override={"train.max_steps": 10}',
    ] + other_args
    worker_args = []
    num_ps = 0
    num_workers = min(2, len(gpus))
    num_dsworkers = 1
    other_env = {}
    self._run_test(f"full_gpu_{name}",
                   args,
                   num_ps,
                   num_workers,
                   num_dsworkers,
                   other_env=other_env,
                   worker_args=worker_args,
                   use_mpi_run=True)

  def test_full_gpu_0(self):
    args = [
        "--enable_fused_layout=True",
        "--use_native_multi_hash_table=False",
    ]
    self.full_gpu_run('0', args)

  def test_full_gpu_1(self):
    args = [
        "--enable_fused_layout=True",
        "--use_native_multi_hash_table=True",
    ]
    self.full_gpu_run('1', args)

  def test_full_gpu_2(self):
    args = [
        "--enable_fused_layout=False",
        "--use_native_multi_hash_table=False",
    ]
    self.full_gpu_run('2', args)

  def test_full_gpu_3(self):
    args = [
        "--enable_fused_layout=False",
        "--use_native_multi_hash_table=True",
    ]
    self.full_gpu_run('3', args)


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
