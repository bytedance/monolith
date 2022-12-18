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
import tensorflow as tf


def bps_init(uuid: str):
  """
  Initialize BytePS.
  Args:
    uuid: uuid of the training job, used to distinguish concurrent BytePS processes across runs.
  """
  # init bps only if needed
  if os.environ.get('BYTEPS_ALLTOALL_SESSION_SIZE') is None:
    os.environ["BYTEPS_ALLTOALL_SESSION_SIZE"] = '3'

  # set size, rank based on OMPI env vars
  if os.environ.get('BYTEPS_LOCAL_SIZE', None) is None:
    os.environ["BYTEPS_LOCAL_SIZE"] = os.environ.get(
        'OMPI_COMM_WORLD_LOCAL_SIZE')
  local_size = int(os.environ.get('BYTEPS_LOCAL_SIZE'))
  rank = int(os.environ.get('OMPI_COMM_WORLD_RANK'))
  size = int(os.environ.get('OMPI_COMM_WORLD_SIZE'))
  local_rank = rank % local_size
  phy_node_id = int(rank / local_size)
  socket_path = f"/tmp/bps_{uuid}_socket_{phy_node_id}"
  gdr_alltoall = os.environ.get('MONOLITH_WITH_BYTEPS_FWD_GDR', '0') == '1' or \
      os.environ.get('MONOLITH_WITH_BYTEPS_BWD_GDR', '0') == '1'

  # gpu_nic_binding_mode: Default False, when True we bind gpu_id (0,1) to eth0, (2,3) to eth1...
  # This is useful for A100 systems where we have topology in which some gpus are closer to some
  # NICs.
  gpu_nic_binding_mode = int(os.environ.get('BYTEPS_GPU_NIC_BINDING_MODE', 0))
  if not gpu_nic_binding_mode:
    # Constant binding mode (default), all GPUs use one NIC
    interface = os.getenv("DMLC_INTERFACE", "eth0")
  else:
    # gpu_nic_binding_mode binding mode
    NUM_GPU_PER_NIC = 2
    nic_id = int(local_rank // NUM_GPU_PER_NIC)
    if gdr_alltoall:
      os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
      numa_id = os.environ['BYTEPS_NUMA_ID']
      print(
          f"GDR: set CUDA_VISIBLE_DEVICES={local_rank}, BYTEPS_NUMA_ID={numa_id}"
      )
      os.environ['BYTEPS_PIN_MEMORY'] = "1"
      os.environ['BYTEPS_PIN_MEMORY_CPU'] = os.environ.get(
          'BYTEPS_PIN_MEMORY_CPU', '1')
      os.environ['DMLC_NUM_CPU_DEV'] = "0"
      os.environ['DMLC_NUM_GPU_DEV'] = "1"
    os.environ['BYTEPS_USE_GDR_ALLREDUCE'] = os.environ.get(
        'BYTEPS_USE_GDR_ALLREDUCE', '1')
    interface = "eth{}".format(nic_id)

    # Add all eth otherwise it may give out "Destination not reachable" error
    # or block in some communication.
    if os.environ.get('BYTEPS_WITH_ALL_NICS', '0') == '1':
      os.environ[
          "UCX_NET_DEVICES"] = "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,eth0,eth1,eth2,eth3"
    else:
      os.environ["UCX_NET_DEVICES"] = "mlx5_{}:1".format(nic_id)

  # scheduler connection info
  cmd = f'ip addr show {interface}'
  hostname = os.popen(
      cmd +
      ' | grep "\<inet\>" | awk \'{ print $2 }\' | awk -F "/" \'{ print $1 }\''
  ).read().strip()
  os.environ["UCX_RDMA_CM_SOURCE_ADDRESS"] = hostname
  os.environ["PSLITE_UCX_TLS"] = os.environ.get('PSLITE_UCX_TLS',
                                                'rc_x,tcp,self,cuda')
  print(
      f"UCX: set PSLITE_UCX_TLS={os.environ['PSLITE_UCX_TLS']} {os.environ['UCX_NET_DEVICES']}"
  )
  os.environ["DMLC_NODE_HOST"] = hostname
  os.environ["DMLC_ROLE"] = 'joint'
  os.environ["DMLC_ENABLE_UCX"] = os.environ.get('DMLC_ENABLE_UCX', '1')

  os.makedirs(socket_path, exist_ok=True)
  os.environ["DMLC_WORKER_ID"] = str(rank)
  os.environ["DMLC_NUM_WORKER"] = str(size)
  os.environ["DMLC_NUM_SERVER"] = str(size)
  os.environ["BYTEPS_UUID"] = uuid
  os.environ["BYTEPS_LOCAL_RANK"] = str(local_rank)
  os.environ["BYTEPS_SOCKET_PATH"] = socket_path
  os.environ["BYTEPS_OMP_THREAD_PER_GPU"] = os.environ.get(
      "BYTEPS_OMP_THREAD_PER_GPU", "1")
  os.environ["BYTEPS_FORCE_DISTRIBUTED"] = '1'
  os.environ["BYTEPS_TELEMETRY_ON"] = os.environ.get("BYTEPS_TELEMETRY_ON", '0')
  os.environ["BYTEPS_LOG_LEVEL"] = os.environ.get('BYTEPS_LOG_LEVEL', 'info')
  os.environ["BYTEPS_SERVER_DIRECT_RESPONSE"] = os.environ.get(
      'BYTEPS_SERVER_DIRECT_RESPONSE', '2')
  os.environ["BYTEPS_UCX_FORCE_REQ_ORDER"] = '1'

  # performance tuning knobs
  os.environ["BYTEPS_KEY_HASH_FN"] = os.environ.get('BYTEPS_KEY_HASH_FN',
                                                    'djb2-colocate')
  os.environ["BYTEPS_UCX_SHORT_THRESH"] = os.environ.get(
      'BYTEPS_UCX_SHORT_THRESH', '0')
  os.environ["PSLITE_UCX_RNDV_THRESH"] = os.environ.get(
      "PSLITE_UCX_RNDV_THRESH", '8192')
  os.environ["BYTEPS_WORKER_LOCAL_ROOT"] = os.environ.get(
      'BYTEPS_WORKER_LOCAL_ROOT', '-1')
  # To enable async alltoall operations, we must reserve memory buffers on the receiver side.
  # BYTEPS_P2P_PARTITION_BYTES sets the receive buffer size for each alltoall operation from each sender.
  # It needs to be large enough such that the actual data sent does not exceed the buffer size, otherwise
  # error message may occur
  if os.environ.get("BYTEPS_P2P_PARTITION_BYTES") is None:
    alltoall_buff_size_per_rank = int(2048000 * 128 * 2 / size)
    os.environ["BYTEPS_P2P_PARTITION_BYTES"] = str(alltoall_buff_size_per_rank)
  if os.environ.get("BYTEPS_PARTITION_BYTES") is None:
    allreduce_partition_size = 1024000 if size < 128 else 512000
    os.environ["BYTEPS_PARTITION_BYTES"] = str(allreduce_partition_size)

  import byteps.tensorflow as bps
  bps.init(lazy=False)


# bps allreduce stress test
def byteps_benchmark_ar(total_len,
                        total_niter=10000,
                        use_cpu=False,
                        op='pushpull'):
  tf.compat.v1.enable_eager_execution()
  import byteps.tensorflow as bps
  import numpy as np
  rank, size = bps.rank(), bps.size()
  niter = 0
  print(
      f'===== start pushpull_benchmark {rank}/{size} total_len={total_len} =====',
      flush=True)
  device = tf.device("/gpu:0" if not use_cpu else "/cpu:0")
  with device:
    tensor = tf.ones([total_len, 1], dtype=tf.float32) * (rank + 1)
  t0 = time.time()
  interval = 20
  name = f'data_len_{total_len}_{op}_' + ('cpu' if use_cpu else 'gpu')
  comm_fn = bps.push_pull
  goodputs = []
  while niter < total_niter:
    with device:
      result = comm_fn(tensor, average=True, name=name)
    niter += 1
    if niter % interval == 0:
      t1 = time.time()
      latency = (t1 - t0) / interval * 1000
      goodput = total_len * 32 / latency / 1000000
      goodputs.append(goodput)
      rank == 0 and print(
          f'DONE iter={niter}, latency={latency:.3} ms, Goodput={goodput:.5} Gb/s, is_cpu={use_cpu}',
          flush=True)
      t0 = time.time()
  print(
      f'===== end pushpull_benchmark {rank}/{size} total_len={total_len} =====',
      flush=True)
  return goodputs[1:]


# bps all2all stress test
def byteps_benchmark_a2a(total_len,
                         total_niter=10000,
                         dst_gpu=True,
                         src_gpu=True):
  tf.compat.v1.enable_eager_execution()
  # the CPU alltoall size is much smaller in real use cases
  if not dst_gpu and not src_gpu:
    total_len /= 8
  import byteps.tensorflow as bps
  import numpy as np
  rank, size = bps.rank(), bps.size()
  niter = 0
  len_per_worker = int(total_len / size)
  assert total_len % size == 0
  p2p_matrix = np.array([len_per_worker] * (size * size)).reshape(size, size)
  splits_list = list(p2p_matrix[rank])
  recv_splits_list = list(p2p_matrix[:, rank])
  print(
      f'===== start all2all_benchmark {rank}/{size} total_len={total_len} =====',
      flush=True)
  with tf.device("/cpu:0"):
    splits = tf.constant(splits_list, dtype=tf.int32)
    recv_splits = tf.constant(recv_splits_list, dtype=tf.int32)
  with tf.device("/gpu:0" if src_gpu else "/cpu:0"):
    tensor = tf.ones([sum(splits_list), 1], dtype=tf.float32) * (rank + 1)
  t0 = time.time()
  interval = 20
  name = f'data_len_{total_len}_'
  alltoall_fn = bps.alltoall
  if dst_gpu:
    if src_gpu:
      name += 'g2g'
    else:
      alltoall_fn = bps.alltoall_cpu2gpu
      name += 'c2g'
  else:
    if src_gpu:
      alltoall_fn = bps.alltoall_gpu2cpu
      name += 'g2c'
    else:
      name += 'c2c'
  goodputs = []
  while niter < total_niter:
    with tf.device("/gpu:0" if src_gpu or dst_gpu else "/cpu:0"):
      result = alltoall_fn(tensor,
                           splits=splits,
                           recv_splits=recv_splits,
                           name=name)
    niter += 1
    if niter % interval == 0:
      t1 = time.time()
      latency = (t1 - t0) / interval * 1000
      goodput = total_len * 32 / latency / 1000000
      goodputs.append(goodput)
      rank == 0 and print(
          f'DONE iter={niter}, latency={latency:.3} ms, Goodput={goodput:.5} Gb/s',
          flush=True)
      t0 = time.time()
  print(
      f'===== end all2all_benchmark {rank}/{size} total_len={total_len} =====',
      flush=True)
  return goodputs[1:]


def bps_comm_benchmark():
  benchmark_bps = os.environ.get("MONOLITH_BENCHMARK_BPS", "none")
  benchmark_iters = int(os.getenv("MONOLITH_BENCHMARK_ITERS", "200"))
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  assert benchmark_bps in ("c2g", "g2g", "c2c", "g2c", "ar",
                           "all"), benchmark_bps
  benchmarks = ["c2g", "g2g", "c2c", "g2c", "ar"
               ] if benchmark_bps == "all" else [benchmark_bps]
  for benchmark in benchmarks:
    results = []
    dst_gpu = benchmark in ("c2g", "g2g")
    src_gpu = benchmark in ("g2c", "g2g")
    if benchmark == "ar":
      total_len = int(os.getenv("MONOLITH_BENCHMARK_BPS_AR_LEN", "65536000"))
      goodputs_cpu = byteps_benchmark_ar(total_len,
                                         total_niter=benchmark_iters,
                                         use_cpu=True)
      results.append((total_len, sum(goodputs_cpu) / len(goodputs_cpu)))
      goodputs_gpu = byteps_benchmark_ar(total_len,
                                         total_niter=benchmark_iters,
                                         use_cpu=False)
      results.append((total_len, sum(goodputs_gpu) / len(goodputs_gpu)))
    else:
      total_len = int(os.getenv("MONOLITH_BENCHMARK_BPS_A2A_LEN", "65536000"))
      for _ in range(3):
        goodputs = byteps_benchmark_a2a(total_len,
                                        total_niter=benchmark_iters,
                                        dst_gpu=dst_gpu,
                                        src_gpu=src_gpu)
        results.append((total_len, sum(goodputs) / len(goodputs)))
        total_len = total_len // 2
    print(benchmark + "_summary:", results)
