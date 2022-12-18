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

from absl import logging, flags
from contextlib import closing
from dataclasses import dataclass
import google.protobuf.text_format as text_format
import google.protobuf.json_format as json_format
import json
import os
import re
import socket
import tempfile
from typing import Dict, List, Union, Optional, get_type_hints
from idl.matrix.proto.proto_parser_pb2 import Instance

from tensorflow.core.framework.tensor_pb2 import TensorProto
from tensorflow.core.framework.types_pb2 import DataType
from tensorflow.core.protobuf.error_codes_pb2 import Code as ErrorCode
from tensorflow_serving.apis import model_pb2
from tensorflow_serving.apis.get_model_status_pb2 import ModelVersionStatus
from tensorflow_serving.config import model_server_config_pb2
from tensorflow_serving.sources.storage_path.file_system_storage_path_source_pb2 import \
  FileSystemStoragePathSourceConfig
from tensorflow_serving.util.status_pb2 import StatusProto
from tensorflow_serving.config import platform_config_pb2
from tensorflow_serving.servables.tensorflow import session_bundle_config_pb2
from tensorflow_serving.servables.tensorflow import saved_model_bundle_source_adapter_pb2
from tensorflow.core.protobuf.config_pb2 import ConfigProto

from monolith.agent_service import constants

ModelState = ModelVersionStatus.State
SEQ = re.compile(r"[ =\t]+")
ServableVersionPolicy = FileSystemStoragePathSourceConfig.ServableVersionPolicy
FeatureKeys = {
    'name', 'fid', 'float_value', 'int64_value', 'bytes_value', 'fid_list',
    'float_list', 'int64_list', 'bytes_list'
}
flags.DEFINE_string("conf", "", "agent conf file")

TFS_HOME = ""
DEFAULT_MODEL_CONFIG = None
DEFAULT_PLATFORM_CONFIG_FILE = ""
old_isabs = os.path.isabs


def isabs(path: str):
  if path.startswith('hdfs:/'):
    return True
  else:
    return old_isabs(path)


os.path.isabs = isabs
DefaultRoughSortModelLocalPath = None
DefaultRoughSortModelP2PPath = None
class TFSServerType:
  PS = 'ps'
  ENTRY = 'entry'
  DENSE = 'dense'
  UNIFIED = 'unified'


class DeployType(TFSServerType):
  MIXED = 'mixed'  # bath ps anf entry are host in one tfs

  def __init__(self, dtype: str):
    assert dtype.lower() in {
        self.ENTRY, self.PS, self.DENSE, self.MIXED, self.UNIFIED
    }
    self._dtype = dtype.lower()

  def __str__(self):
    return self._dtype

  def __hash__(self):
    return hash(self._dtype)

  def __eq__(self, o):
    if isinstance(o, str):
      return self._dtype == o
    elif isinstance(o, DeployType):
      return self._dtype == o._dtype
    else:
      return False

  def compat_server_type(self, server_type: str):
    if server_type is None or server_type == DeployType.MIXED:
      if self._dtype == DeployType.MIXED:
        raise RuntimeError('DeployType and ServerType is not compatable!')
      else:
        return self._dtype
    elif self._dtype == DeployType.MIXED:
      return server_type
    else:
      assert self._dtype == server_type
      return server_type


class RoughSortModelLoadedServer:
  NONE = 'none'
  ENTRY = 'entry'
  PS = 'ps'
  DENSE = 'dense'


class RoughSortModelPrefix:
  PS = 'ps_item_embedding'
  ENTRY = 'entry_item_embedding'
  DENSE = 'dense_item_embedding'


def conf_parser(file_name: str, args: dict):
  if not os.path.exists(file_name):
    return

  with open(file_name) as f:
    for line in f:
      line = line.strip()
      if line.startswith('#') or len(line) == 0:
        continue
      else:
        idx = line.find('#')
        if idx > 0:
          line = line[0:idx]

        if line.startswith('include'):
          conf_parser(line.split()[-1], args)
        else:
          try:
            key, value = SEQ.split(line, maxsplit=1)
            if key in args:
              if type(args[key]) is not list:
                args[key] = [args[key]] + [value]
            elif value is not None and len(value) > 0:
              args[key] = value
          except Exception as e:
            logging.error(f'{e}')


def find_free_port():
  with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
    s.bind(('localhost', 0))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    host, port = s.getsockname()
    return port


def check_port_open(port):
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  try:
    s.connect(('127.0.0.1', port))
    s.close()
  except Exception:
    logging.info(f'port {port} not open!')
    return False
  logging.info(f'port {port} opened!')
  return True


def write_to_tmp_file(content) -> str:
  fd, path = tempfile.mkstemp()
  with os.fdopen(fd, 'w') as fp:
    fp.write(str(content))
  return path


def replica_id_from_pod_name() -> int:
  try:
    if 'MY_POD_NAME' in os.environ:
      md5 = hashlib.md5()
      pod_name = os.environ.get('MY_POD_NAME', 'pod_name')
      md5.update(pod_name.encode('utf-8'))
      return int(md5.hexdigest()[10:20], base=16)
  except Exception as e:
    return -1


@dataclass
class TfServingConfig:
  '''TfServingConfig

  attributes:
    :param enable_batching: enable batching
    :param allow_version_labels_for_unavailable_models: If true, allows assigning unused version labels to models
                                                        that are not available yet.
    :param batching_parameters_file: If non-empty, read an ascii BatchingParameters protobuf from the supplied file name
                                     and use the contained values instead of the defaults.
    :param num_load_threads: The number of threads in the thread-pool used to load servables. If set as 0, we don't use
                             a thread-pool, and servable loads are performed serially in the manager's main work loop,
                             may casue the Serving request to be delayed. Default: 0
    :param num_unload_threads: The number of threads in the thread-pool used to unload servables. If set as 0, we don't
                               use a thread-pool, and servable loads are performed serially in the manager's main work
                               loop, may casue the Serving request to be delayed. Default: 0
    :param max_num_load_retries: maximum number of times it retries loading a model after the first failure, before
                                 giving up. If set to 0, a load is attempted only once. Default: 5
    :param load_retry_interval_micros: The interval, in microseconds, between each servable load retry. If set negative,
                                       it doesn't wait. Default: 1 minute
    :param file_system_poll_wait_seconds: Interval in seconds between each poll of the filesystem for new model version.
                                          If set to zero poll will be exactly done once and not periodically. Setting
                                          this to negative value will disable polling entirely causing ModelServer to
                                          indefinitely wait for a new model at startup. Negative values are reserved for
                                          testing purposes only.
    :param flush_filesystem_caches: If true (the default), filesystem caches will be flushed after the initial load of
                                    all servables, and after each subsequent individual servable reload
                                    (if the number of load threads is 1). This reduces memory consumption of the model
                                    server, at the potential cost of cache misses if model files are accessed after
                                    servables are loaded.
    :param tensorflow_session_parallelism: Number of threads to use for running a Tensorflow session.
                                           Auto-configured by default. Note that this option is ignored if
                                           --platform_config_file is non-empty.
    :param tensorflow_intra_op_parallelism: Number of threads to use to parallelize the execution of an individual op.
                                            Auto-configured by default. Note that this option is ignored
                                            if --platform_config_file is non-empty.
    :param tensorflow_inter_op_parallelism: Controls the number of operators that can be executed simultaneously.
                                            Auto-configured by default. Note that this option is ignored
                                            if --platform_config_file is non-empty.
    :param ssl_config_file: If non-empty, read an ascii SSLConfig protobuf from the supplied file name and set up a
                            secure gRPC channel
    :param per_process_gpu_memory_fraction: Fraction that each process occupies of the GPU memory space the value is
                                            between 0.0 and 1.0 (with 0.0 as the default) If 1.0, the server will
                                            allocate all the memory when the server starts, If 0.0, Tensorflow will
                                            automatically select a value.
    :param allow_growth: allow gpu growth
    :param saved_model_tags: Comma-separated set of tags corresponding to the meta graph def to load from SavedModel.
    :param grpc_channel_arguments: A comma separated list of arguments to be passed to the grpc server.
                                    (e.g. grpc.max_connection_age_ms=2000)
    :param grpc_max_threads: Max grpc server threads to handle grpc messages.
    :param enable_model_warmup: Enables model warmup, which triggers lazy initializations (such as TF optimizations)
                                at load time, to reduce first request latency.
    :param version: Display version
    :param remove_unused_fields_from_bundle_metagraph: Removes unused fields from MetaGraphDef proto message to save
                                                       memory.
    :param enable_signature_method_name_check: Enable method_name check for SignatureDef. Disable this if agent_service native
                                               TF2 regression/classification models.
    :param xla_cpu_compilation_enabled: EXPERIMENTAL; CAN BE REMOVED ANYTIME! Enable XLA:CPU JIT (default is disabled).
                                        With XLA:CPU JIT disabled, models utilizing this feature will return bad Status
                                        on first compilation request.
    :param enable_profiler: Enable profiler service.
  '''

  enable_batching: bool = False
  allow_version_labels_for_unavailable_models: bool = False
  batching_parameters_file: str = None
  num_load_threads: int = 0
  num_unload_threads: int = 0
  max_num_load_retries: int = 5
  load_retry_interval_micros: int = 60 * 1000 * 1000
  file_system_poll_wait_seconds: int = 1
  flush_filesystem_caches: bool = True
  tensorflow_session_parallelism: int = 0
  tensorflow_intra_op_parallelism: int = 0
  tensorflow_inter_op_parallelism: int = 0
  ssl_config_file: str = None
  platform_config_file: str = None
  per_process_gpu_memory_fraction: float = 0
  allow_growth: bool = True
  saved_model_tags: str = None
  grpc_channel_arguments: str = None
  grpc_max_threads: int = 0
  enable_model_warmup: bool = True
  version: str = None
  remove_unused_fields_from_bundle_metagraph: bool = True
  enable_signature_method_name_check: bool = False
  xla_cpu_compilation_enabled: bool = False
  enable_profiler: bool = True


@dataclass
class AgentConfig(TfServingConfig):
  '''AgentConfig

  attributes:
    :param bzid: business id of this agent_service, cannot be None.
    :param base_name: base name of model
    :param base_path: path to export
    :param num_ps: The number of ps.
    :param num_shard: The total number of shard.
    :param deploy_type: Server type, can be ps/entry/dense/mixed.
    :param stand_alone_serving: Whether is stand alone agent_service.
    :param zk_servers: The zk servers.
    :param proxy_port: TODO
    :param tfs_entry_port: TODO
    :param tfs_entry_http_port: TODO
    :param tfs_ps_port: TODO
    :param tfs_ps_http_port: TODO
    :param dense_alone: whether dense alone
    :param tfs_dense_port: TODO
    :param tfs_dense_http_port: TODO
    :param agent_port: TODO
    :param update_model_status_interval: Update model status interval.
    :param max_waiting_sec: The waiting second for PS/DENSE to load, default 600
    :param agent_version: Version of Agent, default 1
    :param version_policy: Tensorflow version_policy, can be latest/specific/all
    :param version_data: saved_model version
    :param preload_jemalloc: preload jemalloc.so
    :param rough_sort_model_name: model name for deep rough sort, which is generated by FeynmanTob
    :param rough_sort_model_local_path: load deep rough sort model from this dir
    :param rough_sort_model_loaded_server: load rough sort model on which server: ps or entry or dense
    :param layout_pattern: layout path format
    :param layout_filters: filter saved_models under layout_pattern to load
    :param tfs_port_archon: service archon port
    :param tfs_port_grpc: service grpc port
    :param tfs_port_http: service http port
    :param use_metrics: whether use metrics
  '''

  bzid: str = None
  base_name: str = None
  base_path: str = None
  num_ps: int = 1
  num_shard: int = None
  deploy_type: str = None
  replica_id: int = None
  stand_alone_serving: bool = False
  zk_servers: str = None
  proxy_port: int = None
  tfs_entry_port: int = None
  tfs_entry_http_port: int = None
  tfs_entry_archon_port: int = None
  tfs_ps_port: int = None
  tfs_ps_http_port: int = None
  tfs_ps_archon_port: int = None
  dense_alone: bool = False
  tfs_dense_port: int = None
  tfs_dense_http_port: int = None
  tfs_dense_archon_port: int = None
  agent_port: int = None
  update_model_status_interval: int = 1
  model_config_file = None
  agent_version: int = 1
  max_waiting_sec: int = 1200
  preload_jemalloc: bool = True
  version_policy: str = 'latest'
  version_data: int = 1
  fetch_ps_timeout_ms: int = 200
  fetch_ps_long_conn_num: int = 100
  fetch_ps_long_conn_enable: bool = True
  fetch_ps_retry: int = 2
  aio_thread_num: int = 30
  # for deep rough sort
  rough_sort_model_name: str = None
  rough_sort_model_local_path: str = DefaultRoughSortModelLocalPath
  rough_sort_model_loaded_server: str = RoughSortModelLoadedServer.ENTRY
  rough_sort_model_p2p_path: str = DefaultRoughSortModelP2PPath
  dc_aware: bool = False
  # for unified container
  layout_pattern: str = None
  layout_filters: List = None
  tfs_port_archon: int = None
  tfs_port_grpc: int = None
  tfs_port_http: int = None
  use_metrics: bool = True

  def __post_init__(self):
    if self.stand_alone_serving:
      self.deploy_type = DeployType(DeployType.MIXED)
    else:
      assert self.deploy_type is not None
      self.deploy_type = DeployType(self.deploy_type)

    if self.num_shard is None:
      self.num_shard = self.num_tce_shard
    else:
      assert self.num_shard == self.num_tce_shard

    # PORT1 reserve for p2p
    # PORT2 reserve for agent
    if self.deploy_type == DeployType.MIXED:
      self.proxy_port = find_free_port()
      self.tfs_entry_archon_port = int(os.environ.get('PORT', find_free_port()))
      self.tfs_entry_port = int(os.environ.get('PORT3', find_free_port()))
      self.tfs_entry_http_port = int(os.environ.get('PORT4', find_free_port()))
      self.tfs_ps_port = int(os.environ.get('PORT5', find_free_port()))
      self.tfs_ps_http_port = int(os.environ.get('PORT6', find_free_port()))
      self.tfs_ps_archon_port = int(os.environ.get('PORT7', find_free_port()))
      if self.dense_alone:
        self.tfs_dense_port = int(os.environ.get('PORT8', find_free_port()))
        self.tfs_dense_http_port = int(os.environ.get('PORT9',
                                                      find_free_port()))
        self.tfs_dense_archon_port = int(
            os.environ.get('PORT10', find_free_port()))
    elif self.deploy_type == DeployType.ENTRY:
      self.proxy_port = find_free_port()
      self.tfs_ps_archon_port = find_free_port()
      self.tfs_ps_port = find_free_port()
      self.tfs_ps_http_port = find_free_port()

      if self.dense_alone:
        self.tfs_dense_port = find_free_port()
        self.tfs_dense_http_port = find_free_port()
        self.tfs_dense_archon_port = find_free_port()

      self.tfs_entry_archon_port = int(os.environ.get('PORT', find_free_port()))
      self.tfs_entry_port = int(os.environ.get('PORT3', find_free_port()))
      self.tfs_entry_http_port = int(os.environ.get('PORT4', find_free_port()))
    elif self.deploy_type == DeployType.PS:
      self.proxy_port = find_free_port()
      self.tfs_entry_archon_port = find_free_port()
      self.tfs_entry_port = find_free_port()
      self.tfs_entry_http_port = find_free_port()

      if self.dense_alone:
        self.tfs_dense_port = find_free_port()
        self.tfs_dense_http_port = find_free_port()
        self.tfs_dense_archon_port = find_free_port()

      self.tfs_ps_archon_port = int(os.environ.get('PORT', find_free_port()))
      self.tfs_ps_port = int(os.environ.get('PORT3', find_free_port()))
      self.tfs_ps_http_port = int(os.environ.get('PORT4', find_free_port()))
    elif self.deploy_type == DeployType.DENSE:
      assert self.dense_alone == True
      self.proxy_port = find_free_port()
      self.tfs_entry_archon_port = find_free_port()
      self.tfs_entry_port = find_free_port()
      self.tfs_entry_http_port = find_free_port()

      self.tfs_ps_archon_port = find_free_port()
      self.tfs_ps_port = find_free_port()
      self.tfs_ps_http_port = find_free_port()

      self.tfs_dense_archon_port = int(os.environ.get('PORT', find_free_port()))
      self.tfs_dense_port = int(os.environ.get('PORT3', find_free_port()))
      self.tfs_dense_http_port = int(os.environ.get('PORT4', find_free_port()))
    else:
      assert self.deploy_type == DeployType.UNIFIED
      self.tfs_port_archon = int(os.environ.get('PORT', find_free_port()))
      self.tfs_port_grpc = int(os.environ.get('PORT3', find_free_port()))
      self.tfs_port_http = int(os.environ.get('PORT4', find_free_port()))

    if self.agent_port is None:
      self.agent_port = int(os.environ.get('PORT2', find_free_port()))

    if self.agent_version == 1:
      self.replica_id = int(os.environ.get('REPLICA_ID', -1))
    else:
      replica_id = int(os.environ.get('REPLICA_ID', -1))
      if replica_id == -1:
        replica_id = replica_id_from_pod_name()
      self.replica_id = replica_id

    if not self.platform_config_file:
      self.platform_config_file = DEFAULT_PLATFORM_CONFIG_FILE

    self.generate_platform_config_file()

  def generate_platform_config_file(self):
    try:
      session_config = ConfigProto()
      session_config.intra_op_parallelism_threads = (
          self.tensorflow_intra_op_parallelism or
          int(os.getenv("MY_CPU_LIMIT", "0"))) or 16
      session_config.inter_op_parallelism_threads = (
          self.tensorflow_inter_op_parallelism or
          int(os.getenv("MY_CPU_LIMIT", "0"))) or 16
      session_config.allow_soft_placement = True
      session_config.gpu_options.allow_growth = self.allow_growth
      if self.dense_alone and self.enable_batching:
        batching_parameters = session_bundle_config_pb2.BatchingParameters()
        batching_parameters.max_batch_size.value = 1024
        batching_parameters.batch_timeout_micros.value = 0
        batching_parameters.max_enqueued_batches.value = 100000
        batching_parameters.num_batch_threads.value = 2
        legacy_config = session_bundle_config_pb2.SessionBundleConfig(
            session_config=session_config,
            batching_parameters=batching_parameters)
      else:
        legacy_config = session_bundle_config_pb2.SessionBundleConfig(
            session_config=session_config)
      legacy_config.enable_model_warmup = self.enable_model_warmup
      adapter = saved_model_bundle_source_adapter_pb2.SavedModelBundleSourceAdapterConfig(
          legacy_config=legacy_config)
      config_map = platform_config_pb2.PlatformConfigMap()
      config_map.platform_configs['tensorflow'].source_adapter_config.Pack(
          adapter)
      text_config_map = text_format.MessageToString(config_map)
      with open(self.platform_config_file, 'w') as f:
        f.write(text_config_map)
    except Exception as e:
      logging.info(e)
      try:
        if os.path.isfile(self.platform_config_file):
          os.remove(self.platform_config_file)
      except Exception as e2:
        logging.info(e2)

  @property
  def num_tce_shard(self) -> int:
    return int(os.environ.get(constants.HOST_SHARD_ENV, 1))

  @property
  def shard_id(self) -> int:
    return int(os.environ.get('SHARD_ID', -1))

  @property
  def idc(self) -> Optional[str]:
    idc = os.environ.get('TCE_INTERNAL_IDC')
    if idc is None:
      return None
    else:
      return idc.lower()

  @property
  def cluster(self) -> Optional[str]:
    cluster = (os.environ.get('TCE_LOGICAL_CLUSTER') or
               os.environ.get('TCE_CLUSTER') or
               os.environ.get('TCE_PHYSICAL_CLUSTER'))
    if cluster is None:
      return None
    else:
      return cluster.lower()

  @property
  def location(self) -> Optional[str]:
    idc, cluster = self.idc, self.cluster
    if idc is None or cluster is None:
      return None
    else:
      return f'{idc}:{cluster}'

  @property
  def path_prefix(self) -> str:
    if self.dc_aware:
      return os.path.join('/', self.bzid, 'service', self.base_name,
                          self.location)
    else:
      return os.path.join('/', self.bzid, 'service', self.base_name)

  @property
  def layout_path(self) -> str:
    if self.layout_pattern.startswith("/"):
      return self.layout_pattern
    else:
      return f"/{self.bzid}/layouts/{self.layout_pattern}"

  @property
  def container_cluster(self) -> str:
    psm = os.environ.get("TCE_PSM", "unknown")
    return f"{psm};{self.idc};{self.cluster}"

  @property
  def container_id(self) -> str:
    return os.environ.get("MY_POD_NAME", get_local_ip())

  def get_cmd_and_port(self,
                       binary,
                       server_type: str = None,
                       config_file: str = None):
    server_type = self.deploy_type.compat_server_type(server_type)

    if config_file is None:
      model_server_config = self._gen_model_server_config(server_type)
      config_file = write_to_tmp_file(model_server_config)

    flags = []
    flags.append(f'--model_config_file={config_file}')
    psm = os.environ.get("TCE_PSM", "")
    cluster = os.environ.get("TCE_CLUSTER", "")
    prefix = psm
    log_conf = '../conf/log4j.properties'

    if self.deploy_type == DeployType.MIXED and server_type != TFSServerType.ENTRY:
      psm = psm + '_' + server_type.lower()
      prefix = psm
      log_conf = '../conf/log4j_{}.properties'.format(server_type.lower())

    flags.append(f"--archon_rpc_psm={psm}")
    flags.append(f"--archon_rpc_cluster={cluster}")
    flags.append(f"--metrics_namespace_prefix={prefix}")
    flags.append(f"--log_conf={log_conf}")

    if server_type == TFSServerType.PS:
      flags.append(f"--port={self.tfs_ps_port}")
      flags.append(f"--rest_api_port={self.tfs_ps_http_port}")
      flags.append(f'--archon_port={self.tfs_ps_archon_port}')
      port = self.tfs_ps_port
    elif server_type == TFSServerType.DENSE:
      flags.append(f"--port={self.tfs_dense_port}")
      flags.append(f"--rest_api_port={self.tfs_dense_http_port}")
      flags.append(f'--archon_port={self.tfs_dense_archon_port}')
      if self.enable_batching:
        flags.append(f'--enable_batching')
      port = self.tfs_dense_port
    else:
      flags.append(f"--port={self.tfs_entry_port}")
      flags.append(f"--rest_api_port={self.tfs_entry_http_port}")
      flags.append(f'--archon_port={self.tfs_entry_archon_port}')
      flags.append(
          f'--archon_entry_to_ps_rpc_timeout={self.fetch_ps_timeout_ms}')
      flags.append(
          f'--archon_entry_to_ps_long_conn_num={self.fetch_ps_long_conn_num}')
      flags.append(f'--archon_entry_to_ps_rpc_retry={self.fetch_ps_retry}')
      flags.append(f'--archon_async_dispatcher_threads={self.aio_thread_num}')
      if not self.fetch_ps_long_conn_enable:
        flags.append(f'--archon_entry_to_ps_long_conn_enable=false')
      port = self.tfs_entry_port

    if self.agent_version != 1:
      flags.append("--model_config_file_poll_wait_seconds=0")

    for key, clz in get_type_hints(TfServingConfig).items():
      default = getattr(TfServingConfig, key)
      value = getattr(self, key)

      if key == 'file_system_poll_wait_seconds':
        if self.agent_version == 1:
          if server_type == TFSServerType.PS:
            flags.append('--file_system_poll_wait_seconds=0')
          elif value != default:  # entry,dense
            flags.append('--file_system_poll_wait_seconds={value}')
      elif value != default:
        if clz == bool:
          flags.append(f'--{key}={str(value).lower()}')
        else:
          flags.append(f'--{key}={value}')

    return f'{binary} {" ".join(flags)}', port

  def get_cmd(self, binary, server_type: str = None) -> str:
    cmd, port = self.get_cmd_and_port(binary, server_type)
    return cmd

  def _gen_model_server_config(
      self,
      server_type: str = None,
      version_policy: str = 'latest',
      version_data: Union[int, List[int]] = 1,
  ) -> model_server_config_pb2.ModelServerConfig:
    server_type = self.deploy_type.compat_server_type(server_type)
    assert server_type is not None

    model_server_config = model_server_config_pb2.ModelServerConfig()
    model_config_list = model_server_config.model_config_list.config

    if server_type == TFSServerType.PS:
      for i in range(self.num_ps):
        if i % self.num_shard == self.shard_id:
          name = f'{server_type}_{i}'
          model_config = model_config_list.add()
          model_config.CopyFrom(
              gen_model_config(name=name,
                               base_path=os.path.join(self.base_path, name),
                               version_policy=version_policy,
                               version_data=version_data))
          if self.rough_sort_model_name and self.rough_sort_model_loaded_server == RoughSortModelLoadedServer.PS:
            name = f'{RoughSortModelPrefix.PS}_{i}'
            model_config = model_config_list.add()
            rough_model_path = os.path.join(self.rough_sort_model_local_path,
                                            self.rough_sort_model_name, name)
            model_config.CopyFrom(
                gen_model_config(name=name,
                                 base_path=rough_model_path,
                                 version_policy=version_policy,
                                 version_data=version_data))
    else:
      if server_type == TFSServerType.DENSE:
        name = f'{server_type}_0'
      else:
        name = server_type
      model_config = model_config_list.add()
      model_config.CopyFrom(
          gen_model_config(name=name,
                           base_path=os.path.join(self.base_path, name),
                           version_policy=version_policy,
                           version_data=version_data))
      if (self.rough_sort_model_name and (self.rough_sort_model_loaded_server
                                          == RoughSortModelLoadedServer.ENTRY or
                                          self.rough_sort_model_loaded_server
                                          == RoughSortModelLoadedServer.DENSE)):
        if self.rough_sort_model_loaded_server == RoughSortModelLoadedServer.ENTRY:
          name = f'{RoughSortModelPrefix.ENTRY}_0'
        elif self.rough_sort_model_loaded_server == RoughSortModelLoadedServer.DENSE:
          name = f'{RoughSortModelPrefix.DENSE}_0'
        model_config = model_config_list.add()
        rough_model_path = os.path.join(self.rough_sort_model_local_path,
                                        self.rough_sort_model_name, name)
        model_config.CopyFrom(
            gen_model_config(name=name,
                             base_path=rough_model_path,
                             version_policy=version_policy,
                             version_data=version_data))

    return model_server_config

  @classmethod
  def from_file(cls, fname):
    kwarg = {}
    conf_parser(fname, kwarg)

    args = {}
    for key, clz in get_type_hints(AgentConfig).items():
      try:
        if key in kwarg:
          if clz == bool and kwarg[key].lower() in {
              'true', 'y', 't', 'yes', '1'
          }:
            args[key] = True
          elif clz == bool and kwarg[key].lower() in {
              'false', 'n', 'f', 'no', '0'
          }:
            args[key] = False
          elif clz in {int, float}:
            args[key] = clz(eval(kwarg[key]))
          elif clz == str:
            if kwarg[key].lower() == 'none':
              args[key] = None
            else:
              args[key] = kwarg[key]
          elif clz == List:
            if type(kwarg[key]) is not list:
              args[key] = [kwarg[key]]
            else:
              args[key] = kwarg[key]
          else:
            args[key] = clz(kwarg[key])
      except:
        logging.error(f'type convert {key} error, the type is {clz}')

    # for compat
    if 'deploy_type' not in args:
      args['deploy_type'] = kwarg.pop('server_type', None)

    return cls(**args)


class ZKPath(object):
  PAT = re.compile(
      r'^/(?P<bzid>[-_0-9A-Za-z]+)/service/(?P<base_name>[-_0-9A-Za-z]+)(/(?P<idc>\w+):(?P<cluster>\w+))?/(?P<server_type>\w+):(?P<index>\d+)(/(?P<replica_id>\d+))?$'
  )

  def __init__(self, path: str):
    self.path = path

    if path is None or len(path) != 0:
      matched = self.PAT.match(self.path)
      if matched:
        self._group_dict = matched.groupdict()
      else:
        self._group_dict = None
    else:
      self._group_dict = None

  def __getattr__(self, name: str):
    assert name in {
        'bzid', 'base_name', 'idc', 'cluster', 'server_type', 'index',
        'replica_id'
    }
    if self._group_dict:
      return self._group_dict.get(name)
    else:
      return None

  @property
  def task(self) -> str:
    server_type, index = self.server_type, self.index

    if server_type is not None and index is not None:
      return f'{server_type}:{index}'
    else:
      return None

  @property
  def location(self) -> Optional[str]:
    idc, cluster = self.idc, self.cluster
    if idc is None or cluster is None:
      return None
    else:
      return f'{idc}:{cluster}'

  def ship_in(self, idc: str, cluster: str) -> bool:
    if idc is None or cluster is None:
      return True
    else:
      return idc == self.idc and cluster == self.cluster


def parse_pattern(pattern_str, init_val, comb_fn, lp='{', rp='}'):
  ret_val = init_val
  while len(pattern_str):
    begin = pattern_str.find(lp)
    end = pattern_str.find(rp, begin)
    if begin == -1 or end == -1:
      ret_val = comb_fn(ret_val, pattern_str, None)
      break
    ret_val = comb_fn(ret_val, pattern_str[:begin], pattern_str[begin + 1:end])
    pattern_str = pattern_str[end + 1:]
  return ret_val


def normalize_regex(pattern_str):

  def comb(val: str, p1: str, p2: str):
    if p2 is None:
      return val + p1
    return val + p1 + f'(?P<{p2}>\d+)'

  return parse_pattern(pattern_str, '', comb_fn=comb)


def expand_pattern(pattern_str):

  def comb(vals: List, p1: str, p2: str):
    if p2 is None:
      return [val + p1 for val in vals]
    l = []

    for t in p2.split(','):
      if '-' in t:
        s, e = t.split('-')
        l.extend(range(int(s), int(e)))
      else:
        l.extend(int(t))
    return [val + p1 + str(i) for val in vals for i in l]

  return parse_pattern(pattern_str, [''], comb, '[', ']')


def gen_model_spec(name: str,
                   version: Union[int, str] = None,
                   signature_name: str = None):
  mode_spec = model_pb2.ModelSpec(name=name)

  if version is not None:
    if isinstance(version, int):
      mode_spec.version.value = version
    else:
      mode_spec.version_label = version

  if signature_name is not None:
    mode_spec.signature_name = signature_name

  return mode_spec


def gen_model_config(
    name: str,
    base_path: str,
    version_policy: str = 'latest',
    version_data: Union[int, List[int]] = 1,
    model_platform: str = 'tensorflow',
    version_labels: Dict[str,
                         int] = None) -> model_server_config_pb2.ModelConfig:
  model_config = model_server_config_pb2.ModelConfig(
      name=name, base_path=base_path, model_platform=model_platform)

  if version_policy.lower() == 'latest':
    assert isinstance(version_data, int)
    model_config.model_version_policy.latest.num_versions = version_data
  elif version_policy.lower() == 'latest_once':
    assert isinstance(version_data, int)
    model_config.model_version_policy.latest_once.num_versions = version_data
  elif version_policy.lower() == 'all':
    model_config.model_version_policy.all.CopyFrom(ServableVersionPolicy.All())
  elif version_policy.lower() == 'specific':
    if isinstance(version_data, int):
      version_data = [version_data]
    assert isinstance(version_data, list)
    model_config.model_version_policy.specific.versions.extend(version_data)
  else:
    raise ValueError(version_policy + " is not allowed!")

  if version_labels is not None:
    model_config.version_labels.update(version_labels)

  return model_config


DEFAULT_MODEL_CONFIG = gen_model_config(name='default',
                                        base_path=os.path.join(
                                            TFS_HOME, 'dat', 'saved_models',
                                            'entry'))


def gen_status_proto(error_code: ErrorCode = ErrorCode.OK,
                     error_message: str = None):
  return StatusProto(error_code=error_code, error_message=error_message)


def gen_model_version_status(version: int,
                             state: ModelState = ModelState.UNKNOWN,
                             error_code: ErrorCode = ErrorCode.OK,
                             error_message: str = None):
  mvs = ModelVersionStatus(version=version, state=state)
  mvs.status.CopyFrom(gen_status_proto(error_code, error_message))

  return mvs


def make_tensor_proto(instances):
  tp = TensorProto(dtype=DataType.DT_STRING)
  dim = tp.tensor_shape.dim.add()
  dim.size = len(instances)
  tp.string_val.extend(instances)

  return tp


class InstanceFormater:

  def __init__(self, inst: Instance):
    self._inst = inst

  def __str__(self):
    return f"{self._inst}"

  def to_tensor_proto(self, batch_size: int):
    serialized = self._inst.SerializeToString()
    instances = [serialized for _ in range(batch_size)]

    return make_tensor_proto(instances)

  def to_pb(self, fname: str = None) -> str:
    content = self._inst.SerializeToString()
    if fname is None:
      fd, path = tempfile.mkstemp()
      with os.fdopen(fd, 'wb') as fp:
        fp.write(content)
      return path
    else:
      with open(fname, 'wb') as fid:
        fid.write(content)
      return fname

  def to_json(self, fname: str = None) -> str:
    content = json_format.MessageToJson(self._inst)
    if fname is None:
      return write_to_tmp_file(content)
    else:
      with open(fname) as fid:
        fid.write(content)
      return fname

  def to_pb_text(self, fname: str = None) -> str:
    if fname is None:
      return write_to_tmp_file(self._inst)
    else:
      with open(fname) as fid:
        fid.write(str(self._inst))
      return fname

  @classmethod
  def from_json(cls, fname: str):
    message = Instance()
    with open(fname) as fid:
      kwargs = json.load(fid)
      return cls(json_format.ParseDict(kwargs, message))

  @classmethod
  def from_pb_text(cls, fname: str):
    message = Instance()
    text = []
    with open(fname) as fid:
      for line in fid:
        text.append(line.strip())
    return cls(text_format.Parse('\n'.join(text), message))

  @classmethod
  def from_dump(cls, fname: str):
    stack, kwargs = [], {}

    def get_item():
      if stack and kwargs:
        arg = kwargs
        for item in stack:
          if item in arg:
            arg = arg[item]
          else:
            return None
        return arg
      else:
        return None

    def set_item(item):
      last_arg, arg = None, kwargs
      for key in stack:
        last_arg = arg
        arg = arg[key]

      if isinstance(item, dict):
        if isinstance(arg, list):
          stack.pop()
          arg = last_arg

        (key, value), = item.items()
        if value is None:
          if key.isnumeric() and int(key) == 0:
            last_arg[stack[-1]] = [value]
            stack.append(0)
          elif key.isnumeric():
            stack[-1] = int(key)
            last_arg.append(value)
          else:
            if arg is None:
              last_arg[stack[-1]] = item
            elif len(stack) >= 2 and isinstance(stack[-1],
                                                int) and stack[-2] == 'feature':
              if key in FeatureKeys and key not in arg:
                arg.update(item)
              else:
                stack.pop()
                stack.pop()
                kwargs.update(item)
            else:
              arg.update(item)
            stack.append(key)
        else:
          if arg is None:
            last_arg[stack[-1]] = item
          else:
            assert isinstance(arg, dict)
            arg.update(item)
      else:
        if arg is None:
          last_arg[stack[-1]] = [item]
        else:
          arg.append(item)

    with open(fname) as fid:
      for line in fid:
        if line.startswith('"root":'):
          continue
        (key, value) = [
            item.strip().strip('"').strip("'")
            for item in line.strip().split(':')
        ]
        if len(value) == 0:
          set_item(item={key: None})
        else:
          if value.isnumeric():
            value = int(value)

          if key.isnumeric():  # list
            set_item(item=value)
          else:  # dict
            set_item(item={key: value})

    message = Instance()
    return cls(json_format.ParseDict(kwargs, message))


def pasre_sub_model_name(sub_model_name: str):
  if sub_model_name is None or len(sub_model_name) == 0:
    raise RuntimeError('sub_model_name is None or empty')

  pasred = sub_model_name.strip().split('_')
  if len(pasred) == 1:
    return pasred[0].lower(), 0
  else:
    assert len(pasred) == 2
    return pasred[0].lower(), int(pasred[1])


def get_local_ip() -> str:
  try:
    local_ip = os.environ.get("MY_HOST_IP",
                              socket.gethostbyname(socket.gethostname()))
    if local_ip is not None and local_ip not in {'', 'localhost', '127.0.0.1'}:
      return local_ip
  except Exception as e:
    logging.warning(e)

  skt = None
  try:
    skt = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    skt.connect(('8.8.8.8', 80))
    local_ip = skt.getsockname()[0]
    if local_ip is not None and local_ip not in {'', 'localhost', '127.0.0.1'}:
      return local_ip
  except Exception as e:
    logging.warning(e)
  finally:
    if skt is not None:
      skt.close()

  return 'localhost'
