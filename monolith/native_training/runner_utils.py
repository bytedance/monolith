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
from contextlib import contextmanager
from dataclasses import dataclass, field, Field
from enum import Enum
import json
import os, sys, traceback
from threading import RLock
import time
from absl.flags import FlagValues
from google.protobuf import text_format

import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training.checkpoint_state_pb2 import CheckpointState

from monolith.native_training.cpu_training import DistributedCpuTrainingConfig
from monolith.native_training.service_discovery import ServiceDiscoveryType, \
  ConsulServiceDiscovery, TfConfigServiceDiscovery, ZKServiceDiscovery
from monolith.native_training import gflags_utils
from monolith.native_training.monolith_checkpoint_state_pb2 import MonolithCheckpointState
from monolith.native_training.net_utils import AddressFamily
from monolith.native_training import save_utils

FLAGS = flags.FLAGS
old_isabs = os.path.isabs
old_get_checkpoint_state = checkpoint_management.get_checkpoint_state


def isabs(path: str):
  if path.startswith('hdfs:/'):
    return True
  else:
    return old_isabs(path)


# [todo](fitz) part of function will move to Rec Platfrom, this is a tem solution
def gen_get_checkpoint_state():
  # ensure get the same value when call in the same process
  _lock = RLock()

  @tf_export("train.get_checkpoint_state")
  def _get_checkpoint_state_internal(checkpoint_dir, latest_filename=None):
    latest_filename = latest_filename or 'checkpoint'
    with _lock:
      checkpoint_state = old_get_checkpoint_state(checkpoint_dir,
                                                  latest_filename)
      try:
        if FLAGS.restore_ckpt is not None:
          if latest_filename != 'checkpoint' or checkpoint_state is None:
            return checkpoint_state

          dirname_from_ckpt_state = os.path.dirname(
              checkpoint_state.model_checkpoint_path)
          restore_ckpt = os.path.join(dirname_from_ckpt_state,
                                      os.path.basename(FLAGS.restore_ckpt))
          restore_ckpt_file = os.path.join(checkpoint_dir, 'restore_ckpt')

          if restore_ckpt != checkpoint_state.model_checkpoint_path and restore_ckpt in checkpoint_state.all_model_checkpoint_paths:
            if FLAGS.mode == tf.estimator.ModeKeys.TRAIN:
              if not tf.io.gfile.exists(restore_ckpt_file):
                checkpoint_state.model_checkpoint_path = restore_ckpt
              else:
                logging.info(
                    f'mode is {FLAGS.mode} and {restore_ckpt_file} file exists, keep {checkpoint_state.model_checkpoint_path}'
                )
            else:
              logging.info(f'mode is {FLAGS.mode}, ignore {restore_ckpt_file}')
              checkpoint_state.model_checkpoint_path = restore_ckpt
          else:
            if restore_ckpt == checkpoint_state.model_checkpoint_path:
              logging.warning(
                  f"model_checkpoint_path and {FLAGS.restore_ckpt} are identity"
              )
            else:
              logging.warning(
                  f"checkpoint {FLAGS.restore_ckpt} not exists in {checkpoint_dir}"
              )

          if FLAGS.mode == tf.estimator.ModeKeys.TRAIN:
            if not tf.io.gfile.exists(restore_ckpt_file):
              checkpoint_state.model_checkpoint_path = restore_ckpt
              with tf.io.gfile.GFile(restore_ckpt_file, 'w') as gfile:
                gfile.write(restore_ckpt)
              checkpoint_filename = os.path.join(checkpoint_dir,
                                                 latest_filename)
              file_io.atomic_write_string_to_file(
                  checkpoint_filename,
                  text_format.MessageToString(checkpoint_state))
              logging.info(
                  f'mode is {FLAGS.mode} and no {restore_ckpt_file} file exists, apply {restore_ckpt}'
              )
      except flags._exceptions.UnparsedFlagAccessError as e:
        pass
      except Exception as e:
        logging.info(f"get_checkpoint_state: {e}")
        exc_type, exc_value, exc_traceback_obj = sys.exc_info()
        logging.error(f"exc_type: {exc_type}")
        logging.error(f"exc_value: {exc_value}")
        traceback.print_tb(exc_traceback_obj, limit=10)

      return checkpoint_state

  return _get_checkpoint_state_internal


os.path.isabs = isabs
checkpoint_management.get_checkpoint_state = gen_get_checkpoint_state()
tf.train.get_checkpoint_state = checkpoint_management.get_checkpoint_state


class ContainerType(Enum):
  DOCKER = 1
  NATIVE = 2


@gflags_utils.extract_flags_decorator(is_nested=True)
@dataclass
class RunnerConfig(DistributedCpuTrainingConfig):
  """RunnerConfig for start a running.
  
  attributes:
    :param task: Name of the task class to run, or the run py file name
    :param tf_config: The TF_CONFIG env variable from primus, a json string.
    :param deep_insight_name: the deep_insight name, which should be identity during the whole job.
    :param discovery_type: service discovery type, which can be primus, consul and zk.
    :param zk_server: The ZK server
    :param zk_watch_address_family: We register both ipv4 and ipv6 when serving,
              and watch either ipv4 or ipv6 when synchronizing parameters.
    :param is_local: Whether is local running.
    :param enable_fid_dedup: Whether enable fid dedup in PS.
    :param bzid: In realtime native training, business id of the job.
    :param ps_replica_num: In realtime native training, the number of online ps replica.
    :param tf_grpc_worker_cache_threads: Env variable for TF_GRPC_WORKER_CACHE_THREADS
    :param monolith_grpc_worker_service_handler_multiplier: the multiplier of the number of default gprc service handler.
    :param params_override: Override to model params. A JSON string.
    :param base_name: Base name while enable realtime training.
    :param data_type: The input data proto type, can be Instance/Example/ExampleBatch.
    :param feature_list: The feature list name
    :param lagrangex_header: Whether has lagrangex_header
    :param sort_id: Whether has sort_id
    :param kafka_dump: Whether has kafka_dump
    :param kafka_dump_prefix: Whether has kafka_dump_prefix
    :param restore_dir: The directory where the model restore.
    :param restore_ckpt: The directory where the model restore.
    :param deep_insight_target: Deep insight target name, if there are multi target, use comma split.
    :param deep_insight_sample_ratio: Deep insight sample ratio.
    :param unified_serving: Whether serving cluster is deployed in unified mode 
    :param use_estimator: Whether use estimator to run a model
    :param kafka_topics: kafka topics for streaming, when no forier and flink
    :param kafka_group_id: kafka group_id for streaming, when no forier and flink
    :param kafka_servers: kafka servers for streaming, when no forier and flink
    :param input_path: The input hdfs path for training/eval.
    :param wildcard: Wildcard for filter input files.
    :param start_date: The start date of training/eval, include.
    :param end_date: The end date of training/eval, exclude.
    :param start_hour: The start hour of training, include.
    :param end_hour: The end hour of training, exclude.
    :param is_hourly: Whether the input data is hourly partitioned.
    :param enable_dynamic_sharding: Whether switch on dynamic_sharding
    :param max_task_num_per_worker: Number of data reader task per worker, the same as primus setting
  """

  task: str = None
  tf_config: str = None
  deep_insight_name: str = None
  discovery_type: ServiceDiscoveryType = ServiceDiscoveryType.CONSUL
  zk_server: str = None
  zk_watch_address_family: str = AddressFamily.IPV4
  is_local: bool = False
  enable_fid_dedup: bool = False
  bzid: str = None
  ps_replica_num: int = None
  tf_grpc_worker_cache_threads: int = 16
  monolith_grpc_worker_service_handler_multiplier: float = 1.0
  params_override: str = None
  base_name: str = None
  data_type: str = None
  feature_list: str = None
  lagrangex_header: bool = False
  sort_id: bool = True
  kafka_dump: bool = False
  kafka_dump_prefix: bool = False
  restore_dir: str = None
  restore_ckpt: str = None
  deep_insight_target: str = None
  deep_insight_sample_ratio: float = None
  unified_serving: bool = False
  use_estimator: bool = False
  kafka_topics: str = None
  kafka_group_id: str = None
  kafka_servers: str = None
  input_path: str = None
  is_hourly: bool = False
  wildcard: str = None
  start_date: str = None
  end_date: str = None
  start_hour: int = None
  end_hour: int = None
  enable_dynamic_sharding: bool = False
  max_task_num_per_worker: int = 1

  def __post_init__(self):
    try:
      gflags_utils.update(self)
    except:
      logging.info("update RunnerConfig failed")

    if self.kafka_topics:
      if isinstance(self.kafka_topics, str):
        self.kafka_topics = self.kafka_topics.split(',')
      FLAGS.kafka_topics = ','.join(self.kafka_topics)
    if self.kafka_group_id:
      FLAGS.kafka_group_id = self.kafka_group_id
    if self.kafka_servers:
      FLAGS.kafka_servers = self.kafka_servers

    assert self.zk_watch_address_family in [
        AddressFamily.IPV4, AddressFamily.IPV6
    ]

    try:
      if self.restore_ckpt != FLAGS.restore_ckpt and FLAGS.restore_ckpt is None:
        FLAGS.restore_ckpt = self.restore_ckpt
    except flags._exceptions.UnparsedFlagAccessError:
      pass

    is_chief = self.is_local or (self.server_type == "worker" and
                                 self.index == 0)
    if self.restore_dir is not None and len(self.restore_dir) > 0:
      if is_chief:
        self._copy_ckpt_file()
      else:
        monolith_checkpoint_filename = os.path.join(
            self.model_dir, save_utils.MONOLITH_CKPT_STATE_FILE_NAME)
        while True:
          if tf.io.gfile.exists(monolith_checkpoint_filename):
            break
          logging.info("Waiting for chief setting up restore_dir...")
          time.sleep(30)

  def _copy_ckpt_file(self):
    logging.info(f"restore_dir is {self.restore_dir}")
    src_file = os.path.join(self.restore_dir, 'checkpoint')
    if tf.io.gfile.exists(src_file):
      if not tf.io.gfile.exists(self.model_dir):
        tf.io.gfile.makedirs(self.model_dir)
        logging.info(f"makedirs {self.model_dir} done!")

      # because we fix os.path.isabs, path startswith 'hdfs:/' is view as abs path
      # 1) get_checkpoint_state will add restore_dir for relative path to make a abs path
      #    if it is already abs path (including hdfs path), keep it as is
      # 2) for path start with 'hdfs:/' will seam as abs path, and do not add prefix any more
      try:
        restore_checkpoint_state = old_get_checkpoint_state(
            self.restore_dir)  # abs path
        if self.restore_ckpt is None:
          model_checkpoint_path = restore_checkpoint_state.model_checkpoint_path
        else:
          dirname = os.path.dirname(
              restore_checkpoint_state.model_checkpoint_path)
          basename = os.path.basename(self.restore_ckpt)
          model_checkpoint_path = os.path.join(dirname, basename)
          if model_checkpoint_path not in restore_checkpoint_state.all_model_checkpoint_paths:
            logging.warning(
                f'{model_checkpoint_path} is not in restore all_model_checkpoint_paths'
            )
            model_checkpoint_path = restore_checkpoint_state.model_checkpoint_path

        checkpoint_state = CheckpointState(
            model_checkpoint_path=model_checkpoint_path)
        checkpoint_state.all_model_checkpoint_paths.append(
            model_checkpoint_path)
      except Exception as e:
        logging.warning(e)
        return

      # we use the checkpoint file as a flag, if it exists, the restore_dir ckpt will not take action
      checkpoint_filename = os.path.join(self.model_dir, 'checkpoint')
      if tf.io.gfile.exists(checkpoint_filename):
        return

      try:
        file_io.atomic_write_string_to_file(
            checkpoint_filename, text_format.MessageToString(checkpoint_state))
        logging.info("write checkpoint file done!")

        # write the restore ckpt to monolith_checkpoint, so that the previous ckpts would not remove by ckpt mamager
        monolith_checkpoint_filename = os.path.join(
            self.model_dir, save_utils.MONOLITH_CKPT_STATE_FILE_NAME)
        monolith_ckpt_state = save_utils.get_monolith_checkpoint_state(
            self.restore_dir,
            remove_invalid_path=True) or MonolithCheckpointState()
        exempt_model_checkpoint_paths = monolith_ckpt_state.exempt_model_checkpoint_paths
        del exempt_model_checkpoint_paths[:]
        if tf.io.gfile.exists(monolith_checkpoint_filename):
          # in case there is a 'monolith_checkpoint' file
          file_content = file_io.read_file_to_string(
              monolith_checkpoint_filename)
          text_format.Merge(file_content, monolith_ckpt_state)

        for restore_ckpt_path in checkpoint_state.all_model_checkpoint_paths:
          if restore_ckpt_path not in exempt_model_checkpoint_paths:
            exempt_model_checkpoint_paths.append(restore_ckpt_path)

        file_io.atomic_write_string_to_file(
            monolith_checkpoint_filename,
            text_format.MessageToString(monolith_ckpt_state),
            overwrite=True)
        logging.info("write monolith checkpoint file done!")
      except Exception as e:
        logging.warning(e)
        logging.warning(f"checkpoint exist in {self.model_dir}")
    else:
      logging.warning(f"no checkpoint in {self.restore_dir}")


def get_discovery(runner_conf: RunnerConfig, psm: str = None):
  if runner_conf.is_local:
    discovery = None
  elif runner_conf.discovery_type == ServiceDiscoveryType.PRIMUS:
    assert runner_conf.tf_config is not None
    tf_config = json.loads(runner_conf.tf_config)
    discovery = TfConfigServiceDiscovery(tf_config)
    runner_conf.server_type = discovery.server_type
    runner_conf.index = discovery.index
  elif runner_conf.discovery_type == ServiceDiscoveryType.CONSUL:
    # For async training, PS discovery is inside the process.
    discovery = ConsulServiceDiscovery(psm)
  else:
    discovery = ZKServiceDiscovery(runner_conf.deep_insight_name,
                                   runner_conf.zk_server)

  return discovery


@contextmanager
def monolith_discovery(runner_conf: RunnerConfig):
  discovery = None
  try:
    if runner_conf.is_local:
      yield None
    else:
      from monolith.native_training import env_utils
      psm = env_utils.generate_psm_from_uuid(runner_conf.uuid)
      discovery = get_discovery(runner_conf, psm)

      logging.info('enter monolith_discovery!')
      yield discovery
  except Exception as e:
    raise e
  finally:
    if discovery is not None:
      discovery.close()

    logging.info('exit monolith_discovery!')
