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
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import os
import copy
import numpy as np
import collections
import getpass
from typing import Dict, List
from kazoo.client import KazooClient
from typing import Optional, Union, get_type_hints

import tensorflow as tf

from monolith.native_training import env_utils
from monolith.agent_service.utils import AgentConfig
from monolith.agent_service.backends import ZKBackend
from monolith.native_training.zk_utils import default_zk_servers
from monolith.agent_service.replica_manager import ReplicaWatcher
from monolith.native_training.cpu_training import CpuTraining, create_exporter
from monolith.native_training.runner_utils import RunnerConfig, monolith_discovery
from monolith.native_training.cpu_training import local_train_internal
from monolith.native_training.cpu_training import distributed_train
from monolith.native_training.service_discovery import ServiceDiscoveryType
from monolith.native_training.monolith_export import monolith_export
from monolith.native_training.model_dump.dump_utils import DumpUtils
from monolith.core.hyperparams import InstantiableParams

from monolith.native_training.data.item_pool_hook import ItemPoolSaveRestoreHook
from monolith.native_training.utils import set_metric_prefix
from monolith.native_training.data.parsers import get_default_parser_ctx, ParserCtx
from monolith.native_training.model_export.export_context import \
  is_exporting, is_exporting_distributed, ExportMode
from monolith.native_training.zk_utils import MonolithKazooClient


@monolith_export
class EstimatorSpec(
    collections.namedtuple(
        'EstimatorSpec',
        ['label', 'pred', 'head_name', 'loss', 'optimizer', 'classification'])):
  """EstimatorSpec???model_fn?????????????????????.
  
  Args:
    label (:obj:`Union[tf.Tensor, List[tf.Tensor]]`): ????????????, multi-head??????????????????
    pred (:obj:`Union[tf.Tensor, List[tf.Tensor]]`): ????????????, multi-head??????????????????
    head_name (:obj:`Union[str, List[str]]`): predict??????, multi-head??????????????????
    loss (:obj:`tf.Tensor`): ??????
    optimizer (:obj:`tf.Optimizer`): dense??????????????????
    classification (:obj:`Union[bool, List[bool]]`): ?????????????????????, multi-head???????????????
  
  """

  def __new__(cls,
              label,
              pred,
              head_name=None,
              loss=None,
              optimizer=None,
              classification=True):
    return super(EstimatorSpec, cls).__new__(cls,
                                             label=label,
                                             pred=pred,
                                             head_name=head_name,
                                             loss=loss,
                                             optimizer=optimizer,
                                             classification=classification)

  def _replace(self, **kwds):
    """Return a new EstimatorSpec replacing specified fields with new values."""
    if 'mode' in kwds:
      if self.mode != kwds['mode']:
        raise ValueError('mode of EstimatorSpec cannot be changed.')
    new_fields = map(kwds.pop, self._fields, list(self))
    return EstimatorSpec(*new_fields)


@monolith_export
@dataclass_json
@dataclass
class RunConfig:
  """Estimator????????????, ?????????????????????????????????
  
  Args: 
    is_local (:obj:`bool`): ?????????????????????, ?????????False
    num_ps (:obj:`int`): PS??????
    num_workers (:obj:`int`): Woeker??????
    chief_timeout_secs (:obj:`int`): chief ????????????, ????????? 1800???
    operation_timeout_in_ms (:obj:`int`): ??????????????????, ????????? 600000, ???600s
    session_creation_timeout_secs (:obj:`int`): session??????????????????, ?????????7200???
    enable_fused_layout (:obj:`bool`): ????????????layout??????, ????????????
    partial_recovery (:obj:`bool`): ????????????????????????
    max_retry_times (:obj:`int`): ???????????????, ??????????????????, ????????? 6
    retry_wait_in_secs (:obj:`int`): ???????????????, ??????????????????, ????????? 5
    bzid (:obj:`str`): serving ??????id, ?????????????????????????????????Online PS
    base_name (:obj:`str`): serving ?????????, ?????????????????????????????????Online PS
    ps_replica_num (:obj:`int`): serving PS ?????????
    enable_parameter_sync (:obj:`bool`): ????????????????????????, ?????????False
    model_dir (:obj:`str`): ??????dump??????
    restore_dir (:obj:`str`): ??????????????????, ???dump??????????????????????????????????????????, ?????????model_dir???????????????
    restore_ckpt (:obj:`str`): ??????checkpoint??????, ?????????????????????
    save_checkpoints_secs (:obj:`int`): ????????????????????????checkpoint
    save_checkpoints_steps (:obj:`int`): ????????????step?????????checkpoint
    max_rpc_deadline_millis (:obj:`int`): prc????????????, ??????30???
    dense_only_save_checkpoints_secs (:obj:`int`): ????????????????????????dense??????checkpoint
    dense_only_save_checkpoints_steps (:obj:`int`): ????????????step?????????dense??????checkpoint
    checkpoints_max_to_keep (:obj:`int`): ?????????????????????checkpoint
    warmup_file (:obj:`str`): serving warmup?????????
    enable_local_profiling (:obj:`bool`): ???????????????????????? profiling
    use_native_multi_hash_table (:obj:`bool`): ????????????????????????????????????2023-1-1??????
    clear_nn (:obj:`bool`): ?????????reload????????????dense?????????????????????, ?????????false.
    continue_training (:obj:`bool`): ???clear_nn???true???, global_step??????????????????, ?????????false.
    reload_alias_map (:obj:`dict`): ??????????????????, ????????????????????????, ?????????????????????, ?????????reload_alias_map?????????????????????????????????
    enable_alias_map_auto_gen: ???????????????????????? alias_map
  """

  # basic
  is_local: bool = False
  num_ps: int = 0
  num_workers: int = 1

  chief_timeout_secs: int = 1800
  operation_timeout_in_ms: int = -1
  session_creation_timeout_secs: int = 7200
  enable_fused_layout: bool = False
  enable_model_dump: bool = False

  # failover
  partial_recovery: bool = False
  max_retry_times: int = 6
  retry_wait_in_secs: int = 5

  # for params sync
  bzid: str = None
  base_name: str = None
  ps_replica_num: int = None
  enable_parameter_sync: bool = False

  # checkpoint and export
  model_dir: str = ""
  restore_dir: str = None
  restore_ckpt: str = None
  save_checkpoints_secs: int = None
  save_checkpoints_steps: int = None
  max_rpc_deadline_millis: int = 30000
  dense_only_save_checkpoints_secs: int = None
  dense_only_save_checkpoints_steps: int = None
  checkpoints_max_to_keep: int = 10

  warmup_file: str = './warmup_file'
  enable_local_profiling: bool = False

  use_native_multi_hash_table: bool = None

  clear_nn: bool = False
  continue_training: bool = False
  reload_alias_map: Dict[str, int] = None
  enable_alias_map_auto_gen: bool = None

  kafka_topics: str = None
  kafka_group_id: str = None
  kafka_servers: str = None

  def to_runner_config(self) -> RunnerConfig:
    conf = RunnerConfig(
        restore_dir=self.restore_dir,
        restore_ckpt=self.restore_ckpt,
        model_dir=self.model_dir,
        enable_fused_layout=self.enable_fused_layout,
        enable_model_dump=self.enable_model_dump,
        warmup_file=self.warmup_file,
        enable_alias_map_auto_gen=self.enable_alias_map_auto_gen,
        kafka_topics=self.kafka_topics,
        kafka_group_id=self.kafka_group_id,
        kafka_servers=self.kafka_servers)
    for name, _ in get_type_hints(RunConfig).items():
      value = getattr(self, name)
      if hasattr(conf, name) and value is not None:
        default = getattr(RunConfig, name)
        # must be and, because RunnerConfig value can be writen by command line
        # we cannot use default value in RunConfig to overwrite command line value
        if value != default and getattr(conf, name) != value:
          setattr(conf, name, value)
    # in case US tearm use CONSUL
    conf.discovery_type = ServiceDiscoveryType.ZK

    # set default value for embedding prefetch/postpush
    if conf.embedding_prefetch_capacity <= 0:
      conf.embedding_prefetch_capacity = 1
    if not conf.enable_embedding_postpush:
      conf.enable_embedding_postpush = True

    # [todo] remove this when enable_realtime_training changed to enable_parameter_sync
    if self.enable_parameter_sync:
      if hasattr(conf, 'enable_realtime_training'):
        conf.enable_realtime_training = True
      elif hasattr(conf, 'enable_parameter_sync'):
        conf.enable_parameter_sync = True
      else:
        raise RuntimeError("enable_parameter_sync not set!")
    return conf

  def __post_init__(self):
    ser_data = self.to_json()
    DumpUtils().add_config(ser_data)
    # get user params
    params = vars(self)
    user_params = []
    for name, _ in get_type_hints(RunConfig).items():
      default_value = getattr(RunConfig, name)
      if default_value != params[name]:
        logging.info("save user param {} with value {}".format(
            name, params[name]))
        user_params.append(name)
    DumpUtils().add_user_params(user_params)


@monolith_export
class Estimator(object):
  """??????Estimator????????????local?????????????????????????????????, ??????, Estimator?????????????????????/save/restore??????, ??????hooks, ???summary???
  
  Args:
    model (:obj:`Model`): NativeModel??????
    conf (:obj:`RunConfig`): ???????????????????????????
    warm_start_from (:obj:`str`): ?????????saved_model???, ????????????warmup??????. warm_start_from????????????warmup???????????????, ??????????????????
  
  """

  def __init__(self,
               model,
               conf: Union[RunConfig, RunnerConfig],
               warm_start_from=None):
    self._runner_conf = conf.to_runner_config() if isinstance(
        conf, RunConfig) else conf
    self._model = model
    self._task = None
    self._warm_start_from = warm_start_from
    self._sync_backend = None
    self._kazoo_client = None

    if isinstance(conf, RunConfig):
      self._enable_loacl_profiling = conf.enable_local_profiling
    else:
      self._enable_loacl_profiling = False

    logging.info(self._runner_conf)
    if self._runner_conf.is_local:
      # local mode cannot asscess deep_insight
      self._model.metrics.enable_deep_insight = False
    else:
      self._model.metrics.enable_deep_insight = True
      if self._runner_conf.deep_insight_name:
        self._model.metrics.deep_insight_name = self._runner_conf.deep_insight_name
      if self._runner_conf.deep_insight_target:
        self._model.metrics.deep_insight_target = self._runner_conf.deep_insight_target
      if self._runner_conf.deep_insight_sample_ratio:
        self._model.metrics.deep_insight_sample_ratio = self._runner_conf.deep_insight_sample_ratio

    if self._runner_conf.enable_realtime_training and self._runner_conf.server_type == 'ps':
      assert self._runner_conf.bzid, "Business id cannot be none while realtime training."
      assert self._runner_conf.base_name, "Base name cannot be none while realtime training."
      zk_servers = self._runner_conf.zk_server or os.environ.get(
          'zk_servers', default_zk_servers())
      if self._runner_conf.unified_serving:
        self._sync_backend = ZKBackend(self._runner_conf.bzid,
                                       zk_servers=zk_servers)
      else:
        assert self._runner_conf.base_name, "Base name cannot be none while realtime training."
        self._kazoo_client = MonolithKazooClient(hosts=zk_servers)
        self._kazoo_client.start()
        agent_config = AgentConfig(bzid=self._runner_conf.bzid,
                                   base_name=self._runner_conf.base_name,
                                   deploy_type='ps',
                                   num_ps=self._runner_conf.num_ps,
                                   dc_aware=self._runner_conf.dc_aware)
        replica_watcher = ReplicaWatcher(
            self._kazoo_client,
            agent_config,
            zk_watch_address_family=self._runner_conf.zk_watch_address_family)
        self._sync_backend = replica_watcher.to_sync_wrapper()

    if self._runner_conf.params_override:
      logging.info("Override: {}".format(self._runner_conf.params_override))
      params_override_dict = json.loads(self._runner_conf.params_override)
      if hasattr(model, 'p'):
        model.p.set(**params_override_dict)
      elif hasattr(model, 'params'):
        model.params.set(**params_override_dict)
      else:
        logging.warning('params_override error!')

    try:
      if not os.environ.get("HADOOP_HDFS_HOME"):
        env_utils.setup_hdfs_env()
    except Exception as e:
      logging.error('setup_hdfs_env fail {}!'.format(e))

    os.environ["TF_GRPC_WORKER_CACHE_THREADS"] = str(
        self._runner_conf.tf_grpc_worker_cache_threads)
    os.environ["MONOLITH_GRPC_WORKER_SERVICE_HANDLER_MULTIPLIER"] = str(
        self._runner_conf.monolith_grpc_worker_service_handler_multiplier)

    # private attr
    self.__est: Optional[tf.estimator.Estimator] = None

  @property
  def _sess_config(self):
    return self._est._session_config

  @property
  def model_dir(self):
    return self._runner_conf.model_dir

  @property
  def config(self):
    return self._est._config

  @property
  def _est(self):
    if self.__est is None:
      model = copy.deepcopy(self._model)
      model.mode = tf.estimator.ModeKeys.PREDICT
      self._task = CpuTraining(config=self._runner_conf,
                               task=model.instantiate())

      # the default estimate for predict/export_saved_model/import_saved_model
      if 'TF_CONF' in os.environ:
        del os.environ['TF_CONF']

      self.__est = tf.estimator.Estimator(
          self._task.create_model_fn(),
          model_dir=self._runner_conf.model_dir,
          config=tf.estimator.RunConfig(log_step_count_steps=1),
          warm_start_from=self._warm_start_from)

    return self.__est

  def _init_fountain_env(self):
    if self._model.train.use_fountain and bool(
        self._runner_conf.fountain_zk_host) and bool(
            self._runner_conf.fountain_model_name):
      logging.info("Override Fountain Params:{}; {}".format(
          self._runner_conf.fountain_model_name,
          self._runner_conf.fountain_zk_host))
      self._model.train.fountain_zk_host = self._runner_conf.fountain_zk_host
      self._model.train.fountain_model_name = self._runner_conf.fountain_model_name

  def close(self):
    if self._sync_backend is not None:
      try:
        self._sync_backend.stop()
      except Exception as e:
        logging.error(e)

    if self._kazoo_client is not None:
      try:
        self._kazoo_client.stop()
      except Exception as e:
        logging.info(e)

      try:
        self._kazoo_client.close()
      except Exception as e:
        logging.info(e)

  def get_variable_value(self, name):
    return self._est.get_variable_value(name)

  def get_variable_names(self):
    return self._est.get_variable_names()

  def latest_checkpoint(self):
    return self._est.latest_checkpoint()

  def train(self, steps=None, max_steps=None):
    set_metric_prefix("monolith.training.{}".format(
        self._runner_conf.deep_insight_name))
    model = copy.deepcopy(self._model)
    if not isinstance(model, InstantiableParams):
      model.file_name = self._model.file_name
    model.mode = tf.estimator.ModeKeys.TRAIN

    if steps is not None:
      model.train.steps = steps
    if max_steps is not None:
      model.train.max_steps = max_steps
    self._init_fountain_env()
    if self._runner_conf.is_local:
      if not self._runner_conf.model_dir:
        model_dir = "/tmp/{}/{}".format(getpass.getuser(), model.name)
      else:
        model_dir = self._runner_conf.model_dir
      DumpUtils().record_params(model)
      self.__est = local_train_internal(model,
                                        self._runner_conf,
                                        model_dir=model_dir,
                                        steps=steps,
                                        profiling=self._enable_loacl_profiling)
      DumpUtils().dump(f'{self._runner_conf.model_dir}/model_dump')
    else:
      DumpUtils().enable = False
      logging.info(f'{model.p}')
      with monolith_discovery(self._runner_conf) as discovery:
        if self._sync_backend is not None:
          self._sync_backend.start()
          self._sync_backend.subscribe_model(self._runner_conf.model_name or
                                             model.metrics.deep_insight_name)
        logging.info("Environment vars: %s", os.environ)
        logging.info("Flags: %s", flags.FLAGS.flag_values_dict())
        self.__est = distributed_train(config=self._runner_conf,
                                       discovery=discovery,
                                       params=model,
                                       sync_backend=self._sync_backend)
    self.close()

  def evaluate(self, steps=None):
    model = copy.deepcopy(self._model)
    model.mode = tf.estimator.ModeKeys.EVAL
    if not isinstance(model, InstantiableParams):
      model.file_name = self._model.file_name
    self._runner_conf.mode = tf.estimator.ModeKeys.EVAL
    if steps is not None:
      model.train.steps = steps
    self._init_fountain_env()
    if self._runner_conf.is_local:
      DumpUtils().record_params(model)
      if not self._runner_conf.model_dir:
        model_dir = "/tmp/{}/{}".format(getpass.getuser(), model.name)
      else:
        model_dir = self._runner_conf.model_dir
      self.__est = local_train_internal(model,
                                        self._runner_conf,
                                        model_dir=model_dir,
                                        steps=steps,
                                        profiling=self._enable_loacl_profiling)
      DumpUtils().dump(f'{self._runner_conf.model_dir}/model_dump')
    else:
      DumpUtils().enable = False
      logging.info(f'{model.p}')
      logging.info("Environment vars: %s", os.environ)
      logging.info("Flags: %s", flags.FLAGS.flag_values_dict())
      with monolith_discovery(self._runner_conf) as discovery:
        self.__est = distributed_train(self._runner_conf,
                                       discovery,
                                       model,
                                       sync_backend=self._sync_backend)
    self.close()

  def predict(self,
              predict_keys=None,
              hooks=None,
              checkpoint_path=None,
              yield_single_examples=True):
    est = self._est  # create tf estimator
    input_fn = self._task.create_input_fn(tf.estimator.ModeKeys.PREDICT)
    est.predict(input_fn, predict_keys, hooks, checkpoint_path,
                yield_single_examples)
    self.close()

  def export_saved_model(self,
                         batch_size=64,
                         name=None,
                         dense_only: bool = False,
                         enable_fused_layout: bool = False):
    model = copy.deepcopy(self._model)
    runner_conf = copy.deepcopy(self._runner_conf)
    runner_conf.enable_fused_layout = enable_fused_layout
    model.name = name or "demo_export"
    model.train.per_replica_batch_size = batch_size
    model.mode = tf.estimator.ModeKeys.PREDICT

    model_dir = runner_conf.model_dir
    export_dir_base = os.path.join(model_dir, model.serving.export_dir_base)
    warmup_file = runner_conf.warmup_file
    task = CpuTraining(config=runner_conf, task=model.instantiate())
    exporter = create_exporter(task, model_dir, warmup_file, export_dir_base,
                               dense_only)
    serving_input_receiver_fn = task.create_serving_input_receiver_fn()
    with ParserCtx(enable_fused_layout=enable_fused_layout):
      exporter.export_saved_model(serving_input_receiver_fn)


@monolith_export
def import_saved_model(saved_model_path: str,
                       input_name: str = "instances",
                       output_name: str = 'output',
                       signature: str = None):
  """??????saved_model
  
  Args:
    saved_model_path (:obj:`str`): saved_model??????
  
  """

  class saved_model(object):

    def __init__(self, saved_model_path, signature, inputs, outputs):
      basename = os.path.basename(saved_model_path)
      if not basename.isnumeric():
        versions = []
        for subitem in tf.io.gfile.listdir(saved_model_path):
          if subitem.isnumeric():
            versions.append(int(subitem))

        if versions:
          versions.sort()
          saved_model_path = os.path.join(saved_model_path, str(versions[-1]))
        else:
          raise RuntimeError(f"no models in dir {saved_model_path}")

      self._saved_model_path = saved_model_path
      if signature:
        self._signature = signature
      else:
        self._signature = tf.compat.v1.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY

      if inputs:
        self._inputs = inputs if isinstance(inputs, (list, tuple)) else [inputs]
      else:
        self._inputs = None

      if outputs:
        self._outputs = outputs if isinstance(outputs,
                                              (list, tuple)) else [outputs]
      else:
        self._outputs = None

    def __enter__(self):

      class infer(object):

        def __init__(self, graph, sess, placeholders, output_dict,
                     output_name_map):
          self._graph = graph
          self._sess = sess

          self._placeholders = placeholders
          self._output_dict = output_dict
          self._output_name_map = output_name_map

        def __call__(self, features: Dict[str, np.ndarray]) -> List[np.ndarray]:
          with self._graph.as_default(), self._sess.as_default():
            if len(self._placeholders) == 1:
              placeholders = next(iter(self._placeholders.values()))
              result = sess.run(self._output_dict,
                                feed_dict={placeholders: features})
            else:
              result = sess.run(self._output_dict,
                                feed_dict={
                                    self._placeholders[name]: feature
                                    for name, feature in features.items()
                                })
            return {
                self._output_name_map[key]: tensor
                for key, tensor in result.items()
            }

      tag = tf.compat.v1.saved_model.tag_constants.SERVING
      graph = tf.compat.v1.Graph()
      sess = tf.compat.v1.Session(graph=graph)
      with graph.as_default(), sess.as_default():
        imported = tf.compat.v1.saved_model.load(sess, {tag},
                                                 self._saved_model_path)
        print(imported.signature_def, flush=True)
        signature_def = imported.signature_def[self._signature]

        placeholders: Dict[str, tf.compat.v1.placeholder] = {}
        for input_name in self._inputs:
          input_ph_name = signature_def.inputs[input_name].name
          input_ph = graph.get_tensor_by_name(input_ph_name)
          placeholders[input_name] = input_ph

        output_dict, output_name_map = {}, {}
        if self._outputs:
          for output_name in self._outputs:
            output_tensor_name = signature_def.outputs[output_name].name
            output_tensor = graph.get_tensor_by_name(output_tensor_name)
            if output_tensor_name.endswith(':0'):
              output_tensor_name = output_tensor_name[0:-2]
            output_dict[output_tensor_name] = output_tensor
            output_name_map[output_tensor_name] = output_name
        else:
          for output_name, tensor in signature_def.outputs.items():
            output_tensor_name = tensor.name
            output_tensor = graph.get_tensor_by_name(output_tensor_name)
            if output_tensor_name.endswith(':0'):
              output_tensor_name = output_tensor_name[0:-2]
            output_dict[output_tensor_name] = output_tensor
            output_name_map[output_tensor_name] = output_name

        logging.info('import_saved_model finished')

        return infer(graph, sess, placeholders, output_dict, output_name_map)

    def __exit__(self, exc_type, exc_val, exc_tb):
      logging.info('exit import_saved_model')

  return saved_model(saved_model_path, signature, input_name, output_name)
