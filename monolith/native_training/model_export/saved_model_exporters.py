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

import abc
import os
import time
from typing import Callable, Dict, List, Union

from absl import logging
from google.protobuf.any_pb2 import Any
import tensorflow as tf
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.framework import ops
from tensorflow.python.saved_model import constants
from tensorflow_estimator.python.estimator.export import export_lib

from monolith.native_training import device_utils
from monolith.native_training import hash_table_ops
from monolith.native_training import multi_hash_table_ops
from monolith.native_training import save_utils
from monolith.native_training.model_export import export_context
from monolith.native_training.monolith_checkpoint_state_pb2 import MonolithCheckpointState
from monolith.native_training.monolith_export import monolith_export
from monolith.native_training.model_export.data_gen_utils import gen_warmup_file
from monolith.native_training.model_dump.dump_utils import DumpUtils


class BaseExporter(abc.ABC):

  _ASSET_BASE = "ASSET_BASE"

  def __init__(self,
               model_fn: Callable,
               model_dir: str,
               export_dir_base: str,
               shared_embedding=False,
               warmup_file: str = None):
    """
    Args:    
      model_fn - the model fn which should have (features, mode, config) as args
      and return a EstimatorSpec
      shared_embedding - instead of exporting a hermetic SavedModel, we will use the embedding
      in the checkpoint instead of copying it.
      warmup_file - the warmup file name.
    """
    self._raw_model_fn = model_fn
    self._model_dir = model_dir
    self._export_dir_base = export_dir_base
    self._shared_embedding = shared_embedding
    self._warmup_file = warmup_file

  @staticmethod
  def create_asset_base():
    """
    This method returns a tensor which represents relative path of the assets folder.
    For example:
    We have a saved model here:
    /tmp/${USER}/saved_models/${model_name}/1622840665
    If we want to ref an asset with path:
    /tmp/${USER}/saved_models/${model_name}/1622840665/assets/MonolithHashTable_1
    We should use `tf.strings.join([create_asset_base(), "MonolithHashTable_1"])` as the asset path
    """

    try:
      return tf.compat.v1.get_default_graph().get_tensor_by_name(
          BaseExporter._ASSET_BASE + ":0")
    except KeyError:
      pass

    asset_dir = "./"
    asset_base = tf.convert_to_tensor(asset_dir,
                                      dtype=tf.string,
                                      name=BaseExporter._ASSET_BASE)
    asset_proto = meta_graph_pb2.AssetFileDef()
    asset_proto.filename = asset_dir
    asset_proto.tensor_info.name = asset_base.name
    asset_any_proto = Any()
    asset_any_proto.Pack(asset_proto)
    ops.add_to_collection(constants.ASSETS_KEY, asset_any_proto)
    return asset_base

  @staticmethod
  def add_ckpt_to_assets(ckpt_to_export, pattern="*"):
    hash_table_ckpts = tf.io.gfile.glob(
        save_utils.SaveHelper.get_ckpt_asset_dir(ckpt_to_export) + pattern)
    for hash_table_ckpt in hash_table_ckpts:
      logging.info(hash_table_ckpt)
      ops.add_to_collection(tf.compat.v1.GraphKeys.ASSET_FILEPATHS,
                            tf.convert_to_tensor(hash_table_ckpt))

  @staticmethod
  def build_signature(input_tensor_dict, output_tensor_dict):

    def ensure_tensor_info(maybe_tensor_info):
      if isinstance(maybe_tensor_info, meta_graph_pb2.TensorInfo):
        return maybe_tensor_info
      else:
        return tf.compat.v1.saved_model.utils.build_tensor_info(
            maybe_tensor_info)

    return tf.compat.v1.saved_model.build_signature_def(
        inputs={k: ensure_tensor_info(v) for k, v in input_tensor_dict.items()},
        outputs={
            k: ensure_tensor_info(v) for k, v in output_tensor_dict.items()
        },
        method_name=tf.compat.v1.saved_model.signature_constants.
        PREDICT_METHOD_NAME)

  def _export_saved_model_from_graph(
      self,
      graph: tf.Graph,
      checkpoint_path: str,
      export_dir_base: str = None,
      export_dir: str = None,
      restore_vars=True,
      restore_hashtable=True,
      assign_hashtable=True,
      export_ctx: export_context.ExportContext = None,
      export_tags=['serve'],
      assets_extra=None,
      clear_devices=False,
      strip_default_attrs=True) -> bytes:
    """
    Export saved_model from a user constructed graph, a graph can have multiple signatures
    Signautres are stored in export_ctx.signatures
    """

    assert export_dir or export_dir_base, "must provide export_dir or export_dir_base"

    if export_ctx is None:
      export_ctx = export_context.get_current_export_ctx()
    assert export_ctx is not None

    if not export_dir:
      export_dir = export_lib.get_timestamped_export_dir(export_dir_base)
    temp_export_dir = export_lib.get_temp_export_dir(export_dir)

    builder = tf.compat.v1.saved_model.Builder(temp_export_dir)
    with graph.as_default():
      tf.compat.v1.train.get_or_create_global_step(graph)
      signature_def_map = {}
      # Add signatures collected in the export_context
      for signature in export_ctx.signatures(graph):
        signature_def_map[signature.name] = BaseExporter.build_signature(
            signature.inputs, signature.outputs)

      if assign_hashtable:
        # assign signature
        assign_inputs, assign_outputs = self.build_hashtable_assign_inputs_outputs(
        )
        signature_def_map["hashtable_assign"] = BaseExporter.build_signature(
            assign_inputs, assign_outputs)
        self.add_multi_hashtable_assign_signatures(signature_def_map)
      '''
      To export CPU-trained saved_model for GPU serving, it requires explicit 
      GPU device placement at the exporting time. But it raised error here on 
      CPU-only machines while exporting graph with ops explicitly placed on GPU 
      (due to the necessity of loading the whole graph at runtime for calling save_op). 
      So we use the soft placement here, to avoid raising runtime exceptions, 
      but still successfully record the correct GPU placements to the saved_model.
      '''
      with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
          allow_soft_placement=True)) as session:

        graph_saver = tf.compat.v1.train.Saver(sharded=True)
        if restore_vars:
          try:
            graph_saver.restore(session, checkpoint_path)
          except tf.errors.NotFoundError as e:
            msg = ('Could not load all requested variables from checkpoint. '
                   'Please make sure your model_fn does not expect variables '
                   'that were not saved in the checkpoint.\n\n'
                   'Encountered error with mode `{}` while restoring '
                   'checkpoint from: `{}`. Full Traceback:\n\n{}').format(
                       tf.estimator.ModeKeys.PREDICT, checkpoint_path, e)
            raise ValueError(msg)
        restore_op = None
        if restore_hashtable:
          restore_op = tf.group(
              self.create_multi_hashtable_restore_ops(checkpoint_path) +
              self.create_hashtable_restore_ops(checkpoint_path))
        restore_op = restore_op or tf.no_op()
        meta_graph_kwargs = dict(tags=export_tags,
                                 signature_def_map=signature_def_map,
                                 assets_collection=tf.compat.v1.get_collection(
                                     tf.compat.v1.GraphKeys.ASSET_FILEPATHS),
                                 clear_devices=clear_devices,
                                 main_op=restore_op,
                                 saver=graph_saver,
                                 strip_default_attrs=strip_default_attrs)

        builder.add_meta_graph_and_variables(session, **meta_graph_kwargs)
      builder.save()

      # Add the extra assets
      if assets_extra:
        assets_extra_path = os.path.join(tf.compat.as_bytes(temp_export_dir),
                                         tf.compat.as_bytes('assets.extra'))
        for dest_relative, source in assets_extra.items():
          dest_absolute = os.path.join(tf.compat.as_bytes(assets_extra_path),
                                       tf.compat.as_bytes(dest_relative))
          dest_path = os.path.dirname(dest_absolute)
          tf.compat.v1.gfile.MakeDirs(dest_path)
          tf.compat.v1.gfile.Copy(source, dest_absolute)

      tf.io.gfile.rename(temp_export_dir, export_dir)
      return export_dir if isinstance(export_dir,
                                      bytes) else export_dir.encode()

  def create_hashtable_restore_ops(self, checkpoint_path):
    """
    Find all the hashtables in the current graph and create restore_op for them.
    When shared_embedding is False, it adds the hashtable ckpt files into the assets folder
    """
    ckpt_asset_base = save_utils.SaveHelper.get_ckpt_asset_dir(checkpoint_path)

    restore_ops = []
    for table in ops.get_collection(hash_table_ops._HASH_TABLE_GRAPH_KEY):
      tensor_prefix = hash_table_ops._table_tensor_prefix(table)
      share_embedding = self._shared_embedding if table.export_share_embedding is None else table.export_share_embedding
      if not share_embedding:
        BaseExporter.add_ckpt_to_assets(checkpoint_path,
                                        pattern=tensor_prefix + "*")
        asset_base = BaseExporter.create_asset_base()
      else:
        asset_base = ckpt_asset_base
      table_prefix = tf.strings.join([asset_base, tensor_prefix])
      table_prefix = tf.strings.join(
          [asset_base, hash_table_ops._table_tensor_prefix(table)])
      restore_ops.append(table.restore(table_prefix).as_op())
    return restore_ops

  def create_multi_hashtable_restore_ops(self, checkpoint_path):
    """
    Find all the multi-hashtables in the current graph and create restore_op for them.
    When shared_embedding is False, it adds the hashtable ckpt files into the assets folder
    """
    ckpt_asset_base = save_utils.SaveHelper.get_ckpt_asset_dir(checkpoint_path)

    restore_ops = []
    for table in ops.get_collection(
        multi_hash_table_ops._MULTI_HASH_TABLE_GRAPH_KEY):
      if not self._shared_embedding:
        BaseExporter.add_ckpt_to_assets(checkpoint_path,
                                        pattern=table.shared_name + "*")
        asset_base = BaseExporter.create_asset_base()
      else:
        asset_base = ckpt_asset_base
      table_basename = tf.strings.join([asset_base, table.shared_name])
      with tf.control_dependencies([table.initializer]):
        restore_op = table.restore(basename=table_basename).as_op()
      restore_ops.append(restore_op)
    return restore_ops

  def build_hashtable_assign_inputs_outputs(self):
    """
    For all hashtables in the current graph, create assign tensors for them
    """
    assign_input_tensors, assign_output_tensors = {}, {}
    for table in ops.get_collection(hash_table_ops._HASH_TABLE_GRAPH_KEY):
      assign_id = tf.compat.v1.placeholder(dtype=tf.int64, shape=(None,))
      assign_value = tf.compat.v1.placeholder(dtype=tf.float32,
                                              shape=(None, table.dim_size))
      assign_input_tensors[table.name + "_id"] = assign_id
      assign_input_tensors[table.name + "_value"] = assign_value
      updated_table = table.assign(assign_id, assign_value)
      with tf.control_dependencies(control_inputs=[updated_table.as_op()]):
        # The size of id tensor is returned here as a dummy value
        assign_output_tensors[table.name + "_result"] = tf.size(assign_id)
    return assign_input_tensors, assign_output_tensors

  def add_multi_hashtable_assign_signatures(self, signature_def_map: Dict):
    """
    For all hashtables in the current graph, create assign tensors for them
    """

    for table in ops.get_collection(
        multi_hash_table_ops._MULTI_HASH_TABLE_GRAPH_KEY):
      name = table.shared_name + "/raw_assign"
      assert name not in signature_def_map, f"{name} has already been defined in signature"
      input_tensors, output_tensors = dict(), dict()
      id = tf.compat.v1.placeholder(dtype=tf.int64, shape=(None,))
      input_tensors["id"] = id
      id_split = tf.compat.v1.placeholder(dtype=tf.int64, shape=(None,))
      input_tensors["id_split"] = id_split
      flat_value = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None,))
      input_tensors["flat_value"] = flat_value
      assign_op = table.raw_assign(
          tf.RaggedTensor.from_row_splits(id, id_split), flat_value).as_op()
      with tf.control_dependencies([assign_op]):
        dummy_tensor = tf.constant(0)
      output_tensors["result"] = dummy_tensor
      signature_def_map[name] = self.build_signature(input_tensors,
                                                     output_tensors)

  def _model_fn_with_input_reveiver(self, serving_input_receiver_fn):
    input_receiver = serving_input_receiver_fn()
    estimator_spec = self._raw_model_fn(input_receiver.features,
                                        mode=tf.estimator.ModeKeys.PREDICT,
                                        config=tf.estimator.RunConfig(
                                            self._model_dir))
    export_outputs = export_lib.get_export_outputs(
        estimator_spec.export_outputs, estimator_spec.predictions)
    signature_def_map = export_lib.build_all_signature_defs(
        input_receiver.receiver_tensors,
        export_outputs,
        getattr(input_receiver, 'receiver_tensors_alternatives', None),
        serving_only=True)
    for signature_name, signature in signature_def_map.items():
      export_context.get_current_export_ctx().add_signature(
          tf.compat.v1.get_default_graph(), signature_name, signature.inputs,
          signature.outputs)

  @abc.abstractmethod
  def export_saved_model(self,
                         serving_input_receiver_fn,
                         checkpoint_path=None,
                         global_step=None) -> Union[bytes, Dict[str, bytes]]:
    """
    Export the saved model and returns the path of exported model.
    Args:
      checkpoint_path - If None, the latest one will be used.
    """
    pass

  def gen_warmup_assets(self) -> Dict[str, str]:
    if not self._warmup_file:
      return None

    if not tf.io.gfile.exists(self._warmup_file):
      try:
        flag = gen_warmup_file(self._warmup_file)
        if flag is None:
          return None
        else:
          return {'tf_serving_warmup_requests': self._warmup_file}
      except Exception as e:
        logging.error(str(e))
        return None
    else:
      return {'tf_serving_warmup_requests': self._warmup_file}


@monolith_export
class StandaloneExporter(BaseExporter):
  """单机模式的saved model导出器
  
  Args:
    model_fn: 和tf.estimator兼容的model_fn, 以(features, mode, config)作为参数并且返回EstimatorSpec
    model_dir: 保存checkpoint的目录
    export_dir_base: 导出saved_model的目标路径
    shared_embedding: 是否复用checkpoint中的 embedding 文件, 
                      False的话会将embedding文件拷贝至saved_model, 可能会降低导出速度
    warmup_file: warmup文件, 参考 https://www.tensorflow.org/tfx/serving/saved_model_warmup
  
  """

  def __init__(self,
               model_fn: Callable,
               model_dir: str,
               export_dir_base: str,
               shared_embedding=False,
               warmup_file: str = None):
    super(StandaloneExporter,
          self).__init__(model_fn, model_dir, export_dir_base, shared_embedding,
                         warmup_file)

  def export_saved_model(self,
                         serving_input_receiver_fn,
                         checkpoint_path=None,
                         global_step=None):
    """ 导出saved_model
    
    Args:
      serving_input_receiver_fn: 
        返回 tf.estimator.export.ServingInputReceiver 的函数, 用来将serving 请求映射到模型输入
      checkpoint_path:
        可选的checkpoint路径, 为空则使用tf.train.latest_checkpoint(self._model_dir)
    """
    if not checkpoint_path:
      checkpoint_path = tf.train.latest_checkpoint(self._model_dir)

    with export_context.enter_export_mode(
        export_context.ExportMode.STANDALONE) as export_ctx:
      saved_tf_config = os.environ.pop("TF_CONFIG", None)
      try:
        with tf.Graph().as_default() as g:
          self._model_fn_with_input_reveiver(serving_input_receiver_fn)
          return self._export_saved_model_from_graph(
              g,
              checkpoint_path=checkpoint_path,
              export_dir_base=self._export_dir_base,
              export_ctx=export_ctx,
              assets_extra=self.gen_warmup_assets())
      finally:
        if saved_tf_config:
          os.environ["TF_CONFIG"] = saved_tf_config


@monolith_export
class DistributedExporter(BaseExporter):
  """分布式模型导出器
  
  Args: 
    model_fn: 和tf.estimator兼容的model_fn, 以(features, mode, config)作为参数并且返回EstimatorSpec   
    model_dir: 保存checkpoint的目录
    export_dir_base: 导出saved_model的目标路径
    shared_embedding: 是否复用checkpoint中的 embedding 文件, 
                      False的话会将embedding文件拷贝至saved_model, 可能会降低导出速度
    warmup_file: warmup文件, 参考 https://www.tensorflow.org/tfx/serving/saved_model_warmup
    include_graphs: Only export saved_models from include_graphs if the param not None, 
                    otherwise export all graphs in export context
    global_step_as_timestamp: whether to use use global_step export folder name, 
                              useful when we do parallel export in sync_training
  
  """

  def __init__(
      self,
      model_fn: Callable,
      model_dir: str,
      export_dir_base: str,
      shared_embedding=False,
      warmup_file: str = None,
      dense_only=False,
      allow_gpu=False,
      with_remote_gpu=False,
      clear_entry_devices=False,
      include_graphs: List[str] = None,
      global_step_as_timestamp: bool = False,
  ):
    super(DistributedExporter,
          self).__init__(model_fn, model_dir, export_dir_base, shared_embedding,
                         warmup_file)
    self._dense_only = dense_only
    self._allow_gpu = allow_gpu
    self._with_remote_gpu = with_remote_gpu
    self._clear_entry_devices = clear_entry_devices
    self._include_graphs = include_graphs
    self._global_step_as_timestamp = global_step_as_timestamp

  def _should_export(self, graph_name, export_dir):
    if tf.io.gfile.exists(export_dir):
      logging.info("skipping duplicated model exportings")
      return False
    return self._include_graphs is None or graph_name in self._include_graphs

  def export_saved_model(self,
                         serving_input_receiver_fn,
                         checkpoint_path=None,
                         global_step=None):
    """ 导出saved_model
    
    Args:
      serving_input_receiver_fn: 
        返回 tf.estimator.export.ServingInputReceiver 的函数, 用来将serving 请求映射到模型输入
      checkpoint_path:
        可选的checkpoint路径, 为空则使用tf.train.latest_checkpoint(self._model_dir)
    """
    if not checkpoint_path:
      checkpoint_path = tf.train.latest_checkpoint(self._model_dir)

    export_ctx = export_context.ExportContext(
        with_remote_gpu=self._with_remote_gpu)

    with export_context.enter_export_mode(export_context.ExportMode.DISTRIBUTED,
                                          export_ctx):

      saved_tf_config = os.environ.pop("TF_CONFIG", None)
      result = {}
      try:
        # Run model fn and export entry part
        if self._allow_gpu:
          device_utils.enable_gpu_training()
        with tf.Graph().as_default() as g, g.device(
            device_utils.default_device_fn):
          self._model_fn_with_input_reveiver(serving_input_receiver_fn)

          if global_step and self._global_step_as_timestamp:
            timestamp = str(global_step)
            entry_export_dir = os.path.join(self._export_dir_base, "entry",
                                            timestamp)
          else:
            entry_export_dir = export_lib.get_timestamped_export_dir(
                os.path.join(self._export_dir_base, "entry")).decode()
            timestamp = os.path.basename(entry_export_dir)

          if self._should_export("entry", entry_export_dir):
            result["entry"] = self._export_saved_model_from_graph(
                g,
                checkpoint_path=checkpoint_path,
                export_dir=entry_export_dir,
                restore_hashtable=False,
                assign_hashtable=False,
                export_ctx=export_ctx,
                assets_extra=self.gen_warmup_assets(),
                clear_devices=self._clear_entry_devices,
            )

        # Export additional dense graph stored in export_ctx
        DumpUtils().restore_sub_model('dense')
        for name, graph in export_ctx.dense_sub_graphs.items():
          DumpUtils().add_sub_model('dense', name, graph)
          ps_export_dir = os.path.join(
              self._export_dir_base, name,
              str(timestamp) + getattr(graph, "export_suffix", ""))
          if not self._should_export(name, ps_export_dir):
            continue
          result[name] = self._export_saved_model_from_graph(
              graph,
              checkpoint_path=checkpoint_path,
              export_dir=ps_export_dir,
              export_ctx=export_ctx,
          )

        if self._dense_only:
          return result

        # Export PS/GPU Dense from graph stored in export_ctx
        DumpUtils().restore_sub_model('ps')
        for name, graph in export_ctx.sub_graphs.items():
          DumpUtils().add_sub_model('ps', name, graph)
          ps_export_dir = os.path.join(
              self._export_dir_base, name,
              str(timestamp) + getattr(graph, "export_suffix", ""))
          if not self._should_export(name, ps_export_dir):
            continue
          result[name] = self._export_saved_model_from_graph(
              graph,
              checkpoint_path=checkpoint_path,
              export_dir=ps_export_dir,
              export_ctx=export_ctx,
          )
      finally:
        if saved_tf_config:
          os.environ["TF_CONFIG"] = saved_tf_config
    return result
