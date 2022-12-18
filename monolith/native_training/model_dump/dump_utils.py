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

from absl import logging
import tensorflow as tf
import copy
import pickle
from io import BytesIO
from inspect import signature, Parameter
from typing import Union, Dict, List, Optional, Any, Set

from tensorflow.python.data.ops import dataset_ops
from tensorflow_estimator.python.estimator import util
from tensorflow.core.framework import variable_pb2
from tensorflow.python.ops.variables import Variable
from tensorflow.python.framework import ops

SaveSliceInfo = Variable.SaveSliceInfo
from monolith.native_training import entry
from monolith.native_training.model_dump.monolith_model_pb2 import ProtoModel, ModelDump, \
  HashTableConfig, FeatureSliceDim, Combiner, FeatureCombiner
from monolith.native_training.model_dump.graph_utils import DatasetInitHook, GraphDefHelper, \
  DRY_RUN, _node_name
from monolith.native_training.data.utils import get_slot_feature_name
from monolith.native_training.embedding_combiners import ReduceMean, ReduceSum, FirstN
from monolith.native_training.runtime.hash_table import embedding_hash_table_pb2
from monolith.native_training.data.parsers import get_default_parser_ctx
from idl.matrix.proto.example_pb2 import OutConfig
from monolith.native_training.data.datasets import POOL_KEY
from monolith.native_training.model_export.export_context import get_current_export_ctx
from monolith.native_training.data.item_pool_hook import ItemPoolSaveRestoreHook
from monolith.native_training.data.feature_list import get_feature_name_and_slot


class DumpUtils(object):
  _instance = None

  def __new__(cls, *agrs, **kwds):
    if cls._instance is None:
      cls._instance = object.__new__(cls)

    return cls._instance

  def __init__(self, enable: bool = False):
    if not hasattr(self, "enable"):
      self.enable = enable
      self._params: Set[str] = set()
      self._run_config = None
      self._user_params = []
      self.train, self.train_graph = None, None
      self.infer, self.infer_graph = None, None
      self._ps_sub_model, self._dense_sub_model = {}, {}
      self._table_configs: List[HashTableConfig] = []
      self._feature_slice_dims: List[FeatureSliceDim] = []
      self._feature_combiners: List[FeatureCombiner] = []

  def add_config(self, run_config: str):
    self._run_config = run_config

  def add_user_params(self, user_params: List):
    self._user_params = user_params

  @property
  def model_dump(self) -> ProtoModel:
    graph = tf.compat.v1.get_default_graph()
    if hasattr(graph, 'monolith_model_dump'):
      monolith_model_dump = graph.monolith_model_dump
      return monolith_model_dump
    else:
      setattr(graph, 'monolith_model_dump', ProtoModel())
      return graph.monolith_model_dump

  def update_kwargs_with_default(self, func, kwargs):
    params = signature(func).parameters
    for key in kwargs:
      if (kwargs[key] is not None) or (key
                                       not in params) or (params[key].default
                                                          == Parameter.empty):
        continue
      kwargs[key] = params[key].default

  def record_feature(self, func):

    def wraper(*args, **kwargs):
      self.update_kwargs_with_default(func, kwargs)
      if self.need_record:
        proto = self.model_dump.features.add()
        if args:
          params = signature(func).parameters.values()
          for p, value in zip(params, args):
            if p.name == 'self':
              continue
            try:
              if p.name == 'feature_name' and isinstance(value, int):
                feature_name, _ = get_feature_name_and_slot(value)
                setattr(proto, p.name, feature_name)
              else:
                setattr(proto, p.name, value)
            except Exception as e:
              logging.warning(f"{p.name} is not in proto, {e}")

        for key, value in kwargs.items():
          try:
            if key == 'feature_name' and isinstance(value, int):
              feature_name, _ = get_feature_name_and_slot(value)
              setattr(proto, key, feature_name)
            elif value is not None:
              setattr(proto, key, value)
          except Exception as e:
            logging.warning(f"{key} is not in proto, func {func}, {e}")

      return func(*args, **kwargs)

    return wraper

  def record_slice(self, func):

    def warper(*args, **kwargs):
      self.update_kwargs_with_default(func, kwargs)
      if self.need_record:
        if kwargs.get('learning_rate_fn', None) is not None:
          raise Exception('for safety purpose learning_rate_fn is not allowed')
        proto = self.model_dump.emb_slices.add()
        if args:
          params = signature(func).parameters.values()
          for p, value in zip(params, args):
            if p.name == 'self':
              continue

            try:
              if value is not None:
                if p.name == 'features':
                  proto.features = repr(value)
                elif p.name in {'initializer', 'optimizer', 'compressor'}:
                  getattr(proto, p.name).CopyFrom(value.as_proto())
                else:
                  setattr(proto, p.name, value)
            except Exception as e:
              logging.warning(f"{p.name} is not in proto, {e}")

        for key, value in kwargs.items():
          try:
            if value is not None:
              if key == 'features':
                proto.features = repr(value)
              elif key in {'initializer', 'optimizer', 'compressor'}:
                getattr(proto, key).CopyFrom(value.as_proto())
              else:
                setattr(proto, key, value)
          except Exception as e:
            logging.warning(f"{key} is not in proto, func {func}, {e}")

        results = func(*args, **kwargs)

        if isinstance(results, (list, tuple)):
          for res in results:
            proto.output_tensor_names.append(res.name)
        else:
          proto.output_tensor_names.append(results.name)

        return results
      else:
        return func(*args, **kwargs)

    return warper

  def record_receiver(self, func):

    def warper(*args, **kwargs):
      self.update_kwargs_with_default(func, kwargs)
      receiver = func(*args, **kwargs)
      if self.need_record:
        proto = self.model_dump.serving_input_receiver_fn
        proto.parser_type = get_default_parser_ctx().parser_type
        for k, ts in receiver.features.items():
          if isinstance(ts, tf.RaggedTensor):
            proto.features[k] = repr({
                "values": ts.values.name,
                "row_splits": ts.row_splits.name,
                "is_ragged": True,
            })
          else:
            if len(ts.shape) > 0:
              last_dim = ts.shape[-1]
              if not isinstance(last_dim, int):
                if hasattr(last_dim, 'value'):
                  last_dim = last_dim.value
            else:
              last_dim = 0

            proto.features[k] = repr({
                "name": ts.name,
                "is_ragged": False,
                "dtype": ts.dtype,
                "last_dim": last_dim
            })

        for k, ts in receiver.receiver_tensors.items():
          proto.receiver_name[k] = ts.name

      return receiver

    return warper

  def record_params(self, model):
    if not self.need_record:
      return
    skip_attrs = {
        '_abc_impl', '_ctx', '_export_outputs', '_global_step', '_losses',
        '_occurrence_threshold', '_private_children', 'children', 'ctx',
        'fc_dict', 'fs_dict', 'losses', 'slice_dict', '_layout_dict',
        '_training_hooks'
    }
    for attr_name in dir(model):
      if attr_name.startswith('__') or attr_name in skip_attrs:
        continue
      attr = getattr(model, attr_name)
      if callable(attr):
        continue
      self._params.add(attr_name)

  def get_params_bytes(self, model) -> bytes:
    if not self.need_record:
      return

    params = {'_layout_dict'} | self._params
    model_params = {}
    for attr_name in params:
      attr = getattr(model, attr_name)
      if attr_name == 'p':
        params = copy.deepcopy(model.p)
        params.cls = None
        model_params['p'] = params
      elif attr_name == '_layout_dict':
        if attr:
          model_params[attr_name] = {
              name: out_cfg.SerializeToString()
              for name, out_cfg in attr.items()
          }
        else:
          model_params[attr_name] = attr
      else:
        model_params[attr_name] = attr

    f = BytesIO()
    pickle.dump(model_params, f)
    return f.getvalue()

  @classmethod
  def add_signature(cls, proto_model, graph: tf.Graph):
    export_ctx = get_current_export_ctx()
    if export_ctx:
      for signature in export_ctx.signatures(graph):
        signature_proto = proto_model.signature.add()
        signature_proto.name = signature.name
        for ip_key, value in signature.inputs.items():
          signature_proto.inputs[ip_key] = value.name
        for op_key, value in signature.outputs.items():
          signature_proto.outputs[op_key] = value.name

  @classmethod
  def restore_signature(cls, proto_model, graph: tf.Graph):
    export_ctx = get_current_export_ctx()
    if export_ctx:
      for signature in proto_model.signature:
        name = signature.name
        inputs = {
            ip_key: graph.get_tensor_by_name(value)
            for ip_key, value in signature.inputs.items()
        }
        outputs = {
            op_key: graph.get_tensor_by_name(value)
            for op_key, value in signature.outputs.items()
        }
        export_ctx.add_signature(graph, name, inputs, outputs)

  def add_model_fn(self,
                   model,
                   mode: str,
                   features: Dict[str, tf.Tensor],
                   label: Union[tf.Tensor, List[tf.Tensor], Dict[str, tf.Tensor]],
                   loss: Optional[tf.Tensor], 
                   pred: Union[tf.Tensor, List[tf.Tensor], Dict[str, tf.Tensor]],
                   head_name: Union[str, List[str]],
                   is_classification: Union[bool, List[bool]]):
    if not self.need_record: return

    model_dump = self.model_dump
    model_dump.params = self.get_params_bytes(model)
    model_fn = model_dump.model_fn
    if label is not None:
      if isinstance(label, (tuple, list)):
        model_fn.label.extend(['' if t is None else t.name for t in label])
      elif isinstance(label, dict):
        for value in label.values():
          model_fn.label.append('' if value is None else value.name)
      else:
        model_fn.label.append(label.name)
    if loss is not None:
      model_fn.loss = loss.name
    if isinstance(pred, (tuple, list)):
      model_fn.predict.extend([t.name for t in pred if t is not None])
    elif isinstance(pred, dict):
      for value in pred.values():
        model_fn.predict.append(value.name)
    else:
      model_fn.predict.append(pred.name)

    if head_name:
      if isinstance(head_name, str):
        model_fn.head_name.append(head_name)
      else:
        assert isinstance(head_name, (list, tuple))
        model_fn.head_name.extend(head_name)
    else:
      if label is not None and isinstance(label, dict):
        model_fn.head_name.extend(list(label.keys()))
      if isinstance(pred, dict):
        model_fn.head_name.extend(list(pred.keys()))

    if is_classification is not None:
      logging.info("dumped is_classification {}".format(is_classification))
      if isinstance(is_classification, bool):
        model_fn.classification.append(is_classification)
      else:
        assert isinstance(is_classification, list)
        model_fn.classification.extend(is_classification)
    
    summaries = [x.op.name for x in ops.get_collection(ops.GraphKeys.SUMMARIES)]
    if len(summaries) > 0:
      logging.info("dumped user summaries {}".format(summaries))
      model_fn.summary.extend(summaries)

    regged_features = {fc.feature_name for fc in self.model_dump.features}
    for name, ts in features.items():
      if name not in regged_features and not isinstance(ts, tf.RaggedTensor):
        model_fn.non_ragged_features[name] = ts.name

    graph = tf.compat.v1.get_default_graph()
    extra_losses = model_fn.extra_losses
    for ts in getattr(graph, '__losses', []):
      extra_losses.append(ts.name)

    export_outputs = getattr(graph, '__export_outputs', {})
    if export_outputs:
      for name, predict_output in export_outputs.items():
        extra_output = self.model_dump.extra_output.add()
        extra_output.signature_name = name
        outputs = predict_output.outputs
        if isinstance(outputs, dict):
          for key, ts in outputs.items():
            extra_output.fetch_dict[key] = ts.name
        else:
          extra_output.fetch_dict[ts.name] = ts.name

    training_hooks = getattr(graph, '__training_hooks', [])
    if training_hooks:
      if len(training_hooks) == 1 and isinstance(training_hooks[0],
                                                 ItemPoolSaveRestoreHook):
        pass
      else:
        raise Exception('For safety purpose, customer hooks is not allowed!')

    self.add_signature(model_dump, graph)

    variables = tf.compat.v1.all_variables()
    if variables:
      for v in variables:
        save_slice_info = v._get_save_slice_info()
        if save_slice_info is not None:
          save_slice_info_bytes = save_slice_info.to_proto().SerializeToString()
          model_dump.save_slice_info[_node_name(v.name)] = save_slice_info_bytes

    if hasattr(graph, 'monolith_model_dump'):
      graph_def = graph.as_graph_def()
      if mode == tf.estimator.ModeKeys.TRAIN:
        self.train = graph.monolith_model_dump
        self.train_graph = copy.deepcopy(graph_def)
      else:
        self.infer = graph.monolith_model_dump
        self.infer_graph = copy.deepcopy(graph_def)

  def add_input_fn(self, results: Dict[str, Union[tf.Tensor, tf.RaggedTensor]]):
    if not self.need_record:
      return

    input_fn = self.model_dump.input_fn
    if isinstance(results, (list, tuple)):
      features, label = results[0], results[1]
    else:
      features, label = results, None

    assert isinstance(features, dict)
    for key, ts in features.items():
      if isinstance(ts, tf.RaggedTensor):
        input_fn.output_features[key] = repr({
            "name": ts.values.name.split(':')[0],
            "is_ragged": True,
        })
      else:
        input_fn.output_features[key] = repr({
            "name": ts.name,
            "is_ragged": False
        })
    if label is not None:
      input_fn.label = label.name

    input_fn.parser_type = get_default_parser_ctx().parser_type
    pools = tf.compat.v1.get_collection(POOL_KEY)
    if pools:
      assert len(pools) == 1
      input_fn.item_pool = pools[0].name

  def add_sub_model(self, sub_model_type: str, name: str, graph: tf.Graph):
    if not self.need_record:
      return
    assert sub_model_type in {'ps', 'dense'}
    if sub_model_type == 'ps' and name in self._ps_sub_model:
      return
    if sub_model_type == 'dense' and name in self._dense_sub_model:
      return
    logging.info(f'add_sub_model: {sub_model_type}-{name}')

    proto = ProtoModel()
    graph_def = graph.as_graph_def()
    proto.graph_def = graph_def.SerializeToString()
    self.add_signature(proto, graph)

    if sub_model_type == 'ps':
      self._ps_sub_model[name] = proto
    elif sub_model_type == 'dense':
      self._dense_sub_model[name] = proto
    else:
      raise Exception('sub_model error!')

  def restore_sub_model(self, sub_model_type: str):
    export_ctx = get_current_export_ctx()
    if export_ctx:
      if sub_model_type == 'ps':
        for name, sub_model in self._ps_sub_model.items():
          with export_ctx.sub_graph(name).as_default() as g:
            sub_graph = tf.compat.v1.GraphDef()
            sub_graph.ParseFromString(sub_model.graph_def)
            tf.import_graph_def(sub_graph, name="")
            self.restore_signature(sub_model, g)
            logging.info(f'restore_sub_model: {sub_model_type}-{name}')
      elif sub_model_type == 'dense':
        for name, sub_model in self._dense_sub_model.items():
          with export_ctx.dense_sub_graph(name).as_default() as g:
            sub_graph = tf.compat.v1.GraphDef()
            sub_graph.ParseFromString(sub_model.graph_def)
            tf.import_graph_def(sub_graph, name="")
            self.restore_signature(sub_model, g)
            logging.info(f'restore_sub_model: {sub_model_type}-{name}')
      else:
        raise Exception(f'sub_model_type: {sub_model_type} error ')

  def add_optimizer(self, optimizer):
    if not self.need_record:
      return
    f = BytesIO()
    pickle.dump(optimizer, f)
    value = f.getvalue()
    self.model_dump.optimizer = value

  def dump(self, fname: str):
    if not self.enable:
      return

    md = ModelDump()
    if self._run_config:
      md.run_config = self._run_config

    if self._user_params:
      logging.info("xxx dump to md with user_params {}".format(
          self._user_params))
      md.user_params.extend(self._user_params)

    md.model_dump['train'].CopyFrom(self.train)
    md.model_dump['train'].graph_def = self.train_graph.SerializeToString()

    if self.infer is not None:
      md.model_dump['infer'].CopyFrom(self.infer)
      md.model_dump['infer'].graph_def = self.infer_graph.SerializeToString()

    for name, sub_model in self._ps_sub_model.items():
      if sub_model is not None:
        print('dump ps_sub_model: ', name, flush=True)
        md.ps_sub_model_dump[name].CopyFrom(sub_model)

    for name, sub_model in self._dense_sub_model.items():
      if sub_model is not None:
        print('dump dense_sub_model: ', name, flush=True)
        md.dense_sub_model_dump[name].CopyFrom(sub_model)

    for table_config in self._table_configs:
      md.table_configs.add().CopyFrom(table_config)

    for feature_slice_dim in self._feature_slice_dims:
      md.feature_slice_dims.add().CopyFrom(feature_slice_dim)

    for feature_combiner in self._feature_combiners:
      md.feature_combiners.add().CopyFrom(feature_combiner)

    with tf.io.gfile.GFile(fname, 'wb') as ostream:
      ostream.write(file_content=md.SerializeToString())

  def load(self, fname: str):
    with tf.io.gfile.GFile(fname, 'rb') as ostream:
      md = ModelDump()
      md.ParseFromString(ostream.read())
      self._run_config = md.run_config

      self.train = md.model_dump['train']
      train_graph = tf.compat.v1.GraphDef()
      train_graph.ParseFromString(self.train.graph_def)
      self.train_graph = train_graph
      self.train.graph_def = b''

      if 'infer' in md.model_dump:
        self.infer = md.model_dump['infer']
        infer_graph = tf.compat.v1.GraphDef()
        infer_graph.ParseFromString(self.infer.graph_def)
        self.infer_graph = infer_graph
        self.infer.graph_def = b''
      else:
        self.infer = None
        self.infer_graph = None

      self._table_configs.extend(md.table_configs)
      self._feature_slice_dims.extend(md.feature_slice_dims)
      self._feature_combiners.extend(md.feature_combiners)
      self._user_params.extend(md.user_params)
      logging.info("xxx load from md with user_params {}".format(
          self._user_params))

      for name, sub_model in md.ps_sub_model_dump.items():
        self._ps_sub_model[name] = sub_model

      for name, sub_model in md.dense_sub_model_dump.items():
        self._dense_sub_model[name] = sub_model

  def get_proto_model(self,
                      mode: str = tf.estimator.ModeKeys.TRAIN) -> ProtoModel:
    graph = tf.compat.v1.get_default_graph()
    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
      return self.train
    else:
      return self.infer

  def get_graph_helper(self, mode: str) -> GraphDefHelper:
    graph = tf.compat.v1.get_default_graph()
    if hasattr(graph, 'graph_def_helper'):
      return graph.graph_def_helper
    else:
      save_slice_info = {}
      if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
        for name, info_bytes in self.train.save_slice_info.items():
          save_slice_info_def = variable_pb2.SaveSliceInfoDef()
          save_slice_info_def.ParseFromString(info_bytes)
          save_slice_info[name] = SaveSliceInfo(
              save_slice_info_def=save_slice_info_def)
        helper = GraphDefHelper(self.train_graph, save_slice_info)
      else:
        for name, info_bytes in self.infer.save_slice_info.items():
          save_slice_info_def = variable_pb2.SaveSliceInfoDef()
          save_slice_info_def.ParseFromString(info_bytes)
          save_slice_info[name] = SaveSliceInfo(
              save_slice_info_def=save_slice_info_def)
        helper = GraphDefHelper(self.infer_graph, save_slice_info)

      setattr(graph, 'graph_def_helper', helper)
      return helper

  def restore_params(self) -> Dict[str, Any]:
    params = self.train.params
    if params is None or len(params) == 0:
      return None

    f = BytesIO(params)
    model_params = pickle.load(f)
    layout_dict = model_params.get('_layout_dict')
    if layout_dict:
      layout_dict_tmp = {}
      for name, out_cfg_str in layout_dict.items():
        out_cfg = OutConfig()
        out_cfg.ParseFromString(out_cfg_str)
        layout_dict_tmp[name] = out_cfg
      layout_dict.update(layout_dict_tmp)
    else:
      raise Exception('layout_dict is empty')
    if '_training_hooks' in model_params:
      del model_params['_training_hooks']
    return model_params

  def get_config(self):
    return self._run_config

  def get_user_params(self):
    return self._user_params

  @property
  def need_record(self) -> bool:
    graph = tf.compat.v1.get_default_graph()
    return self.enable and not hasattr(graph, DRY_RUN)

  @property
  def table_configs(self) -> Dict[str, entry.HashTableConfigInstance]:
    result = {}
    for tcfg in self._table_configs:
      table_config = embedding_hash_table_pb2.EmbeddingHashTableConfig()
      table_config.ParseFromString(tcfg.table_config)
      result[tcfg.name] = entry.HashTableConfigInstance(
          table_config, list(tcfg.learning_rates),
          list(tcfg.extra_restore_names))
    return result

  @table_configs.setter
  def table_configs(self, table_confs: Dict[str,
                                            entry.HashTableConfigInstance]):
    if table_confs:
      assert isinstance(table_confs, dict)
      self._table_configs.clear()
      for name, tcfg in table_confs.items():
        hash_table_config = HashTableConfig(
            name=name, table_config=tcfg._table_config.SerializeToString())
        if tcfg.extra_restore_names:
          hash_table_config.extra_restore_names.extend(tcfg.extra_restore_names)
        if tcfg.learning_rate_fns:
          if all(isinstance(lr, (float, int)) for lr in tcfg.learning_rate_fns):
            hash_table_config.learning_rates.extend(tcfg.learning_rate_fns)
          else:
            raise Exception('learning_rate_fn is not support!')
        self._table_configs.append(hash_table_config)

  @property
  def feature_slice_dims(self) -> Dict[str, List[int]]:
    slice_dims: Dict[str, List[int]] = {}
    for fsd in self._feature_slice_dims:
      slice_dims[fsd.name] = list(fsd.dims)
    return slice_dims

  @feature_slice_dims.setter
  def feature_slice_dims(self, slice_dims: Dict[str, List[int]]):
    if slice_dims:
      assert isinstance(slice_dims, dict)
      self._feature_slice_dims.clear()
      for name, dims in slice_dims.items():
        fsd = FeatureSliceDim(name=name)
        if dims:
          fsd.dims.extend(dims)

        self._feature_slice_dims.append(fsd)

  @property
  def feature_combiners(self):
    fcombs = {}
    for fcomb in self._feature_combiners:
      if fcomb.combiner == Combiner.ReduceSum:
        fcombs[fcomb.name] = ReduceSum()
      elif fcomb.combiner == Combiner.ReduceMean:
        fcombs[fcomb.name] = ReduceMean()
      else:
        fcombs[fcomb.name] = FirstN(seq_length=fcomb.max_seq_length)
    return fcombs

  @feature_combiners.setter
  def feature_combiners(self, fcombs):
    if fcombs:
      assert isinstance(fcombs, dict)
      self._feature_combiners.clear()
      for name, fcomb in fcombs.items():
        fc_proto = FeatureCombiner(name=name,
                                   max_seq_length=fcomb.max_seq_length)
        if isinstance(fcomb, ReduceSum):
          fc_proto.combiner = Combiner.ReduceSum
        elif isinstance(fcomb, ReduceMean):
          fc_proto.combiner = Combiner.ReduceMean
        else:
          fc_proto.combiner = Combiner.FirstN

        self._feature_combiners.append(fc_proto)

  def get_slot_to_occurrence_threshold(self, mode: str = 'train'):
    if mode == tf.estimator.ModeKeys.TRAIN:
      model_dump = self.train
    else:
      model_dump = self.infer
    slot_to_ot = {}
    for feature in model_dump.features:
      feature_name, slot = get_feature_name_and_slot(feature.feature_name)
      if slot is not None:
        slot_to_ot[slot] = feature.occurrence_threshold
      else:
        logging.warning(
            "feature[{}] slot is None. pls check feature_list.conf".format(
                feature_name))
    return slot_to_ot

  def get_slot_to_expire_time(self, mode: str = 'train'):
    if mode == tf.estimator.ModeKeys.TRAIN:
      model_dump = self.train
    else:
      model_dump = self.infer
    slot_to_et = {}
    for feature in model_dump.features:
      feature_name, slot = get_feature_name_and_slot(feature.feature_name)
      if slot is not None:
        slot_to_et[slot] = feature.expire_time
      else:
        logging.warning(
            "feature[{}] slot is None. pls check feature_list.conf".format(
                feature_name))
    return slot_to_et

  @property
  def has_collected(self) -> bool:
    if self._table_configs and self._feature_slice_dims and self._feature_combiners:
      return True
    else:
      assert self._table_configs is None or len(self._table_configs) == 0
      assert self._feature_slice_dims is None or len(
          self._feature_slice_dims) == 0
      assert self._feature_combiners is None or len(
          self._feature_combiners) == 0
      return False


def parse_input_fn_result(result):
  input_hooks = []
  if isinstance(result, dataset_ops.DatasetV2):
    iterator = dataset_ops.make_initializable_iterator(result)
    input_hooks.append(util._DatasetInitializerHook(iterator))
    result = iterator.get_next()
  else:
    initializer = tf.compat.v1.get_collection('mkiter')
    if isinstance(initializer, (list, tuple)) and len(initializer) > 0:
      initializer = initializer[0]
    assert initializer is not None
    input_hooks.append(DatasetInitHook(initializer))

  DumpUtils().add_input_fn(result)
  return util.parse_iterator_result(result) + (input_hooks,)


util.parse_input_fn_result = parse_input_fn_result
