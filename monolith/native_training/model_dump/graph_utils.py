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

import os, re, time
import six, copy, time
from absl import logging
from inspect import signature
import pickle
from io import BytesIO
from typing import Dict, Any, Optional, Union, List, Set
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from google.protobuf import text_format
from tensorflow.keras import initializers
from tensorflow.python.framework import ops
from tensorflow.python.ops.variables import Variable

SaveSliceInfo = Variable.SaveSliceInfo

from idl.matrix.proto.line_id_pb2 import LineId
from tensorflow.python.ops.variables import PartitionedVariable
from tensorflow.python.ops.gen_resource_variable_ops import read_variable_op
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.framework import ops
from monolith.native_training.utils import add_to_collections

DRY_RUN = 'dry_run'


class DatasetInitHook(tf.compat.v1.train.SessionRunHook):

  def __init__(self, initializer):
    self._initializer = initializer

  def after_create_session(self, session, coord):
    del coord
    session.run(self._initializer)


def _node_name(name):
  if name.startswith("^"):
    return name[1:]
  else:
    return name.split(":")[0]


def _colocated_node_name(name):
  """Decodes colocated node name and returns it without loc:@ prepended."""
  colocated_node_decoded = name.decode("utf-8")
  if colocated_node_decoded.startswith("loc:@"):
    return colocated_node_decoded[5:]
  return colocated_node_decoded


class EchoInitializer(tf.keras.initializers.Initializer):

  def __init__(self, init_value):
    self._init_value = init_value

  def __call__(self, shape, dtype=None, **kwargs):
    if isinstance(self._init_value, (list, tuple)):
      assert len(self._init_value) == 1
      init_value = self._init_value[0]
    else:
      init_value = self._init_value

    if isinstance(init_value, tf.Tensor):
      return init_value
    else:
      assert len(init_value.outputs) == 1
      return init_value.outputs[0]


class VariableDef(object):

  def __init__(self,
               node: tf.compat.v1.NodeDef = None,
               helper: 'GraphDefHelper' = None):
    self.node = node
    self._helper = helper
    self._name_to_node = helper.name_to_node
    self._read_ops: List[tf.compat.v1.NodeDef] = []

    self._variable = None
    self._initializer = None

  @property
  def dtype(self):
    return tf.dtypes.DType(self.node.attr['dtype'].type)

  @property
  def shape(self):
    return tuple(dim.size for dim in self.node.attr['shape'].shape.dim)

  @property
  def device(self):
    return self.node.device

  @property
  def name(self):
    return _node_name(self.node.name)

  @property
  def initializer(self):
    if self._initializer is None:
      assign = _node_name(f'{self.node.name}/Assign')
      assert assign in self._name_to_node
      assign_node = self._name_to_node[assign]
      assert len(assign_node.input) == 2

      initializer = None
      vname = _node_name(self.node.name)
      for name in assign_node.input:
        if _node_name(name) != vname:
          initializer = name
          break
      assert initializer is not None

      sub_graph, _ = self._helper.sub_graph(dest_nodes=[initializer],
                                            source_nodes=None,
                                            with_library=False)
      init_ops = tf.compat.v1.import_graph_def(sub_graph,
                                               return_elements=[initializer],
                                               name="")
      self._initializer = EchoInitializer(init_ops)

    return self._initializer

  @property
  def variable(self):
    if self._variable is None:
      vs = tf.compat.v1.get_variable_scope()
      partitioner = vs._partitioner
      vs._partitioner = None
      if self.device is not None and len(self.device) > 0:
        with tf.compat.v1.device(self.device):
          self._variable = tf.compat.v1.get_variable(
              dtype=self.dtype,
              shape=self.shape,
              initializer=self.initializer,
              name=self.node.name)
      else:
        self._variable = tf.compat.v1.get_variable(dtype=self.dtype,
                                                   shape=self.shape,
                                                   initializer=self.initializer,
                                                   name=self.node.name)
      vs._partitioner = partitioner
      if isinstance(self._variable, PartitionedVariable):
        self._variable = self._variable._variable_list[0]

    return self._variable

  def add_read(self, node: tf.compat.v1.NodeDef):
    assert node.op == 'ReadVariableOp'
    self._read_ops.append(node)

  @property
  def read_ops(self):
    return self._read_ops


class PartitionVariableDef(object):
  PVName = re.compile(r'^(.*)/part_\d+$')

  def __init__(self, base_name: str, helper: 'GraphDefHelper' = None):
    self.base_name = base_name
    self._helper = helper
    self._name_to_node = helper.name_to_node
    self._partitions: Dict[str, tf.compat.v1.NodeDef] = {}
    self._read_ops: Dict[str, List[tf.compat.v1.NodeDef]] = {}

    self._variable = None
    self._initializer = None
    self._partitioned_variable = None
    self._save_slice_info = self._helper._save_slice_info

  def add_partition(self, node: tf.compat.v1.NodeDef):
    assert node.op == 'VarHandleOp'
    self._partitions[_node_name(node.name)] = node

  def add_read(self, node: tf.compat.v1.NodeDef):
    assert node.op == 'ReadVariableOp'
    name = _node_name(node.input[0])
    if name in self._read_ops:
      self._read_ops[name].append(node)
    else:
      self._read_ops[name] = [node]

  @classmethod
  def get_base_name(cls, node: tf.compat.v1.NodeDef) -> Optional[str]:
    name, op = node.name, node.op
    if op == "VarHandleOp":
      matched = cls.PVName.match(name)
      if matched:
        return matched.group(1)
    elif op == "ReadVariableOp":
      inputs = [name for name in node.input if not name.startswith('^')]
      assert len(inputs) == 1
      matched = cls.PVName.match(inputs[0])
      if matched:
        return matched.group(1)
    return None

  @property
  def dtype(self):
    return [
        tf.dtypes.DType(node.attr['dtype'].type)
        for node in self._partitions.values()
    ]

  @property
  def shape(self):
    return [
        tuple(dim.size
              for dim in node.attr['shape'].shape.dim)
        for node in self._partitions.values()
    ]

  @property
  def device(self):
    return [node.device for node in self._partitions.values()]

  @property
  def initializer(self):
    if self._initializer is None:
      if len(self._partitions) > 1:
        dest_nodes = [
            _node_name(f'{pname}/PartitionedInitializer/Slice')
            for pname in self._partitions
        ]
      else:
        node_name = _node_name(f'{next(iter(self._partitions))}')
        slice_node_name = f'{node_name}/PartitionedInitializer/Slice'
        if slice_node_name in self._name_to_node:
          dest_nodes = [slice_node_name]
        else:
          assign_node_name = f'{node_name}/Assign'
          assert assign_node_name in self._name_to_node
          assign_node = self._name_to_node[assign_node_name]
          assert len(assign_node.input) == 2

          initializer = None
          for name in assign_node.input:
            if _node_name(name) != node_name:
              initializer = name
              break
          assert initializer is not None
          dest_nodes = [initializer]

      sub_graph, _ = self._helper.sub_graph(dest_nodes=dest_nodes,
                                            source_nodes=None,
                                            with_library=False)
      init_ops = tf.compat.v1.import_graph_def(sub_graph,
                                               return_elements=dest_nodes,
                                               name="")
      self._initializer = [EchoInitializer(init_op) for init_op in init_ops]
    return self._initializer

  @property
  def variable(self):
    if self._variable is None:
      self._variable = {}
      dtypes, shapes, inits = self.dtype, self.shape, self.initializer
      group_device = None
      vs = tf.compat.v1.get_variable_scope()
      partitioner = vs._partitioner
      vs._partitioner = None
      for i, (name, device) in enumerate(zip(self._partitions, self.device)):
        group_device = device if i == 0 else group_device
        if group_device is not None and len(group_device) > 0:
          with tf.compat.v1.device(None):
            with tf.compat.v1.device(group_device):
              variable = tf.compat.v1.get_variable(dtype=dtypes[i],
                                                   shape=shapes[i],
                                                   initializer=inits[i],
                                                   name=name)
              if isinstance(variable, PartitionedVariable):
                variable = variable._variable_list[0]
              else:
                save_slice_info = self._save_slice_info.get(name)
                variable._set_save_slice_info(save_slice_info)
        else:
          variable = tf.compat.v1.get_variable(dtype=dtypes[i],
                                               shape=shapes[i],
                                               initializer=inits[i],
                                               name=name)
          if isinstance(variable, PartitionedVariable):
            variable = variable._variable_list[0]
          else:
            save_slice_info = self._save_slice_info.get(name)
            variable._set_save_slice_info(save_slice_info)
          group_device = variable.device

        self._variable[name] = variable
      vs._partitioner = partitioner

      # make PartitionedVariable, to check save_slice_info
      if len(self._variable) > 1:
        names = sorted(self._variable,
                       key=lambda x: int(_node_name(x).rsplit('_')[-1]))
        name = names[0].rsplit('/', maxsplit=1)[0]
        partitions = [
            len(shapes) if i == 0 else 1 for i, d in enumerate(shapes[0])
        ]
        first_dim = sum(s[0] for s in shapes)
        if len(shapes[0]) > 1:
          shape = [first_dim] + list(shapes[0][1:])
        else:
          shape = [first_dim]
        self._partitioned_variable = PartitionedVariable(
            name=name,
            shape=shape,
            dtype=dtypes[0],
            variable_list=[self._variable[name] for name in names],
            partitions=partitions)

    return self._variable

  def read_ops(self, pname: str) -> List[tf.compat.v1.NodeDef]:
    return self._read_ops[pname]


class GraphDefHelper(object):

  def __init__(self, graph_def: tf.compat.v1.GraphDef,
               save_slice_info: Dict[str, SaveSliceInfo]):
    if not isinstance(graph_def, tf.compat.v1.GraphDef):
      raise TypeError("graph_def must be a graph_pb2.GraphDef proto.")

    self.graph_def = graph_def
    self.name_to_vardef: Dict[str, Union[VariableDef,
                                         PartitionVariableDef]] = {}
    self.name_to_input_name: Dict[str,
                                  Set[str]] = {}  # Keyed by the dest node name.
    self.name_to_node: Dict[str, tf.compat.v1.NodeDef] = {}
    self.seq_num_to_node: Dict[int, tf.compat.v1.NodeDef] = {}
    self.name_to_seq_num: Dict[str, int] = {}
    self._save_slice_info = save_slice_info
    self._file_name = None

    seq = 0
    for node in graph_def.node:
      node.device = b''
      name = _node_name(node.name)
      self.name_to_input_name[name] = set(_node_name(x) for x in node.input)
      if "_class" in node.attr:
        for colocated_node_name in node.attr["_class"].list.s:
          self.name_to_input_name[name].add(
              _colocated_node_name(colocated_node_name))
        del node.attr["_class"]
      self.name_to_node[name] = node
      self.name_to_seq_num[name] = seq
      self.seq_num_to_node[seq] = node
      seq += 1

      if node.name == "PBDataset/file_name" and node.op == "Const":
        self._file_name = node

      stop_names = {'global_step', 'WorkerCkptMetaInfo'}
      if node.op == "VarHandleOp" and name not in stop_names:
        base_name = PartitionVariableDef.get_base_name(node)
        if base_name is not None:
          if base_name in self.name_to_vardef:
            self.name_to_vardef[base_name].add_partition(node)
          else:
            pvd = PartitionVariableDef(base_name, self)
            pvd.add_partition(node)
            self.name_to_vardef[base_name] = pvd
        else:
          if name in self.name_to_vardef:
            if self.name_to_vardef[name].node is None:
              self.name_to_vardef[name].node = node
            else:
              logging.info("maybe error, because node is not None")
          else:
            self.name_to_vardef[name] = VariableDef(node, self)

      if node.op == "ReadVariableOp":
        inputs = [name for name in node.input if not name.startswith('^')]
        assert len(inputs) == 1
        vname = inputs[0]
        if vname in stop_names:
          continue

        base_name = PartitionVariableDef.get_base_name(node)
        if base_name is not None:
          if base_name in self.name_to_vardef:
            self.name_to_vardef[base_name].add_read(node)
          else:
            pvd = PartitionVariableDef(base_name, self)
            pvd.add_read(node)
            self.name_to_vardef[base_name] = pvd
        else:
          base_name = _node_name(node.input[0])
          if base_name in self.name_to_vardef:
            self.name_to_vardef[base_name].add_read(node)
          else:
            dummy = VariableDef(None, self)
            dummy.add_read(node)
            self.name_to_vardef[base_name] = dummy

  @property
  def library(self):
    return self.graph_def.library

  @property
  def versions(self):
    return self.graph_def.versions

  @classmethod
  def _check_invalidate_node(clz, graph_def: tf.compat.v1.GraphDef,
                             input_map: Dict[str, tf.Tensor]):
    if input_map is None or len(input_map) == 0:
      return

    exists = set()
    for node in graph_def.node:
      for ts_name in node.input:
        if ts_name.startswith('^'):
          ts_name = ts_name[1:]

        exists.add(ts_name)
        if ":" not in ts_name:
          exists.add(f'{ts_name}:0')

    invalidate = set(input_map.keys()) - exists
    for name in invalidate:
      del input_map[name]
      logging.warning(f"{name} is not used in model")

  def _create_variables(self, variables: Set[str]) -> Dict[str, tf.Tensor]:
    vread_map = {}
    graph = tf.compat.v1.get_default_graph()
    for vardef in self.name_to_vardef.values():
      if isinstance(vardef, PartitionVariableDef):
        # remove variable that outside the graph
        if len(set(vardef._partitions.keys()) - variables) != 0:
          continue
        part_var = vardef.variable
        for pname, v in part_var.items():
          # v._handle -> Tensor("dense/kernel/part_0:0", shape=(), dtype=resource)
          # v.value() -> Tensor("ReadVariableOp:0", shape=(48, 512), dtype=float32)
          # v.read_value() -> Tensor("Identity:0", shape=(48, 512), dtype=float32)
          # graph.get_tensor_by_name(f'{pname}:0') -> Tensor("dense/kernel/part_0:0",
          #                                                  shape=(), dtype=resource)
          for reader in vardef.read_ops(pname):
            if reader.name == f'{reader.input[0]}/Read/ReadVariableOp':
              continue
            vread_map[reader.name] = read_variable_op(resource=v._handle,
                                                      dtype=v.dtype,
                                                      name=_node_name(
                                                          reader.name))
      else:
        # remove variable that outside the graph
        if vardef.name not in variables:
          continue

        v = vardef.variable
        for reader in vardef.read_ops:
          if reader.name == f'{reader.input[0]}/Read/ReadVariableOp':
            continue
          vread_map[reader.name] = read_variable_op(resource=v._handle,
                                                    dtype=v.dtype,
                                                    name=_node_name(
                                                        reader.name))
    return vread_map

  def sub_graph(self,
                dest_nodes: List[str],
                source_nodes: Optional[List[str]] = None,
                with_library: bool = True):
    if isinstance(dest_nodes, six.string_types):
      raise TypeError("dest_nodes must be a list.")

    if source_nodes is not None:
      source_nodes = list(set([_node_name(sn) for sn in source_nodes]))
      for sn in source_nodes:
        assert sn in self.name_to_node, f"{sn} is not in graph"

    dest_nodes = list(set([_node_name(dn) for dn in dest_nodes]))
    for dn in dest_nodes:
      assert dn in self.name_to_node, f"{dn} is not in graph"

    # Breadth first search to find all the nodes that we should keep.
    nodes_to_keep = set()
    stop_nodes = set() if source_nodes is None else set(source_nodes)
    next_to_visit = dest_nodes[:]
    while next_to_visit:
      node = next_to_visit[0]
      del next_to_visit[0]
      if node in nodes_to_keep or node in stop_nodes:
        # Already visited/stop this node.
        continue
      nodes_to_keep.add(node)
      if node in self.name_to_input_name:
        next_to_visit += list(self.name_to_input_name[node])
    nodes_to_keep_list = sorted(list(nodes_to_keep),
                                key=lambda name: self.name_to_seq_num[name])

    # Now construct the output GraphDef
    sub_gd = tf.compat.v1.GraphDef()
    variables = set()
    for n in nodes_to_keep_list:
      node = self.name_to_node[n]
      if node.op not in {"VarHandleOp", "ReadVariableOp"}:
        sub_gd.node.extend([copy.deepcopy(node)])
      elif node.op == "VarHandleOp":
        variables.add(n)

    if with_library:
      func_names = set()
      for node in sub_gd.node:
        for key, value in node.attr.items():
          if value.func is not None:
            name = value.func.name
            if name:
              func_names.add(name)
      for func in self.graph_def.library.function:
        if "Dataset" in func.signature.name or func.signature.name in func_names:
          ofunc = sub_gd.library.function.add()
          ofunc.CopyFrom(func)
          for node_def in ofunc.node_def:
            node_def.device = b''
    # out.versions.CopyFrom(self.graph_def.versions)

    return sub_gd, variables

  def import_input_fn(self, input_conf, file_name: str):
    graph = tf.compat.v1.get_default_graph()
    dry_run: bool = hasattr(graph, DRY_RUN)

    dest_nodes = []
    for feat_name, ts_repr in input_conf.output_features.items():
      ts_dict = eval(ts_repr)
      dest_nodes.append(ts_dict['name'])
    if input_conf.label is not None and len(input_conf.label) > 0:
      dest_nodes.append(input_conf.label)
    if not dry_run:
      if "IteratorToStringHandle" in self.name_to_node:
        dest_nodes.append("IteratorToStringHandle")
      if "MakeIterator" in self.name_to_node:
        dest_nodes.append("MakeIterator")
    del self._file_name.attr['value'].tensor.string_val[:]
    file_name_bytes = bytes(file_name, encoding='utf-8')
    self._file_name.attr['value'].tensor.string_val.append(file_name_bytes)
    sub_graph, _ = self.sub_graph(dest_nodes=dest_nodes,
                                  source_nodes=None,
                                  with_library=True)
    return_elements = tf.import_graph_def(sub_graph,
                                          input_map=None,
                                          return_elements=dest_nodes,
                                          name="")
    if not dry_run:
      if "IteratorToStringHandle" in self.name_to_node:
        idx = dest_nodes.index("IteratorToStringHandle")
        tf.compat.v1.add_to_collection("iterators", return_elements[idx])
      if "MakeIterator" in self.name_to_node:
        idx = dest_nodes.index("MakeIterator")
        tf.compat.v1.add_to_collection("mkiter", return_elements[idx])

    result = {}
    for i, (feat_name,
            ts_repr) in enumerate(input_conf.output_features.items()):
      ts_dict = eval(ts_repr)
      if ts_dict['is_ragged']:
        row_splits, values = return_elements[i].outputs
        assert ts_dict['name'] == values.name.split(':')[0]
        row_partition = RowPartition.from_row_splits(row_splits=tf.reshape(
            row_splits, shape=(-1,)),
                                                     validate=False,
                                                     preferred_dtype=None)
        result[feat_name] = tf.RaggedTensor(tf.reshape(values, shape=(-1,)),
                                            row_partition,
                                            internal=True)
      else:
        assert ts_dict['name'] == return_elements[i].name
        result[feat_name] = return_elements[i]

    if input_conf.label is not None and len(input_conf.label) > 0:
      idx = dest_nodes.index(input_conf.label)
      result['label'] = return_elements[idx]
    return result

  def import_model_fn(self, input_map: Dict[str, tf.Tensor], proto_model):
    source_nodes = list(input_map.keys()) if input_map else None

    model_fn = proto_model.model_fn
    outputs = list(model_fn.predict)
    if len(proto_model.extra_output) > 0:
      for extra_output in proto_model.extra_output:
        for ts_name in extra_output.fetch_dict.values():
          node_name = _node_name(ts_name)
          full_name = ts_name if ':' in ts_name else f'{node_name}:0'
          if node_name not in outputs and full_name not in outputs:
            outputs.append(ts_name)
    if model_fn.loss is not None and len(model_fn.loss) > 0:
      outputs.append(model_fn.loss)
    if model_fn.label is not None and len(model_fn.label) > 0:
      outputs.extend([l for l in model_fn.label if l])
    for extra_loss in model_fn.extra_losses:
      if extra_loss not in outputs:
        outputs.append(extra_loss)
    signature_input_names = []
    if len(proto_model.signature) > 0:
      for signature in proto_model.signature:
        for ts_name in signature.inputs.values():
          if ts_name not in signature_input_names:
            signature_input_names.append(ts_name)
        for ts_name in signature.outputs.values():
          node_name = _node_name(ts_name)
          full_name = ts_name if ':' in ts_name else f'{node_name}:0'
          if node_name not in outputs and full_name not in outputs:
            outputs.append(ts_name)

    if len(model_fn.summary) > 0:
      logging.info("load user summaries {}".format(model_fn.summary))
      outputs.extend(list(model_fn.summary))
      summaries = model_fn.summary

    sub_graph, variables = self.sub_graph(dest_nodes=outputs,
                                          source_nodes=source_nodes,
                                          with_library=True)
    self._check_invalidate_node(sub_graph, input_map)
    vread_map = self._create_variables(variables)
    if input_map is not None and len(input_map) > 0:
      vread_map.update(input_map)

    # check input_map for import_graph_def
    nodes = {_node_name(node.name) for node in sub_graph.node}
    graph = tf.compat.v1.get_default_graph()
    if vread_map:
      ts_names = set()
      for node in sub_graph.node:
        for ip_ts_name in node.input:
          if _node_name(ip_ts_name) not in nodes:
            ts_names.add(ip_ts_name)
      vread_map = {
          key if key in ts_names else _node_name(key): value
          for key, value in vread_map.items()
          if key in ts_names or _node_name(key) in ts_names
      }
      unknown_input = ts_names - set(vread_map)
      if unknown_input:
        logging.info(f"Debug. unknown_input {unknown_input}")
        for ts_name in unknown_input:
          vread_map[ts_name] = graph.get_tensor_by_name(
              ts_name if ':' in ts_name else f'{ts_name}:0')
    else:
      vread_map = None

    # in case some output tensor not include in graph
    direct_out, real_out = {}, []
    for op_ts_name in outputs:
      if _node_name(op_ts_name) not in nodes:
        try:
          direct_out[op_ts_name] = graph.get_tensor_by_name(
              op_ts_name if ':' in op_ts_name else f'{op_ts_name}:0')
        except:
          logging.warning(f'Cannot find {op_ts_name} in both graph and inputs')
          direct_out[op_ts_name] = None
      else:
        real_out.append(op_ts_name)
    real_result = tf.import_graph_def(sub_graph,
                                      input_map=vread_map,
                                      return_elements=real_out,
                                      name="")
    real_result = {name: value for name, value in zip(real_out, real_result)}
    result = [real_result.get(name, direct_out.get(name)) for name in outputs]
    if len(model_fn.summary) > 0:
      for summary in summaries:
        for sum_ts in real_result.get(summary).outputs:
          logging.info("[INFO] add summary {} to collection".format(sum_ts))
          ops.add_to_collection(ops.GraphKeys.SUMMARIES, sum_ts)

    # check sig_input in graph
    for op_ts_name in signature_input_names:
      graph.get_tensor_by_name(op_ts_name if ':' in
                               op_ts_name else f'{op_ts_name}:0')

    if model_fn.label is not None and len(model_fn.label) > 0:
      labels = [None] * len(model_fn.label)
      for i, label in enumerate(model_fn.label):
        if label:
          idx = outputs.index(label)
          labels[i] = result[idx]
      label = labels if len(labels) > 1 else labels[0]
    else:
      label = None

    if model_fn.loss is not None and len(model_fn.loss) > 0:
      idx = outputs.index(model_fn.loss)
      loss = result[idx]
    else:
      loss = None

    predict = result[:len(list(model_fn.predict))]
    if len(predict) == 1:
      predict = predict[0]

    if model_fn.head_name is not None and len(model_fn.head_name) > 0:
      if len(model_fn.head_name) == 1:
        head_name = model_fn.head_name[0]
      else:
        head_name = list(model_fn.head_name)
    else:
      head_name = None

    if model_fn.classification is not None:
      logging.info("load is_classificaiton {}".format(model_fn.classification))
      if len(model_fn.classification) == 1:
        is_classification = model_fn.classification[0]
      else:
        is_classification = list(model_fn.classification)

    extra_output_dict = {}
    if len(proto_model.extra_output) > 0:
      for extra_output in proto_model.extra_output:
        real_extra = {}
        for key, ts_name in extra_output.fetch_dict.items():
          idx = outputs.index(ts_name)
          real_extra[key] = result[idx]

        if len(extra_output.fetch_dict) == 1 and key == result[idx].name:
          extra_output_dict[extra_output.signature_name] = next(
              iter(real_extra.values()))
        else:
          extra_output_dict[extra_output.signature_name] = real_extra

    return label, loss, predict, head_name, extra_output_dict, is_classification

  def import_receiver_fn(self, receiver_conf):
    dest_nodes, sparse_features, dense_features, extra_features = [], [], [], []
    dense_feature_shapes, dense_feature_types, extra_feature_shapes = [], [], []
    for feat_name, ts_repr in receiver_conf.features.items():
      ts_dict = eval(ts_repr)
      if ts_dict['is_ragged']:
        dest_nodes.append(ts_dict['values'])
        dest_nodes.append(ts_dict['row_splits'])
        sparse_features.append(feat_name)
      else:
        dest_nodes.append(ts_dict['name'])
        if hasattr(LineId, feat_name):
          extra_features.append(feat_name)
          extra_feature_shapes.append(ts_dict['last_dim'])
        else:
          dense_features.append(feat_name)
          dense_feature_types.append(ts_dict['dtype'])
          dense_feature_shapes.append(ts_dict['last_dim'])
    add_to_collections('sparse_features', sparse_features)
    add_to_collections('dense_features', dense_features)
    add_to_collections('dense_feature_shapes', dense_feature_shapes)
    add_to_collections('dense_feature_types', dense_feature_types)
    add_to_collections('extra_features', extra_features)
    add_to_collections('extra_feature_shapes', extra_feature_shapes)

    num_feature_tensors = len(dest_nodes)
    for name, ph_name in receiver_conf.receiver_name.items():
      dest_nodes.append(ph_name)

    sub_graph, _ = self.sub_graph(dest_nodes=dest_nodes,
                                  source_nodes=None,
                                  with_library=True)
    return_elements = tf.import_graph_def(sub_graph,
                                          input_map=None,
                                          return_elements=dest_nodes,
                                          name="")

    idx, features = 0, {}
    for feat_name, ts_repr in receiver_conf.features.items():
      ts_dict = eval(ts_repr)
      if ts_dict['is_ragged']:
        values, row_splits = return_elements[idx], return_elements[idx + 1]
        features[feat_name] = tf.RaggedTensor.from_row_splits(values,
                                                              row_splits,
                                                              validate=False)
        idx += 2
      else:
        features[feat_name] = return_elements[idx]
        idx += 1

    receiver_tensors = {}
    for name in receiver_conf.receiver_name:
      receiver_tensors[name] = return_elements[idx]
      idx += 1

    return features, receiver_tensors

  @classmethod
  def get_optimizer(cls, proto_model):
    ser_opt = proto_model.optimizer
    if ser_opt is not None and len(ser_opt) > 0:
      f = BytesIO(ser_opt)
      return pickle.load(f)
    else:
      return None
