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

from collections import namedtuple
from collections import defaultdict
from enum import Enum
from typing import List

import tensorflow as tf
from tensorflow.python.util import tf_contextlib
from monolith.native_training.monolith_export import monolith_export
from monolith.native_training.utils import add_to_collections


class ExportMode(Enum):
  NONE = 0
  STANDALONE = 1
  DISTRIBUTED = 2


SavedModelSignature = namedtuple('SavedModelSignature',
                                 ['name', 'inputs', 'outputs'])


@monolith_export
class ExportContext:
  """保存模型导出的上下文"""

  def __init__(self, with_remote_gpu=False):
    self._sub_graphs = defaultdict(lambda: tf.Graph())
    self._dense_sub_graphs = defaultdict(lambda: tf.Graph())
    self._signatures = defaultdict(lambda: {})
    self._with_remote_gpu = with_remote_gpu

  def sub_graph(self, name: str) -> tf.Graph:
    return self._sub_graphs[name]

  def dense_sub_graph(self, name: str) -> tf.Graph:
    return self._dense_sub_graphs[name]

  @property
  def dense_sub_graphs(self):
    return self._dense_sub_graphs

  @property
  def sub_graphs(self):
    return self._sub_graphs

  @property
  def with_remote_gpu(self):
    return self._with_remote_gpu

  def signatures(self, graph: tf.Graph) -> List[SavedModelSignature]:
    return self._signatures[id(graph)].values()

  def add_signature(self, graph: tf.Graph, name: str, inputs, outputs):
    add_to_collections('signature_name', name)
    self._signatures[id(graph)][name] = SavedModelSignature(name=name,
                                                            inputs=inputs,
                                                            outputs=outputs)

  def merge_signature(self, graph: tf.Graph, name: str, inputs, outputs):
    if name not in self._signatures[id(graph)]:
      self._signatures[id(graph)][name] = SavedModelSignature(name=name,
                                                              inputs={},
                                                              outputs={})
    self._signatures[id(graph)][name].inputs.update(inputs)
    self._signatures[id(graph)][name].outputs.update(outputs)

  @property
  def sub_graph_num(self):
    """得到当前export_context中sub graph的数量"""
    return len(self._sub_graphs)


EXPORT_MODE = ExportMode.NONE
EXPORT_CTX = None


@monolith_export
def is_exporting():
  """是否在导出模式中"""
  return EXPORT_MODE != ExportMode.NONE


@monolith_export
def is_exporting_standalone():
  """是否在导出单机模型"""
  return EXPORT_MODE == ExportMode.STANDALONE


@monolith_export
def is_exporting_distributed():
  """是否正在导出分布式模型"""
  return EXPORT_MODE == ExportMode.DISTRIBUTED


@monolith_export
def get_current_export_ctx() -> ExportContext:
  """获取当前的上下文"""
  return EXPORT_CTX


@monolith_export
@tf_contextlib.contextmanager
def enter_export_mode(mode: ExportMode, export_ctx=None):
  """进入模型导出模式，会根据mode构图

  Args:
    mode (:obj:`ExportMode`): 导出模式，可选ExportMode.DISTRIBUTED, ExportMode.STANDALONE
    export_ctx (:obj:`ExportContext`, optional): 模型导出上下文
  """

  global EXPORT_MODE, EXPORT_CTX
  assert EXPORT_MODE is ExportMode.NONE and EXPORT_CTX is None, "export mode can't be nested"
  if export_ctx is None:
    export_ctx = ExportContext()
  EXPORT_MODE = mode
  EXPORT_CTX = export_ctx
  try:
    yield export_ctx
  finally:
    EXPORT_MODE = ExportMode.NONE
    EXPORT_CTX = None
