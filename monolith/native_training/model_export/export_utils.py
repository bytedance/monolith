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

from typing import Callable

import tensorflow as tf

from monolith.native_training import nested_tensors
from monolith.native_training.model_export import export_context
from monolith.native_training import distributed_serving_ops

remote_predict = distributed_serving_ops.remote_predict


def _get_tensor_signature_name(t: tf.Tensor):
  return t.name.replace(":", "_")


class RemotePredictHelper:

  def __init__(self, name: str, input_tensors: object,
               remote_func: Callable[[object], object]):
    self._name = name
    self._input_tensors = nested_tensors.NestedTensors(input_tensors)
    self._remote_func = remote_func
    self._define_remote_func()

  def _define_remote_func(self):
    """Defines the remote func"""
    self._func_defined = True
    flat_input_tensors = self._input_tensors.get_tensors()
    phs = []
    for tensor in flat_input_tensors:
      phs.append(
          tf.compat.v1.placeholder(dtype=tensor.dtype,
                                   shape=tensor.shape,
                                   name=_get_tensor_signature_name(tensor) +
                                   "_remote_input_ph"))
    func_input = self._input_tensors.get_nested_result(phs)
    func_output = self._remote_func(func_input)
    self._output_tensors = nested_tensors.NestedTensors(func_output)
    flat_output_tensors = self._output_tensors.get_tensors()
    self._sig_input = {
        _get_tensor_signature_name(t): ph
        for t, ph in zip(flat_input_tensors, phs)
    }
    assert len(self._sig_input) == len(
        flat_input_tensors), f"Name conflicts: {flat_input_tensors}"
    self._sig_output = {
        _get_tensor_signature_name(t): t for t in flat_output_tensors
    }
    assert len(self._sig_output) == len(
        flat_output_tensors), f"Name conflicts: {flat_input_tensors}"

    export_context.get_current_export_ctx().add_signature(
        tf.compat.v1.get_default_graph(), self._name, self._sig_input,
        self._sig_output)

  def call_remote_predict(self,
                          model_name: str,
                          input_tensors: object = None,
                          old_model_name: str = None,
                          task: int = 0):
    """
    Calls the remote function.

    Args:
    model_name - the remote model_name that will be used.
    input_tensors - if None, will use tensors in the __init__
    old_model_name & task - A deprecated args to support old remote predict
    """
    flat_input_tensors = None
    if input_tensors:
      flat_input_tensors = nested_tensors.NestedTensors(
          input_tensors).get_tensors()
    else:
      flat_input_tensors = self._input_tensors.get_tensors()
    results = remote_predict(
        list(self._sig_input.keys()),
        flat_input_tensors,
        list(self._sig_output.keys()),
        model_name,
        task,
        old_model_name,
        output_types=[t.dtype for t in self._sig_output.values()],
        signature_name=self._name)
    return self._output_tensors.get_nested_result(results)
