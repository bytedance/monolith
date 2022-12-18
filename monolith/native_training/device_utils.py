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

"""
Device Utils.

Provide device placement utils and strategies.
"""

import contextlib

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.compiler.xla import xla

_GPU_PLACEMENT_ALLOWED = False


def enable_gpu_training():
  global _GPU_PLACEMENT_ALLOWED
  _GPU_PLACEMENT_ALLOWED = True


def disable_gpu_training():
  global _GPU_PLACEMENT_ALLOWED
  _GPU_PLACEMENT_ALLOWED = False


def get_visible_gpus(local_rank, processes_per_gpu=1):
  """
  Visible GPU devices string for session config.
  
  Args:
    local_rank: the process local rank, for example `hvd.local_rank()`.
    process_per_gpu: the integer number of processes per gpu.
  
  Return:
    String compatible for session_config.gpu_options.visible_device_list,
    for example, "2" indicates TensorFlow session will map the physical
    gpu:2 into the TensorFlow virtual string-specified device "GPU:0".
  """
  # TODO: processes_per_gpu :float < 0 allows str of gpus.
  assert isinstance(processes_per_gpu, int) and processes_per_gpu >= 1
  return str(int(local_rank / processes_per_gpu))


_default_device = tf.DeviceSpec.from_string("/device:CPU:0")


def _device_rule(device_name):
  d = tf.DeviceSpec.from_string(device_name)
  if (d.device_type == "GPU" and
      not _GPU_PLACEMENT_ALLOWED) or not d.device_type:
    # If GPU is illegally assigned, or, device type is empty,
    # Merge with the _default_device while keep the assigned job,task,replica
    return d.make_merged_spec(_default_device).to_string()
  # Don't override the assigned and allowed device string
  return device_name


def default_device_fn(op: tf.Operation):
  """Default device_fn for Estimator RunConfig."""
  # Enforce commonly-used summary op on CPU.
  if op.type.startswith("Write") or op.type.endswith("Summary") or \
    (op.type == "Const" and op.get_attr("dtype") == tf.string):
    return _default_device.to_string()
  # Enforce general placement rule.
  if op.device:
    return _device_rule(op.device)
  # Guarantee default CPU:0 at op creation. Because
  # otherwise if any GPU is visiable and kernel is available,
  # op would be placed on GPU when no device string specified.
  return _default_device.to_string()


@contextlib.contextmanager
def maybe_device_if_allowed(device_name):
  """
  Monolith disallows soft device placement for training.
  This is an insurance when default_device_fn is missed/not-enforced in Estimator Runconfig.
  """
  dev = _device_rule(device_name)
  with tf.device(dev):
    yield


class _FakeNodeDef(object):
  """A fake NodeDef for _FakeOperation."""

  __slots__ = ["op", "name"]

  def __init__(self):
    self.op = ""
    self.name = ""


class _FakeOp(object):
  """A helper class to determine the current device.

  Supports only the type and device set/get methods needed to run the
  graph's _apply_device_function method.
  """

  def __init__(self):
    self._device = ""
    self.type = "FakeOpPyObj"
    self.name = ""
    self.node_def = _FakeNodeDef()

  @property
  def device(self):
    return self._device

  def _set_device(self, device):
    self._device = ops._device_string(device)  # pylint: disable=protected-access

  def _set_device_from_string(self, device_str):
    self._device = device_str


def within_placement_context_of(device_name):
  """Check if the current placement context is ."""
  fake_op = _FakeOp()
  ops.get_default_graph()._apply_device_functions(fake_op)
  return tf.DeviceSpec.from_string(
      fake_op.device).device_type == device_name.upper()
