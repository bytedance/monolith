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

from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Set, Tuple
import os
import platform
import re
import socket
import types
import threading
import six
from inspect import signature, Parameter
from numpy.lib.arraysetops import isin

from absl import logging

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables

from monolith.core.base_layer import get_uname
from monolith.core.hyperparams import allowed_kwargs, InstantiableParams, Params

PS_JOB_NAME = "ps"


def ps_device(index: int) -> str:
  return "/job:{}/task:{}/device:CPU:0".format(PS_JOB_NAME, index)


def propagate_back_gradients(
    grads_and_vars: Iterable[Tuple[tf.Tensor, tf.Tensor]],
    xs: Iterable[tf.Tensor],
    valid_var_set: Set[tf.Tensor] = None) -> List[tf.Tensor]:
  """
  Propagate the gradients from vars back to the xs and return a list of gradients (dxs).
  Args:
    xs: tensors we want to get the gradient for. 
    valid_var_set: if non empty, we will verify if var in grad_and_vars is in this set.
  """
  combined_vars = []
  combined_grads = []
  for grad, var in grads_and_vars:
    if valid_var_set and (not var in valid_var_set):
      raise RuntimeError("Invalid variables in the input", var, valid_var_set)
    combined_vars.append(var)
    combined_grads.append(grad)
  return tf.gradients(combined_vars, list(xs), combined_grads)


def propagate_back_dict_gradients(
    grads_and_vars: Iterable[Tuple[tf.Tensor, tf.Tensor]],
    x_to_key: Dict[tf.Tensor, Any],
    valid_var_set: Set[tf.Tensor] = None
) -> Dict[Any, List[Tuple[tf.Tensor, tf.Tensor]]]:
  """
  Similar to above. But xs is replaced by x_to_key, and the returned gradients will 
  be grouped by key.
  """
  dxs = propagate_back_gradients(grads_and_vars, x_to_key.keys(), valid_var_set)
  grouped = defaultdict(list)
  for dx, (x, key) in zip(dxs, x_to_key.items()):
    grouped[key].append((dx, x))
  return grouped


def get_ndim(x: tf.Tensor):
  dims = x.get_shape()._dims
  if dims is not None:
    return len(dims)
  return None


def int_shape(x):
  try:
    shapes = []
    for dim in x.get_shape().as_list():
      if dim is None:
        shapes.append(-1)
      elif isinstance(dim, int):
        shapes.append(dim)
      elif isinstance(dim, tf.compat.v1.Dimension):
        shapes.append(dim.value)
      else:
        raise ValueError(f'dim {dim} is error')
    return tuple(shapes)
  except ValueError:
    return None


def extend_as_list(x, n):
  """This is a helper function to extend x as list, it will do:
    1. If x is a list, padding it to specified length n with None, if the length
    is less than n;
    2. If x is not a list, create a list with n elements x, please note that,
    these n elements are the same object, not a copy of x.
    """
  if isinstance(x, (list, tuple)):
    if len(x) < n:
      return x + [None] * (n - len(x))
    else:
      return x
  else:
    try:
      return [x if i == 0 else deepcopy(x) for i in range(n)]
    except:
      return [x] * n


def check_list(candidate, length_checker, could_be_none=False):
  """Checks whether a list has valid length
    Args:
        length_checker: a callable object takes a single integer
            return T/F on whether the candidate in the range or not
        could_be_none: None type is acceptable

    Returns: candidate

    Raises:
        TypeError
        ValueError
    """
  if not could_be_none and candidate is None:
    raise TypeError('ListChecker cannot accept None candidate')
  if type(candidate) not in [type(None), list]:
    raise TypeError('ListChecker got candidate '
                    'in the wrong type[{}]'.format(type(candidate)))
  if candidate is not None and not length_checker(len(candidate)):
    raise ValueError('ListChecker got candidate beyonds the range')
  return candidate


def to_snake_case(name):
  intermediate = re.sub('(.)([A-Z][a-z0-9]+)', r'\1_\2', name)
  insecure = re.sub('([a-z])([A-Z])', r'\1_\2', intermediate).lower()
  # If the class is private the name starts with "_" which is not secure
  # for creating scopes. We prefix the name with "private" in this case.
  if insecure[0] != '_':
    return insecure
  return 'private' + insecure


def to_list(x):
  """Normalizes a list/tensor into a list.

    If a tensor is passed, we return
    a list of size 1 containing the tensor.

    # Arguments
        x: target object to be normalized.

    # Returns
        A list.
    """
  if isinstance(x, list):
    return x
  return [x]


def _get_parameters(cls, parameters):
  for p in signature(cls.__init__).parameters.values():
    if p.name in {'self', 'cls'} or \
        p.kind in {Parameter.VAR_KEYWORD, Parameter.VAR_POSITIONAL}:
      continue
    else:
      parameters[p.name] = p


def _get_all_parameters(cls, parameters):
  if cls is not object:
    for base in cls.__bases__:
      _get_all_parameters(base, parameters)
  _get_parameters(cls, parameters)


def _inverted_index(ips: InstantiableParams, idx_dict):
  for name, item in ips.iter_params():
    if isinstance(item, (InstantiableParams, Params)):
      _inverted_index(item, idx_dict)
    else:
      idx_dict[name] = ips


def params(cls):
  """Returns the layer params."""

  ips = None
  for base in cls.__mro__:
    if base is cls:
      continue

    if hasattr(base, 'params'):
      ips = base.params()
      ips.cls = cls
      break
  ips = ips or InstantiableParams(cls)

  parameters = {}
  _get_all_parameters(cls, parameters)

  reversed_dict = {}
  _inverted_index(ips, reversed_dict)

  try:
    ips.define('name', get_uname(cls.__name__), "name")
  except:
    pass

  for p in parameters.values():
    if p.name in {'cls', 'self'}:
      continue

    if p.name in reversed_dict:
      _ips = reversed_dict[p.name]
      if p.default != Parameter.empty:
        _ips[p.name] = p.default
    else:
      try:
        ips.define(p.name, None if p.default == Parameter.empty else p.default,
                   p.name)
      except:
        if p.default != Parameter.empty and p.default != None:
          ips[p.name] = p.default

  for kw in allowed_kwargs:
    try:
      ips.define(kw, None, kw)
    except:
      pass

  return ips


def check_ops_dependence(op_names_1, op_names_2):
  """Check whether op_names_1 depend on op_names_2.

  Raises:
    Exception: If op_names_1 depend on op_names_2.

  """
  op_names_1 = to_list(op_names_1)
  graph_def = tf.compat.v1.get_default_graph().as_graph_def()
  sub_graph_1 = tf.compat.v1.graph_util.extract_sub_graph(graph_def, op_names_1)

  op_names_2 = set(to_list(op_names_2))

  depended_op_names = [
      node.name for node in sub_graph_1.node if node.name in op_names_2
  ]
  if depended_op_names:
    raise Exception(
        "Checking ops dependence, the ops [%s] depend on ops [%s], which may cause ops [%s] to be run twice."
        % (",".join(op_names_1), ",".join(depended_op_names),
           ",".join(depended_op_names)))


def with_params(cls):
  cls.params = types.MethodType(params, cls)
  return cls


def get_local_host():
  if platform.system() in ("Windows", "Linux"):
    local_host = socket.gethostbyname(socket.gethostname())
  else:
    local_host = socket.gethostbyname(socket.gethostname() + ".local")

  return local_host


def get_test_tmp_dir():
  return os.environ.get("TEST_TMPDIR", "/tmp")


def get_debugging_info_file_name(model_dir: str):
  return os.path.join(model_dir, "debugging_info.pb")


def get_meta_graph_file_name(model_dir: str):
  return os.path.join(model_dir, "meta_graph_for_debugging.pb")


def add_to_collections(names, value):
  if isinstance(value, (bool, int, float, str)):
    tf.compat.v1.add_to_collections(names, value)
  elif value:
    tf.compat.v1.add_to_collections(names, value)
  else:
    logging.info(f'value is {value}, skip')


def get_collection(name):
  collection = tf.compat.v1.get_collection(name)
  if isinstance(collection, (bool, int, float, str)):
    return collection
  elif collection:
    return collection
  else:
    return None


def set_metric_prefix(prefix: str):
  os.environ["MONOLITH_METRIC_PREFIX"] = prefix


def get_metric_prefix():
  return os.environ.get("MONOLITH_METRIC_PREFIX", "monolith.training")
