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

import collections
import dataclasses
from typing import Dict, List

import tensorflow as tf

from tensorflow.python.types import core
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.ops import resource_variable_ops

from monolith.native_training import graph_meta

_CACHED_VARIABLES = "monolith_cached_variables"


@dataclasses.dataclass
class CachedVariableAssociates:
  async_fetched_var: tf.Variable
  async_cached_var: tf.Variable


@dataclasses.dataclass
class CachedVariableMeta:

  var_id_to_assoc: Dict[int, CachedVariableAssociates] = dataclasses.field(
      default_factory=dict)


def _get_meta() -> CachedVariableMeta:
  return graph_meta.get_meta("cached_variables_meta", CachedVariableMeta)


@tf.custom_gradient
def cached_value(var, async_cached_var):

  def grad(dy):
    return dy, None

  return async_cached_var, grad


def _get_valid_op_name(name: str):
  return name.replace(":", "_").replace("/", "_")


def cached_variable_creator(next_creator, **kwargs):
  var = next_creator(**kwargs)
  if not isinstance(var, resource_variable_ops.ResourceVariable):
    raise ValueError("Only ResourceVariable is supported. "
                     "Do you disable V2 behavior or use strategy?")

  if not var._cached_value is None:
    raise ValueError("The variable has already been cached. "
                     "Consider about removing cache_device.")

  with tf.device(None):
    async_cached_var = resource_variable_ops.ResourceVariable(
        initial_value=var.initial_value,
        trainable=False,
        collections=[tf.compat.v1.GraphKeys.LOCAL_VARIABLES],
        shape=var.shape,
        dtype=var.dtype)
    async_fetched_var = resource_variable_ops.ResourceVariable(
        initial_value=var.initial_value,
        trainable=False,
        collections=[tf.compat.v1.GraphKeys.LOCAL_VARIABLES],
        shape=var.shape,
        dtype=var.dtype)

  if async_cached_var.device == var.device:
    # In this case, we shouldn't do the cache since we try assign vars
    # on the remote machines.
    #
    # This is commmon when cached_var is forced to colocate with var.
    # For example, var is optimizer's slot variables.
    return var

  tf.compat.v1.add_to_collection(_CACHED_VARIABLES, var)
  var._cached_value = cached_value(var, async_cached_var)

  meta = _get_meta()
  meta.var_id_to_assoc[id(var)] = CachedVariableAssociates(
      async_fetched_var=async_fetched_var, async_cached_var=async_cached_var)
  return var


def fetch_all_cached_variables():
  meta = _get_meta()
  ops = []
  for var in tf.compat.v1.get_collection(_CACHED_VARIABLES):
    fetched_var = meta.var_id_to_assoc[id(var)].async_fetched_var
    ops.append(
        fetched_var.assign(var._read_variable_op(),
                           name="fetch_from_{}".format(
                               _get_valid_op_name(str(var.device))),
                           read_value=False))
  return tf.group(ops)


def assign_all_cached_variables():
  meta = _get_meta()
  ops = []
  for var in tf.compat.v1.get_collection(_CACHED_VARIABLES):
    associates = meta.var_id_to_assoc[id(var)]
    ops.append(
        associates.async_cached_var.assign(associates.async_fetched_var,
                                           name="assign_cached_var",
                                           read_value=False))
  return tf.group(ops, name="assign_all_cached_variables")


class FetchAllCachedVariablesHook(tf.estimator.SessionRunHook):
  """Fetch variables."""

  def __init__(self):
    self._fetch_op = fetch_all_cached_variables()
    self._assign_op = assign_all_cached_variables()
    self._first_run = True

  def after_create_session(self, session, coord):
    self._first_run = True

  def before_run(self, run_context: tf.estimator.SessionRunContext):
    if self._first_run:
      # For the first run, we do a sync fetch since the local values might be
      # super stale.
      run_context.session.run(self._fetch_op)
      run_context.session.run(self._assign_op)
      self._first_run = False
    return tf.estimator.SessionRunArgs(self._fetch_op)

  def after_run(self, run_context, run_values):
    run_context.session.run(self._assign_op)
