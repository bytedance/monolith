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

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import traceback
import inspect
import sys

import tensorflow as tf
from absl import logging

from monolith.core import model_imports
from monolith.core.base_model_params import SingleTaskModelParams


class _ModelRegistryHelper(object):
  # Global dictionary mapping subclass name to registered ModelParam subclass.
  _MODEL_PARAMS = {}
  # Global set of modules from which ModelParam subclasses have been registered.
  _REGISTERED_MODULES = set()

  @classmethod
  def _ClassPathPrefix(cls):
    return 'monolith.tasks.'

  @classmethod
  def _ModelParamsClassKey(cls, src_cls, shortcut=False):
    """Returns a string key used for `src_cls` in the model registry.
    Args:
      src_cls: A subclass of `BaseModel`.
      shortcut: (Deprecated) generate shortcut version of given task.
    """
    path = src_cls.__module__
    if shortcut:
      # Removes the prefix.
      path_prefix = cls._ClassPathPrefix()
      path = path.replace(path_prefix, '')
      # Removes 'params.' if exists.
      if 'params.' in path:
        path = path.replace('params.', '')
    return '{}.{}'.format(path, src_cls.__name__)

  @classmethod
  def _GetSourceInfo(cls, src_cls):
    """Gets a source info string given a source class."""
    return '%s@%s:%d' % (cls._ModelParamsClassKey(src_cls),
                         inspect.getsourcefile(src_cls),
                         inspect.getsourcelines(src_cls)[-1])

  @classmethod
  def _RegisterModel(cls, src_cls):
    """Registers a ModelParams subclass in the global registry."""
    for key in set([
        cls._ModelParamsClassKey(src_cls, shortcut=False),
        cls._ModelParamsClassKey(src_cls, shortcut=True)
    ]):
      module = src_cls.__module__
      if key in cls._MODEL_PARAMS:
        raise ValueError('Duplicate model registered for key {}: {}.{}'.format(
            key, module, src_cls.__name__))

      logging.debug('Registering model %s', key)
      # Log less frequently (once per module) but at a higher verbosity level.
      if module not in cls._REGISTERED_MODULES:
        logging.info('Registering models from module: %s', module)
        cls._REGISTERED_MODULES.add(module)

      # Decorate param methods to add source info metadata.
      cls._MODEL_PARAMS[key] = src_cls
    return cls._ModelParamsClassKey(src_cls, shortcut=False)

  @classmethod
  def RegisterSingleTaskModel(cls, src_cls):
    """Class decorator that registers a `.SingleTaskModelParams` subclass."""
    logging.info("Register {} Start".format(src_cls.__name__))
    if not issubclass(src_cls, SingleTaskModelParams):
      raise TypeError('src_cls %s is not a SingleTaskModelParams!' %
                      src_cls.__name__)

    cls._RegisterModel(src_cls)
    all_params = _ModelRegistryHelper._MODEL_PARAMS
    logging.info("Register {} successfully".format(src_cls.__name__))
    return src_cls

  @staticmethod
  def GetAllRegisteredClasses():
    """Returns global registry map from model names to their param classes."""
    all_params = _ModelRegistryHelper._MODEL_PARAMS
    if not all_params:
      logging.warning('No classes registered.')
    return all_params

  @classmethod
  def GetClass(cls, class_key):
    """Returns a ModelParams subclass with the given `class_key`.

    Args:
      class_key: string key of the ModelParams subclass to return.

    Returns:
      A subclass of `SingleTaskModelParams`.

    Raises:
      LookupError: If no class with the given key has been registered.
    """
    all_params = cls.GetAllRegisteredClasses()
    if class_key not in all_params:
      for k in sorted(all_params):
        logging.info('Known model: %s', k)
      raise LookupError('Model %s not found from list of above known models.' %
                        class_key)
    return all_params[class_key]

  @classmethod
  def GetParams(cls, class_key):
    """Constructs a `Params` object for given model.

    Args:
      class_key: String class key.

    Returns:
      Full `~.hyperparams.Params` for the model class.
    """
    model_params_cls = cls.GetClass(class_key)
    model_params = model_params_cls()
    cfg = model_params.task()
    return cfg


RegisterSingleTaskModel = _ModelRegistryHelper.RegisterSingleTaskModel


def GetAllRegisteredClasses():
  model_imports.ImportAllParams()
  return _ModelRegistryHelper.GetAllRegisteredClasses()


def GetClass(class_key):
  model_imports.ImportParams(class_key)
  return _ModelRegistryHelper.GetClass(class_key)


def GetParams(class_key):
  model_imports.ImportParams(class_key)
  return _ModelRegistryHelper.GetParams(class_key)
