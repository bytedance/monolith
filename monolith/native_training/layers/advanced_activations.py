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

import types

import tensorflow.keras.initializers as initializers
import tensorflow.keras.constraints as constraints
from tensorflow.python.keras.activations import exponential
from tensorflow.python.keras.activations import gelu
from tensorflow.python.keras.activations import hard_sigmoid
from tensorflow.python.keras.activations import linear
from tensorflow.python.keras.activations import selu
from tensorflow.python.keras.activations import sigmoid
from tensorflow.python.keras.activations import softplus
from tensorflow.python.keras.activations import softsign
from tensorflow.python.keras.activations import swish
from tensorflow.python.keras.activations import tanh
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.layers.advanced_activations import ReLU
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.layers.advanced_activations import ELU
from tensorflow.python.keras.layers.advanced_activations import Softmax
from tensorflow.python.keras.layers.advanced_activations import ThresholdedReLU
from tensorflow.python.keras.layers.advanced_activations import PReLU

from monolith.native_training.utils import params as _params
from monolith.native_training.monolith_export import monolith_export

__all__ = [
    'ReLU', 'LeakyReLU', 'ELU', 'Softmax', 'ThresholdedReLU', 'PReLU',
    'Exponential', 'Gelu', 'HardSigmoid', 'Linear', 'Selu', 'Sigmoid',
    'Sigmoid2', 'Softplus', 'Softsign', 'Swish', 'Tanh'
]

Tanh = type('Tanh', (Layer,), {'call': lambda self, x: tanh(x)})
Sigmoid = type('Sigmoid', (Layer,), {'call': lambda self, x: sigmoid(x)})
Sigmoid2 = type('Sigmoid2', (Layer,), {'call': lambda self, x: sigmoid(x) * 2})
Linear = type('Linear', (Layer,), {'call': lambda self, x: linear(x)})
Gelu = type('Gelu', (Layer,), {'call': lambda self, x: gelu(x)})
Selu = type('Selu', (Layer,), {'call': lambda self, x: selu(x)})
Softsign = type('Softsign', (Layer,), {'call': lambda self, x: softsign(x)})
Softplus = type('Softplus', (Layer,), {'call': lambda self, x: softplus(x)})
Exponential = type('Exponential', (Layer,),
                   {'call': lambda self, x: exponential(x)})
HardSigmoid = type('HardSigmoid', (Layer,),
                   {'call': lambda self, x: hard_sigmoid(x)})
Swish = type('Swish', (Layer,), {'call': lambda self, x: swish(x)})

ReLU.params = types.MethodType(_params, ReLU)
PReLU.params = types.MethodType(_params, PReLU)
LeakyReLU.params = types.MethodType(_params, LeakyReLU)
ELU.params = types.MethodType(_params, ELU)
Softmax.params = types.MethodType(_params, Softmax)
ThresholdedReLU.params = types.MethodType(_params, ThresholdedReLU)
Tanh.params = types.MethodType(_params, Tanh)
Sigmoid.params = types.MethodType(_params, Sigmoid)
Sigmoid2.params = types.MethodType(_params, Sigmoid2)
Linear.params = types.MethodType(_params, Linear)
Gelu.params = types.MethodType(_params, Gelu)
Selu.params = types.MethodType(_params, Selu)
Softsign.params = types.MethodType(_params, Softsign)
Softplus.params = types.MethodType(_params, Softplus)
Exponential.params = types.MethodType(_params, Exponential)
HardSigmoid.params = types.MethodType(_params, HardSigmoid)
Swish.params = types.MethodType(_params, Swish)

__all_activations = {
    'exponential': Exponential,
    'gelu': Gelu,
    'hard_sigmoid': HardSigmoid,
    'hardsigmoid': HardSigmoid,
    'linear': Linear,
    'selu': Selu,
    'sigmoid': Sigmoid,
    'sigmoid2': Sigmoid2,
    'softplus': Softplus,
    'softsign': Softsign,
    'swish': Swish,
    'tanh': Tanh,
    'leakyrelu': LeakyReLU,
    'relu': ReLU,
    'elu': ELU,
    'softmax': Softmax,
    'thresholdedrelu': ThresholdedReLU,
    'prelu': PReLU
}
ALL_ACTIVATION_NAMES = set(__all_activations.keys())


@monolith_export
def get(identifier):
  """获取函数, 可以用名字获取, 也可以用序列化的Json获取

  Args:
    identifier (:obj:`Any`): 标识, 可以是name, 获序列化的Json, None等
  
  Returns:
    激活函数
  """

  if identifier is None:
    return None
  if isinstance(identifier, str):
    if identifier.lower() in __all_activations:
      return __all_activations[identifier.lower()]()
    else:
      evaled = eval(identifier)
      if isinstance(evaled, dict):
        return deserialize(evaled)
      raise TypeError(
          'Could not interpret activation function identifier: {}'.format(
              identifier))
  elif isinstance(identifier, dict):
    return deserialize(identifier)
  elif callable(identifier):
    if hasattr(identifier, 'params'):
      try:
        if issubclass(identifier, Layer):
          return identifier()
        else:
          return identifier
      except:
        return identifier
    elif isinstance(identifier, Layer):
      name = identifier.__class__.__name__.lower()
      return __all_activations[name]()
    else:
      try:
        name = identifier.__name__
        return __all_activations[name]()
      except:
        return identifier
  else:
    raise TypeError(
        'Could not interpret activation function identifier: {}'.format(
            identifier))


@monolith_export
def serialize(activation):
  """序列化激活函数
  
  Args:
    activation (:obj:`tf.activation`): keras激活函数
  
  Returns:
    Dict/Json 获序列化的激活函数
  """

  if isinstance(activation, (Linear, Exponential, Selu, HardSigmoid, Gelu,
                             Sigmoid, Softplus, Softsign, Swish, Tanh)):
    return repr({'name': activation.__class__.__name__})
  elif isinstance(activation, (LeakyReLU, ELU)):
    return repr({
        'name': activation.__class__.__name__,
        'alpha': float(activation.alpha)
    })
  elif isinstance(activation, ReLU):
    return repr({
        'name': 'ReLU',
        'max_value': activation.max_value,
        'negative_slope': float(activation.negative_slope),
        'threshold': float(activation.threshold)
    })
  elif isinstance(activation, PReLU):
    return repr({
        'name':
            'PReLU',
        'alpha_initializer':
            initializers.serialize(activation.alpha_initializer),
        'alpha_regularizer':
            initializers.serialize(activation.alpha_regularizer),
        'alpha_constraint':
            constraints.serialize(activation.alpha_constraint),
        'shared_axes':
            activation.shared_axes
    })
  elif isinstance(activation, Softmax):
    return repr({'name': 'Softmax', 'axis': activation.axis})
  elif isinstance(activation, ThresholdedReLU):
    return repr({'name': 'ThresholdedReLU', 'theta': float(activation.theta)})
  else:
    return None


@monolith_export
def deserialize(identifier):
  """反序列化激活函数
  
  Args:
    identifier (:obj:`Any`): 标识, 可以是name, 获序列化的Json, None等
  
  Returns:
    激活函数
  """

  if identifier is None:
    return None
  else:
    if not isinstance(identifier, dict):
      identifier = eval(identifier)
    assert isinstance(identifier, dict)
    name = identifier['name'].lower()
    assert name in __all_activations
    identifier.pop('name')
    return __all_activations[name](**identifier)
