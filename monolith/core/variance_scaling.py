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
"Code to implement a custom Variance Scaling initializer that returns numpy arrays."

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.stats as stats


def _compute_fans(shape, data_format='channels_last'):
  """Computes the number of input and output units for a weight shape.

    Args:
        shape: Integer shape tuple.
        data_format: Image data format to use for convolution kernels.
            Note that all kernels in Keras are standardized on the
            `channels_last` ordering (even when inputs are set
            to `channels_first`).

    Returns:
        A tuple of scalars, `(fan_in, fan_out)`.

    # Raises
        ValueError: in case of invalid `data_format` argument.
    """
  if len(shape) == 2:
    fan_in = shape[0]
    fan_out = shape[1]
  elif len(shape) in {3, 4, 5}:
    # Assuming convolution kernels (1D, 2D or 3D).
    # TH kernel shape: (depth, input_depth, ...)
    # TF kernel shape: (..., input_depth, depth)
    if data_format == 'channels_first':
      receptive_field_size = np.prod(shape[2:])
      fan_in = shape[1] * receptive_field_size
      fan_out = shape[0] * receptive_field_size
    elif data_format == 'channels_last':
      receptive_field_size = np.prod(shape[:-2])
      fan_in = shape[-2] * receptive_field_size
      fan_out = shape[-1] * receptive_field_size
    else:
      raise ValueError('Invalid data_format: ' + data_format)
  else:
    # No specific assumptions.
    fan_in = np.sqrt(np.prod(shape))
    fan_out = np.sqrt(np.prod(shape))
  return fan_in, fan_out


class VarianceScaling():
  """Initializer capable of adapting its scale to the shape of weights.

    With `distribution="truncated_normal"`, samples are drawn from a truncated
    normal distribution centered on zero, with `stddev = sqrt(scale / n)`
    where n is:

        - number of input units in the weight tensor, if mode = "fan_in"
        - number of output units, if mode = "fan_out"
        - average of the numbers of input and output units, if mode = "fan_avg"

    With `distribution="uniform"`,
    samples are drawn from a uniform distribution
    within [-limit, limit], with `limit = sqrt(3 * scale / n)`.

    With `distribution="untrucated_normal"`, samples are drawn from a truncated
    normal distribution centered on zero, with `stddev = sqrt(scale / n)`.

    When called, this initializer produces a numpy array, instead of Tensorflow
    tensors.

    Args:
        scale (float, optional): Scaling factor (positive float).
        mode (str, optional): One of "fan_in", "fan_out", "fan_avg".
        distribution (str, optional): Random distribution to use. One of
            "truncated_normal", "untruncated_normal", and "uniform".
        seed (int, optional): A Python integer. Used to seed the random generator.

    Raises:
        ValueError: In case of an invalid value for the "scale", mode" or
          "distribution" arguments.
    """

  def __init__(self,
               scale=1.0,
               mode='fan_in',
               distribution='truncated_normal',
               seed=None):
    if scale <= 0.:
      raise ValueError('`scale` must be a positive float. Got:', scale)
    mode = mode.lower()
    if mode not in {'fan_in', 'fan_out', 'fan_avg'}:
      raise ValueError(
          'Invalid `mode` argument: '
          'expected on of {"fan_in", "fan_out", "fan_avg"} '
          'but got', mode)
    distribution = distribution.lower()
    if distribution not in {
        'truncated_normal', 'untruncated_normal', 'uniform'
    }:
      raise ValueError(
          'Invalid `distribution` argument: '
          'expected one of {"truncated_normal", "untruncated_normal", "uniform"} '
          'but got', distribution)
    self.scale = scale
    self.mode = mode
    self.distribution = distribution
    self.seed = seed

  def __call__(self, shape, dtype=np.float32):
    fan_in, fan_out = _compute_fans(shape)
    scale = self.scale
    if self.mode == 'fan_in':
      scale /= max(1., fan_in)
    elif self.mode == 'fan_out':
      scale /= max(1., fan_out)
    else:
      scale /= max(1., float(fan_in + fan_out) / 2)

    np.random.seed(self.seed)
    if self.distribution == 'truncated_normal':
      mean = 0.0
      # 0.879... = scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
      stddev = np.sqrt(scale) / .87962566103423978
      # Mimic the behavior of tf.random.truncated_normal, which truncates
      # at mean +/- 2 standard deviations
      lower_clip = mean - 2 * stddev
      upper_clip = mean + 2 * stddev
      a = (lower_clip - mean) / stddev
      b = (upper_clip - mean) / stddev
      return stats.truncnorm.rvs(
          a=a,
          b=b,
          loc=mean,
          scale=stddev,
          size=shape,
      ).astype(dtype)
    elif self.distribution == 'untruncated_normal':
      mean = 0.0
      stddev = np.sqrt(scale)
      return np.random.normal(
          loc=mean,
          scale=stddev,
          size=shape,
      ).astype('float32')
    else:
      limit = np.sqrt(3. * scale)
      return np.random.uniform(
          low=-limit,
          high=limit,
          size=shape,
      ).astype(dtype)

  def get_config(self):
    return {
        'scale': self.scale,
        'mode': self.mode,
        'distribution': self.distribution,
        'seed': self.seed
    }
