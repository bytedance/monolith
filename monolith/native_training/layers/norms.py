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

import tensorflow as tf
import tensorflow.keras.initializers as initializers
from tensorflow.python.keras import regularizers
from tensorflow.keras.layers import Layer, InputSpec

from monolith.core.base_layer import add_layer_loss
from monolith.native_training.utils import with_params
from monolith.native_training.monolith_export import monolith_export
from monolith.native_training.layers.utils import check_dim, dim_size


@with_params
class BatchNorm(Layer):

  def __init__(self,
               momentum=0.99,
               center=True,
               scale=True,
               moving_mean_initializer=initializers.Zeros(),
               moving_variance_initializer=initializers.Ones(),
               beta_initializer=initializers.Zeros(),
               gamma_initializer=initializers.Ones(),
               regularizer=None,
               training_use_global_dist=False,
               global_dist_momentum=1.0,
               stop_grad_of_var_mean=False,
               epsilon=1e-6,
               mode=tf.estimator.ModeKeys.TRAIN,
               **kwargs):
    super(BatchNorm, self).__init__(**kwargs)

    self.momentum = momentum
    self.epsilon = epsilon
    self.center = center
    self.scale = scale
    self.beta_initializer = initializers.get(beta_initializer)
    self.gamma_initializer = initializers.get(gamma_initializer)
    self.moving_mean_initializer = initializers.get(moving_mean_initializer)
    self.moving_variance_initializer = initializers.get(
        moving_variance_initializer)
    self.training_use_global_dist = training_use_global_dist
    self.global_dist_momentum = global_dist_momentum
    self.stop_grad_of_var_mean = stop_grad_of_var_mean
    self.mode = mode
    self.regularizer = regularizers.get(regularizer)

    self.input_spec = InputSpec(min_ndim=2)

  def build(self, input_shape):
    assert len(input_shape) >= 2
    self.input_dim = check_dim(input_shape[-1])

    self.moving_mean = self.add_weight(name='moving_mean',
                                       shape=[self.input_dim],
                                       dtype=self.dtype,
                                       initializer=self.moving_mean_initializer)
    self.moving_variance = self.add_weight(
        name='moving_variance',
        dtype=self.dtype,
        shape=[self.input_dim],
        initializer=self.moving_variance_initializer)

    if self.center:
      self.beta_offset = self.add_weight(name='beta_offset',
                                         dtype=self.dtype,
                                         shape=[self.input_dim],
                                         initializer=self.beta_initializer,
                                         regularizer=self.regularizer)
    else:
      self.beta_offset = tf.constant(0.0, dtype=tf.float32)
    if self.scale:
      self.gamma_scale = self.add_weight(name='gamma_scale',
                                         dtype=self.dtype,
                                         shape=[self.input_dim],
                                         initializer=self.gamma_initializer,
                                         regularizer=self.regularizer)
    else:
      self.gamma_scale = tf.constant(1.0, dtype=tf.float32)

    self.input_spec = InputSpec(min_ndim=2, axes={-1: self.input_dim})
    super(BatchNorm, self).build(input_shape)

  def call(self, inputs, **kwargs):

    @tf.custom_gradient
    def replace_gradient(original_moving_average, gradient):

      def grad(dy):
        return gradient, None

      return tf.identity(original_moving_average), grad

    if self.mode == tf.estimator.ModeKeys.TRAIN:
      if self.stop_grad_of_var_mean:
        self.mean, self.variance = tf.nn.moments(tf.stop_gradient(inputs),
                                                 axes=[0])
      else:
        self.mean, self.variance = tf.nn.moments(inputs, axes=[0])
      # replace moving average gradient by mean & variance in current minibatch

      moving_mean = replace_gradient(self.moving_mean, self.mean)
      moving_variance = replace_gradient(self.moving_variance, self.variance)
      moving_variance = tf.maximum(moving_variance,
                                   tf.constant(0, dtype=tf.float32))
      add_layer_loss('{}_moving_mean'.format(self.name),
                     tf.reduce_sum(moving_mean))
      add_layer_loss('{}_moving_variance'.format(self.name),
                     tf.reduce_sum(moving_variance))

      if self.training_use_global_dist:
        mean = self.global_dist_momentum * moving_mean + \
               (1.0 - self.global_dist_momentum) * self.mean
        variance = self.global_dist_momentum * moving_variance + \
                   (1.0 - self.global_dist_momentum) * self.variance
      else:
        mean, variance = self.mean, self.variance

      tf.compat.v1.summary.histogram(self.name + '/mean_train', mean)
      tf.compat.v1.summary.scalar(self.name + '/mean_train',
                                  tf.reduce_mean(mean))
      tf.compat.v1.summary.histogram(self.name + '/var_train', variance)
      tf.compat.v1.summary.scalar(self.name + '/var_train',
                                  tf.reduce_mean(variance))

    else:
      moving_variance = tf.maximum(self.moving_variance,
                                   tf.constant(0, dtype=tf.float32))
      mean, variance = tf.stop_gradient(
          self.moving_mean), tf.stop_gradient(moving_variance)
      tf.compat.v1.summary.histogram(self.name + '/mean_test', mean)
      tf.compat.v1.summary.scalar(self.name + '/mean_test',
                                  tf.reduce_mean(mean))
      tf.compat.v1.summary.histogram(self.name + '/var_test', variance)
      tf.compat.v1.summary.scalar(self.name + '/var_test',
                                  tf.reduce_mean(variance))

    output = tf.nn.batch_normalization(inputs, mean, variance, self.beta_offset,
                                       self.gamma_scale, self.epsilon)
    return output

  def set_use_global_dist(self, training_use_global_dist):
    assert type(training_use_global_dist) is bool
    self.training_use_global_dist = training_use_global_dist

  def get_config(self):
    config = {
        'momentum':
            self.momentum,
        'epsilon':
            self.epsilon,
        'center':
            self.center,
        'scale':
            self.scale,
        'beta_initializer':
            initializers.serialize(self.beta_initializer),
        'gamma_initializer':
            initializers.serialize(self.gamma_initializer),
        'moving_mean_initializer':
            initializers.serialize(self.moving_mean_initializer),
        'moving_variance_initializer':
            initializers.serialize(self.moving_variance_initializer),
        'training_use_global_dist':
            self.training_use_global_dist,
        'global_dist_momentum':
            self.global_dist_momentum,
        'stop_grad_of_var_mean':
            self.stop_grad_of_var_mean,
        'mode':
            self.mode,
        'regularizer':
            regularizers.serialize(self.regularizer),
    }
    base_config = super(BatchNorm, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@monolith_export
@with_params
class LayerNorm(Layer):
  """与BatchNorm类似, 但是是在样本内做归一化. 

  与BatchNorm不同的是LayerNorm在训练与推理时, 使用同一套逻辑, 不用分别处理

  Args:
    initializer (:obj:`tf.initializer`): gamma的初始化器
    regularizer (:obj:`tf.regularizer`): beta/gamma的变量的正则化

  """

  def __init__(self, initializer, regularizer=None, **kwargs):
    super(LayerNorm, self).__init__(**kwargs)
    self.beta, self.gamma = None, None
    self.initializer = initializers.get(initializer) or initializers.Ones()
    self.regularizer = regularizers.get(regularizer)

  def build(self, input_shape):
    params_shape = [check_dim(input_shape[-1])]
    self.beta = self.add_weight(name='beta',
                                dtype=tf.float32,
                                shape=params_shape,
                                initializer=initializers.Zeros(),
                                regularizer=self.regularizer)
    self.gamma = self.add_weight(name='gamma',
                                 dtype=tf.float32,
                                 shape=params_shape,
                                 initializer=self.initializer,
                                 regularizer=self.regularizer)
    super(LayerNorm, self).build(input_shape)

  def call(self, inputs, **kwargs):
    mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
    output = tf.nn.batch_normalization(inputs,
                                       mean,
                                       variance,
                                       self.beta,
                                       self.gamma,
                                       variance_epsilon=1e-6)

    return output

  def get_config(self):
    config = {
        'initializer': initializers.serialize(self.initializer),
        'regularizer': regularizers.serialize(self.regularizer),
    }
    base_config = super(LayerNorm, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@monolith_export
@with_params
class GradNorm(Layer):
  """GradNorm提出通过将不同任务的梯度控制在一定的范围来进行多任务学习, 论文可参考 https://arxiv.org/abs/1711.02257

  GradNorm是通过构造辅助loss实现, 辅助的构造过程如下:
    - 选择shared bottom的最顶层变量W, 然后计算每个head对它的梯度 (如果顶层有多个W, 则分别计算梯度, 再concat起来)
    - 对上一步得到的梯度取L2范数, 得到gnorms, gnorms是一个n维向量, 长度与task的个数相同
    - gnorms加权weight, 得到wgnorms,  wgnorms平均, 得到avgnorm
    - gnorm_loss = scale * sum([(wgnorms - avgnorm) / (avgnorm + epsilon)]^loss_pow), relative_diff = True
    - gnorm_loss = scale * sum([wgnorms - avgnorm]^loss_pow), relative_diff = False
    - weighted_loss = sum(weight * losses)

  Args:
    loss_names (:obj:`str`): loss名称, 用于确定loss的个数, 写相关日志
    scale (:obj:`float`): 缩放因子, 用于缩放
    loss_pow (:obj:`float`): gnorm diff的指数因子
    relative_diff (:obj:`bool`): gnorm diff的计算方式, 如果为True, 会计算相对值
    epsilon (:obj:`float`): 一个非常小的常数, 防止除以0
  
  """

  def __init__(self,
               loss_names,
               scale=1.0,
               loss_pow=2.0,
               relative_diff=False,
               epsilon=1e-6,
               **kwargs):
    super(GradNorm, self).__init__(**kwargs)
    self.loss_names = loss_names
    self.scale = scale
    self.loss_pow = loss_pow
    self.relative_diff = relative_diff
    self.epsilon = epsilon

  def build(self, input_shape):
    n = len(self.loss_names)
    self.weight = self.add_weight(name='grad_norm_weights',
                                  shape=[n],
                                  dtype=tf.float32,
                                  initializer=tf.initializers.Zeros())
    self._weights = tf.nn.softmax(self.weight)

    for i in range(n):
      tf.compat.v1.summary.scalar(
          'gradnorm_weight/{}'.format(self.loss_names[i]), self._weights[i])

    super(GradNorm, self).build(input_shape)

  def _get_norm(self, grad):
    return (tf.reduce_sum(tf.multiply(grad, grad)))**0.5

  def get_weights(self):
    return self._weights

  def call(self, inputs, **kwargs):
    losses, shared_inputs = inputs
    if not isinstance(shared_inputs, list):
      shared_inputs = [shared_inputs]

    grads = [tf.gradients(loss, shared_inputs) for loss in losses]
    grads = [tf.concat(gs, axis=1) for gs in grads]

    gnorms = [self._get_norm(g) for g in grads]
    gnorms = tf.stop_gradient(tf.stack(gnorms, axis=0))

    weights = self._weights
    n = len(self.loss_names)

    avgnorm = tf.reduce_sum(gnorms * weights) / n
    wgnorms = gnorms * weights

    grad_diff = tf.abs(wgnorms - avgnorm)
    if self.relative_diff:
      grad_diff = grad_diff / (avgnorm + self.epsilon)

    gnorm_loss = tf.reduce_sum(grad_diff**self.loss_pow) * self.scale
    weighted_loss = tf.reduce_sum(
        tf.stack(losses, axis=0) * tf.stop_gradient(weights))

    for i in range(n):
      tf.compat.v1.summary.scalar(
          'gradnorm_gnorm/{}'.format(self.loss_names[i]), gnorms[i])
      tf.compat.v1.summary.scalar(
          'gradnorm_wgnorm/{}'.format(self.loss_names[i]), wgnorms[i])

    return gnorm_loss, weighted_loss

  def get_config(self):
    config = {
        'loss_names': self.loss_names,
        'scale': self.scale,
        'loss_pow': self.loss_pow,
        'relative_diff': self.relative_diff,
        'epsilon': self.epsilon
    }
    base_config = super(GradNorm, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
