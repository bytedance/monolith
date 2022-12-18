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

# -*- encoding=utf-8 -*-

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

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec


class TeacherEmbeddingTransform(Layer):
  """Combined.

    Example::

        # as first layer in a sequential model:
        #  x is a compatible tensor
        x = layers.Dense(32, input_shape=(16,))(x)
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)

    Args:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
        bias_initializer: Initializer for the bias vector
        allow_kernel_norm: T/F
            kernel normalization is only applicable when TRAINING
        kernel_normalization_trainable:
            If True, a trainable weight norm variable is allocated

    Input shapes:
        nD tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    Output shapes:
        nD tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

  def __init__(self, max_choice_per_embedding, teacher_embedding_sizes_list,
               **kwargs):
    super(TeacherEmbeddingTransform, self).__init__(**kwargs)
    assert len(max_choice_per_embedding) == len(teacher_embedding_sizes_list)
    self._max_choice_per_embedding = np.array(max_choice_per_embedding)
    self._teacher_embedding_sizes_list = np.array(teacher_embedding_sizes_list)

    self.input_spec = InputSpec(ndim=2)

  def build(self, input_shape):
    assert len(input_shape) == 2
    input_dim = input_shape[-1]
    assert input_dim == np.sum(self._teacher_embedding_sizes_list)

    total_teacher_embedding_transform_weight_size = np.sum(
        self._max_choice_per_embedding * self._teacher_embedding_sizes_list)
    self._teacher_embedding_transform_weight = self.add_weight(
        initial_value=tf.keras.initializers.TruncatedNormal(stddev=0.15)(
            [total_teacher_embedding_transform_weight_size, 1], self.dtype),
        name='teacher_embedding_transform_weight')
    self._snapshot_for_serving(self._teacher_embedding_transform_weight,
                               'teacher_embedding_transform_weight')
    self._teacher_embedding_transform_bias = self.add_weight(
        initial_value=tf.keras.initializers.Zeros()(
            [np.sum(self._max_choice_per_embedding)], self.dtype),
        name='teacher_embedding_transform_bias')
    self._snapshot_for_serving(self._teacher_embedding_transform_bias,
                               'teacher_embedding_transform_bias')
    self.built = True

  def call(self, inputs):
    teacher_embedding = inputs
    current_weight_idx = 0
    current_teacher_idx = 0
    teacher_transformed = []
    for i in range(self._max_choice_per_embedding.shape[0]):
      teacher_embedding_slice = teacher_embedding[:, current_teacher_idx:
                                                  current_teacher_idx + self.
                                                  _teacher_embedding_sizes_list[
                                                      i]]
      transform_weight_slice = self._teacher_embedding_transform_weight[
          current_weight_idx:current_weight_idx +
          self._teacher_embedding_sizes_list[i] *
          self._max_choice_per_embedding[i]]
      teacher_transformed.append(
          tf.matmul(
              teacher_embedding_slice,
              tf.reshape(transform_weight_slice, [
                  self._teacher_embedding_sizes_list[i],
                  self._max_choice_per_embedding[i]
              ])))
      current_weight_idx += self._teacher_embedding_sizes_list[
          i] * self._max_choice_per_embedding[i]
      current_teacher_idx += self._teacher_embedding_sizes_list[i]

    return tf.concat(teacher_transformed,
                     axis=1) + self._teacher_embedding_transform_bias

  def compute_output_shape(self, input_shape):
    raise NotImplementedError("I don't think I need to implement this one.")

  def get_config(self):
    config = {
        'max_choice_per_embedding': self._max_choice_per_embedding,
        'teacher_embedding_sizes_list': self._teacher_embedding_sizes_list
    }
    base_config = super(TeacherEmbeddingTransform, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class MixedEmbedOpComb(Layer):
  """Combined.

    Example::

        # as first layer in a sequential model:
        #  x is a compatible tensor
        x = layers.Dense(32, input_shape=(16,))(x)
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)

    Args:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
        bias_initializer: Initializer for the bias vector
        allow_kernel_norm: T/F
            kernel normalization is only applicable when TRAINING
        kernel_normalization_trainable:
            If True, a trainable weight norm variable is allocated

    Input shapes:
        nD tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    Output shapes:
        nD tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

  def __init__(self,
               slot_names,
               embedding_size_choices_list,
               warmup_steps,
               pretraining_steps,
               teacher_embedding_sizes_list=None,
               distillation_mask=False,
               **kwargs):
    super(MixedEmbedOpComb, self).__init__(**kwargs)
    print(len(slot_names))
    print(len(embedding_size_choices_list))
    assert len(slot_names) == len(embedding_size_choices_list)
    self._slot_names = slot_names
    self._embedding_size_choices_list = embedding_size_choices_list
    self._num_choices_per_embedding = []
    self._max_choice_per_embedding = []
    self._max_num_choices = 0
    self._total_emb_size = 0
    self._warmup_steps = warmup_steps
    self._pretraining_steps = pretraining_steps
    for embedding_size_choices in embedding_size_choices_list:
      self._num_choices_per_embedding.append(len(embedding_size_choices))
      self._max_num_choices = max(self._max_num_choices,
                                  len(embedding_size_choices))
      self._max_choice_per_embedding.append(sum(embedding_size_choices))
      self._total_emb_size += self._max_choice_per_embedding[-1]
    self._teacher_embedding_sizes_list = None

    self._arch_embedding_weights_multipler = None
    self._arch_embedding_weights = None

    # allowed input specification
    if self._teacher_embedding_sizes_list is not None:
      self.input_spec = [InputSpec(ndim=2), InputSpec(ndim=2)]
    else:
      self.input_spec = InputSpec(ndim=2)

    self._distillation_mask = distillation_mask

  def build(self, input_shape):
    assert len(input_shape) == 2
    if self._teacher_embedding_sizes_list is not None:
      assert len(input_shape[0]) == 2 and len(input_shape[1]) == 2
      input_dim = input_shape[0][-1]
      assert input_shape[1][-1] == sum(self._teacher_embedding_sizes_list)
    else:
      input_dim = input_shape[-1]
    print(input_dim)
    print(self._total_emb_size)
    assert input_dim == self._total_emb_size

    # kernel
    self._arch_embedding_weights = self.add_weight(
        shape=(sum(self._num_choices_per_embedding),),
        initializer=tf.random_uniform_initializer(minval=-1e-3, maxval=1e-3),
        trainable=True,
        name='arch_embedding_weights')
    print("arch embedding weights: {}".format(self._arch_embedding_weights))

    current_idx = 0
    arch_embedding_masks_list = []
    arch_embedding_weights_multiplier_list = []
    arch_entropy_list = []
    expected_emb_dims_list = []
    expected_zero_embedding_size_weights_list = []
    arch_embedding_weights_after_softmax_list = []

    for i in range(len(self._slot_names)):
      num_choices = self._num_choices_per_embedding[i]
      max_emb_choice = sum(self._embedding_size_choices_list[i])
      arch_embedding_weights_slice = self._arch_embedding_weights[
          current_idx:current_idx + num_choices]
      arch_embedding_weights_after_softmax = tf.nn.softmax(
          arch_embedding_weights_slice)  #softmax selection
      #arch_embedding_weights_after_softmax = tf.math.sigmoid(arch_embedding_weights_slice) #sigmoid selection like FairNAS
      arch_entropy_list.append(
          tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
              labels=arch_embedding_weights_after_softmax,
              logits=arch_embedding_weights_slice))
      expected_emb_dims_list.append(
          tf.reduce_sum(arch_embedding_weights_after_softmax *
                        self._embedding_size_choices_list[i]))

      if self._embedding_size_choices_list[i][0] == 0:
        expected_zero_embedding_size_weights_list.append(
            arch_embedding_weights_after_softmax[0])
        arch_embedding_weights_after_softmax = tf.concat([
            arch_embedding_weights_after_softmax[0:1] * tf.cast(
                tf.minimum(
                    tf.maximum(
                        tf.compat.v1.train.get_global_step() -
                        self._warmup_steps, 0.0), 1.0), self.dtype),
            arch_embedding_weights_after_softmax[1:]
        ],
                                                         axis=0)
      embedding_masks = []
      lower = 0
      upper = 0
      for j, embedding_size_choice in enumerate(
          self._embedding_size_choices_list[i]):
        name = 'arch_embedding_weights_after_softmax/{}_{}'.format(
            self._slot_names[i], embedding_size_choice)
        data = arch_embedding_weights_after_softmax[j]
        arch_embedding_weights_after_softmax_list.append((name, data))
        upper += embedding_size_choice
        mask = tf.constant([
            1.0 if jj < upper and jj >= lower else 0.0
            for jj in range(max_emb_choice)
        ],
                           dtype=self.dtype)
        #mask = tf.constant([
        #    1.0 / embedding_size_choice if jj < upper and jj >= lower else 0.0
        #    for jj in range(max_emb_choice)
        #],
        #                   dtype=self.dtype) # Balance the gradient of each slot choices
        lower += embedding_size_choice
        embedding_masks.append(mask)
      # [self._max_num_choices, max_emb_choice]
      embedding_mask = tf.pad(
          tf.stack(embedding_masks,
                   0), [[0, self._max_num_choices - num_choices], [0, 0]])
      #print('embedding_mask: {}'.format(tf.keras.backend.eval(embedding_mask)))
      # [self._max_num_choices]
      arch_embedding_weights_after_softmax_per_slot_padded = tf.pad(
          arch_embedding_weights_after_softmax,
          [[0, self._max_num_choices - num_choices]])
      probability = tf.where(
          tf.cast(tf.compat.v1.train.get_or_create_global_step(), tf.float32) <
          tf.cast(self._pretraining_steps, tf.float32),
          [0.5, 0.5],  #[0.25, 0.25, 0.25, 0.25]
          [
              arch_embedding_weights_after_softmax_per_slot_padded[i]
              for i in range(self._max_num_choices)
          ])
      # [max_emb_choice] # method 1: Sampling-based nws
      indices = tf.random.categorical(
          tf.math.log(tf.expand_dims(probability, 0)), 1)
      index = tf.reduce_sum(indices)
      index_one_hot = tf.one_hot(index, self._max_num_choices)
      embedding_masks_selected = tf.reduce_sum(
          embedding_masks * tf.expand_dims(index_one_hot, -1), 0)
      arch_embedding_weights_after_softmax_per_slot_padded_chosen = tf.reduce_sum(
          arch_embedding_weights_after_softmax_per_slot_padded * index_one_hot,
          0)
      arch_embedding_masks_list.append(
          embedding_masks_selected *
          (1 + arch_embedding_weights_after_softmax_per_slot_padded_chosen -
           tf.stop_gradient(
               arch_embedding_weights_after_softmax_per_slot_padded_chosen)))

      # [max_emb_choice] # method 2: MixedOp-based nws
      #arch_embedding_masks_list.append(embedding_mask)
      #arch_embedding_weights_multiplier_list.append(
      #    tf.broadcast_to(
      #        tf.expand_dims(
      #            probability,
      #            -1), tf.shape(embedding_mask)
      #            )
      #           )
      current_idx += num_choices

    # [total_emb_dims]
    self._arch_embedding_masks_multipler = tf.concat(
        arch_embedding_masks_list, 0)  # method 1: Sampling-based nws
    #self._arch_embedding_masks_multipler = tf.concat(
    #    arch_embedding_masks_list, 1) #method 2: MixedOp-based nws
    #self._arch_embedding_weights_multipler = tf.concat(
    #    arch_embedding_weights_multiplier_list, 1)
    self._arch_entropy = tf.add_n(arch_entropy_list)
    self._expected_emb_dims = tf.add_n(expected_emb_dims_list)
    self._expected_zero_embedding_size_weights = tf.add_n(
        expected_zero_embedding_size_weights_list
    ) if expected_zero_embedding_size_weights_list else 0
    self._arch_embedding_weights_after_softmax_list = arch_embedding_weights_after_softmax_list
    self.built = True

  def call(self, inputs):
    if self._teacher_embedding_sizes_list is not None:
      embedding = inputs[0]
      teacher_embedding = inputs[1]
    else:
      embedding = inputs

    # [batch_size, total_emb_dims]
    masked_embedding = embedding * self._arch_embedding_masks_multipler  # method 1: Sampling-based nws
    #masked_embedding = tf.expand_dims(
    #    embedding, 1) * self._arch_embedding_masks_multipler # method 2: MixedOp-based nws
    #mixed_embedding = tf.reduce_sum(
    #    masked_embedding * self._arch_embedding_weights_multipler, 1)

    if self._teacher_embedding_sizes_list is not None:
      print("TeacherEmbeddingTransform: {} {}".format(
          self._max_choice_per_embedding, self._teacher_embedding_sizes_list))
      teacher_embedding_transform = TeacherEmbeddingTransform(
          self._max_choice_per_embedding,
          self._teacher_embedding_sizes_list,
          dtype=self.dtype)
      teacher_embedding_transformed = teacher_embedding_transform(
          teacher_embedding)
      # [batch_size, total_emb_dims] -> [batch_size, 1, total_emb_dims]
      # -> [batch_size, self._max_num_choices, total_emb_dims]
      if not self._distillation_mask:
        distillation_loss = tf.losses.mean_squared_error(
            tf.broadcast_to(tf.expand_dims(teacher_embedding_transformed, 1),
                            tf.shape(masked_embedding)), masked_embedding)
      else:
        masked_teacher_embedding_transformed = tf.expand_dims(
            teacher_embedding_transformed,
            1) * self._arch_embedding_masks_multipler
        distillation_loss = tf.losses.mean_squared_error(
            masked_teacher_embedding_transformed, masked_embedding)
      return mixed_embedding, distillation_loss, teacher_embedding_transform.name
    else:
      return masked_embedding  # method 1: Sampling-based nws
      #return mixed_embedding # method 2: MixedOp-based nws

  def compute_output_shape(self, input_shape):
    raise NotImplementedError("I don't think I need to implement this one.")

  def get_config(self):
    config = {
        'slot_names': self._slot_names,
        'embedding_size_choices_list': self._embedding_size_choices_list,
        'warmup_steps': self._warmup_steps,
        'teacher_embedding_sizes_list': self._teacher_embedding_sizes_list,
    }
    base_config = super(MixedEmbedOpComb, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_arch_embedding_weights(self):
    return self._arch_embedding_weights

  def get_summaries(self):
    return {
        'arch_entropy':
            self._arch_entropy,
        'expected_emb_dims':
            self._expected_emb_dims,
        'arch_weights_after_softmax_list':
            self._arch_embedding_weights_after_softmax_list,
    }
