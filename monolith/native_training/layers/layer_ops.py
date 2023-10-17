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
from absl import logging
from typing import Tuple

from monolith.native_training.runtime.ops import gen_monolith_ops

layer_ops_lib = gen_monolith_ops


def ffm(left: tf.Tensor,
        right: tf.Tensor,
        dim_size: int,
        int_type: str = 'multiply') -> tf.Tensor:
  output = layer_ops_lib.FFM(left=left,
                             right=right,
                             dim_size=dim_size,
                             int_type=int_type)
  return output


@tf.RegisterGradient('FFM')
def _ffm_grad(op, grad: tf.Tensor) -> tf.Tensor:
  left, right = op.inputs[0], op.inputs[1]
  dim_size = op.get_attr('dim_size')
  int_type = op.get_attr('int_type')

  (left_grad, right_grad) = layer_ops_lib.FFMGrad(grad=grad,
                                                  left=left,
                                                  right=right,
                                                  dim_size=dim_size,
                                                  int_type=int_type)
  return left_grad, right_grad


def feature_insight(input_embedding,
                    weight,
                    segment_sizes,
                    aggregate: bool = False) -> tf.Tensor:
  assert segment_sizes
  assert input_embedding.shape.as_list()[-1] == weight.shape.as_list()[0]
  out = layer_ops_lib.FeatureInsight(input=input_embedding,
                                     weight=weight,
                                     segment_sizes=segment_sizes)
  if aggregate:
    k, num_feature = weight.shape.as_list()[-1], len(segment_sizes)
    segment_ids = []
    for i in range(num_feature):
      segment_ids.extend([i] * k)
    segment_ids_tensor = tf.constant(value=segment_ids,
                                     shape=(k * num_feature,),
                                     dtype=tf.int32)
    return tf.transpose(
        tf.math.segment_sum(tf.transpose(out * out),
                            segment_ids=segment_ids_tensor))
    pass
  else:
    return out


@tf.RegisterGradient('FeatureInsight')
def _feature_insight(op, grad: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
  input_embedding, weight = op.inputs[0], op.inputs[1]
  segment_sizes = op.get_attr('segment_sizes')
  k = weight.shape.as_list()[-1]
  input_embedding_grad, weight_grad = layer_ops_lib.FeatureInsightGrad(
      grad=grad,
      input=input_embedding,
      weight=weight,
      segment_sizes=segment_sizes,
      K=k)
  return input_embedding_grad, weight_grad




def fid_counter(counter: tf.Tensor, counter_threshold: int, step: float = 1.0):
  """Count element value(e.g. embedding/label), will consume 1-size vector as counter
  Args:
    counter(Tensor): feature slice to store counter
    counter_threshold(int): threshold set step to 0
    step(Tensor): value add to counter
  Returns:
    counter: counter value
  Attention:
    1. fid_counter's input embedding MUST use SgdOptimizer(1.0).
    2. We recommend using Fp32Compressor() for counter slice.
    3. If you use Fp16Compressor(), for precision reason, we recommend setting counter_threshold to 60000.

  Example::
      >>> item_count = self.embedding_lookup(slice_name='item_count',
                                             slots=[534],
                                             dim=1,
                                             initializer= ConstantsInitializer(1.0),
                                             optimizer= SgdOptimizer(1.0),
                                             compressor= Fp32Compressor())
      >>> item_count = layer_ops.fid_counter(item_count, step=1)
      >>> item_count = tf.reshape(item_count, shape=(-1, ))
      >>> item_weights = 1 / (1 + tf.math.exp(4 - 0.03 * item_count))
  """
  counter = layer_ops_lib.MonolithFidCounter(
      counter=counter, step=step, counter_threshold=counter_threshold)
  counter = counter + tf.cast(step, counter.dtype)
  counter = tf.where(
      counter > counter_threshold,
      tf.ones_like(counter) * tf.cast(counter_threshold, counter.dtype),
      counter)
  return counter


@tf.RegisterGradient('MonolithFidCounter')
def _fid_counter_grad(op, grad: tf.Tensor) -> tf.Tensor:
  counter = op.inputs[0]
  step = op.get_attr('step')
  grad = tf.ones_like(counter) * tf.cast(-step, counter.dtype)
  counter_threshold = op.get_attr('counter_threshold')
  grad = tf.where(counter >= counter_threshold, tf.zeros_like(grad), grad)
  return grad
