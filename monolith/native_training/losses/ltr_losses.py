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

import abc
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.losses import losses as core_losses

# The smallest probability that is used to derive smallest logit for invalid or
# padding entries.
_EPSILON = 1e-10


def label_valid_fn(labels):
  """Returns a boolean `Tensor` for label validity."""
  labels = ops.convert_to_tensor(labels)
  return math_ops.greater_equal(labels, 0.)


def sort_by_scores(scores, features_list, topn=None):
  """Sorts example features according to per-example scores.

  Args:
    scores: A `Tensor` of shape [batch_size, list_size] representing the
      per-example scores.
    features_list: A list of `Tensor`s with the same shape as scores to be
      sorted.
    topn: An integer as the cutoff of examples in the sorted list.

  Returns:
    A list of `Tensor`s as the list of sorted features by `scores`.
  """
  scores = ops.convert_to_tensor(scores)
  scores.get_shape().assert_has_rank(2)
  batch_size, list_size = array_ops.unstack(array_ops.shape(scores))
  if topn is None:
    topn = list_size
  topn = math_ops.minimum(topn, list_size)
  _, indices = nn_ops.top_k(scores, topn, sorted=True)
  list_offsets = array_ops.expand_dims(
      math_ops.range(batch_size) * list_size, 1)
  # The shape of `indices` is [batch_size, topn] and the shape of
  # `list_offsets` is [batch_size, 1]. Broadcasting is used here.
  gather_indices = array_ops.reshape(indices + list_offsets, [-1])
  output_shape = array_ops.stack([batch_size, topn])
  # Each feature is first flattened to a 1-D vector and then gathered by the
  # indices from sorted scores and then re-shaped.
  return [
      array_ops.reshape(
          array_ops.gather(array_ops.reshape(feature, [-1]), gather_indices),
          output_shape) for feature in features_list
  ]


def organize_valid_indices(is_valid, shuffle=True, seed=None):
  """Organizes indices in such a way that valid items appear first.

  Args:
    is_valid: A boolen `Tensor` for entry validity with shape [batch_size,
      list_size].
    shuffle: A boolean indicating whether valid items should be shuffled.
    seed: An int for random seed at the op level. It works together with the
      seed at global graph level together to determine the random number
      generation. See `tf.set_random_seed`.

  Returns:
    A tensor of indices with shape [batch_size, list_size, 2]. The returned
    tensor can be used with `tf.gather_nd` and `tf.scatter_nd` to compose a new
    [batch_size, list_size] tensor. The values in the last dimension are the
    indices for an element in the input tensor.
  """
  is_valid = ops.convert_to_tensor(is_valid)
  is_valid.get_shape().assert_has_rank(2)
  output_shape = array_ops.shape(is_valid)

  if shuffle:
    values = random_ops.random_uniform(output_shape, seed=seed)
  else:
    values = (array_ops.ones_like(is_valid, dtypes.float32) * array_ops.reverse(
        math_ops.to_float(math_ops.range(output_shape[1])), [-1]))

  rand = array_ops.where(is_valid, values, array_ops.ones(output_shape) * -1e-6)
  # shape(indices) = [batch_size, list_size]
  _, indices = nn_ops.top_k(rand, output_shape[1], sorted=True)
  # shape(batch_ids) = [batch_size, list_size]
  batch_ids = array_ops.ones_like(indices) * array_ops.expand_dims(
      math_ops.range(output_shape[0]), 1)
  return array_ops.concat(
      [
          array_ops.expand_dims(batch_ids, 2),  #[[0,...0], [1, ..., 1]]
          array_ops.expand_dims(indices, 2)
      ],  # shuffle之后的indices，dim0=batch  dim1=shuffle后的index,例如[0, 1, 2, 3] 变为[2,3,0,1]，为list的长度
      axis=2)


def shuffle_valid_indices(is_valid, seed=None):
  """Returns a shuffle of indices with valid ones on top."""
  return organize_valid_indices(is_valid, shuffle=True, seed=seed)


def reshape_first_ndims(tensor, first_ndims, new_shape):
  """Reshapes the first n dims of the input `tensor` to `new shape`.

  Args:
    tensor: The input `Tensor`.
    first_ndims: A int denoting the first n dims.
    new_shape: A list of int representing the new shape.

  Returns:
    A reshaped `Tensor`.
  """
  assert tensor.get_shape().ndims is None or tensor.get_shape(
  ).ndims >= first_ndims, (
      'Tensor shape is less than {} dims.'.format(first_ndims))
  new_shape = array_ops.concat(
      [new_shape, array_ops.shape(tensor)[first_ndims:]], 0)
  if isinstance(tensor, sparse_tensor.SparseTensor):
    return sparse_ops.sparse_reshape(tensor, new_shape)

  return array_ops.reshape(tensor, new_shape)


def approx_ranks(logits, alpha=10.):
  r"""Computes approximate ranks given a list of logits.

  Given a list of logits, the rank of an item in the list is simply
  one plus the total number of items with a larger logit. In other words,

    rank_i = 1 + \sum_{j \neq i} I_{s_j > s_i},

  where "I" is the indicator function. The indicator function can be
  approximated by a generalized sigmoid:

    I_{s_j < s_i} \approx 1/(1 + exp(-\alpha * (s_j - s_i))).

  This function approximates the rank of an item using this sigmoid
  approximation to the indicator function. This technique is at the core
  of "A general approximation framework for direct optimization of
  information retrieval measures" by Qin et al.

  Args:
    logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item.
    alpha: Exponent of the generalized sigmoid function.

  Returns:
    A `Tensor` of ranks with the same shape as logits.
  """
  list_size = array_ops.shape(logits)[1]
  x = array_ops.tile(array_ops.expand_dims(logits, 2), [1, 1, list_size])
  y = array_ops.tile(array_ops.expand_dims(logits, 1), [1, list_size, 1])
  pairs = math_ops.sigmoid(alpha * (y - x))
  return math_ops.reduce_sum(pairs, -1) + .5


def inverse_max_dcg(labels,
                    gain_fn=lambda labels: math_ops.pow(2.0, labels) - 1.,
                    rank_discount_fn=lambda rank: 1. / math_ops.log1p(rank),
                    topn=None):
  """Computes the inverse of max DCG.

  Args:
    labels: A `Tensor` with shape [batch_size, list_size]. Each value is the
      graded relevance of the corresponding item.
    gain_fn: A gain function. By default this is set to: 2^label - 1.
    rank_discount_fn: A discount function. By default this is set to:
      1/log(1+rank).
    topn: An integer as the cutoff of examples in the sorted list.
  Returns:
    A `Tensor` with shape [batch_size, 1].
  """
  ideal_sorted_labels, = sort_by_scores(labels, [labels], topn=topn)
  rank = math_ops.range(array_ops.shape(ideal_sorted_labels)[1]) + 1
  discounted_gain = gain_fn(ideal_sorted_labels) * rank_discount_fn(
      math_ops.to_float(rank))
  discounted_gain = math_ops.reduce_sum(discounted_gain, 1, keepdims=True)
  return array_ops.where(math_ops.greater(discounted_gain, 0.),
                         1. / discounted_gain,
                         array_ops.zeros_like(discounted_gain))


def get_batch_idx_size(logits, labels, rank_id, name_prefix):
  batch_size = tf.shape(logits)[0]
  rank_key, rank_idx, count = tf.unique_with_counts(rank_id)
  max_count = tf.reduce_max(count)
  unique_rank_id_num = tf.shape(count)[0]
  rank_idx_tile = tf.tile(tf.expand_dims(rank_idx, 0), [unique_rank_id_num, 1])
  range_count_tile = tf.tile(tf.expand_dims(tf.range(unique_rank_id_num), 1),
                             [1, batch_size])
  list_id_mask = tf.cast(tf.equal(rank_idx_tile, range_count_tile), tf.int32)
  cum_mask = tf.cumsum(list_id_mask, axis=1, exclusive=True)
  masked_list_id = list_id_mask * cum_mask
  col_cor = tf.reduce_sum(masked_list_id, axis=0)
  row_cor = rank_idx
  batch_idx = tf.concat(
      [tf.expand_dims(row_cor, 1),
       tf.expand_dims(col_cor, 1)], axis=1)
  output_shape = [tf.shape(count)[0], max_count]

  logits_idx = tf.scatter_nd(updates=logits,
                             indices=batch_idx,
                             shape=output_shape,
                             name=name_prefix + "logits_idx")
  label_idx = tf.scatter_nd(updates=labels,
                            indices=batch_idx,
                            shape=output_shape,
                            name=name_prefix + "label_idx") - 1e-6
  mask_idx = tf.scatter_nd(updates=tf.ones_like(logits),
                           indices=batch_idx,
                           shape=[batch_size, batch_size],
                           name=name_prefix + 'mask_idx')
  unique_idx = tf.argmax(list_id_mask, axis=1)
  return logits_idx, label_idx, mask_idx, unique_idx


class RankingLossKey(object):
  """Ranking loss key strings."""
  # Names for the ranking based loss functions.
  PAIRWISE_HINGE_LOSS = 'pairwise_hinge_loss'
  PAIRWISE_LOGISTIC_LOSS = 'pairwise_logistic_loss'
  PAIRWISE_SOFT_ZERO_ONE_LOSS = 'pairwise_soft_zero_one_loss'
  SOFTMAX_LOSS = 'softmax_loss'
  SIGMOID_CROSS_ENTROPY_LOSS = 'sigmoid_cross_entropy_loss'
  MEAN_SQUARED_LOSS = 'mean_squared_loss'
  LIST_MLE_LOSS = 'list_mle_loss'
  APPROX_NDCG_LOSS = 'approx_ndcg_loss'


def make_loss_fn(loss_keys,
                 loss_weights=None,
                 weights_feature_name=None,
                 lambda_weight=None,
                 reduction=core_losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
                 name=None,
                 seed=None,
                 extra_args=None):
  """Makes a loss function using a single loss or multiple losses.

  Args:
    loss_keys: A string or list of strings representing loss keys defined in
      `RankingLossKey`. Listed loss functions will be combined in a weighted
      manner, with weights specified by `loss_weights`. If `loss_weights` is
      None, default weight of 1 will be used.
    loss_weights: List of weights, same length as `loss_keys`. Used when merging
      losses to calculate the weighted sum of losses. If `None`, all losses are
      weighted equally with weight being 1.
    weights_feature_name: A string specifying the name of the weights feature in
      `features` dict.
    lambda_weight: A `_LambdaWeight` object created by factory methods like
      `create_ndcg_lambda_weight()`.
    reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch.
    name: A string used as the name for this loss.
    seed: A randomization seed used in computation of some loss functions such
      as ListMLE and pListMLE.
    extra_args: A string-keyed dictionary that contains any other loss-specific
      arguments.

  Returns:
    A function _loss_fn(). See `_loss_fn()` for its signature.

  Raises:
    ValueError: If `reduction` is invalid.
    ValueError: If `loss_keys` is None or empty.
    ValueError: If `loss_keys` and `loss_weights` have different sizes.
  """
  if (reduction not in core_losses.Reduction.all() or
      reduction == core_losses.Reduction.NONE):
    raise ValueError('Invalid reduction: {}'.format(reduction))

  if not loss_keys:
    raise ValueError('loss_keys cannot be None or empty.')

  if loss_weights:
    if len(loss_keys) != len(loss_weights):
      raise ValueError('loss_keys and loss_weights must have the same size.')

  if not isinstance(loss_keys, list):
    loss_keys = [loss_keys]

  def _loss_fn(labels, logits, features):
    """Computes a single loss or weighted combination of losses.

    Args:
      labels: A `Tensor` of the same shape as `logits` representing relevance.
      logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
        ranking score of the corresponding item.
      features: Dict of Tensors of shape [batch_size, list_size, ...] for
        per-example features and shape [batch_size, ...] for non-example context
        features.

    Returns:
      An op for a single loss or weighted combination of multiple losses.

    Raises:
      ValueError: If `loss_keys` is invalid.
    """
    weights = features[weights_feature_name] if weights_feature_name else None
    loss_kwargs = {
        'labels': labels,
        'logits': logits,
        'weights': weights,
        'reduction': reduction,
        'name': name,
    }
    if extra_args is not None:
      loss_kwargs.update(extra_args)

    loss_kwargs_with_lambda_weight = loss_kwargs.copy()
    loss_kwargs_with_lambda_weight['lambda_weight'] = lambda_weight

    loss_kwargs_with_lambda_weight_and_seed = (
        loss_kwargs_with_lambda_weight.copy())
    loss_kwargs_with_lambda_weight_and_seed['seed'] = seed

    key_to_fn = {
        RankingLossKey.PAIRWISE_HINGE_LOSS:
            (_pairwise_hinge_loss, loss_kwargs_with_lambda_weight),
        RankingLossKey.PAIRWISE_LOGISTIC_LOSS:
            (_pairwise_logistic_loss, loss_kwargs_with_lambda_weight),
        RankingLossKey.PAIRWISE_SOFT_ZERO_ONE_LOSS:
            (_pairwise_soft_zero_one_loss, loss_kwargs_with_lambda_weight),
        RankingLossKey.SOFTMAX_LOSS:
            (_softmax_loss, loss_kwargs_with_lambda_weight),
        RankingLossKey.SIGMOID_CROSS_ENTROPY_LOSS:
            (_sigmoid_cross_entropy_loss, loss_kwargs),
        RankingLossKey.MEAN_SQUARED_LOSS: (_mean_squared_loss, loss_kwargs),
        RankingLossKey.LIST_MLE_LOSS:
            (_list_mle_loss, loss_kwargs_with_lambda_weight_and_seed),
        RankingLossKey.APPROX_NDCG_LOSS: (_approx_ndcg_loss, loss_kwargs),
    }

    # Obtain the list of loss ops.
    loss_ops = []
    for loss_key in loss_keys:
      if loss_key not in key_to_fn:
        raise ValueError('Invalid loss_key: {}.'.format(loss_key))
      loss_fn, kwargs = key_to_fn[loss_key]
      loss_ops.append(loss_fn(**kwargs))

    # Compute weighted combination of losses.
    if loss_weights:
      weighted_losses = []
      for loss_op, loss_weight in zip(loss_ops, loss_weights):
        weighted_losses.append(math_ops.multiply(loss_op, loss_weight))
    else:
      weighted_losses = loss_ops

    return math_ops.add_n(weighted_losses)

  return _loss_fn


def create_ndcg_lambda_weight(topn=None, smooth_fraction=0.):
  """Creates _LambdaWeight for NDCG metric."""
  return DCGLambdaWeight(
      topn,
      gain_fn=lambda labels: math_ops.pow(2.0, labels) - 1.,
      rank_discount_fn=lambda rank: 1. / math_ops.log1p(rank),
      normalized=True,
      smooth_fraction=smooth_fraction)


def create_reciprocal_rank_lambda_weight(topn=None, smooth_fraction=0.):
  """Creates _LambdaWeight for MRR-like metric."""
  return DCGLambdaWeight(topn,
                         gain_fn=lambda labels: labels,
                         rank_discount_fn=lambda rank: 1. / rank,
                         normalized=True,
                         smooth_fraction=smooth_fraction)


def create_p_list_mle_lambda_weight(list_size):
  """Creates _LambdaWeight based on Position-Aware ListMLE paper.

  Produces a weight based on the formulation presented in the
  "Position-Aware ListMLE" paper (Lan et al.) and available using
  create_p_list_mle_lambda_weight() factory function above.

  Args:
    list_size: Size of the input list.

  Returns:
    A _LambdaWeight for Position-Aware ListMLE.
  """
  return ListMLELambdaWeight(
      rank_discount_fn=lambda rank: math_ops.pow(2., list_size - rank) - 1.)


class _LambdaWeight(object):
  """Interface for ranking metric optimization.

  This class wraps weights used in the LambdaLoss framework for ranking metric
  optimization (https://ai.google/research/pubs/pub47258). Such an interface is
  to be instantiated by concrete lambda weight models. The instance is used
  together with standard loss such as logistic loss and softmax loss.
  """

  __metaclass__ = abc.ABCMeta

  def _get_valid_pairs_and_clean_labels(self, sorted_labels):
    """Returns a boolean Tensor for valid pairs and cleaned labels."""
    sorted_labels = ops.convert_to_tensor(sorted_labels)
    sorted_labels.get_shape().assert_has_rank(2)
    is_label_valid = label_valid_fn(sorted_labels)
    valid_pairs = math_ops.logical_and(array_ops.expand_dims(is_label_valid, 2),
                                       array_ops.expand_dims(is_label_valid, 1))
    sorted_labels = array_ops.where(is_label_valid, sorted_labels,
                                    array_ops.zeros_like(sorted_labels))
    return valid_pairs, sorted_labels

  @abc.abstractmethod
  def pair_weights(self, sorted_labels):
    """Returns the weight adjustment `Tensor` for example pairs.

    Args:
      sorted_labels: A dense `Tensor` of labels with shape [batch_size,
        list_size] that are sorted by logits.

    Returns:
      A `Tensor` that can weight example pairs.
    """
    raise NotImplementedError('Calling an abstract method.')

  def individual_weights(self, sorted_labels):
    """Returns the weight `Tensor` for individual examples.

    Args:
      sorted_labels: A dense `Tensor` of labels with shape [batch_size,
        list_size] that are sorted by logits.

    Returns:
      A `Tensor` that can weight individual examples.
    """
    return sorted_labels


class IdentityLambdaWeight(_LambdaWeight):

  def __init__(self,):
    pass

  def pair_weights(self, sorted_labels):
    return 1.0


class DCGLambdaWeight(_LambdaWeight):
  """LambdaWeight for Discounted Cumulative Gain metric."""

  def __init__(self,
               topn=None,
               gain_fn=lambda label: label,
               rank_discount_fn=lambda rank: 1. / rank,
               normalized=False,
               smooth_fraction=0.):
    """Constructor.

    Ranks are 1-based, not 0-based. Given rank i and j, there are two types of
    pair weights:
      u = |rank_discount_fn(|i-j|) - rank_discount_fn(|i-j| + 1)|
      v = |rank_discount_fn(i) - rank_discount_fn(j)|
    where u is the newly introduced one in LambdaLoss paper
    (https://ai.google/research/pubs/pub47258) and v is the original one in the
    LambdaMART paper "From RankNet to LambdaRank to LambdaMART: An Overview".
    The final pair weight contribution of ranks is
      (1-smooth_fraction) * u + smooth_fraction * v.

    Args:
      topn: (int) The topn for the DCG metric.
      gain_fn: (function) Tranforms labels.
      rank_discount_fn: (function) The rank discount function.
      normalized: (bool) If True, normalize weight by the max DCG.
      smooth_fraction: (float) parameter to control the contribution from
        LambdaMART.
    """
    self._topn = topn
    self._gain_fn = gain_fn
    self._rank_discount_fn = rank_discount_fn
    self._normalized = normalized
    assert 0. <= smooth_fraction and smooth_fraction <= 1., (
        'smooth_fraction %s should be in range [0, 1].' % smooth_fraction)
    self._smooth_fraction = smooth_fraction

  def pair_weights(self, sorted_labels):
    """See `_LambdaWeight`."""
    with ops.name_scope(None, 'dcg_lambda_weight', (sorted_labels,)):
      valid_pair, sorted_labels = self._get_valid_pairs_and_clean_labels(
          sorted_labels)
      gain = self._gain_fn(sorted_labels)
      if self._normalized:
        gain *= inverse_max_dcg(sorted_labels,
                                gain_fn=self._gain_fn,
                                rank_discount_fn=self._rank_discount_fn,
                                topn=self._topn)
      pair_gain = array_ops.expand_dims(gain, 2) - array_ops.expand_dims(
          gain, 1)
      pair_gain *= math_ops.to_float(valid_pair)

      list_size = array_ops.shape(sorted_labels)[1]
      topn = self._topn or list_size
      rank = math_ops.range(list_size) + 1

      def _discount_for_relative_rank_diff():
        """Rank-based discount in the LambdaLoss paper."""
        # The LambdaLoss is not well defined when topn is active and topn <
        # list_size. We cap the rank of examples to topn + 1 so that the rank
        # differene is capped to topn. This is just a convenient upperbound
        # when topn is active. We need to revisit this.
        capped_rank = array_ops.where(math_ops.greater(rank, topn),
                                      array_ops.ones_like(rank) * (topn + 1),
                                      rank)
        rank_diff = math_ops.to_float(
            math_ops.abs(
                array_ops.expand_dims(capped_rank, 1) -
                array_ops.expand_dims(capped_rank, 0)))
        pair_discount = array_ops.where(
            math_ops.greater(rank_diff, 0),
            math_ops.abs(
                self._rank_discount_fn(rank_diff) -
                self._rank_discount_fn(rank_diff + 1)),
            array_ops.zeros_like(rank_diff))
        return pair_discount

      def _discount_for_absolute_rank():
        """Standard discount in the LambdaMART paper."""
        # When the rank discount is (1 / rank) for example, the discount is
        # |1 / r_i - 1 / r_j|. When i or j > topn, the discount becomes 0.
        rank_discount = array_ops.where(
            math_ops.greater(rank, topn),
            array_ops.zeros_like(math_ops.to_float(rank)),
            self._rank_discount_fn(math_ops.to_float(rank)))
        pair_discount = math_ops.abs(
            array_ops.expand_dims(rank_discount, 1) -
            array_ops.expand_dims(rank_discount, 0))
        return pair_discount

      u = _discount_for_relative_rank_diff()
      v = _discount_for_absolute_rank()
      pair_discount = (1. -
                       self._smooth_fraction) * u + self._smooth_fraction * v
      pair_weight = math_ops.abs(pair_gain) * pair_discount
      if self._topn is None:
        return pair_weight
      pair_mask = math_ops.logical_or(
          array_ops.expand_dims(math_ops.less_equal(rank, self._topn), 1),
          array_ops.expand_dims(math_ops.less_equal(rank, self._topn), 0))
      return pair_weight * math_ops.to_float(pair_mask)

  def individual_weights(self, sorted_labels):
    """See `_LambdaWeight`."""
    with ops.name_scope(None, 'dcg_lambda_weight', (sorted_labels,)):
      sorted_labels = ops.convert_to_tensor(sorted_labels)
      sorted_labels = array_ops.where(label_valid_fn(sorted_labels),
                                      sorted_labels,
                                      array_ops.zeros_like(sorted_labels))
      gain = self._gain_fn(sorted_labels)
      if self._normalized:
        gain *= inverse_max_dcg(sorted_labels,
                                gain_fn=self._gain_fn,
                                rank_discount_fn=self._rank_discount_fn,
                                topn=self._topn)
      rank_discount = self._rank_discount_fn(
          math_ops.to_float(
              math_ops.range(array_ops.shape(sorted_labels)[1]) + 1))
      return gain * rank_discount


class PrecisionLambdaWeight(_LambdaWeight):
  """LambdaWeight for Precision metric."""

  def __init__(self,
               topn,
               positive_fn=lambda label: math_ops.greater_equal(label, 1.0)):
    """Constructor.

    Args:
      topn: (int) The K in Precision@K metric.
      positive_fn: (function): A function on `Tensor` that output boolean True
        for positive examples. The rest are negative examples.
    """
    self._topn = topn
    self._positive_fn = positive_fn

  def pair_weights(self, sorted_labels):
    """See `_LambdaWeight`.

    The current implementation here is that for any pairs of documents i and j,
    we set the weight to be 1 if
      - i and j have different labels.
      - i <= topn and j > topn or i > topn and j <= topn.
    This is exactly the same as the original LambdaRank method. The weight is
    the gain of swapping a pair of documents.

    Args:
      sorted_labels: A dense `Tensor` of labels with shape [batch_size,
        list_size] that are sorted by logits.

    Returns:
      A `Tensor` that can weight example pairs.
    """
    with ops.name_scope(None, 'precision_lambda_weight', (sorted_labels,)):
      valid_pair, sorted_labels = self._get_valid_pairs_and_clean_labels(
          sorted_labels)
      binary_labels = math_ops.to_float(self._positive_fn(sorted_labels))
      label_diff = math_ops.abs(
          array_ops.expand_dims(binary_labels, 2) -
          array_ops.expand_dims(binary_labels, 1))
      label_diff *= math_ops.to_float(valid_pair)
      # i <= topn and j > topn or i > topn and j <= topn, i.e., xor(i <= topn, j
      # <= topn).
      list_size = array_ops.shape(sorted_labels)[1]
      rank = math_ops.range(list_size) + 1
      rank_mask = math_ops.logical_xor(
          array_ops.expand_dims(math_ops.less_equal(rank, self._topn), 1),
          array_ops.expand_dims(math_ops.less_equal(rank, self._topn), 0))
      return label_diff * math_ops.to_float(rank_mask)


class ListMLELambdaWeight(_LambdaWeight):
  """LambdaWeight for ListMLE cost function."""

  def __init__(self, rank_discount_fn):
    """Constructor.

    Ranks are 1-based, not 0-based.

    Args:
      rank_discount_fn: (function) The rank discount function.
    """
    self._rank_discount_fn = rank_discount_fn

  def pair_weights(self, sorted_labels):
    """See `_LambdaWeight`."""
    return sorted_labels

  def individual_weights(self, sorted_labels):
    """See `_LambdaWeight`."""
    with ops.name_scope(None, 'p_list_mle_lambda_weight', (sorted_labels,)):
      sorted_labels = ops.convert_to_tensor(sorted_labels)
      rank_discount = self._rank_discount_fn(
          math_ops.to_float(
              math_ops.range(array_ops.shape(sorted_labels)[1]) + 1))
      return array_ops.ones_like(sorted_labels) * rank_discount


def _sort_and_normalize(labels, logits, weights=None):
  """Sorts `labels` and `logits` and normalize `weights`.

  Args:
    labels: A `Tensor` of the same shape as `logits` representing graded
      relevance.
    logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item.
    weights: A scalar, a `Tensor` with shape [batch_size, 1], or a `Tensor` with
      the same shape as `labels`.

  Returns:
    A tuple of (sorted_labels, sorted_logits, sorted_weights).
  """
  labels = ops.convert_to_tensor(labels)
  logits = ops.convert_to_tensor(logits)
  logits.get_shape().assert_has_rank(2)
  logits.get_shape().assert_is_compatible_with(labels.get_shape())
  weights = 1.0 if weights is None else ops.convert_to_tensor(weights)
  weights = array_ops.ones_like(labels) * weights
  _, topn = array_ops.unstack(array_ops.shape(logits))

  # Only sort entries with valid labels that are >= 0.
  scores = array_ops.where(
      math_ops.greater_equal(labels,
                             0.), logits, -1e-6 * array_ops.ones_like(logits) +
      math_ops.reduce_min(logits, axis=1, keepdims=True))
  sorted_labels, sorted_logits, sorted_weights = sort_by_scores(
      scores, [labels, logits, weights], topn=topn)
  return sorted_labels, sorted_logits, sorted_weights


def _pairwise_comparison(sorted_labels,
                         sorted_logits,
                         sorted_weights,
                         lambda_weight=None):
  r"""Returns pairwise comparison `Tensor`s.

  Given a list of n items, the labels of graded relevance l_i and the logits
  s_i, we sort the items in a list based on s_i and obtain ranks r_i. We form
  n^2 pairs of items. For each pair, we have the following:

                        /
                        | 1   if l_i > l_j
  * `pairwise_labels` = |
                        | 0   if l_i <= l_j
                        \
  * `pairwise_logits` = s_i - s_j
                         /
                         | 0              if l_i <= l_j,
  * `pairwise_weights` = | |l_i - l_j|    if lambda_weight is None,
                         | lambda_weight  otherwise.
                         \

  The `sorted_weights` is item-wise and is applied non-symmetrically to update
  pairwise_weights as
    pairwise_weights(i, j) = w_i * pairwise_weights(i, j).
  This effectively applies to all pairs with l_i > l_j. Note that it is actually
  symmetric when `sorted_weights` are constant per list, i.e., listwise weights.

  Args:
    sorted_labels: A `Tensor` with shape [batch_size, list_size] of labels
      sorted.
    sorted_logits: A `Tensor` with shape [batch_size, list_size] of logits
      sorted.
    sorted_weights: A `Tensor` with shape [batch_size, list_size] of item-wise
      weights sorted.
    lambda_weight: A `_LambdaWeight` object.

  Returns:
    A tuple of (pairwise_labels, pairwise_logits, pairwise_weights) with each
    having the shape [batch_size, list_size, list_size].
  """
  # Compute the difference for all pairs in a list. The output is a Tensor with
  # shape [batch_size, list_size, list_size] where the entry [-1, i, j] stores
  # the information for pair (i, j).
  pairwise_label_diff = array_ops.expand_dims(
      sorted_labels, 2) - array_ops.expand_dims(sorted_labels, 1)
  pairwise_logits = array_ops.expand_dims(
      sorted_logits, 2) - array_ops.expand_dims(sorted_logits, 1)
  pairwise_labels = math_ops.to_float(math_ops.greater(pairwise_label_diff, 0))
  is_label_valid = label_valid_fn(sorted_labels)
  valid_pair = math_ops.logical_and(array_ops.expand_dims(is_label_valid, 2),
                                    array_ops.expand_dims(is_label_valid, 1))
  # Only keep the case when l_i > l_j.
  pairwise_weights = pairwise_labels * math_ops.to_float(valid_pair)
  # Apply the item-wise weights along l_i.
  pairwise_weights *= tf.cast(array_ops.expand_dims(sorted_weights, 2),
                              tf.float32)
  if lambda_weight is not None:
    pairwise_weights *= lambda_weight.pair_weights(sorted_labels)
  else:
    pairwise_weights *= math_ops.abs(pairwise_label_diff)
  pairwise_weights = array_ops.stop_gradient(pairwise_weights,
                                             name='weights_stop_gradient')
  return pairwise_labels, pairwise_logits, pairwise_weights


def _pairwise_loss(loss_fn,
                   labels,
                   logits,
                   weights=None,
                   lambda_weight=None,
                   lambda_scale=True,
                   reduction=core_losses.Reduction.SUM_BY_NONZERO_WEIGHTS):
  """Template to compute pairwise loss.

  Args:
    loss_fn: A function that computes loss from the pairwise logits with l_i >
      l_j.
    labels: A `Tensor` of the same shape as `logits` representing graded
      relevance.
    logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item.
    weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
      weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
      weights.
    lambda_weight: A `_LambdaWeight` object.
    reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch.

  Returns:
    An op for the pairwise loss.
  """
  sorted_labels, sorted_logits, sorted_weights = _sort_and_normalize(
      labels, logits, weights)
  _, pairwise_logits, pairwise_weights = _pairwise_comparison(
      sorted_labels, sorted_logits, sorted_weights, lambda_weight)

  if lambda_weight is not None and lambda_scale:
    # For LambdaLoss with relative rank difference, the scale of loss becomes
    # much smaller when applying LambdaWeight. This affects the training can
    # make the optimal learning rate become much larger. We use a heuristic to
    # scale it up to the same magnitude as standard pairwise loss.
    pairwise_weights *= math_ops.to_float(array_ops.shape(sorted_labels)[1])
  return core_losses.compute_weighted_loss(loss_fn(pairwise_logits),
                                           weights=pairwise_weights,
                                           reduction=reduction)


def _pairwise_hinge_loss(labels,
                         logits,
                         weights=None,
                         lambda_weight=None,
                         lambda_scale=True,
                         reduction=core_losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
                         name=None):
  """Computes the pairwise hinge loss for a list.

  The hinge loss is defined as Hinge(l_i > l_j) = max(0, 1 - (s_i - s_j)). So a
  correctly ordered pair has 0 loss if (s_i - s_j >= 1). Otherwise the loss
  increases linearly with s_i - s_j. When the list_size is 2, this reduces to
  the standard hinge loss.

  Args:
    labels: A `Tensor` of the same shape as `logits` representing graded
      relevance.
    logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item.
    weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
      weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
      weights.
    lambda_weight: A `_LambdaWeight` object.
    reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch.
    name: A string used as the name for this loss.

  Returns:
    An op for the pairwise hinge loss.
  """

  def _loss(logits):
    """The loss of pairwise logits with l_i > l_j."""
    # TODO(xuanhui, pointer-team): Consider pass params object into the loss and
    # put a margin here.
    return nn_ops.relu(1. - logits)

  with ops.name_scope(name, 'pairwise_hinge_loss', (labels, logits, weights)):
    return _pairwise_loss(_loss,
                          labels,
                          logits,
                          weights,
                          lambda_weight,
                          lambda_scale,
                          reduction=reduction)


def _pairwise_logistic_loss(
    labels,
    logits,
    weights=None,
    lambda_weight=None,
    lambda_scale=True,
    reduction=core_losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    name=None):
  """Computes the pairwise logistic loss for a list.

  The preference probability of each pair is computed as the sigmoid function:
  P(l_i > l_j) = 1 / (1 + exp(s_j - s_i)) and the logistic loss is log(P(l_i >
  l_j)) if l_i > l_j and 0 otherwise.

  Args:
    labels: A `Tensor` of the same shape as `logits` representing graded
      relevance.
    logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item.
    weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
      weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
      weights.
    lambda_weight: A `_LambdaWeight` object.
    reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch.
    name: A string used as the name for this loss.

  Returns:
    An op for the pairwise logistic loss.
  """

  def _loss(logits):
    """The loss of pairwise logits with l_i > l_j."""
    # The following is the same as log(1 + exp(-pairwise_logits)).
    return nn_ops.relu(-logits) + math_ops.log1p(
        math_ops.exp(-math_ops.abs(logits)))

  with ops.name_scope(name, 'pairwise_logistic_loss',
                      (labels, logits, weights)):
    return _pairwise_loss(_loss,
                          labels,
                          logits,
                          weights,
                          lambda_weight,
                          lambda_scale,
                          reduction=reduction)


def _pairwise_soft_zero_one_loss(
    labels,
    logits,
    weights=None,
    lambda_weight=None,
    reduction=core_losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    name=None):
  """Computes the pairwise soft zero-one loss.

  Note this is different from sigmoid cross entropy in that soft zero-one loss
  is a smooth but non-convex approximation of zero-one loss. The preference
  probability of each pair is computed as the sigmoid function: P(l_i > l_j) = 1
  / (1 + exp(s_j - s_i)). Then 1 - P(l_i > l_j) is directly used as the loss.
  So a correctly ordered pair has a loss close to 0, while an incorrectly
  ordered pair has a loss bounded by 1.

  Args:
    labels: A `Tensor` of the same shape as `logits` representing graded
      relevance.
    logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item.
    weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
      weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
      weights.
    lambda_weight: A `_LambdaWeight` object.
    reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch.
    name: A string used as the name for this loss.

  Returns:
    An op for the pairwise soft zero one loss.
  """

  def _loss(logits):
    """The loss of pairwise logits with l_i > l_j."""
    return array_ops.where(math_ops.greater(logits, 0),
                           1. - math_ops.sigmoid(logits),
                           math_ops.sigmoid(-logits))

  with ops.name_scope(name, 'pairwise_soft_zero_one_loss',
                      (labels, logits, weights)):
    return _pairwise_loss(_loss,
                          labels,
                          logits,
                          weights,
                          lambda_weight,
                          reduction=reduction)


def _softmax_loss(labels,
                  logits,
                  weights=None,
                  lambda_weight=None,
                  reduction=core_losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
                  name=None):
  """Computes the softmax cross entropy for a list.

  Given the labels l_i and the logits s_i, we sort the examples and obtain ranks
  r_i. The standard softmax loss doesn't need r_i and is defined as
      -sum_i l_i * log(exp(s_i) / (exp(s_1) + ... + exp(s_n))).
  The `lambda_weight` re-weight examples based on l_i and r_i.
      -sum_i w(l_i, r_i) * log(exp(s_i) / (exp(s_1) + ... + exp(s_n))).abc
  See 'individual_weights' in 'DCGLambdaWeight' for how w(l_i, r_i) is computed.

  Args:
    labels: A `Tensor` of the same shape as `logits` representing graded
      relevance.
    logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item.
    weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
      weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
      weights.
    lambda_weight: A `DCGLambdaWeight` instance.
    reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch.
    name: A string used as the name for this loss.

  Returns:
    An op for the softmax cross entropy as a loss.
  """
  with ops.name_scope(name, 'softmax_loss', (labels, logits, weights)):
    sorted_labels, sorted_logits, sorted_weights = _sort_and_normalize(
        labels, logits, weights)
    is_label_valid = label_valid_fn(sorted_labels)
    # Reset the invalid labels to 0 and reset the invalid logits to a logit with
    # ~= 0 contribution in softmax.
    sorted_labels = array_ops.where(is_label_valid, sorted_labels,
                                    array_ops.zeros_like(sorted_labels))
    sorted_logits = array_ops.where(
        is_label_valid, sorted_logits,
        math_ops.log(_EPSILON) * array_ops.ones_like(sorted_logits))
    if lambda_weight is not None and isinstance(lambda_weight, DCGLambdaWeight):
      sorted_labels = lambda_weight.individual_weights(sorted_labels)
    sorted_labels *= sorted_weights
    label_sum = math_ops.reduce_sum(sorted_labels, 1, keepdims=True)
    nonzero_mask = math_ops.greater(array_ops.reshape(label_sum, [-1]), 0.0)
    label_sum, sorted_labels, sorted_logits = [
        array_ops.boolean_mask(x, nonzero_mask)
        for x in [label_sum, sorted_labels, sorted_logits]
    ]
    return core_losses.softmax_cross_entropy(sorted_labels / label_sum,
                                             sorted_logits,
                                             weights=array_ops.reshape(
                                                 label_sum, [-1]),
                                             reduction=reduction)


def _sigmoid_cross_entropy_loss(
    labels,
    logits,
    weights=None,
    reduction=core_losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    name=None):
  """Computes the sigmoid_cross_entropy loss for a list.

  Given the labels of graded relevance l_i and the logits s_i, we calculate
  the sigmoid cross entropy for each ith position and aggregate the per position
  losses.

  Args:
    labels: A `Tensor` of the same shape as `logits` representing graded
      relevance.
    logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item.
    weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
      weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
      weights.
    reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch.
    name: A string used as the name for this loss.

  Returns:
    An op for the sigmoid cross entropy as a loss.
  """
  with ops.name_scope(name, 'sigmoid_cross_entropy_loss',
                      (labels, logits, weights)):
    is_label_valid = array_ops.reshape(label_valid_fn(labels), [-1])
    weights = 1.0 if weights is None else ops.convert_to_tensor(weights)
    weights = array_ops.ones_like(labels) * weights
    label_vector, logit_vector, weight_vector = [
        array_ops.boolean_mask(array_ops.reshape(x, [-1]), is_label_valid)
        for x in [labels, logits, weights]
    ]
    return core_losses.sigmoid_cross_entropy(label_vector,
                                             logit_vector,
                                             weights=weight_vector,
                                             reduction=reduction)


def _mean_squared_loss(labels,
                       logits,
                       weights=None,
                       reduction=core_losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
                       name=None):
  """Computes the mean squared loss for a list.

  Given the labels of graded relevance l_i and the logits s_i, we calculate
  the squared error for each ith position and aggregate the per position
  losses.

  Args:
    labels: A `Tensor` of the same shape as `logits` representing graded
      relevance.
    logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item.
    weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
      weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
      weights.
    reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch.
    name: A string used as the name for this loss.

  Returns:
    An op for the mean squared error as a loss.
  """
  with ops.name_scope(name, 'mean_squared_loss', (labels, logits, weights)):
    is_label_valid = array_ops.reshape(label_valid_fn(labels), [-1])
    weights = 1.0 if weights is None else ops.convert_to_tensor(weights)
    weights = array_ops.ones_like(labels) * weights
    label_vector, logit_vector, weight_vector = [
        array_ops.boolean_mask(array_ops.reshape(x, [-1]), is_label_valid)
        for x in [labels, logits, weights]
    ]
    return core_losses.mean_squared_error(label_vector,
                                          logit_vector,
                                          weights=weight_vector,
                                          reduction=reduction)


def _list_mle_loss(labels,
                   logits,
                   weights=None,
                   lambda_weight=None,
                   reduction=core_losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
                   name=None,
                   seed=None):
  """Computes the ListMLE loss [Xia et al. 2008] for a list.

  Given the labels of graded relevance l_i and the logits s_i, we calculate
  the ListMLE loss for the given list.

  The `lambda_weight` re-weights examples based on l_i and r_i.
  The recommended weighting scheme is the formulation presented in the
  "Position-Aware ListMLE" paper (Lan et al.) and available using
  create_p_list_mle_lambda_weight() factory function above.

  Args:
    labels: A `Tensor` of the same shape as `logits` representing graded
      relevance.
    logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item.
    weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
      weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
      weights.
    lambda_weight: A `DCGLambdaWeight` instance.
    reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch.
    name: A string used as the name for this loss.
    seed: A randomization seed used when shuffling ground truth permutations.

  Returns:
    An op for the ListMLE loss.
  """
  with ops.name_scope(name, 'list_mle_loss', (labels, logits, weights)):
    is_label_valid = label_valid_fn(labels)
    # Reset the invalid labels to 0 and reset the invalid logits to a logit with
    # ~= 0 contribution.
    labels = array_ops.where(is_label_valid, labels,
                             array_ops.zeros_like(labels))
    logits = array_ops.where(
        is_label_valid, logits,
        math_ops.log(_EPSILON) * array_ops.ones_like(logits))
    weights = 1.0 if weights is None else ops.convert_to_tensor(weights)
    weights = array_ops.squeeze(weights)

    # Shuffle labels and logits to add randomness to sort.
    shuffled_indices = shuffle_valid_indices(is_label_valid, seed)
    shuffled_labels = array_ops.gather_nd(labels, shuffled_indices)
    shuffled_logits = array_ops.gather_nd(logits, shuffled_indices)

    sorted_labels, sorted_logits = sort_by_scores(
        shuffled_labels, [shuffled_labels, shuffled_logits])

    raw_max = math_ops.reduce_max(sorted_logits, axis=1, keepdims=True)
    sorted_logits = sorted_logits - raw_max
    sums = math_ops.cumsum(math_ops.exp(sorted_logits), axis=1, reverse=True)
    sums = math_ops.log(sums) - sorted_logits

    if lambda_weight is not None and isinstance(lambda_weight,
                                                ListMLELambdaWeight):
      sums *= lambda_weight.individual_weights(sorted_labels)

    negative_log_likelihood = math_ops.reduce_sum(sums, 1)

    return core_losses.compute_weighted_loss(negative_log_likelihood,
                                             weights=weights,
                                             reduction=reduction)


def _approx_ndcg_loss(labels,
                      logits,
                      weights=None,
                      reduction=core_losses.Reduction.SUM,
                      name=None,
                      alpha=10.):
  """Computes ApproxNDCG loss.

  ApproxNDCG ["A general approximation framework for direct optimization of
  information retrieval measures" by Qin et al.] is a smooth approximation
  to NDCG.

  Args:
    labels: A `Tensor` of the same shape as `logits` representing graded
      relevance.
    logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item.
    weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
      weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
      weights. If None, the weight of a list in the mini-batch is set to
      the sum of the labels of the items in that list.
    reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch.
    name: A string used as the name for this loss.
    alpha: The exponent in the generalized sigmoid function.

  Returns:
    An op for the ApproxNDCG loss.
  """
  with ops.name_scope(name, 'approx_ndcg_loss', (labels, logits, weights)):
    is_label_valid = label_valid_fn(labels)
    labels = array_ops.where(is_label_valid, labels,
                             array_ops.zeros_like(labels))
    logits = array_ops.where(
        is_label_valid, logits, -1e3 * array_ops.ones_like(logits) +
        math_ops.reduce_min(logits, axis=-1, keepdims=True))

    label_sum = math_ops.reduce_sum(labels, 1, keepdims=True)
    if weights is None:
      weights = array_ops.ones_like(label_sum)
    weights = array_ops.squeeze(weights)

    nonzero_mask = math_ops.greater(array_ops.reshape(label_sum, [-1]), 0.0)
    labels, logits, weights = [
        array_ops.boolean_mask(x, nonzero_mask)
        for x in [labels, logits, weights]
    ]

    gains = math_ops.pow(2., math_ops.to_float(labels)) - 1.
    ranks = approx_ranks(logits, alpha=alpha)
    discounts = 1. / math_ops.log1p(ranks)
    dcg = math_ops.reduce_sum(gains * discounts, -1)
    cost = -dcg * array_ops.squeeze(inverse_max_dcg(labels))

    return core_losses.compute_weighted_loss(cost,
                                             weights=weights,
                                             reduction=reduction)
