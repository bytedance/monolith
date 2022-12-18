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

"""Sail like API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import tensorflow as tf
from collections import namedtuple


class FeatureSlice(object):
  """Sail-like FeatureSlice implementation."""

  def __init__(
      self,
      feature_slot,
      dim,
      slice_index,
      optimizer=None,
      initializer=None,
      learning_rate_fn=None,
  ):
    """Initialize a FeatureSlice object.

    Args:
      feature_slot: A FeatureSlot object that this FeatureSlice object belongs
        to.
      dim: The dim of the FeatureSlice.
      slice_index: The index of this FeatureSlice in the FeatureSlot list of
        FeatureSlice.
      optimizer: TensorFlow optimization parameters (e.g.,
        tf.compat.v1.tpu.experimental.FtrlParameters).
      initializer: TensorFlow initializer (e.g., tf.random_uniform_initializer).
      learning_rate_fn: A function that changes the learning rate over the
        training period (e.g., tf.compat.v1.train.polynomial_decay).
    """
    self._feature_slot = feature_slot
    self._dim = dim
    self._slice_index = slice_index
    self._optimizer = optimizer
    self._initializer = initializer
    self._learning_rate_fn = learning_rate_fn

  def __repr__(self):
    """The default __repr__() method is overwritten to enable FeatureSlice as dict key."""
    return '[FeatureSlice][slot_{}][{}]'.format(self._feature_slot.slot_id(),
                                                self._slice_index)

  def __hash__(self):
    """The default __hash__() method is overwritten to enable FeatureSlice as dict key."""
    return hash((self._feature_slot.slot_id(), self._slice_index))

  @property
  def dim(self):
    return self._dim

  @property
  def slice_index(self):
    return self._slice_index

  @property
  def optimizer(self):
    return self._optimizer

  @property
  def initializer(self):
    return self._initializer

  @property
  def learning_rate_fn(self):
    return self._learning_rate_fn


class FeatureSlot(object):
  """Sail like FeatureSlot implementation."""

  def __init__(
      self,
      env,
      slot_id,
      has_bias=False,
      bias_optimizer=tf.compat.v1.tpu.experimental.FtrlParameters(
          learning_rate=0.01),
      bias_initializer=tf.zeros_initializer(),
      bias_learning_rate_fn=None,
      default_vec_optimizer=tf.compat.v1.tpu.experimental.AdagradParameters(
          learning_rate=0.01),
      default_vec_initializer=tf.random_uniform_initializer(minval=-0.001,
                                                            maxval=0.001),
      default_vec_learning_rate_fn=None,
      occurrence_threshold=None,
      expire_time=None,
  ):
    """Initialize a FeatureSlot object.

    Args:
      env: An Env object that this FeatureSlot belongs to.
      slot_id: The slot_id of the FeatureSlot.
      has_bias: A Boolean on whether this FeatureSlot has a bias FeatureSlice.
      bias_optimizer: TensorFlow optimization parameters for bias slice (e.g.,
        tf.compat.v1.tpu.experimental.FtrlParameters).
      bias_initializer: TensorFlow initializer for bias slice (e.g.,
        tf.random_uniform_initializer).
      bias_learning_rate_fn: A function that changes the learning rate of the
        bias over the training period (e.g.,
        tf.compat.v1.train.polynomial_decay).
      default_vec_optimizer: The default TensorFlow optimization parameters for
        vector slices.
      default_vec_initializer: The default TensorFlow initializer for vector
        slices.
      default_vec_learning_rate_fn: The default TensorFlow learning rate
        function for vector slices.
      occurrence_threshold: The number of occurrences that an FID of the slot_id
        must occur in order to be recorded into the training data.
      expire_time: FID may be evicted from the model if not been updated
        for expire_time days.
    """
    self._env = env
    self._slot_id = slot_id
    self._has_bias = has_bias
    self._bias_optimizer = bias_optimizer
    self._bias_initializer = bias_initializer
    self._bias_learning_rate_fn = bias_learning_rate_fn
    self._default_vec_optimizer = default_vec_optimizer
    self._default_vec_initializer = default_vec_initializer
    self._default_vec_learning_rate_fn = default_vec_learning_rate_fn
    self._occurrence_threshold = occurrence_threshold
    self._expire_time = expire_time
    self._feature_slices = []
    self._merged_feature_slices = []
    self._feature_columns = []
    self._env.set_feature_slot(slot_id, self)

    if self._has_bias:
      feature_slice = FeatureSlice(
          feature_slot=self,
          dim=1,
          slice_index=0,
          optimizer=self._bias_optimizer,
          initializer=self._bias_initializer,
          learning_rate_fn=self._bias_learning_rate_fn,
      )
      self._feature_slices.append(feature_slice)

  def get_env(self):
    return self._env

  def slot_id(self):
    return self._slot_id

  def has_bias(self):
    return self._has_bias

  def add_feature_slice(
      self,
      dim,
      optimizer=None,
      initializer=None,
      learning_rate_fn=None,
  ):
    """Create a FeatureSlice and add to _feature_slices."""
    optimizer = optimizer if optimizer is not None else self._default_vec_optimizer
    initializer = initializer if initializer is not None else self._default_vec_initializer
    learning_rate_fn = learning_rate_fn if learning_rate_fn is not None else self._default_vec_learning_rate_fn
    feature_slice = FeatureSlice(
        feature_slot=self,
        dim=dim,
        slice_index=len(self._feature_slices),
        optimizer=optimizer,
        initializer=initializer,
        learning_rate_fn=learning_rate_fn,
    )
    self._feature_slices.append(feature_slice)

    return feature_slice

  def add_merged_feature_slice(
      self,
      dim,
      optimizer=None,
      initializer=None,
      learning_rate_fn=None,
  ):
    """Create a FeatureSlice for merged embedding and add to _merged_feature_slices."""
    optimizer = optimizer if optimizer is not None else self._default_vec_optimizer
    initializer = initializer if initializer is not None else self._default_vec_initializer
    learning_rate_fn = learning_rate_fn if learning_rate_fn is not None else self._default_vec_learning_rate_fn
    feature_slice = FeatureSlice(
        feature_slot=self,
        dim=dim,
        slice_index=len(self._merged_feature_slices),
        optimizer=optimizer,
        initializer=initializer,
        learning_rate_fn=learning_rate_fn,
    )
    self._merged_feature_slices.append(feature_slice)

    return feature_slice

  def _add_feature_column(self, feature_column):
    """Add a FeatureColumn object to the FeatureSlot."""
    self._feature_columns.append(feature_column)
    # If the FeatureSlot has bias, add the bias FeatureSlice to all
    # FeatureColumn objects
    if self._has_bias:
      for feature_column in self._feature_columns:
        feature_column._bias = feature_column.embedding_lookup(
            self._feature_slices[0])

  @property
  def bias_optimizer(self):
    return self._bias_optimizer

  @property
  def bias_initializer(self):
    return self._bias_initializer

  @property
  def bias_learning_rate_fn(self):
    return self._bias_learning_rate_fn

  @property
  def default_vec_optimizer(self):
    return self._default_vec_optimizer

  @property
  def default_vec_initializer(self):
    return self._default_vec_initializer

  @property
  def default_vec_learning_rate_fn(self):
    return self._default_vec_learning_rate_fn

  @property
  def feature_slices(self):
    return self._feature_slices

  @property
  def merged_feature_slices(self):
    return self._merged_feature_slices

  @property
  def feature_columns(self):
    return self._feature_columns


class FeatureColumnV1(object):
  """Sail like class FeatureColumnV1 implementation."""

  def __init__(self, feature_slot, fc_name):
    """Initialize a FeatureSlot object.

    Args:
      feature_slot: A FeatureSlot object that this FeatureColumnV1 belongs to.
      fc_name: The name of the feature column (e.g, "slot_1_0").
    """
    self._feature_slot = feature_slot
    self._fc_name = fc_name
    self._feature_slice_to_tf_placeholder = {}
    self._merged_feature_slice_to_tf_placeholder = {}
    self._bias = None
    self._feature_slot._add_feature_column(self)

  def get_env(self):
    return self._feature_slot.get_env()

  def embedding_lookup(
      self,
      feature_slice,
      init_minval_for_oov=None,
      init_maxval_for_oov=None,
  ):
    return self.get_env()._embedding_lookup(self, feature_slice,
                                            init_minval_for_oov,
                                            init_maxval_for_oov)

  def get_bias(self):
    assert self._bias is not None
    return self._bias

  @property
  def feature_slot(self):
    return self._feature_slot

  @property
  def fc_name(self):
    return self._fc_name

  @property
  def feature_slice_to_tf_placeholder(self):
    env = self.get_env()
    assert env.is_finalized, "is_finalized must be true which means \
      _feature_slice_to_tf_placeholder must be initialized before using this \
      _feature_slice_to_tf_placeholder"

    if env._merge_vector:
      return self._merged_feature_slice_to_tf_placeholder
    else:
      return self._feature_slice_to_tf_placeholder


class FeatureColumn3D(object):
  """Sail like class FeatureColumn3D implementation."""

  def __init__(self, feature_slot, max_seq_length, fc_name):
    self._feature_slot = feature_slot
    self._fc_name = fc_name
    self._feature_slice_to_tf_placeholder = {}
    self._bias = None
    logging.info("max_seq_length {}".format(max_seq_length))
    self._max_seq_length = max_seq_length
    self._feature_slot._add_feature_column(self)

  def get_env(self):
    return self._feature_slot.get_env()

  def embedding_lookup(
      self,
      feature_slice,
      max_seq_length,
      init_minval_for_oov=None,
      init_maxval_for_oov=None,
  ):
    return self.get_env()._seq_embedding_lookup(self, feature_slice,
                                                self._max_seq_length,
                                                init_minval_for_oov,
                                                init_maxval_for_oov)

  def get_bias(self):
    assert self._bias is not None
    return self._bias

  @property
  def feature_slot(self):
    return self._feature_slot

  @property
  def fc_name(self):
    return self._fc_name

  @property
  def feature_slice_to_tf_placeholder(self):
    return self._feature_slice_to_tf_placeholder

  @property
  def max_seq_length(self):
    return self._max_seq_length

  def size_tensor_lookup(self):
    """Name with '_size' as the suffix of ${fc_name}"""
    return self.get_env()._size_tensor_lookup(self)


class Env(object):
  """Environment which holds important information and track the embedding tables."""

  def __init__(self, vocab_size_dict, params):
    self._vocab_size_dict = vocab_size_dict
    self._slot_id_to_feature_slot = {}  # {1: FeatureSlot}
    self._tpu_features = None
    self._is_finalized = False

    self.set_params(params)

  def set_tpu_features(self, tpu_features):
    self._tpu_features = tpu_features

    if self._merge_vector:
      for slot_id, feature_slot in self._slot_id_to_feature_slot.items():
        # Split the merged embeddings
        self._split_merged_embedding(feature_slot)

  def set_feature_slot(self, slot_id, feature_slot):
    # Set feature slot only be called during the first round of calling logits in which env is not finalized yet.
    if self._is_finalized == True:
      return
    assert slot_id not in self._slot_id_to_feature_slot, "Feature slot with id: {} can not be created more than once.".format(
        slot_id)
    self._slot_id_to_feature_slot[slot_id] = feature_slot

  def set_params(self, params):
    self._QR_multi_hashing = params.qr_multi_hashing
    self._QR_hashing_threshold = params.qr_hashing_threshold
    self._QR_collision_rate = params.qr_collision_rate
    self._use_random_init_embedding_for_oov = params.use_random_init_embedding_for_oov
    self._merge_vector = params.merge_vector

  def is_finalized(self):
    return self.is_finalized

  def _embedding_lookup(self,
                        feature_column,
                        feature_slice,
                        init_minval_for_oov=None,
                        init_maxval_for_oov=None):
    assert feature_column._feature_slot.slot_id(
    ) == feature_slice._feature_slot.slot_id()

    if self._tpu_features:
      logging.vlog(2, "__embedding_loopup with features exist.")

      if self._QR_multi_hashing and self._vocab_size_dict[
          slot_id] > self._QR_hashing_threshold:
        logging.vlog(
            2, "__embedding_lookup of QR hashing for slot {}.".format(slot_id))
        # taking quotient feature and remainder feature
        keyR = "{}_{}_0".format(feature_column.fc_name,
                                feature_slice.slice_index)
        keyQ = "{}_{}_1".format(feature_column.fc_name,
                                feature_slice.slice_index)
        assert keyR in self._tpu_features and keyQ in self._tpu_features, \
            "keyR: {} or keyQ: {} not in tpu features, probably need to check core.base_embedding_task._post_process_example()".format(keyR, keyQ)

        # combining quotient feature and remainder feature
        # element-wise addition performs better than element-wise multiplication
        return self._tpu_features[keyR] + self._tpu_features[keyQ]
      else:
        key = "{}_{}".format(feature_column.fc_name, feature_slice.slice_index)
        if not self._use_random_init_embedding_for_oov or init_minval_for_oov is None:
          return self._tpu_features[key]
        norm = tf.norm(self._tpu_features[key], axis=1)
        random = tf.random.uniform(
            tf.shape(self._tpu_features[key]),
            minval=init_minval_for_oov,
            maxval=init_maxval_for_oov,
        )
        cond = tf.expand_dims(tf.less(norm, 1e-10), -1)
        return tf.where(cond, random, self._tpu_features[key])
    else:
      logging.vlog(2, "__embedding_lookup with no features exist.")
      if feature_slice not in feature_column._feature_slice_to_tf_placeholder:
        feature_column._feature_slice_to_tf_placeholder[
            feature_slice] = tf.compat.v1.placeholder(tf.float32,
                                                      [None, feature_slice.dim])

      return feature_column._feature_slice_to_tf_placeholder[feature_slice]

  def _seq_embedding_lookup(self,
                            feature_column,
                            feature_slice,
                            max_seq_length,
                            init_minval_for_oov=None,
                            init_maxval_for_oov=None):
    assert feature_column._feature_slot.slot_id(
    ) == feature_slice._feature_slot.slot_id()

    if self._tpu_features:
      logging.vlog(2, "__embedding_loopup with features exist.")
      key = "{}_{}".format(feature_column.fc_name, feature_slice.slice_index)
      if not self._use_random_init_embedding_for_oov or init_minval_for_oov is None:
        return self._tpu_features[key]
      norm = tf.norm(self._tpu_features[key], axis=1)
      random = tf.random.uniform(
          tf.shape(self._tpu_features[key]),
          minval=feature_slice.init_minval_for_oov,
          maxval=feature_slice.init_maxval_for_oov,
      )
      cond = tf.expand_dims(tf.less(norm, 1e-10), -1)
      return tf.where(cond, random, self._tpu_features[key])
    else:
      logging.vlog(2, "__embedding_lookup with no features exist.")
      if feature_slice not in feature_column._feature_slice_to_tf_placeholder:
        feature_column._feature_slice_to_tf_placeholder[
            feature_slice] = tf.compat.v1.placeholder(
                tf.float32, [None, max_seq_length, feature_slice.dim])

      return feature_column._feature_slice_to_tf_placeholder[feature_slice]

  def _size_tensor_lookup(self, feature_column):
    if self._tpu_features:
      key = "{}_0_row_lengths".format(feature_column.fc_name)
      row_lengths = self._tpu_features[key]
      # Convert row_lengths to [B, max_seq_length] Tensor, in which
      # the first row_length elements of each row are 1, and the rest are
      # 0. This is used as the size_tensor
      batch_size = tf.size(row_lengths)  # 0-D Tensor
      boolean_mask = tf.less(
          tf.reshape(
              tf.tile(tf.range(0, feature_column.max_seq_length), [batch_size]),
              [batch_size, -1],
          ), tf.expand_dims(row_lengths, 1))  # [B, max_seq_length] Tensor
      return tf.cast(boolean_mask, tf.int32)
    else:
      return tf.compat.v1.placeholder(
          tf.float32,
          [None, feature_column.max_seq_length],
          '{}_size'.format(feature_column.fc_name),
      )

  def finalize(self):
    """Finalize the env after slot to dims dict has been initialized."""
    assert self._is_finalized == False, "Env can't be finalized more than once"
    self._is_finalized = True

    if self._merge_vector:
      self._merge_vector_in_same_slot()

  def _split_merged_embedding(self, feature_slot):
    """Split merged embedding into embedding splits of corresponding dim.

    Currently, this assumes all vector FeatureSlice embeddings are shared among
    all the FeatureColumns, so all vector FeatureSlice embeddings can be merged
    into one embedding.
    """
    # Iterate through feature columns
    for feature_column in feature_slot.feature_columns:
      merged_embedding = None
      for merged_feature_slice in feature_column._merged_feature_slice_to_tf_placeholder:
        if merged_feature_slice.slice_index == 0 and feature_slot.has_bias():
          # For bias, keep it as bias will not be merged. Nothing to do.
          assert merged_feature_slice.dim == 1, "Bias in {} must have dim equal to 1, but actual dim is {}.".format(
              feature_column.fc_name, merged_feature_slice.dim)
        else:
          merged_feature_name = "{}_{}".format(feature_column.fc_name,
                                               merged_feature_slice.slice_index)
          merged_embedding = self._tpu_features[merged_feature_name]
          # del self._tpu_features[merged_feature_name]

      if merged_embedding is not None:
        # Split embeddings will be written to the position starting from the previous merged embedding position.
        dim_splits = [
            feature_slice.dim for feature_slice in feature_slot.feature_slices
        ]
        if feature_slot.has_bias():
          dim_splits = [
              feature_slice.dim for feature_slice in feature_slot.feature_slices
          ][1:]
        embedding_splits = tf.split(merged_embedding, dim_splits, axis=1)

      for feature_slice in feature_column._feature_slice_to_tf_placeholder:
        if feature_slice.slice_index == 0 and feature_slot.has_bias():
          # For bias, keep it as bias will not be merged. Nothing to do.
          assert feature_slice.dim == 1, "Bias in {} must have dim equal to 1, but actual dim is {}.".format(
              feature_column.fc_name, feature_slice.dim)
        else:
          if merged_embedding is not None:
            if feature_slot.has_bias():
              split_index = feature_slice.slice_index - 1
            else:
              split_index = feature_slice.slice_index
            split = embedding_splits[split_index]
            self._tpu_features["{}_{}".format(
                feature_column.fc_name, feature_slice.slice_index)] = split

  def _merge_vector_in_same_slot(self):
    """Merge vectors in the same slot.

    Currently, this assumes all vector FeatureSlice embeddings are shared among
    all the FeatureColumns, so all vector FeatureSlice embeddings can be merged
    into one embedding.
    """
    # TODO (long): Support the case where only a subset of FeatureSlices are
    # shared, so some vector FeatureSlices cannot be merged
    for slot_id, feature_slot in self._slot_id_to_feature_slot.items():
      merged_vector_dim = 0
      # Iterate through all FeatureSlices in FeatureSlot to calculate the merged
      # embedding dimension
      for feature_slice in feature_slot.feature_slices:
        # Bias will not be merged
        if feature_slot.has_bias() and feature_slice.slice_index == 0:
          assert feature_slice.dim == 1, "Bias in {} must have dim equal to 1, but actual dim is {}.".format(
              slot_id, feature_slice.dim)
          feature_slot._merged_feature_slices.append(feature_slice)
          for feature_column in feature_slot.feature_columns:
            feature_column._merged_feature_slice_to_tf_placeholder[
                feature_slice] = feature_column._feature_slice_to_tf_placeholder[
                    feature_slice]
        else:
          merged_vector_dim += feature_slice.dim
      # Created a merged slice whose dim is the sum of the dims of all vector
      # FeatureSlices
      if merged_vector_dim > 0:
        # Create the merged FeatureSlice with the merged embedding dimension
        merged_feature_slice = feature_slot.add_merged_feature_slice(
            merged_vector_dim)
        # Add the merged FeatureSlice and its corresponding tf.placeholder to
        # each FeatureColumn in the FeatureSlot
        for feature_column in feature_slot.feature_columns:
          feature_column._merged_feature_slice_to_tf_placeholder[
              merged_feature_slice] = tf.compat.v1.placeholder(
                  tf.float32, [None, merged_feature_slice.dim])

  @property
  def vocab_size_dict(self):
    return self._vocab_size_dict

  @property
  def slot_id_to_feature_slot(self):
    assert self.is_finalized, "is_finalized must be true which means _slot_id_to_feature_slot \
      must be initialized before using this _slot_id_to_feature_slot"

    return self._slot_id_to_feature_slot

  @property
  def features(self):
    return self._tpu_features
