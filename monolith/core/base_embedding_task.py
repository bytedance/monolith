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

"""Base class for TPU embedding task"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math
import numpy as np
import os
import shutil
import subprocess

from absl import logging

import tensorflow.compat.v1 as tf
from tensorflow.python.tpu import tpu_embedding

from monolith.core import base_task
import monolith.core.auto_checkpoint_feed_hook as fh
import monolith.core.base_embedding_host_call as hs
from monolith.core.base_embedding_host_call import BaseEmbeddingHostCall
from monolith.core.feature import FeatureSlot, FeatureColumnV1, FeatureColumn3D, Env
from monolith.core import base_task
import monolith.core.util as util


class BaseEmbeddingTask(base_task.BaseTask):
  """A embedding task which trains Sail-like model on TPU."""

  @classmethod
  def params(cls):
    p = super(BaseEmbeddingTask, cls).params()
    p.define(
        'vocab_size_per_slot', None,
        'Fixed vocab_size for all the slots, this mainly '
        'for quick testing purpose.')
    p.define(
        'custom_vocab_size_mapping',
        None,
        'Fixed vocab size for some slots',
    )
    p.define(
        'vocab_size_offset', None,
        'Manually increase the vocab_size of each slot by a constant. '
        'This is used to provide a quick fix for issues/problems in '
        'generating GCP data, e.g., incorrect vocab_id assignment or '
        'incorrect vocab_size calculation.')
    p.define(
        'qr_multi_hashing', False,
        'If True, enable QR multi hashing trick for the slots with vocab_id larger than the threshold.'
    )
    p.define('qr_hashing_threshold', 100000000,
             'Threshold for the the QR slot.')
    p.define('qr_collision_rate', 4,
             'The hashing collision rate for the QR slot.')
    p.define('vocab_file_path', None, 'Path for the vocab file.')
    p.define('enable_deepinsight', False,
             'Whether connect to deepinsight to show the results.')
    p.define(
        'enable_host_call_scalar_metrics', False,
        'If True, enable host call computation of scalar metrics, including \
          average AUC per core, average loss, average label, learning rate, \
          (potentially) weight and gradient norms, etc. These metrics are \
          useful for model development and debugging. If False, only compute \
          basic metrics such as AUC, sample rate, etc.')
    p.define(
        'enable_host_call_norm_metrics', False,
        'Whether to enable host call computation of weight and gradient \
          norms. If enable_host_call_scalar_metrics is False, this param is \
          NOT used.')
    p.define('files_interleave_cycle_length', 4,
             'The number of input files that will be processed concurrently.')
    p.define(
        'deterministic', False,
        'Whether enable deterministic mini-batch training for comparable experiments.'
    )
    p.define("gradient_multiplier", 1.0, "Gradient multiplier for embeddings.")
    p.define('enable_caching_with_tpu_var_mode', False,
             'Whether enable host call with tpu variables mode.')
    # TODO(youlong): Revisit top_k_sampling_num_per_core
    p.define(
        'top_k_sampling_num_per_core', 6,
        'The number of samples to use per core for DeepInsight AUC \
              calculation. A lower number means fewer samples used and faster \
              training, and a bigger number means more samples used and slower \
              training.')
    p.define('use_random_init_embedding_for_oov', False,
             'Whether use random initialized embedding for oov ids')
    p.define('merge_vector', False,
             'If True, enable merging vector tables of the same slot.')
    # If there is file_pattern specified, file_pattern will override file_folder.
    p.train.define('file_folder', None,
                   'Training input data folder before date string.')
    p.train.define('date_and_file_name_format', "*/*/part*",
                   "Training file's date and name pattern.")
    p.train.define(
        'start_date', None,
        'Training input data start date inclusively, for example: 20201201.')
    p.train.define(
        'end_date', None,
        'Training input data end date inclusively, for example: 20201210.')
    p.train.define('vocab_file_folder_prefix', None,
                   'Prefix of hdfs folder to keep per day vocab size file.')
    return p

  def __init__(self, params):
    """Constructs a BaseAttentionLayer object."""
    super(BaseEmbeddingTask, self).__init__(params)
    self.p = params
    self._enable_deepinsight = self.p.enable_deepinsight
    self._enable_host_call_scalar_metrics = self.p.enable_host_call_scalar_metrics
    self._enable_caching_with_tpu_var_mode = self.p.enable_caching_with_tpu_var_mode
    self._top_k_sampling_num_per_core = self.p.top_k_sampling_num_per_core

    if params.vocab_size_per_slot is not None:
      logging.info("Set fixed vocab_size: {} for all the slots.".format(
          params.vocab_size_per_slot))
    if params.custom_vocab_size_mapping is not None:
      logging.info("Set fixed vocab size for some slots: {}".format(
          params.custom_vocab_size_mapping))
    vocab_size_dict = self._create_vocab_dict()

    self._env = Env(vocab_size_dict=vocab_size_dict, params=self.p)

    self._feature_to_config_dict = {}
    self._table_to_config_dict = {}

  def download_vocab_size_file_from_hdfs(self):
    tmp_folder = "temp"
    if os.path.exists(tmp_folder):
      shutil.rmtree(tmp_folder)
    os.mkdir(tmp_folder)
    hdfs_vocab_size_file_path = "{}{}/part*.csv".format(
        self.p.train.vocab_file_folder_prefix, self.p.train.end_date)
    cmd = "hadoop fs -copyToLocal {} {}".format(hdfs_vocab_size_file_path,
                                                tmp_folder)
    logging.info(
        "Hdfs path prefix: {}, end_date: {}, download vocab size file from hdfs cmd: {}"
        .format(self.p.train.vocab_file_folder_prefix, self.p.train.end_date,
                cmd))
    ret = subprocess.run(cmd, shell=True)
    downloaded_files = os.listdir(tmp_folder)
    if ret.returncode == 0 and len(downloaded_files) == 1:
      self.p.vocab_file_path = os.path.join(tmp_folder, downloaded_files[0])
      logging.info(
          "Download vocab size file successfully from hdfs: {}, use downloaded vocab size file: {}."
          .format(hdfs_vocab_size_file_path, self.p.vocab_file_path))
    else:
      logging.info("Downloaded files: {}".format(downloaded_files))
      logging.info("Use default vocab size file: {}".format(
          self.p.vocab_file_path))

  def _create_vocab_dict(self):
    """Create vocab dict from a tsv file."""
    vocab_size_per_slot = self.p.vocab_size_per_slot
    custom_vocab_size_mapping = self.p.custom_vocab_size_mapping

    vocab_size_dict = {}
    if self.p.train.end_date is not None and self.p.train.vocab_file_folder_prefix is not None:
      self.download_vocab_size_file_from_hdfs()

    assert self.p.vocab_file_path is not None, \
      "Either provide vocab_file_path or vocab_file_folder_prefix and end date."

    with open(self.p.vocab_file_path) as f:
      for line in f:
        fields = line.strip().split("\t")
        assert len(fields) == 2, "each line in {} must have 2 fields".format(
            fields)
        if fields[0].isdigit() == False:
          continue

        slot_id = int(fields[0])
        if vocab_size_per_slot is not None:
          distinct_count = vocab_size_per_slot
        else:
          distinct_count = int(fields[1])
          if custom_vocab_size_mapping is not None and slot_id in custom_vocab_size_mapping:
            distinct_count = custom_vocab_size_mapping[slot_id]

        if self.p.vocab_size_offset is not None:
          distinct_count += self.p.vocab_size_offset

        vocab_size_dict[slot_id] = distinct_count

    logging.info("Slot and vocab size: {}".format(vocab_size_dict))
    return vocab_size_dict

  def _parse_inputs(return_values):
    if isinstance(return_values, tuple):
      features, labels = return_values
    else:
      features, labels = return_values, None
    return features, labels

  def create_input_fn(self, mode=tf.estimator.ModeKeys.TRAIN):
    """Create input_fn given the mode.

        Args:
            mode: tf.estimator.ModeKeys.TRAIN/EVAL/PREDICT.
        Returns:
            An input fn for Estimator.
        """
    # TODO(youlong.cheng): support eval and predict.
    assert mode == tf.estimator.ModeKeys.TRAIN
    file_pattern = self.p.train.file_pattern

    def tf_example_parser(examples):
      """Parse multiple examples."""
      feature_map = self._get_feature_map()
      example = tf.io.parse_example(serialized=examples, features=feature_map)
      return self._post_process_example(example)

    def insert_stopping_signal(stop, batch_size, stopping_signals_name):

      def _map_fn(features):
        empty_sparse_tensor = tf.sparse.SparseTensor(
            indices=tf.zeros([0, 2], dtype=tf.int64),
            values=tf.zeros([0], dtype=tf.int64),
            dense_shape=(batch_size, 1))

        shape = [batch_size]
        if stop is True:
          for name, tensor in features.items():
            # For sparse tensors, set them to empty.
            if isinstance(tensor, tf.sparse.SparseTensor) is True:
              features[name] = empty_sparse_tensor
          features[stopping_signals_name] = tf.ones(shape=shape,
                                                    dtype=tf.dtypes.bool)
        else:
          features[stopping_signals_name] = tf.zeros(shape=shape,
                                                     dtype=tf.dtypes.bool)

        return features

      return _map_fn

    def input_fn(params):
      """Returns training or eval examples, batched as specified in params."""
      logging.info("Model input_fn")

      if params["cpu_test"] is True:
        dataset = tf.data.TFRecordDataset(file_pattern,
                                          compression_type=None,
                                          buffer_size=None)
        dataset = dataset.batch(params["batch_size"], drop_remainder=True).map(
            tf_example_parser,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            deterministic=self.p.deterministic)
        return dataset.repeat()

      # By shuffle=False, list_files will get all files already in time sorted order.
      if file_pattern is not None:
        files = tf.data.Dataset.list_files(file_pattern, shuffle=False)
      else:
        assert self.p.train.file_folder is not None, \
          "p.train.file_folder must be defined if file_pattern is None."
        assert self.p.train.date_and_file_name_format is not None, \
          "p.train.date_and_file_name_format must be defined if file_pattern is None."
        file_pattern_ = os.path.join(self.p.train.file_folder,
                                     self.p.train.date_and_file_name_format)
        logging.info("file_pattern: {}, file_folder: {}".format(
            file_pattern_, self.p.train.file_folder))
        files = tf.data.Dataset.list_files(file_pattern_, shuffle=False)
        assert self.p.train.end_date is not None, \
          "end_date in config or flag must be defined if file_pattern is None"
        assert params["enable_stopping_signals"] is not None, \
          "When using end_date of input data, enable_stopping_signals need to be provided."
        files = util.range_dateset(files,
                                   self.p.train.file_folder,
                                   start_date=self.p.train.start_date,
                                   end_date=self.p.train.end_date)

      # This function will get called once per TPU task. Each task will process the files
      # with indexs which modulo num_calls equals to call_index.
      _, call_index, num_calls, _ = (
          params["context"].current_input_fn_deployment())
      files = files.shard(num_calls, call_index)

      def fetch_dataset(filename):
        dataset = tf.data.TFRecordDataset(filename,
                                          compression_type=None,
                                          buffer_size=None)
        return dataset

      # Read the data from disk in parallel.
      # Files will be process from the beginning to the end. With a local parallel of interleaving
      # multiple files currently. Number of interleaving files are defined by the cycle.
      dataset = files.interleave(
          fetch_dataset,
          cycle_length=self.p.files_interleave_cycle_length,
          num_parallel_calls=tf.data.experimental.AUTOTUNE,
          deterministic=self.p.deterministic)
      dataset = dataset.batch(params["batch_size"], drop_remainder=True).map(
          tf_example_parser,
          num_parallel_calls=tf.data.experimental.AUTOTUNE,
          deterministic=self.p.deterministic)

      if self.p.train.repeat:
        assert params["enable_stopping_signals"] is False
        dataset = dataset.repeat()

      enable_stopping_signals = params["enable_stopping_signals"]
      if enable_stopping_signals:
        logging.info("Adding stop signals to original data set.")

        # Add stop signal to help handling end of stream.
        user_provided_dataset = dataset.map(
          insert_stopping_signal(
              stop=False, batch_size=params["batch_size"], \
              stopping_signals_name=fh._USER_PROVIDED_SIGNAL_NAME),
          num_parallel_calls=tf.data.experimental.AUTOTUNE,
          deterministic=False)

        final_batch_dataset = dataset.repeat().map(
          insert_stopping_signal(
              stop=True, batch_size=params["batch_size"], \
              stopping_signals_name=fh._USER_PROVIDED_SIGNAL_NAME),
          num_parallel_calls=tf.data.experimental.AUTOTUNE,
          deterministic=False)

        dataset = user_provided_dataset.concatenate(final_batch_dataset)

      dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

      # Test only
      # dataset = dataset.take(128).cache().repeat()

      return dataset

    return input_fn

  def logits_fn(self):
    """Calculate logits."""
    # Inherated class must implement this function.
    raise NotImplementedError

  def init_slot_to_env(self):
    """Run this in the beginning to initialize the slot and its embedding dims information."""
    logging.info("Model init_slot_to_env")
    logging.info("init_slot_to_env, self: {}".format(self))
    self.logits_fn()
    self._env.finalize()

  def create_model_fn(self,):
    """Create model fn."""
    raise NotImplementedError('Abstract method.')

  def _get_feature_map(self):
    """Returns data format of the serialized tf record file."""
    # Inherated class must implement this function.
    raise NotImplementedError

  def _post_process_example(self, example):
    """Postprocess example."""
    # build tensors for each embeddings in each slot
    for slot_id, feature_slot in list(
        self._env.slot_id_to_feature_slot.items()):
      # Check if feature_slot has at least one FeatureColumn associated with it
      # If not, it means that the slot only has ZeroFeatureColumn, so we
      # ignore it.
      if len(feature_slot.feature_columns) > 0:
        for feature_column in feature_slot.feature_columns:
          # If the vocab size per slot is set, we need to adjust the
          # vocab_id so that no vocab_id exceed this vocab size per slot
          embedding_tensor = example["{}_0".format(feature_column.fc_name)]

          if isinstance(feature_column, FeatureColumn3D):
            new_embedding_tensor = embedding_tensor.to_sparse()
            new_embedding_tensor = tf.SparseTensor(
                indices=new_embedding_tensor.indices,
                values=tf.maximum(new_embedding_tensor.values, 0),
                dense_shape=new_embedding_tensor.dense_shape)
          else:
            new_embedding_tensor = tf.SparseTensor(
                indices=embedding_tensor.indices,
                values=tf.maximum(embedding_tensor.values, 0),
                dense_shape=embedding_tensor.dense_shape)

          if self.p.vocab_size_per_slot is not None:
            new_embedding_tensor = tf.SparseTensor(
                indices=new_embedding_tensor.indices,
                values=tf.math.mod(new_embedding_tensor.values,
                                   self.p.vocab_size_per_slot),
                dense_shape=new_embedding_tensor.dense_shape)
            vocab_size = self.p.vocab_size_per_slot
          else:
            vocab_size = self._env._vocab_size_dict.get(slot_id, 10)
            if self.p.custom_vocab_size_mapping is not None and slot_id in self.p.custom_vocab_size_mapping:
              new_embedding_tensor = tf.SparseTensor(
                  indices=new_embedding_tensor.indices,
                  values=tf.math.mod(new_embedding_tensor.values,
                                     self.p.custom_vocab_size_mapping[slot_id]),
                  dense_shape=new_embedding_tensor.dense_shape)
              vocab_size = self.p.custom_vocab_size_mapping[slot_id]

          if self.p.qr_multi_hashing and vocab_size > self.p.qr_hashing_threshold:
            # setting quotient/remainder vocab size
            R_vocab_size = vocab_size // self.p.qr_collision_rate + 1
            Q_vocab_size = self.p.qr_collision_rate + 1

            embedding_tensor = example["{}_0".format(feature_column.fc_name)]
            del example["{}_0".format(feature_column.fc_name)]

            # creating two features for remainder/quotient
            for feature_slice in feature_column.feature_slice_to_tf_placeholder:
              example["{}_{}_0".format(
                  feature_column.fc_name,
                  feature_slice.slice_index)] = tf.SparseTensor(
                      indices=embedding_tensor.indices,
                      values=tf.math.floormod(embedding_tensor.values,
                                              R_vocab_size),
                      dense_shape=embedding_tensor.dense_shape)
              example["{}_{}_1".format(
                  feature_column.fc_name,
                  feature_slice.slice_index)] = tf.SparseTensor(
                      indices=embedding_tensor.indices,
                      values=tf.math.floordiv(embedding_tensor.values,
                                              R_vocab_size),
                      dense_shape=embedding_tensor.dense_shape)

          else:
            if isinstance(feature_column, FeatureColumn3D):
              # Get row_lengths from embedding_tensor, which is RaggedTensor
              # for FeatureColumn3D
              row_lengths = tf.cast(
                  embedding_tensor.row_lengths(),
                  tf.int32,
              )  # [B] Tensor
              example["{}_0_row_lengths".format(
                  feature_column.fc_name)] = row_lengths
              # seq feature, dims[0][0] is max seq length
              new_embedding_tensor = tf.sparse.slice(
                  new_embedding_tensor, [0, 0], [
                      new_embedding_tensor.dense_shape[0],
                      feature_column.max_seq_length
                  ])

            example["{}_0".format(feature_column.fc_name)] = tf.sparse.reorder(
                new_embedding_tensor)

            for feature_slice in feature_column.feature_slice_to_tf_placeholder:
              if feature_slice.slice_index != 0:
                example["{}_{}".format(
                    feature_column.fc_name,
                    feature_slice.slice_index)] = example["{}_0".format(
                        feature_column.fc_name)]
    # This logic is to calculate AUC which follows current DeepInsight sampling logic.
    # Basically we distribute samples by their UIDs in _RATIO_N buckets.
    # Like we get their UID_BUCKET = UID % _RATIO_N.
    # Later we choose only the examples if their UID_BUCKET < _RATIO_N * _UID_SAMPLE_RATE
    if hs._UID in example:
      example[hs._UID_BUCKET] = tf.cast(
          tf.math.mod(example[hs._UID], hs._RATIO_N), tf.int32)
    return example

  def create_feature_and_table_config_dict(self):
    """Prepares the table and feature config given the parameters."""
    env = self._env
    assert env.is_finalized()

    for slot_id, feature_slot in list(
        self._env.slot_id_to_feature_slot.items()):
      vocab_size = env.vocab_size_dict.get(slot_id, 1)
      # Check if feature_slot has at least one FeatureColumn associated with it
      # If not, it means that the slot only has ZeroFeatureColumn, so we
      # ignore it.
      if len(feature_slot.feature_columns) > 0:
        # Iterate through feature columns to create TableConfig and FeatureConfig
        for feature_column in feature_slot.feature_columns:
          for feature_slice in feature_column.feature_slice_to_tf_placeholder:
            if self.p.qr_multi_hashing and vocab_size > self.p.qr_hashing_threshold:
              # creating quotient/remainder embedding table
              logging.info('Setting QR table for slot {}'.format(slot_id))

              R_vocab_size = vocab_size // self.p.qr_collision_rate + 1
              Q_vocab_size = self.p.qr_collision_rate + 1

              # remainder embedding table
              table_name = "table_{}_{}_0".format(slot_id,
                                                  feature_slice.slice_index)
              if table_name not in self._table_to_config_dict:
                Rtable = tpu_embedding.TableConfig(
                    vocabulary_size=R_vocab_size,
                    dimension=feature_slice.dim,
                    initializer=feature_slice.initializer,
                    combiner="sum",
                    learning_rate_fn=feature_slice.learning_rate_fn,
                    optimization_parameters=feature_slice.optimizer)

                self._table_to_config_dict[table_name] = Rtable
              # remainder feature config
              feature_name = "{}_{}_0".format(feature_column.fc_name,
                                              feature_slice.slice_index)
              self._feature_to_config_dict[
                  feature_name] = tpu_embedding.FeatureConfig(table_name)

              # quotient embedding table
              table_name = "table_{}_{}_1".format(slot_id,
                                                  feature_slice.slice_index)
              if table_name not in self._table_to_config_dict:
                Qtable = tpu_embedding.TableConfig(
                    vocabulary_size=Q_vocab_size,
                    dimension=feature_slice.dim,
                    initializer=feature_slice.initializer,
                    combiner="sum",
                    learning_rate_fn=feature_slice.learning_rate_fn,
                    optimization_parameters=feature_slice.optimizer)
                self._table_to_config_dict[table_name] = Qtable

              # quotient feature config
              feature_name = "{}_{}_1".format(feature_column.fc_name,
                                              feature_slice.slice_index)
              self._feature_to_config_dict[
                  feature_name] = tpu_embedding.FeatureConfig(table_name)

            table_name = "table_{}_{}".format(slot_id,
                                              feature_slice.slice_index)
            if table_name not in self._table_to_config_dict:
              table = tpu_embedding.TableConfig(
                  vocabulary_size=vocab_size,
                  dimension=feature_slice.dim,
                  initializer=feature_slice.initializer,
                  combiner="sum",
                  learning_rate_fn=feature_slice.learning_rate_fn,
                  optimization_parameters=feature_slice.optimizer)
              self._table_to_config_dict[table_name] = table
            feature_name = "{}_{}".format(feature_column.fc_name,
                                          feature_slice.slice_index)
            # Multiple feature configs can share the same table config
            if isinstance(feature_column, FeatureColumn3D):
              self._feature_to_config_dict[
                  feature_name] = tpu_embedding.FeatureConfig(
                      table_name,
                      max_sequence_length=feature_column.max_seq_length)
            else:
              self._feature_to_config_dict[
                  feature_name] = tpu_embedding.FeatureConfig(table_name)
    return self._feature_to_config_dict, self._table_to_config_dict

  def cross_shard_optimizer(self, optimizer, params):
    if params["cpu_test"]:
      return optimizer
    else:
      return tf.compat.v1.tpu.CrossShardOptimizer(optimizer)

  def process_features_for_cpu_test(self, features):
    processed_features = {}
    for feature_name, feature_value in features.items():
      if isinstance(feature_value, tf.sparse.SparseTensor):
        feature_config = self._feature_to_config_dict[feature_name]
        table_config = self._table_to_config_dict[feature_config.table_id]

        dim = table_config.dimension
        max_sequence_length = feature_config.max_sequence_length
        vocab_size = table_config.vocabulary_size

        if feature_config.max_sequence_length == 0:
          initvalue = (np.random.rand(vocab_size, dim) - 0.5) / (vocab_size *
                                                                 dim)
        else:
          initvalue = (np.random.rand(vocab_size, max_sequence_length * dim) -
                       0.5) / (vocab_size * max_sequence_length * dim)

        initvalue = tf.cast(initvalue, tf.float32)
        embedding_variable = tf.compat.v1.get_variable(name=feature_name,
                                                       initializer=initvalue,
                                                       dtype=tf.float32)

        # Get new feature ids based on vocab_size. We mod
        # by vocab_size to make sure new feature ids will be
        # within vocab_size. This is only for test purpose.
        new_feature_ids = tf.SparseTensor(indices=feature_value.indices,
                                          values=tf.math.mod(
                                              feature_value.values, vocab_size),
                                          dense_shape=feature_value.dense_shape)

        # Get embeddings.
        embeddings = tf.nn.safe_embedding_lookup_sparse(embedding_variable,
                                                        new_feature_ids,
                                                        sparse_weights=None,
                                                        combiner="sum")

        if max_sequence_length != 0:
          embeddings = tf.reshape(embeddings, [-1, max_sequence_length, dim])

        processed_features[feature_name] = embeddings
      else:
        processed_features[feature_name] = feature_value

    # For CPU test, we will clear this state from now on. so later some host_call
    # will not use do use them to do tpu specific operation.
    self._feature_to_config_dict.clear()
    self._table_to_config_dict.clear()

    return processed_features
