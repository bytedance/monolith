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

"""Check in TPU embedding feature from TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math

from absl import logging
import tensorflow as tf
from tensorflow.python.tpu import tpu_embedding

from monolith.core.feature import FeatureSlot, FeatureColumnV1, Env

_FLOAT32_BYTES = 4


### This code will be deprecated. Please update and usebase_embedding_task.py.
class Model(object):
  """Sail-like TPU Model."""

  def __init__(self, params):
    self._params = params
    self._vocab_size_per_slot = params["vocab_size_per_slot"]
    if self._vocab_size_per_slot is not None:
      logging.info("Set fixed vocab_size: {} for all the slots.".format(
          self._vocab_size_per_slot))
    vocab_size_dict = self._create_vocab_dict(params["vocab_file_path"],
                                              self._vocab_size_per_slot)
    self._env = Env(vocab_size_dict=vocab_size_dict)
    # Run this to initialize slot, embedding dim information
    self.init_slot_to_dims()

  def _create_vocab_dict(self, file_path, vocab_size_per_slot=None):
    """Create vocab dict from a tsv file.

        Args:
            file_path: the path to the vocab dict
            vocab_size_per_slot: If None, this is set to the number of unique
                FIDs for each slot, obtained from the vocab_size file.
                Otherwise, this value is used to manually set the vocab sise per
                slot (this option is to speed up testing and modeling iteration).

        """
    vocab_size_dict = {}
    with open(file_path) as f:
      for line in f:
        fields = line.strip().split("\t")
        assert len(fields) == 2, "each line in {} must have 2 fields".format(
            fields)
        if fields[0].isdigit() == False:
          continue

        slot_id = int(fields[0])

        distinct_count = vocab_size_per_slot
        if vocab_size_per_slot is None:
          distinct_count = int(fields[1])

        vocab_size_dict[slot_id] = distinct_count

    return vocab_size_dict

  def _get_feature_map(self):
    """Returns data format of the serialized tf record file."""
    # Inherated class must implement this function.
    raise NotImplementedError

  def _post_process_example(self, example):
    """Postprocess example."""
    # build tensors for each embeddings in each slot
    for slot_id, dims in self._env.slot_to_dims.items():
      # If the vocab size per slot is set, we need to adjust the
      # vocab_id so that no vocab_id exceed this vocab size per slot
      if self._vocab_size_per_slot:
        embedding_tensor = example["slot_{}_0".format(slot_id)]
        new_embedding_tensor = tf.SparseTensor(
            indices=embedding_tensor.indices,
            values=tf.math.mod(embedding_tensor.values,
                               self._vocab_size_per_slot),
            dense_shape=embedding_tensor.dense_shape)
        example["slot_{}_0".format(slot_id)] = new_embedding_tensor

      for i in range(1, len(dims)):
        example["slot_{}_{}".format(slot_id,
                                    i)] = example["slot_{}_0".format(slot_id)]
    return example

  def create_input_fn(self, file_pattern, repeat=True):

    def tf_example_parser(examples):
      """Parse multiple examples."""
      feature_map = self._get_feature_map()
      example = tf.io.parse_example(serialized=examples, features=feature_map)
      return self._post_process_example(example)

    def input_fn(params):
      """Returns training or eval examples, batched as specified in params."""
      logging.info("Model input_fn")

      # By shuffle=False, list_files will get all files already in time sorted order.
      files = tf.data.Dataset.list_files(file_pattern, shuffle=False)
      # This function will get called once per TPU task. Each task will process the files
      # with indexs which modulo num_calls equals to call_index.
      _, call_index, num_calls, _ = (
          params["context"].current_input_fn_deployment())
      files = files.shard(num_calls, call_index)

      skip_files_number = 0
      if params["shard_skip_file_number"] is not None:
        skip_files_number = params["shard_skip_file_number"][call_index]

      logging.info("Shard {} skipped {} files.".format(call_index,
                                                       skip_files_number))
      files = files.skip(skip_files_number)

      def fetch_dataset(filename):
        dataset = tf.data.TFRecordDataset(
            filename,
            compression_type=params["compression_type"],
            buffer_size=None)
        return dataset

      # Read the data from disk in parallel.
      # Files will be process from the beginning to the end. With a local parallel of interleaving
      # multiple files currently. Number of interleaving files are defined by the cycle.
      dataset = files.interleave(
          fetch_dataset,
          cycle_length=params["cycle_length"],
          num_parallel_calls=params["num_parallel_calls"],
          deterministic=False)
      dataset = dataset.batch(params["batch_size"], drop_remainder=True).map(
          tf_example_parser,
          num_parallel_calls=tf.data.experimental.AUTOTUNE,
          deterministic=False)

      # The tensors returned from this dataset will be directly used as the ids
      # for the embedding lookup. If you want to have a separate vocab, apply a
      # '.map' here to the dataset which contains you vocab lookup.
      if repeat:
        dataset = dataset.repeat()

      dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

      return dataset

    return input_fn

  def _padding_8(self, dim):
    return math.ceil(dim / 8) * 8

  def _get_slot_number(self, optimizer, use_gradient_accumulation):
    slot_num = 0
    if isinstance(optimizer, tf.compat.v1.tpu.experimental.FtrlParameters):
      slot_num = 3
    elif isinstance(optimizer, tf.compat.v1.tpu.experimental.AdagradParameters):
      slot_num = 2
    elif isinstance(optimizer, tf.compat.v1.tpu.experimental.AdamParameters):
      slot_num = 3
    elif isinstance(
        optimizer,
        tf.compat.v1.tpu.experimental.StochasticGradientDescentParameters):
      slot_num = 1
    else:
      assert ("We don't support this optimizer type yet: {}".format(
          type(optimizer)))

    if use_gradient_accumulation == True:
      slot_num += 1

    return slot_num

  def _get_max_slot_number(self):
    max_slot_number = 0
    for slot_id, dims in self._env.slot_to_dims.items():
      feature_slot = self._env.slot_to_config[slot_id]
      for index, dim in enumerate(dims):
        optimizer = None
        # If index is 0 and feature slot uses bias, then we will use bias optimizer and table initializer.
        # Also please note if slot has bias, bias will always use index = 0.
        if index == 0 and feature_slot.use_bias:
          optimizer = feature_slot.bias_optimizer
        else:
          optimizer = feature_slot.vec_optimizer
        max_slot_number = max(
            max_slot_number,
            self._get_slot_number(optimizer,
                                  self._params["use_gradient_accumulation"]))
    return max_slot_number

  def create_feature_and_table_config_dict(self):
    """Prepares the table and feature config given the parameters."""
    env = self._env
    assert env.is_finalized()
    feature_to_config_dict = {}
    table_to_config_dict = {}

    embedding_table_size = 0
    embedding_table_size_after_padding_8 = 0
    embedding_table_size_after_padding_8_and_use_max_auxiliary_parameters = 0
    max_slot_number = self._get_max_slot_number()

    for slot_id, dims in env.slot_to_dims.items():
      assert slot_id in env.vocab_size_dict, "slot_id {} must be in vocab file".format(
          slot_id)
      vocab_size = env.vocab_size_dict[slot_id]

      assert slot_id in env.slot_to_config, "slot_id {} must be in slot_to_config".format(
          slot_id)
      feature_slot = env.slot_to_config[slot_id]

      for index, dim in enumerate(dims):
        optimizer = None
        table_initializer = None

        # If index is 0 and feature slot uses bias, then we will use bias optimizer and table initializer.
        # Also please note if slot has bias, bias will always use index = 0.
        if index == 0 and feature_slot.use_bias():
          optimizer = feature_slot.bias_optimizer
          table_initializer = feature_slot.bias_initializer
        else:
          optimizer = feature_slot.vec_optimizer
          table_initializer = feature_slot.vec_initializer

        table = tpu_embedding.TableConfig(vocabulary_size=vocab_size,
                                          dimension=dim,
                                          initializer=table_initializer,
                                          combiner="sum",
                                          optimization_parameters=optimizer)
        table_name = "table_{}_{}".format(slot_id, index)
        table_to_config_dict[table_name] = table
        feature_name = "slot_{}_{}".format(slot_id, index)
        feature_to_config_dict[feature_name] = tpu_embedding.FeatureConfig(
            table_name)

        slot_num = self._get_slot_number(
            optimizer, self._params["use_gradient_accumulation"])
        embedding_table_size += vocab_size * dim * _FLOAT32_BYTES * slot_num
        embedding_table_size_after_padding_8 += vocab_size * self._padding_8(
            dim) * _FLOAT32_BYTES * slot_num
        embedding_table_size_after_padding_8_and_use_max_auxiliary_parameters += vocab_size * self._padding_8(
            dim) * _FLOAT32_BYTES * max_slot_number

    logging.info("Size of all embedding tables in bytes: {}".format(
        embedding_table_size))
    logging.info(
        "Size after padding the width of all tables to 8 float multiples in bytes: {}"
        .format(embedding_table_size_after_padding_8))
    logging.info(
        "Size after padding to 8 float multiples and using max auxiliary parameters: {}"
        .format(
            embedding_table_size_after_padding_8_and_use_max_auxiliary_parameters
        ))

    return feature_to_config_dict, table_to_config_dict

  def sum_pooling(self,
                  fc_dict,
                  input_map,
                  features,
                  dim,
                  total_embeddings,
                  add_into_embeddings=True):
    slot_embeddings = []
    dims = 0
    for slot in features:
      #allocate embedding
      embedding = fc_dict[slot].add_vector(dim)
      dims += dim
      if add_into_embeddings:
        total_embeddings.append((embedding, dim))
      slot_embeddings.append(embedding)
      if slot in input_map:
        input_slots = input_map.keys()
        c = 0
        for item in input_slots:
          if isinstance(item, str):
            if str(slot) + '_' in item:
              c += 1
          if isinstance(item, int):
            if item == slot:
              c += 1
        input_map[str(slot) + '_' + str(c)] = embedding
      else:
        input_map[slot] = embedding
    if len(features) == 1:  #单特征无需sum
      return slot_embeddings[0]
    return tf.add_n(slot_embeddings)

  def logits_fn(self):
    """Calculate logits."""
    # Inherated class must implement this function.
    raise NotImplementedError

  def init_slot_to_dims(self):
    """Run this in the beginning to initialize the slot and its embedding dims information."""
    logging.info("Model init_slot_to_dims")
    self.logits_fn()
    self._env.finalize()
    logging.info("_slot_to_dims: {}".format(self._env.slot_to_dims))

  def create_model_fn(self):
    """Creates the model_fn to be used by the TPUEstimator."""
    # Inherated class must implement this function.
    raise NotImplementedError
