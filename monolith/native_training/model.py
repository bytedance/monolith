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

from datetime import datetime
import numpy as np
from typing import Callable, Dict, List

from absl import logging

import tensorflow as tf

from monolith.core import model_registry
from monolith.core.base_model_params import SingleTaskModelParams
from monolith.native_training import entry, feature
from monolith.native_training.input import (generate_ffm_example, slot_to_key)
import monolith.native_training.metric.deep_insight_ops as deep_insight_ops
from monolith.native_training.native_task import NativeTask

_NUM_SLOTS = 6
_FFM_SLOT = ((0, 3, 16), (0, 4, 16), (1, 5, 16), (2, 3, 16), (2, 5, 16))
_VOCAB_SIZES = [5, 5, 5, 5, 5, 5]
_NUM_EXAMPLES = 64


def _parse_example(example: str) -> Dict[str, tf.Tensor]:

  def _get_feature_map():
    feature_map = {}
    feature_map["label"] = tf.io.FixedLenFeature([], dtype=tf.float32)
    for i in range(len(_VOCAB_SIZES)):
      feature_map[slot_to_key(i)] = tf.io.VarLenFeature(dtype=tf.int64)
    return feature_map

  features = tf.io.parse_example(example, _get_feature_map())
  for k, v in features.items():
    if isinstance(v, tf.sparse.SparseTensor):
      features[k] = tf.RaggedTensor.from_sparse(v)
  return features


class TestFFMModel(NativeTask):

  def __init__(self, params):
    super().__init__(params)
    self.p = params

  def create_input_fn(self, mode):

    def input_fn():
      # This keeps the training data stability so we can resume training from the
      # checkpoint.
      np.random.seed(0)
      examples = [
          generate_ffm_example(_VOCAB_SIZES) for i in range(_NUM_EXAMPLES)
      ]
      dataset = tf.data.Dataset.from_tensor_slices(examples)
      dataset = dataset.batch(self.p.train.per_replica_batch_size,
                              drop_remainder=True)
      dataset = dataset.map(_parse_example,
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
      dataset = dataset.cache().repeat()
      dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

      return dataset

    return input_fn

  def create_model_fn(self):

    def model_fn(features: Dict, mode: tf.estimator.ModeKeys,
                 config: tf.estimator.RunConfig) -> tf.estimator.EstimatorSpec:
      del config
      global_step = tf.compat.v1.train.get_or_create_global_step()
      slots = {}
      fc = {}
      bias_list = []
      for i in range(_NUM_SLOTS):
        slots.update({
            i:
                self.ctx.create_feature_slot(
                    feature.FeatureSlotConfig(
                        name=str(i),
                        has_bias=i <= (_NUM_SLOTS // 2),
                        bias_optimizer=entry.FtrlOptimizer(
                            learning_rate=0.1,
                            initial_accumulator_value=1e-6,
                            beta=1.0),
                        default_vec_optimizer=entry.SgdOptimizer(
                            learning_rate=0.1)))
        })
        fc.update({i: feature.FeatureColumnV1(slots[i], slot_to_key(i))})
        if i <= (_NUM_SLOTS // 2):
          bias_list.append(fc[i].embedding_lookup(slots[i].get_bias_slice()))

      bias_input = tf.concat(bias_list,
                             axis=1,
                             name='concatenate_tensor_from_{}_bias'.format(
                                 len(bias_list)))
      sum_bias = tf.reduce_sum(bias_input, axis=1)
      dot_res = []
      for user, item, dim in _FFM_SLOT:
        user_vec = fc[user].embedding_lookup(slots[user].add_feature_slice(dim))
        item_vec = fc[item].embedding_lookup(slots[item].add_feature_slice(dim))
        dot_res.append(tf.reduce_sum(tf.multiply(user_vec, item_vec), 1))
      ffm_out = tf.add_n(dot_res) + sum_bias
      pred = tf.nn.sigmoid(ffm_out)
      if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred)
      loss = tf.reduce_sum(
          tf.losses.binary_crossentropy(features["label"], pred))

      # Write deep insight
      if self.p.metrics.enable_deep_insight and self.p.metrics.deep_insight_sample_ratio > 0:
        deep_insight_client = deep_insight_ops.deep_insight_client(False)
        now = datetime.now()
        model_name = self.p.metrics.deep_insight_name
        uids = tf.cast(tf.fill([self.p.train.per_replica_batch_size], 0),
                       dtype=tf.int64)
        req_times = tf.cast(tf.fill([self.p.train.per_replica_batch_size],
                                    int(datetime.timestamp(now))),
                            dtype=tf.int64)
        sample_rates = tf.fill([self.p.train.per_replica_batch_size], 0.1)
        target = "ctr_head"

        deep_insight_op = deep_insight_ops.write_deep_insight(
            deep_insight_client_tensor=deep_insight_client,
            uids=uids,
            req_times=req_times,
            labels=features["label"],
            preds=pred,
            sample_rates=sample_rates,
            model_name=model_name,
            target=target,
            sample_ratio=0.01)
        logging.info("model_name: {}, target: {}.".format(model_name, target))
      else:
        deep_insight_op = tf.no_op()

      update_global_step = tf.compat.v1.assign_add(global_step, 1)
      all_embeddings = [v.get_all_embeddings_concat() for v in fc.values()]
      emb_grads = tf.gradients(loss, all_embeddings)
      with tf.control_dependencies([update_global_step]):
        train_op = tf.group(
            self.ctx.apply_embedding_gradients(zip(emb_grads, all_embeddings)),
            deep_insight_op)
      return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    return model_fn

  def create_serving_input_receiver_fn(self):

    def serving_input_receiver_fn():
      receiver_tensors = {}
      instances_placeholder = tf.compat.v1.placeholder(dtype=tf.string,
                                                       shape=(None,))
      receiver_tensors["instances"] = instances_placeholder
      parsed_results = _parse_example(instances_placeholder)

      return tf.estimator.export.ServingInputReceiver(parsed_results,
                                                      receiver_tensors)

    return serving_input_receiver_fn


@model_registry.RegisterSingleTaskModel
class FFMParams(SingleTaskModelParams):

  def task(self):
    p = TestFFMModel.params()
    p.train.per_replica_batch_size = 64
    return p
