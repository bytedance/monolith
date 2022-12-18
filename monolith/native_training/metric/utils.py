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

import datetime
import logging
from typing import Dict, List

import tensorflow as tf

from monolith.native_training.metric import deep_insight_ops


def write_deep_insight(features: Dict[str, tf.Tensor],
                       sample_ratio: float,
                       model_name: str,
                       labels: tf.Tensor = None,
                       preds: tf.Tensor = None,
                       target: str = None,
                       targets: List[str] = None,
                       labels_list: List[tf.Tensor] = None,
                       preds_list: List[tf.Tensor] = None,
                       sample_rates_list: List[tf.Tensor] = None,
                       extra_fields_keys: List[str] = [],
                       enable_deep_insight_metrics=True,
                       enable_kafka_metrics=False) -> tf.Tensor:
  """ Writes the data into deepinsight
  Requires 'uid', 'req_time', and 'sample_rate' in features. 
  sample_ratio is deepinsight sample ratio, set value like 0.01.
  
  If targets is non-empty, MonolithWriteDeepInsightV2 will be used, enabling:
  - Multi-target sent as one message;
  - Dump extra fields.
  When using MonolithWriteDeepInsightV2, labels/preds/sample_rates should be
  shape (num_targets, batch_size). sample_rates is optional.
  Extra fields specified in extra_fields_keys must be present in features, and
  must have batch_size numbers of values.
  """
  deep_insight_client = deep_insight_ops.deep_insight_client(
      enable_deep_insight_metrics, enable_kafka_metrics)
  req_times = tf.reshape(features["req_time"], [-1])

  if not targets:
    uids = tf.reshape(features["uid"], [-1])
    sample_rates = tf.reshape(features["sample_rate"], [-1])
    deep_insight_op = deep_insight_ops.write_deep_insight(
        deep_insight_client_tensor=deep_insight_client,
        uids=uids,
        req_times=req_times,
        labels=labels,
        preds=preds,
        sample_rates=sample_rates,
        model_name=model_name,
        target=target,
        sample_ratio=sample_ratio)
  else:
    labels = tf.stack(labels_list)
    preds = tf.stack(preds_list)
    if not sample_rates_list:
      sample_rates_list = [tf.reshape(features["sample_rate"], [-1])
                          ] * len(targets)
    sample_rates = tf.stack(sample_rates_list)
    if "uid" not in extra_fields_keys:
      extra_fields_keys.append("uid")
    extra_fields_values = []
    for key in extra_fields_keys:
      extra_fields_values.append(tf.reshape(features[key], [-1]))
    deep_insight_op = deep_insight_ops.write_deep_insight_v2(
        deep_insight_client_tensor=deep_insight_client,
        req_times=req_times,
        labels=labels,
        preds=preds,
        sample_rates=sample_rates,
        model_name=model_name,
        sample_ratio=sample_ratio,
        extra_fields_values=extra_fields_values,
        extra_fields_keys=extra_fields_keys,
        targets=targets)
  return deep_insight_op
