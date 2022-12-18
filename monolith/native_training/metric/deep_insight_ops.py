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

import os
import socket
from typing import List

import tensorflow as tf
from tensorflow.python.framework import ops

from monolith.native_training.runtime.ops import gen_monolith_ops

deep_insight_ops = gen_monolith_ops

_FEATURE_REQ_TIME = "req_time"
_SAMPLE_RATE = "sample_rate"
_UID = "uid"

def deep_insight_client(enable_metrics_counter: bool=False, is_fake: bool = False, container: str=socket.gethostname()) \
  -> tf.Tensor:
  """
  Create a deep insight client
  Args:
    enable_metrics_counter - whether enable metrics counter for using deepinsight.
    container - Use host name as the container name. So that each container will
                create and use a seperate deepinsight resource.
  """
  return deep_insight_ops.monolith_create_deep_insight_client(
      enable_metrics_counter, is_fake, container)


def write_deep_insight(deep_insight_client_tensor: tf.Tensor,
                       uids: tf.Tensor,
                       req_times: tf.Tensor,
                       labels: tf.Tensor,
                       preds: tf.Tensor,
                       sample_rates: tf.Tensor,
                       model_name: str,
                       target: str = "ctr_head",
                       sample_ratio: float = 0.01,
                       return_msgs: bool = False,
                       use_zero_train_time=False) -> tf.Tensor:
  """
  Write one instance's metrics to deep insight. Internal it includes parse and 
  build JSON format deep insight message. And send to databus channel using 
  unblock API.
  Args:
    uid - a 1-D int64 tensor.
    req_time - a 1-D int64 tensor.
    labels - a 1-D float tensor.
    preds - a 1-D float tensor.
    sample_rates - a 1-D float tensor.
    model_name - model name of string type.
    target - target of string type.
    sample_ratio - sample ratio of float type.
    return_msg - whether return the msg sent to deepinsight for debugging.
    use_zero_train_time - Use True if you want to use training time (0) in deepinsight.
                          this is actually used only in test. Use false if
                          you want to use real training time to write to deepinsight.
  Returns:
    1-D string tensor.
  """
  return deep_insight_ops.monolith_write_deep_insight(
      deep_insight_client_handle=deep_insight_client_tensor,
      uids=uids,
      req_times=req_times,
      labels=labels,
      preds=preds,
      sample_rates=sample_rates,
      model_name=model_name,
      target=target,
      sample_ratio=sample_ratio,
      return_msgs=return_msgs,
      use_zero_train_time=use_zero_train_time)


def write_deep_insight_v2(deep_insight_client_tensor: tf.Tensor,
                          req_times: tf.Tensor,
                          labels: tf.Tensor,
                          preds: tf.Tensor,
                          sample_rates: tf.Tensor,
                          extra_fields_values: List[tf.Tensor],
                          extra_fields_keys: List[str],
                          model_name: str,
                          targets: List[str],
                          sample_ratio: float = 0.01,
                          return_msgs: bool = False,
                          use_zero_train_time=False) -> tf.Tensor:
  """
  Write one instance's metrics to deep insight. Internal it includes parse and 
  build JSON format deep insight message. And send to databus channel using 
  unblock API.
  Args:
    deep_insight_client_tensor: MonolithCreateDeepInsightClient
    req_times: 1-D int64 tensor, shape = (batch_size,)
    labels: 2-D float tensor, shape = (num_targets, batch_size)
    preds: 2-D float tensor, shape = (num_targets, batch_size)
    sample_rates: 2-D float tensor, shape = (num_targets, batch_size)
    extra_fields_values: List of 1-D tensors, each shape = (batch_size,)
    extra_fields_keys: List of strings.
    model_name: model name of string type.
    targets: List of target names.
    sample_ratio: sample ratio of float type.
    return_msgs: whether return the msg sent to deepinsight for debugging.
    use_zero_train_time: Use True if you want to use training time (0) in deepinsight.
                         this is actually used only in test. Use false if
                         you want to use real training time to write to deepinsight.
  Returns:
    1-D string tensor.
  """
  return deep_insight_ops.monolith_write_deep_insight_v2(
      deep_insight_client_handle=deep_insight_client_tensor,
      req_times=req_times,
      labels=labels,
      preds=preds,
      sample_rates=sample_rates,
      extra_fields_values=extra_fields_values,
      extra_fields_keys=extra_fields_keys,
      model_name=model_name,
      targets=targets,
      sample_ratio=sample_ratio,
      return_msgs=return_msgs,
      use_zero_train_time=use_zero_train_time)
