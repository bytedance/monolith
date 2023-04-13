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

import json
from typing import List, Dict, Union, Optional

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.util.tf_export import tf_export
from monolith.native_training.summary import utils
from monolith.native_training.summary.utils import SummaryType
from monolith.native_training.layers.layer_ops import feature_insight


@tf_export(v1=["summary.nas_data"])
def nas_data(weight, segment_names=None, segment_sizes=None, group_info=None, raw_tag=None,
             collections=None, description=None, name=None):
  meta_content, summaty_type = utils.prepare_head(segment_names, segment_sizes, group_info,
                                                  raw_tag, out_type='bytes')
  name = f'{name}_{summaty_type}' if name else summaty_type
  description = description or summaty_type
  with tf.name_scope(name):
    return tf.compat.v1.summary.tensor_summary(
      name=utils.MONOLITH_NAS_DATA,
      tensor=weight,
      collections=collections or [ops.GraphKeys.SUMMARIES],
      summary_metadata=utils.create_summary_metadata(description, meta_content),
    )


@tf_export(v1=["summary.feature_insight_data"])
def feature_insight_data(input_tensor: tf.Tensor, segment_names: List[str], segment_sizes: List[int],
                         weight: tf.Tensor = None, group_info: Union[List[int], List[List[int]]] = None,
                         label: tf.Tensor = None, collections: List[str] = None,
                         description: str = None, name: str = None):
  assert segment_sizes is not None and len(segment_names) == len(segment_sizes)
  aggregate = True if label is None else False
  raw_tag = SummaryType.FEATURE_INSIGHT_DIRECT if aggregate else SummaryType.FEATURE_INSIGHT_TRAIN
  if weight is None:
    summary_data = input_tensor
  else:
    summary_data = feature_insight(
      input_embedding=input_tensor, weight=weight, segment_sizes=segment_sizes, aggregate=aggregate)
    segment_sizes = [1 if aggregate else weight.shape.as_list()[-1]] * len(segment_sizes)
  meta_content, summaty_type = utils.prepare_head(segment_names, segment_sizes, group_info,
                                                  raw_tag=raw_tag, out_type='json')
  name = f'{name}_{summaty_type}' if name else summaty_type
  description = description or summaty_type
  if label is not None:
    if label.dtype != tf.float32:
      label = tf.cast(label, dtype=tf.float32)
    if label.shape.rank == 1:
      label = tf.reshape(label, shape=(-1, 1))
      meta_content['label_size'] = 1
    else:
      meta_content['label_size'] = label.shape.as_list()[-1]
    summary_data = tf.concat(values=[summary_data, label], axis=1)
  else:
    meta_content['label_size'] = 0

  with tf.name_scope(name):
    return tf.compat.v1.summary.tensor_summary(
      name=utils.MONOLITH_FI_DATA,
      tensor=summary_data,
      collections=collections or [ops.GraphKeys.SUMMARIES],
      summary_metadata=utils.create_summary_metadata(description, json.dumps(meta_content)),
    )
