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
from functools import lru_cache
from typing import Any, Dict, List, Tuple, Union

import tensorflow as tf
from tensorboard.compat.proto import summary_pb2


PLUGIN_NAME = 'monolith'
MONOLITH_NAS_DATA = f'{PLUGIN_NAME}_nas_weight'
MONOLITH_FI_DATA = f'{PLUGIN_NAME}_feature_insight'
KTYPE, KMETA, KDATA = 'tag_type', 'meta', 'data'


class SummaryType(object):
  GATING = 'gating'
  SELECTING = 'selecting'
  MIXED = 'mixed'
  SIMPLE = 'simple'
  FEATURE_INSIGHT_DIRECT = 'fi_direct'
  FEATURE_INSIGHT_TRAIN = 'fi_train'


# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto
def create_summary_metadata(description: str = None, meta_content=b''):
  return summary_pb2.SummaryMetadata(
    summary_description=description,
    plugin_data=summary_pb2.SummaryMetadata.PluginData(
      plugin_name=PLUGIN_NAME,
      content=meta_content.encode('utf-8') if isinstance(meta_content, str) else meta_content,
    ),
    data_class=summary_pb2.DATA_CLASS_TENSOR,
  )


def _name_to_group_id(segment_names: List[str], group_info: List[List[str]]):
  if group_info:
    name_to_group: Dict[str, int] = {}
    for i, group in enumerate(group_info):
      for name in group:
        name_to_group[name] = i

    group_id_to_names: Dict[int, List[str]] = {}
    for name in segment_names:
      assert name in name_to_group
      group_id = name_to_group[name]
      if group_id in group_id_to_names:
        group_id_to_names[group_id].append(name)
      else:
        group_id_to_names[group_id] = [name]

    name_to_reorder_id = {}
    for idx, group_id in enumerate(sorted(group_id_to_names)):
      for name in group_id_to_names[group_id]:
        name_to_reorder_id[name] = idx
    name_to_reorder_id = name_to_reorder_id
  else:
    name_to_reorder_id = {name: idx for idx, name in enumerate(segment_names)}

  return name_to_reorder_id


def prepare_head(segment_names: List[str], segment_sizes: Union[List[int], List[List[int]]],
                 group_info: List[List[str]] = None, raw_tag: str = None, out_type: str = 'tensor'
                 ) -> Tuple[Any, str]:
  assert out_type in {'bytes', 'tensor', 'json'}
  if not (segment_names or segment_sizes):
    if out_type == 'tensor':
      return tf.constant(value=[b''], dtype=tf.string, shape=tuple()), raw_tag
    else:
      return b'', raw_tag

  raw_tag = raw_tag or (
    SummaryType.GATING if all(isinstance(s, int) for s in segment_sizes) else SummaryType.SELECTING)
  data = {
    KTYPE: raw_tag,
    'segment_names': segment_names,
    'segment_sizes': segment_sizes,
  }
  if raw_tag in {SummaryType.GATING, SummaryType.FEATURE_INSIGHT_TRAIN}:
    name_to_reorder_id = _name_to_group_id(segment_names, group_info)
    data['group_index'] = [name_to_reorder_id[name] for name in segment_names]

  if out_type == 'tensor':
    return tf.constant(value=[json.dumps(data)], dtype=tf.string, shape=tuple()), raw_tag
  elif out_type == 'json':
    return data, raw_tag
  else:
    return json.dumps(data), raw_tag


@lru_cache
def get_nas_weight_json(ckpt_dir_or_file, prefix=None) -> List[str]:
  prefix = prefix or ARCH_TENSOR_PREFIX
  ckpt = tf.train.load_checkpoint(ckpt_dir_or_file=ckpt_dir_or_file)
  if ckpt:
    for name in ckpt.get_variable_to_dtype_map():
      if prefix in name:
        return [str(v) for v in ckpt.get_tensor(name).flat]
  raise Exception('not arch_weights in ckpt')
