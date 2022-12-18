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

import numpy as np
from typing import List
import tensorflow as tf


def slot_to_key(slot: int):
  return "feature_{}".format(slot)


def generate_ffm_example(vocab_sizes: List[int], length=5) -> str:
  """Generate a random training example."""

  def _int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

  def _float32_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))

  feature = {}
  feature["label"] = _float32_feature([np.random.randint(low=0, high=1)])
  max_vocab = max(vocab_sizes)

  for i, vocab_size in enumerate(vocab_sizes):
    num_ids = np.random.randint(length) + 1
    ids = np.random.randint(max_vocab * i,
                            max_vocab * i + vocab_size,
                            size=num_ids).tolist()
    feature[slot_to_key(i)] = _int64_feature(ids)

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()
