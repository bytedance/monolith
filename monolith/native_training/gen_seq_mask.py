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

import tensorflow as tf
from typing import Union
from monolith.native_training.runtime.ops import gen_monolith_ops

ops = gen_monolith_ops


def gen_seq_mask(splits: Union[tf.Tensor, tf.RaggedTensor],
                 max_seq_length: int) -> tf.Tensor:
  if isinstance(splits, tf.RaggedTensor):
    splits = splits.row_splits()
  return ops.gen_seq_mask(splits=splits, max_seq_length=max_seq_length)
