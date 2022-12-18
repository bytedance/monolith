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
from typing import List, Tuple, Optional, NamedTuple

import tensorflow as tf

from monolith.native_training.runtime.ops import gen_monolith_ops

inbatch_auc_loss_ops = gen_monolith_ops


def inbatch_auc_loss(label: tf.Tensor,
                     logit: tf.Tensor,
                     neg_weight=1.0) -> tf.Tensor:
  return inbatch_auc_loss_ops.inbatch_auc_loss(label=label,
                                               logit=logit,
                                               neg_weight=neg_weight)


@tf.RegisterGradient(op_type='InbatchAucLoss')
def _inbatch_auc_loss_grad(op: tf.Operation, grad: tf.Tensor):
  label, logit = op.inputs[0], op.inputs[1]
  neg_weight = op.get_attr(name='neg_weight')
  logit_grad = inbatch_auc_loss_ops.inbatch_auc_loss_grad(label=label,
                                                          logit=logit,
                                                          grad=grad,
                                                          neg_weight=neg_weight)
  return None, logit_grad
