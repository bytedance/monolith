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
from math import log, exp
import tensorflow as tf
from monolith.native_training.losses import inbatch_auc_loss


class InbatchAucLossTest(tf.test.TestCase):

  def test_inbatch_auc_loss(self):
    label = [1, 0, 0, 1]
    logit = [0.5, -0.2, -0.4, 0.8]
    loss = inbatch_auc_loss.inbatch_auc_loss(label=label, logit=logit)

    loss_truth = 0
    pos, neg = [], []
    for i, l in enumerate(label):
      if l > 0:
        pos.append(i)
      else:
        neg.append(i)

    for i in pos:
      for j in neg:
        diff = logit[i] - logit[j]
        loss_truth += log(1 / (1 + exp(-diff)))

    self.assertAlmostEqual(loss, tf.constant(loss_truth), delta=0.000001)

  def test_inbatch_auc_loss_grad(self):
    label = [1, 0, 0, 1]
    logit = [0.5, -0.2, -0.4, 0.8]
    logit_grad = inbatch_auc_loss.inbatch_auc_loss_ops.inbatch_auc_loss_grad(
        label=label, logit=logit, grad=2, neg_weight=1.0)

    pos, neg = [], []
    for i, l in enumerate(label):
      if l > 0:
        pos.append(i)
      else:
        neg.append(i)

    logit_grad_truth = [0] * len(logit)
    for i in pos:
      for j in neg:
        diff = logit[i] - logit[j]
        grad_ij = 1 - 1 / (1 + exp(-diff))

        logit_grad_truth[i] += grad_ij
        logit_grad_truth[j] -= grad_ij

    logit_grad_truth = [2 * x for x in logit_grad_truth]
    self.assertAllClose(logit_grad, tf.constant(logit_grad_truth))


if __name__ == "__main__":
  # tf.compat.v1.disable_eager_execution()
  tf.test.main()
