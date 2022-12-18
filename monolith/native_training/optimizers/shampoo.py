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
from tensorflow.python.ops import state_ops, io_ops


@tf.function
def eigen_inverse_root(mat, p, head, tail, damping=1e-3):

  alpha = -1.0 / p
  dim = mat.shape[0]

  eval, evec = tf.linalg.eigh(mat)
  non_zero = tf.where(tf.greater(eval, damping))
  zeros = tf.cond(
      tf.greater(tf.size(non_zero),
                 0), lambda: tf.cast(tf.reduce_min(non_zero), dtype="int32"),
      lambda: tf.constant(0, dtype="int32"))  # Count the number of zeros
  eval_p = tf.pow(tf.maximum(eval, damping), alpha)

  if tf.greater(head + tail, dim):
    zeros = 0
    head = dim
    tail = 0
  elif tf.greater(zeros + head + tail, dim):
    zeros = dim - head - tail

  eval_ht = tf.concat([eval_p[zeros:zeros + head], eval_p[dim - tail:]], 0)
  # selected eigenvalues
  evec_ht = tf.concat([evec[:, zeros:zeros + head], evec[:, dim - tail:]], 1)
  # selected eigenvectors

  if tf.equal(zeros + head + tail, dim):
    offset = 0.0
  else:
    offset = tf.reduce_mean(eval[zeros + head:dim - tail])

  return evec_ht, eval_ht - offset, offset


def apply_sparse_precond(tensor, pvec, pval, offset):

  tensor_tmp_1 = tf.tensordot(tensor, pvec, axes=[[0], [0]])
  tensor_tmp_2 = tf.multiply(tensor_tmp_1, pval)
  tensor_tmp_3 = tf.tensordot(tensor_tmp_2, pvec, axes=[[-1], [-1]])

  rank = len(tensor.shape)
  tensor_transpose = tf.transpose(tensor, perm=list(range(1, rank)) + [0])

  return tensor_tmp_3 + tensor_transpose * offset


class ShampooOptimizer(tf.compat.v1.train.Optimizer):

  def __init__(self,
               learning_rate=0.03,
               beta_1: float = 0.9,
               beta_2: float = 1.0,
               warmup: int = 5000,
               tau_1: int = 200,
               tau_2: int = 20,
               eigen_head: int = 100,
               eigen_tail: int = 100,
               damping_epsilon: float = 1e-3,
               use_locking: bool = False,
               name="Shampoo",
               **kwargs):
    super().__init__(use_locking, name, **kwargs)
    self._learning_rate = learning_rate
    self._beta_1 = beta_1
    self._beta_2 = beta_2
    self._warmup = warmup
    self._tau_1 = tau_1
    self._tau_2 = tau_2
    self._eigen_head = eigen_head
    self._eigen_tail = eigen_tail
    self._damping_epsilon = damping_epsilon

  def _create_slots(self, var_list):

    for var in var_list:
      for i, dim in enumerate(var.shape):
        eigens = min(dim, self._eigen_head + self._eigen_tail)
        self._get_or_make_slot(var, tf.zeros([dim, dim]), "s" + str(i),
                               self._name + "/s" + str(i))
        self._get_or_make_slot(var, tf.zeros([dim, dim]), "g" + str(i),
                               self._name + "/g" + str(i))
        self._get_or_make_slot(var, tf.zeros([dim, eigens]), "pvec" + str(i),
                               self._name + "/pvec" + str(i))
        self._get_or_make_slot(var, tf.zeros([eigens]), "pval" + str(i),
                               self._name + "/pval" + str(i))
        self._get_or_make_slot(var, tf.zeros([]), "o" + str(i),
                               self._name + "/o" + str(i))

      self._zeros_slot(var, 'd', self._name + "/d")
      self._zeros_slot(var, 'm', self._name + "/m")
      self._zeros_slot(var, 'pm', self._name + "/pm")

  def _resource_apply_dense(self, grad, var):

    lr = self._learning_rate
    beta_1 = self._beta_1
    beta_2 = self._beta_2
    warmup = self._warmup
    tau_1 = tf.cast(self._tau_1, dtype='int32')
    tau_2 = tf.cast(self._tau_2, dtype='int32')
    eigen_head = self._eigen_head
    eigen_tail = self._eigen_tail
    damping_epsilon = self._damping_epsilon

    global_step = tf.cast(tf.compat.v1.train.get_global_step(), dtype='int32')
    if_update_stat = tf.equal(tf.math.mod(global_step, tau_2), 0)
    if_warmed_up = tf.greater(global_step, warmup)
    if_update_precond = tf.math.logical_and(
        if_warmed_up, tf.equal(tf.math.mod(global_step, tau_1), 0))

    global_step_f = tf.cast(global_step, dtype='float32')
    warmup_f = tf.cast(self._warmup, dtype='float32')
    warmup_rate = tf.minimum(tf.maximum(global_step_f / warmup_f - 1.0, 0.0),
                             1.0)

    if_stat_momentum = tf.less(
        beta_2,
        1.0 - 1e-10)  # if beta_2 = 1.0, do not use momentum on statistics

    ops = []
    rank = len(grad.shape)
    grad_precond = grad

    for i in range(rank):
      axes = list(range(i)) + list(range(i + 1, rank))

      g = self.get_slot(var, 'g' + str(i))
      g_t = tf.cond(
          if_update_stat, lambda: state_ops.assign(
              g, tf.tensordot(grad, grad, axes=[axes, axes])),
          lambda: tf.identity(g))

      s = self.get_slot(var, 's' + str(i))
      s_t = tf.cond(
          if_stat_momentum,
          lambda: state_ops.assign(s, beta_2 * s + (1 - beta_2) * g_t),
          lambda: state_ops.assign_add(s, g_t))

      pvec = self.get_slot(var, 'pvec' + str(i))
      pval = self.get_slot(var, 'pval' + str(i))
      offset = self.get_slot(var, 'o' + str(i))

      def update_precond():
        pvec_t, pval_t, offset_t = eigen_inverse_root(s_t, 2 * rank, eigen_head,
                                                      eigen_tail,
                                                      damping_epsilon)
        return (state_ops.assign(pvec, pvec_t), state_ops.assign(pval, pval_t),
                state_ops.assign(offset, offset_t))

      pvec_t, pval_t, offset_t = tf.cond(
          if_update_precond, lambda: update_precond(), lambda:
          (tf.identity(pvec), tf.identity(pval), tf.identity(offset)))

      grad_precond = apply_sparse_precond(grad_precond, pvec_t, pval_t,
                                          offset_t)
      ops += [
          g_t,
          s_t,
          pvec_t,
          pval_t,
          offset_t,
      ]

    d = self.get_slot(var, 'd')
    d_t = state_ops.assign_add(d, grad * grad)

    m = self.get_slot(var, 'm')
    m_t = state_ops.assign(
        m, beta_1 * m + (1 - beta_1) * grad * tf.math.rsqrt(d_t + 1e-30))

    pm = self.get_slot(var, 'pm')
    pm_t = state_ops.assign(pm, beta_1 * pm + (1.0 - beta_1) * grad_precond)

    update_diag = lr * m_t
    # AdaGrad gradient used in warmup steps

    update_second = lr * tf.norm(m_t) / (tf.norm(pm_t) + 1e-10) * pm_t
    # Shampoo gradient normalized by AdaGrad

    var_t = tf.cond(
        if_warmed_up, lambda: state_ops.assign_sub(var, (
            1.0 - warmup_rate) * update_diag + warmup_rate * update_second),
        lambda: state_ops.assign_sub(var, update_diag))

    ops += [d_t, m_t, pm_t, var_t]
    return tf.group(*ops)

  def _resource_apply_sparse(self, grad, var):
    raise tf.no_op()
