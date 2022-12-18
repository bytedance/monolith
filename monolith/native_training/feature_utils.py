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

from enum import Enum
import os
from typing import Iterable, Dict

from absl import logging

import tensorflow as tf
from tensorflow.python.training import training_util

from monolith.native_training import clip_ops
from monolith.native_training import feature
from monolith.native_training.native_task import NativeContext

enable_hvd = os.getenv("MONOLITH_WITH_HOROVOD")
enable_bps = int(os.getenv("MONOLITH_WITH_BYTEPS", '0'))
enable_bps_allreduce = int(os.getenv("MONOLITH_WITH_BYTEPS_ALLREDUCE", '1'))
enable_allreduce_fusion = str(
    os.getenv("MONOLITH_WITH_ALLREDUCE_FUSION", 'none'))
enable_allreduce_fp16 = int(os.getenv("MONOLITH_WITH_ALLREDUCE_FP16",
                                      '0'))  # for hvd
skip_allreduce = int(os.getenv("MONOLITH_SKIP_ALLREDUCE", '0'))
# enable (limited) fusion functionality for byteccl where bias tensors are fused into one
# tensor before performing allreduce.

if enable_hvd != None:
  import horovod.tensorflow as hvd
  from horovod.tensorflow.compression import FP16Compressor


def allreduce_cond(grads):
  if enable_bps and enable_bps_allreduce:
    import byteps.tensorflow as bps
    from byteps.tensorflow.compression import FP16Compressor as BPSFP16Compressor

  if enable_allreduce_fusion == 'one':
    logging.info('Enabled allreduce fusion (one op)')
    grads_wo_none = [grad for grad in grads if grad is not None]
    if len(grads_wo_none) == 0:
      return grads
    reshaped_grads = []
    grad_shapes = []
    grad_lens = []
    # reshape to 1D
    for idx in range(len(grads_wo_none)):
      grad = grads_wo_none[idx]
      grad_shapes.append(grad.shape)
      reshaped_grad = tf.reshape(grad, [-1], name='ar_reshape_' +
                                 str(idx)) if len(grad.shape) != 1 else grad
      grad_lens.append(int(reshaped_grad.shape[0]))
      reshaped_grads.append(reshaped_grad)
    # concat
    grads_fused = tf.concat(reshaped_grads, axis=0, name='concat_ar')
    # AR
    if enable_bps and enable_bps_allreduce:
      grads_fused_avg = bps.push_pull(grads_fused, average=True, compression=BPSFP16Compressor, name="bps_ar_fuse_one") \
                          if enable_allreduce_fp16 else bps.push_pull(grads_fused, average=True, name="bps_ar_fuse_one")
    else:
      grads_fused_avg = hvd.allreduce(grads_fused, op=hvd.Average, compression=FP16Compressor, name="hvd_ar_fuse_one") \
                          if enable_allreduce_fp16 else hvd.allreduce(grads_fused, op=hvd.Average, name="hvd_ar_fuse_one")
    # split to 1D
    grads_avg_split = tf.split(grads_fused_avg,
                               grad_lens,
                               axis=0,
                               name='split_fuse_one')
    # to original shape
    num_grads = len(grads)
    results = [None for _ in range(num_grads)]
    for idx in range(num_grads):
      if grads[idx] is not None:
        results[idx] = tf.reshape(grads_avg_split[idx], grad_shapes[idx], name='ar_reshape_back_'+str(idx)) \
                      if len(grad_shapes[idx]) !=1 else grads_avg_split[idx]

    return results
  elif enable_allreduce_fusion == 'multi':
    logging.info('Enabled allreduce fusion (based on shape)')
    grads_wo_none = [grad for grad in grads if grad is not None]
    if len(grads_wo_none) == 0:
      return grads

    grads_1d = []
    grads_1d_dim0 = []
    # cur model has the following 2-D grads:
    #   [x,1],[x,16],[x,64],[x,128],[x,256],[x,512],[x,1024],[x,2048]
    grads_2d_dict = {
        1: [],
        16: [],
        64: [],
        128: [],
        256: [],
        512: [],
        1024: [],
        2048: []
    }
    grads_2d_dim0 = {
        1: [],
        16: [],
        64: [],
        128: [],
        256: [],
        512: [],
        1024: [],
        2048: []
    }
    for g in grads_wo_none:
      if len(g.shape) == 1:
        grads_1d.append(g)
        grads_1d_dim0.append(int(g.shape[0]))
      else:
        grads_2d_dict[int(g.shape[1])].append(g)
        grads_2d_dim0[int(g.shape[1])].append(int(g.shape[0]))

    grads_1d_fused = tf.concat(grads_1d, 0, name="concat_1d")
    if enable_bps and enable_bps_allreduce:
      grads_1d_fused_avg = bps.push_pull(grads_1d_fused, average=True, compression=BPSFP16Compressor, name="bps_ar_fuse_1d") \
                          if enable_allreduce_fp16 else bps.push_pull(grads_1d_fused, average=True, name="bps_ar_fuse_1d")
    else:
      grads_1d_fused_avg = hvd.allreduce(grads_1d_fused, op=hvd.Average, compression=FP16Compressor, name="hvd_ar_fuse_1d") \
                            if enable_allreduce_fp16 else hvd.allreduce(grads_1d_fused, op=hvd.Average, name="hvd_ar_fuse_1d")
    grads_1d_split = tf.split(grads_1d_fused_avg,
                              grads_1d_dim0,
                              axis=0,
                              name='split_1d_grads')

    grads_2d_fused = {}
    grads_2d_fused_avg = {}
    grads_2d_split = {}
    for k in grads_2d_dict.keys():
      if len(grads_2d_dict[k]) == 0:
        continue
      grads_2d_fused[k] = tf.concat(grads_2d_dict[k],
                                    0,
                                    name="concat_2d_" + str(k))
      if enable_bps and enable_bps_allreduce:
        grads_2d_fused_avg[k] = bps.push_pull(grads_2d_fused[k], average=True, compression=BPSFP16Compressor, name="bps_ar_fuse_2d_"+str(k)) \
                            if enable_allreduce_fp16 else bps.push_pull(grads_2d_fused[k], average=True, name="bps_ar_fuse_2d_"+str(k))
      else:
        grads_2d_fused_avg[k] = hvd.allreduce(grads_2d_fused[k], op=hvd.Average, compression=FP16Compressor, name="hvd_ar_fuse_2d_"+str(k)) \
                            if enable_allreduce_fp16 else hvd.allreduce(grads_2d_fused[k], op=hvd.Average, name="hvd_ar_fuse_2d_"+str(k))
      grads_2d_split[k] = tf.split(grads_2d_fused_avg[k],
                                   grads_2d_dim0[k],
                                   axis=0,
                                   name='split_2d_grads_' + str(k))

    num_grads = len(grads)
    results = [None for _ in range(num_grads)]
    for idx in range(num_grads - 1, -1, -1):
      if grads[idx] is None:
        continue
      if len(grads[idx].shape) == 1:
        results[idx] = grads_1d_split.pop(-1)
      else:
        dim1 = int(grads[idx].shape[1])
        results[idx] = grads_2d_split[dim1].pop(-1)

    return results
  else:
    # without fusion
    logging.info('Enabled allreduce without fusion')
    if enable_bps and enable_bps_allreduce:
      if enable_allreduce_fp16:
        return [
            bps.push_pull(grad, average=True, compression=BPSFP16Compressor)
            if grad is not None else grad for grad in grads
        ]
      else:
        return [
            bps.push_pull(grad, average=True) if grad is not None else grad
            for grad in grads
        ]
    else:
      if enable_allreduce_fp16:
        return [
            hvd.allreduce(grad, op=hvd.Average, compression=FP16Compressor)
            if grad is not None else grad for grad in grads
        ]
      else:
        return [
            hvd.allreduce(grad, op=hvd.Average) if grad is not None else grad
            for grad in grads
        ]


class GradClipType(Enum):
  ClipByNorm = 1
  ClipByGlobalNorm = 2
  ClipByValue = 3
  ClipByDenseAndSparse = 4
  NoClip = 5


def apply_gradients_with_var_optimizer(
    ctx: NativeContext,
    feature_columns: Iterable[feature.FeatureColumnV1],
    var_opt: tf.compat.v1.train.Optimizer,
    loss: tf.Tensor,
    clip_type: GradClipType = GradClipType.ClipByGlobalNorm,
    clip_norm: float = None,
    global_step=None,
    grads_and_vars_summary: bool = False,
    use_allreduce: bool = False,
    ue_gradient_check: bool = False,
    ue_fc_names: list = [],
    ue_euclidean_norm_threshold: float = 0.0,
    dense_weight_decay: float = 0.0,
    features: Dict[str, tf.Tensor] = {},
    sparse_clip_norm: float = None,
    dense_reduce_mean: bool = False,
    batch_size: int = 1) -> tf.Operation:
  """
  A helper function that applies gradient to both dense params and embedding params.
  Args:
    clip_type - clip type
    clip_norm - norm will be used by clip
    global_step - is not None, will be added by 1.
    grads_and_vars_summary - when True, will print summary of grads and vars
    dense_weight_decay - dense weight decay, l2 norm
  """
  assert isinstance(var_opt, tf.compat.v1.train.Optimizer)
  feature_columns = list(feature_columns)
  all_embeddings = [fc.get_all_embeddings_concat() for fc in feature_columns]
  variables = tf.compat.v1.trainable_variables()
  grads_and_vars = var_opt.compute_gradients(loss,
                                             variables + all_embeddings,
                                             colocate_gradients_with_ops=True)

  # Some variables are created but unused and we need to filter them out.
  unused_filter = [
      True if gv[0] is not None else False for gv in grads_and_vars
  ]
  unused_dense_filter = unused_filter[:len(variables)]
  unused_emb_filter = unused_filter[len(variables):]
  variables = [v for v, used in zip(variables, unused_dense_filter) if used]
  all_embeddings = [
      e for e, used in zip(all_embeddings, unused_emb_filter) if used
  ]
  feature_columns = [
      fc for fc, used in zip(feature_columns, unused_emb_filter) if used
  ]
  grads_and_vars = [
      gv for gv, used in zip(grads_and_vars, unused_filter) if used
  ]

  grads = [grad_and_var[0] for grad_and_var in grads_and_vars]
  # TODO(zouxuan): this is a quick workaround to fix the empty grads issue.
  if len(grads) == 0:
    return tf.no_op()

  # UE conditional gradient check
  if ue_gradient_check:
    grads = []
    for grad_and_var in grads_and_vars:
      found = False
      for fc_name in ue_fc_names:
        if fc_name in grad_and_var[1].name or 'uue' in grad_and_var[1].name:
          grads.append(
              tf.where(
                  tf.norm(features[fc_name]) > ue_euclidean_norm_threshold,
                  grad_and_var[0], tf.zeros_like(grad_and_var[0])))
          logging.info("UE Vars: {}".format(grad_and_var[1].name))
          found = True
          break
      if not found:
        grads.append(grad_and_var[0])

  dense_grads = grads[:len(variables)]
  emb_grads = grads[len(variables):]

  if dense_reduce_mean:
    dense_grads = [g / batch_size for g in dense_grads]

  if grads is not None and len(grads) > 0:
    if clip_type == GradClipType.ClipByGlobalNorm and clip_norm is not None:
      clipped_grads, global_g_norm = clip_ops.clip_by_global_norm(
          grads, clip_norm, use_norm=tf.linalg.global_norm(grads))
      logging.info('clip_by_global_norm: %s', clip_norm)
      with tf.device('/device:CPU:0'):
        tf.compat.v1.summary.scalar("global_gradient_norm", global_g_norm)
    elif clip_type == GradClipType.ClipByValue and clip_norm is not None:
      clipped_grads = [
          tf.clip_by_value(g,
                           clip_value_min=-clip_norm,
                           clip_value_max=clip_norm) for g in grads
      ]
    elif clip_type == GradClipType.ClipByNorm and clip_norm is not None:
      clipped_grads = [tf.clip_by_norm(g, clip_norm) for g in grads]
    elif clip_type == GradClipType.ClipByDenseAndSparse:
      logging.info("Grads are: {}".format(grads))
      global_emb_norm = tf.linalg.global_norm(emb_grads)
      global_dense_norm = tf.linalg.global_norm(dense_grads)
      if len(dense_grads) > 0:
        dense_clipped_grads, _ = clip_ops.clip_by_global_norm(
            dense_grads, clip_norm, use_norm=global_dense_norm)
      else:
        dense_clipped_grads = dense_grads

      if len(emb_grads) > 0 and sparse_clip_norm != None:
        embedding_clipped_grads, _ = clip_ops.clip_by_global_norm(
            emb_grads, sparse_clip_norm, use_norm=global_emb_norm)
      else:
        embedding_clipped_grads = emb_grads
      clipped_grads = dense_clipped_grads + embedding_clipped_grads
      with tf.device('/device:CPU:0'):
        tf.compat.v1.summary.scalar("global_gradient_norm/dense",
                                    global_dense_norm)
        tf.compat.v1.summary.scalar("global_gradient_norm/sparse",
                                    global_emb_norm)
    else:
      clipped_grads = grads
  else:
    clipped_grads = grads

  dense_clipped_grads = clipped_grads[:len(variables)]
  embedding_clipped_grads = clipped_grads[len(variables):]

  if grads_and_vars_summary:
    with tf.device("/device:CPU:0"):
      if len(dense_clipped_grads) > 0:
        tf.compat.v1.summary.histogram(
            "variable_gradient",
            tf.concat([tf.reshape(grad, [-1]) for grad in dense_clipped_grads],
                      0))
        dense_grad_sizes = []
        for grad, var in zip(dense_clipped_grads, variables):
          summary_var_name = var.name.replace(":", "_")
          tf.compat.v1.summary.scalar(
              "trainable_variable_norm/{}".format(summary_var_name),
              tf.norm(var))
          tf.compat.v1.summary.histogram(
              "trainable_variable/{}".format(summary_var_name), var)
          tf.compat.v1.summary.scalar(
              "gradient_norm/{}".format(summary_var_name), tf.norm(grad))
          tf.compat.v1.summary.histogram("gradient/{}".format(summary_var_name),
                                         grad)
          dense_grad_sizes.append(tf.size(grad))
        tf.compat.v1.summary.histogram("dense_grad_sizes", dense_grad_sizes)
        tf.compat.v1.summary.scalar("dense_grad_total_size",
                                    tf.reduce_sum(dense_grad_sizes))
        tf.compat.v1.summary.scalar("dense_grad_total_num",
                                    len(dense_grad_sizes))

      for i, fc in enumerate(feature_columns):
        tf.compat.v1.summary.histogram("{}_gradient".format(fc.feature_name),
                                       embedding_clipped_grads[i])
  if skip_allreduce:
    use_allreduce = False

  dense_clipped_grads = allreduce_cond(
      dense_clipped_grads
  ) if use_allreduce and enable_hvd else dense_clipped_grads

  if dense_weight_decay and variables:
    dense_clipped_grads = [
        g + dense_weight_decay * v
        for g, v in zip(dense_clipped_grads, variables)
    ]
  train_ops = []
  grads_and_vars_without_optimizer = []
  if variables:
    for i, var in enumerate(variables):
      if hasattr(var, 'optimizer') and var.optimizer:
        train_ops.append(
            ctx.add_async_function(var.optimizer.apply_gradients,
                                   ([(dense_clipped_grads[i], var)],)))
        logging.info("var {} uses a custom optimizer: {}".format(
            var.name, var.optimizer))
      else:
        grads_and_vars_without_optimizer.append((dense_clipped_grads[i], var))
    train_ops.append(
        ctx.add_async_function(var_opt.apply_gradients,
                               (grads_and_vars_without_optimizer,)))

  if global_step is not None:
    # The control dependency here ensures that
    # when the StepCounterHook tries to get the global_step
    # from the training session at the same time of training,
    # the read_value should be consistent (before assign_add).
    with tf.control_dependencies(
        [training_util._get_or_create_global_step_read()]):
      train_ops.append(
          ctx.add_async_function(tf.compat.v1.assign_add, (global_step, 1)))

  with tf.device('/device:CPU:0'):
    train_ops.append(
        ctx.apply_embedding_gradients(
            (list(zip(embedding_clipped_grads, all_embeddings)))))
    return tf.group(*train_ops)


def apply_gradients(ctx: NativeContext,
                    var_opt: tf.compat.v1.train.Optimizer,
                    loss: tf.Tensor,
                    clip_type: GradClipType = GradClipType.ClipByGlobalNorm,
                    clip_norm: float = None,
                    dense_weight_decay: float = 0.0,
                    global_step=None):
  layout_factory: feature.EmbeddingLayoutFactory = ctx.layout_factory
  variables = tf.compat.v1.trainable_variables()
  layout_embeddings = layout_factory.flattened_layout()
  grads_and_vars = var_opt.compute_gradients(loss,
                                             variables + layout_embeddings,
                                             colocate_gradients_with_ops=True)
  # clip grads
  flag = False
  for g, v in grads_and_vars:
    if g is None:
      flag = True
      logging.info(f'grad of {v} is None, maybe it not used in the graph')
  if flag:
    grads_and_vars = [(g, v) for (g, v) in grads_and_vars if g is not None]
    variables = [v for (g, v) in grads_and_vars if v in variables]
    layout_embeddings = [
        v for (g, v) in grads_and_vars if v in layout_embeddings
    ]
    assert len(grads_and_vars) == len(variables) + len(layout_embeddings)

  grads = [g for (g, _) in grads_and_vars]
  if grads and clip_norm is not None and clip_norm > 0:
    if clip_type == GradClipType.ClipByGlobalNorm:
      clipped_grads, global_g_norm = clip_ops.clip_by_global_norm(
          grads, clip_norm, use_norm=tf.linalg.global_norm(grads))
      logging.info('clip_by_global_norm: %s', clip_norm)
      with tf.device('/device:CPU:0'):
        tf.compat.v1.summary.scalar("global_gradient_norm", global_g_norm)
    elif clip_type == GradClipType.ClipByNorm:
      clipped_grads = [tf.clip_by_norm(g, clip_norm) for g in grads]
    else:
      raise Exception(f"{clip_type} is not supported yet!")
  else:
    clipped_grads = grads

  train_ops = []

  # dense apply_gradients
  if variables:
    dense_clipped_grads = clipped_grads[:len(variables)]
    if dense_weight_decay > 0:
      grads_and_vars = [(g + dense_weight_decay * v, v)
                        for g, v in zip(dense_clipped_grads, variables)]
    else:
      grads_and_vars = list(zip(dense_clipped_grads, variables))
    train_ops.append(
        var_opt.apply_gradients(grads_and_vars, global_step=global_step))
  else:
    with tf.control_dependencies(
        [training_util._get_or_create_global_step_read()]):
      train_ops.append(tf.compat.v1.assign_add(global_step, 1))

  # sparse apply_gradients
  if layout_embeddings:
    sparse_clipped_grads = clipped_grads[len(variables):]
    grads_and_vars = list(zip(sparse_clipped_grads, layout_embeddings))
    train_ops.append(ctx.apply_embedding_gradients(grads_and_vars).as_op())

  return tf.group(*train_ops)
