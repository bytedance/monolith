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
from monolith.native_training.distribution_ops import gen_distribution_ops
from monolith.native_training import device_utils
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
  from horovod.tensorflow.compression import FP16Compressor, NoneCompressor

control_ops = []
dense_opt_ops = []

def allreduce_cond(grads, scale = 1):
  if enable_bps and enable_bps_allreduce:
    import byteps.tensorflow as bps
    from byteps.tensorflow.compression import FP16Compressor as BPSFP16Compressor, NoneCompressor as BPSNoneCompressor
    compression = BPSFP16Compressor if enable_allreduce_fp16 else BPSNoneCompressor
  else:
    compression = FP16Compressor if enable_allreduce_fp16 else NoneCompressor

  grads_wo_none = [grad for grad in grads if grad is not None]
  num_grads = len(grads)
  results = [None for _ in range(num_grads)]
  if len(grads_wo_none) == 0:
    return grads

  def map_to_output(reduced):
    r_idx = 0
    for i in range(num_grads):
      if grads[i] is not None:
        results[i] = reduced[r_idx]
        r_idx += 1
    assert r_idx == len(reduced), "Something is wrong"
    return results

  global control_ops
  if enable_allreduce_fusion == 'one':
    # note: one allreduce fusion does not yet support CPU
    # note: concat -> allreduce -> split is noticeably faster than hvd.grouped_allreduce
    grads_fused = gen_distribution_ops.monolith_aligned_flat_concat(grads_wo_none, scale)
    control_ops = [grads_fused]
    if enable_bps and enable_bps_allreduce:
      grads_fused_avg = bps.push_pull(grads_fused, average=True, compression=compression, name="bps_ar_fuse_one")
    else:
      grads_fused_avg = hvd.allreduce(grads_fused, op=hvd.Average, compression=compression, name="hvd_ar_fuse_one")
    return map_to_output(gen_distribution_ops.monolith_aligned_flat_split(grads_wo_none, grads_fused_avg))
  elif enable_allreduce_fusion == "grouped":
    assert not enable_bps or not enable_bps_allreduce
    return map_to_output(
      hvd.grouped_allreduce([grad * scale for grad in grads_wo_none], op=hvd.Average, compression=compression))
  elif enable_allreduce_fusion == 'multi':
    raise RuntimeError("Support for multi is dropped. Please use 'one' as the fusion strategy")
  else:
    logging.info('Enabled allreduce without fusion using Average Op!')
    if enable_bps and enable_bps_allreduce:
      return [
        bps.push_pull(grad * scale, average=True, compression=compression)
        if grad is not None else grad for grad in grads
      ]
    else:
      return [
        hvd.allreduce(grad * scale, op=hvd.Average, compression=compression) 
        if grad is not None else grad for grad in grads
      ]


class GradClipType(Enum):
  ClipByNorm = 1
  ClipByGlobalNorm = 2
  ClipByValue = 3
  ClipByDenseAndSparse = 4
  NoClip = 5


def _gen_norm_warmup(clip_norm: float, global_step_var: tf.Tensor,
                     warmup_step: int):
  if not warmup_step:
    return clip_norm
  return tf.cond(
      tf.less(global_step_var, warmup_step), lambda: tf.compat.v1.div(
          tf.cast(global_step_var, dtype=tf.float32), float(warmup_step)),
      lambda: 1.0) * clip_norm


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
    sparse_norm_warmup_steps: int = None,
    dense_reduce_mean: bool = False,
    batch_size: int = 1,
    is_fused_layout: bool = False) -> tf.Operation:
  """
  A helper function that applies gradient to both dense params and embedding params.
  Args:
    clip_type - clip type
    clip_norm - norm will be used by clip
    global_step - is not None, will be added by 1.
    grads_and_vars_summary - when True, will print summary of grads and vars
    dense_weight_decay - dense weight decay, l2 norm
  """
  with device_utils.maybe_device_if_allowed('/device:GPU:0'):
    assert isinstance(var_opt, tf.compat.v1.train.Optimizer)
    feature_columns = list(feature_columns)
    if is_fused_layout:
      layout_factory: feature.EmbeddingLayoutFactory = ctx.layout_factory
      all_embeddings = layout_factory.flattened_layout()
    else:
      all_embeddings = [fc.get_all_embeddings_concat() for fc in feature_columns]
    variables = tf.compat.v1.trainable_variables()
    grads_and_vars = var_opt.compute_gradients(loss,
                                               variables + all_embeddings,
                                               colocate_gradients_with_ops=True)

    # Some variables are created but unused and we need to filter them out.
    if is_fused_layout:
      grads_and_vars_tmp = grads_and_vars[:len(variables)]
      for gv in grads_and_vars[len(variables):]:
        grads_and_vars_tmp.append((gv[0] if gv[0] is not None else tf.zeros_like(gv[1]), gv[1]))
      grads_and_vars = grads_and_vars_tmp

    dense_gvs = [gv for gv in grads_and_vars[:len(variables)] if gv[0] is not None]
    sparse_gvs = [gv for gv in grads_and_vars[len(variables):] if gv[0] is not None]
    if is_fused_layout:
      feature_columns = []
    else:
      feature_columns = [
          fc for fc, gv in zip(feature_columns, grads_and_vars[len(variables):]) if gv[0] is not None
      ]
    
    variables = [gv[1] for gv in dense_gvs]
    all_embeddings = [gv[1] for gv in sparse_gvs]
    grads_and_vars = dense_gvs + sparse_gvs
    grads = [grad_and_var[0] for grad_and_var in grads_and_vars]
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

    # TODO(zouxuan): this is a quick workaround to fix the empty grads issue.
    if len(grads) == 0:
      return tf.no_op()

    dense_grads = grads[:len(variables)]
    sparse_grads = grads[len(variables):]

    if dense_reduce_mean:
      dense_grads = [g / batch_size for g in dense_grads]

    global_dense_norm = None
    global_sparse_norm = None
    norm_fn = clip_ops._global_norm if device_utils.within_placement_context_of(
      "GPU") else tf.linalg.global_norm
    if clip_type == GradClipType.ClipByGlobalNorm and clip_norm is not None:
      global_dense_norm = norm_fn(grads)
      global_sparse_norm = global_dense_norm # use the same norm for sparse and dense
      sparse_clip_norm = sparse_clip_norm or clip_norm
      if sparse_norm_warmup_steps is not None:
        sparse_clip_norm = _gen_norm_warmup(sparse_clip_norm, global_step,
                                            sparse_norm_warmup_steps)
        logging.info('sparse_norm_warmup_steps: %s', sparse_norm_warmup_steps)
      with tf.device('/device:CPU:0'):
        tf.compat.v1.summary.scalar("global_gradient_norm", global_dense_norm)
    elif clip_type == GradClipType.ClipByValue and clip_norm is not None:
      clipped_grads = [
          tf.clip_by_value(g,
                            clip_value_min=-clip_norm,
                            clip_value_max=clip_norm) for g in grads
      ]
    elif clip_type == GradClipType.ClipByNorm and clip_norm is not None:
      clipped_grads = [tf.clip_by_norm(g, clip_norm) for g in grads]
    elif clip_type == GradClipType.ClipByDenseAndSparse:
      global_dense_norm = norm_fn(dense_grads)
      if sparse_clip_norm is not None:
        global_sparse_norm = norm_fn(sparse_grads)
      with tf.device('/device:CPU:0'):
        tf.compat.v1.summary.scalar("global_gradient_norm/dense",
                                    global_dense_norm)
        if global_sparse_norm is not None:
          tf.compat.v1.summary.scalar("global_gradient_norm/sparse",
                                      global_sparse_norm)
    else:
      clipped_grads = grads
    
    if skip_allreduce:
      use_allreduce = False      

    # Conditionally perform clip by global norm.
    # If we're using synchronous (allreduce=True) distributed GPU training,
    # we defer clip and only calculate a scale factor. The scaling is fused 
    # with later concat/gather kernels for better performance
    def cond_defer_clip(norm, clip_norm, grads):
      defer_clip = device_utils.within_placement_context_of("GPU") and \
        use_allreduce and not grads_and_vars_summary and not is_fused_layout
      scale = 1
      if norm is not None:
        if not defer_clip:
          grads, _ = clip_ops.clip_by_global_norm(grads, clip_norm, use_norm=norm)
        else:
          scale = tf.minimum(clip_norm / norm, 1)
      return grads, scale

    if clip_type in (GradClipType.ClipByGlobalNorm, GradClipType.ClipByDenseAndSparse):
      dense_clipped_grads, dense_scale = cond_defer_clip(global_dense_norm, clip_norm, dense_grads)
      sparse_clipped_grads, sparse_scale = cond_defer_clip(global_sparse_norm, sparse_clip_norm, sparse_grads)
    else:
      dense_scale = 1
      sparse_scale = 1
      dense_clipped_grads = clipped_grads[:len(variables)]
      sparse_clipped_grads = clipped_grads[len(variables):]

    if grads_and_vars_summary:
      with tf.device("/device:CPU:0"):
        if len(dense_clipped_grads) > 0:
          tf.compat.v1.summary.histogram(
              "variable_gradient",
              tf.concat(
                  [tf.reshape(grad, [-1]) for grad in dense_clipped_grads], 0))
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
            tf.compat.v1.summary.histogram(
                "gradient/{}".format(summary_var_name), grad)
            dense_grad_sizes.append(tf.size(grad))
          tf.compat.v1.summary.histogram("dense_grad_sizes", dense_grad_sizes)
          tf.compat.v1.summary.scalar("dense_grad_total_size",
                                      tf.reduce_sum(dense_grad_sizes))
          tf.compat.v1.summary.scalar("dense_grad_total_num",
                                      len(dense_grad_sizes))

        for i, fc in enumerate(feature_columns):
          tf.compat.v1.summary.histogram("{}_gradient".format(fc.feature_name),
                                         sparse_clipped_grads[i])

    logging.info('use_allreduce: %s', use_allreduce)
    dense_clipped_grads = allreduce_cond(
        dense_clipped_grads, dense_scale
    ) if use_allreduce and enable_hvd else dense_clipped_grads

    if dense_weight_decay and variables:
      dense_clipped_grads = [
          g + dense_weight_decay * v
          for g, v in zip(dense_clipped_grads, variables)
      ]
    logging.info('dense_weight_decay: %s', dense_weight_decay)
    train_ops = []
    grads_and_vars_without_optimizer = []
    if variables:
      global dense_opt_ops
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
      dense_opt_ops = train_ops.copy()

    with tf.device('/device:CPU:0'):
      train_ops.append(
          ctx.apply_embedding_gradients(
              list(zip(sparse_clipped_grads, all_embeddings)), sparse_scale))

    if global_step is not None:
      # The control dependency here ensures that
      # when the StepCounterHook tries to get the global_step
      # from the training session at the same time of training,
      # the read_value should be consistent (before assign_add).
      # Also makes sure that the global step is incremented after the optimize ops, 
      # since embedding optimizer requires this global step as input
      with tf.control_dependencies(
          train_ops + [training_util._get_or_create_global_step_read()]):
        train_ops.append(
            ctx.add_async_function(tf.compat.v1.assign_add, (global_step, 1)))
    return tf.group(*train_ops)


def apply_gradients(ctx: NativeContext,
                    var_opt: tf.compat.v1.train.Optimizer,
                    loss: tf.Tensor,
                    clip_type: GradClipType = GradClipType.ClipByGlobalNorm,
                    clip_norm: float = None,
                    dense_weight_decay: float = 0.0,
                    global_step=None,
                    use_allreduce: bool = False):
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
    if use_allreduce and enable_hvd:
      dense_clipped_grads = allreduce_cond(
        dense_clipped_grads)

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
    train_ops.append(ctx.apply_embedding_gradients(grads_and_vars))

  return tf.group(*train_ops)
