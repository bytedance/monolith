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

import sys

from absl import logging
import tensorflow.compat.v1 as tf
import tensorflow as tf2

from monolith.core.base_host_call import BaseHostCall
from monolith.core.tpu_variable import ReplicatedVariable

_LABLES_FOR_AUC_CALCULATION = "labels_for_auc_calculation"
_Y_PRED_FOR_AUC_CALCULATION = "y_pred_for_auc_calculation"
_REQ_TIME = "req_time"
_SAMPLE_RATE = "sample_rate"
_UID = "uid"
_UID_BUCKET = 'uid_bucket'

_DEEPINSIGHT_SAMPLE_RATES = "di_example_sample_rates"
_DEEPINSIGHT_LABELS = "di_labels"
_DEEPINSIGHT_PREDS = "di_preds"
_DEEPINSIGHT_REQ_TIMES = "di_req_times"

_RATIO_N = 1000
_UID_SAMPLE_RATE = 0.01

_HOST_CALL_AUC_METRICS = set([
    _LABLES_FOR_AUC_CALCULATION, _Y_PRED_FOR_AUC_CALCULATION, _SAMPLE_RATE,
    _REQ_TIME, _UID_BUCKET
])

# TPU variables names
_LABELS_TPU_VARIABLE = "labels_tpu_variable"
_PREDS_TPU_VARIABLE = "preds_tpu_variable"
_UID_BUCKETS_TPU_VARIABLE = "uid_buckets_tpu_variable"
_REQ_TIMES_TPU_VARIABLE = "req_times_tpu_variable"
_SAMPLE_RATES_TPU_VARIABLE = "sample_rates_tpu_variable"
_ACCUMULATED_COUNTER_TPU_VARIABLE = "tpu_variables_accumulated_times"

_DEPRECATED_METRIC_NAMES = [
    _LABLES_FOR_AUC_CALCULATION, _Y_PRED_FOR_AUC_CALCULATION, _UID_BUCKET,
    _REQ_TIME, _SAMPLE_RATE
]


class TPUVariableRestoreHook(tf.estimator.SessionRunHook):
  """Initialize variables on TPU devices."""

  def __init__(self, op):
    self._op = op

  def after_create_session(self, session, coord):
    logging.info("Initialize variables on TPU devices.")
    session.run(self._op)


class BaseEmbeddingHostCall(BaseHostCall):

  def __init__(self, output_dir, enable_host_call, enable_deepinsight,
               enable_host_call_scalar_metrics,
               enable_caching_with_tpu_var_mode, top_k_sampling_num_per_core,
               params):
    super(BaseEmbeddingHostCall, self).__init__(output_dir, enable_host_call)
    self._enable_host_call = params["enable_host_call"]
    self._enable_deepinsight = enable_deepinsight
    self._enable_host_call_scalar_metrics = enable_host_call_scalar_metrics
    self._enable_caching_with_tpu_var_mode = enable_caching_with_tpu_var_mode
    self._top_k_sampling_num_per_core = top_k_sampling_num_per_core
    if params["cpu_test"] is True:
      self._context = None
    else:
      self._context = params["context"]
    self._host_call_every_n_steps = params["host_call_every_n_steps"]

    # Each TPU core uses these tpu variables to record metrics.
    #   labels tpu variable, shape is (topk_num * host_call_steps, )
    #   preds tpu variable, shape is (topk_num * host_call_steps, )
    #   uid_buckets tpu variable, shape is (topk_num * host_call_steps, )
    #   req_times tpu variable, shape is (host_call_steps, )
    #   sample_rates tpu variable, shape is (host_call_steps, )
    self._labels_tpu_variable = None
    self._preds_tpu_variable = None
    self._uid_buckets_tpu_variable = None
    self._req_times_tpu_variable = None
    self._sample_rates_tpu_variable = None
    # Counter of accumulating times for next host call.
    self._accumulated_counter_tpu_variable = None

    self.tpu_var_restore_hooks = []

    # Create TPU variables.
    self._create_all_tpu_variables()

  # Use TPU variables to reach each step's metrics and process all metrics
  # accumulated in each host call for deepinsight usage.
  def _create_all_tpu_variables(self):
    if self._enable_host_call is False:
      logging.info("enable_host_call is False, do not create tpu variables.")
      return

    if self._enable_caching_with_tpu_var_mode is False:
      logging.info(
          "enable_caching_with_tpu_var_mode is False, do not create tpu variables."
      )
      return

    assert self._host_call_every_n_steps > 1, "If tpu variables caching is enabled, we need host_call_every_n_steps bigger than 1."

    logging.info("Create all tpu variables.")

    # Create TPU variables for metrics.
    max_accumulated_samples_per_host_call = self._top_k_sampling_num_per_core * self._host_call_every_n_steps
    self._labels_tpu_variable = self._create_tpu_var(
        _LABELS_TPU_VARIABLE, [max_accumulated_samples_per_host_call],
        tf.float32)
    self._preds_tpu_variable = self._create_tpu_var(
        _PREDS_TPU_VARIABLE, [max_accumulated_samples_per_host_call],
        tf.float32)
    self._uid_buckets_tpu_variable = self._create_tpu_var(
        _UID_BUCKETS_TPU_VARIABLE, [max_accumulated_samples_per_host_call],
        tf.int32)

    self._req_times_tpu_variable = self._create_tpu_var(
        _REQ_TIMES_TPU_VARIABLE, [self._host_call_every_n_steps], tf.int64)
    self._sample_rates_tpu_variable = self._create_tpu_var(
        _SAMPLE_RATES_TPU_VARIABLE, [self._host_call_every_n_steps], tf.float32)

    # Create meta to maintain TPU varaibles.
    self._accumulated_counter_tpu_variable = self._create_tpu_var(
        _ACCUMULATED_COUNTER_TPU_VARIABLE, [], tf.int32)

  def _create_tpu_var(self, var_name, var_shape, var_type):
    ctx = self._context
    master = ctx._internal_ctx.master_job
    job_device = '' if master is None else ('/job:%s' % master)
    slices = []
    tpu_host_placement_fn = ctx.tpu_host_placement_function
    with tf.control_dependencies(None):
      assign_ops = []
      for h in range(ctx.num_hosts):
        with tf.device(tpu_host_placement_fn(h)):
          zero_tensor = tf.zeros(shape=var_shape, dtype=var_type)
        for d in range(ctx.num_of_replicas_per_host):
          with tf.device('%s/task:%d/device:TPU:%d' % (job_device, h, d)):
            slice_var = tf.Variable(initial_value=zero_tensor,
                                    trainable=False,
                                    name="slice_{}_{}_{}".format(
                                        var_name, h, d),
                                    dtype=var_type,
                                    expected_shape=var_shape,
                                    collections=["TPU_VAR"])
            slices.append(slice_var)
          assign_ops.append(tf.assign(slice_var, zero_tensor))

      tpu_var = ReplicatedVariable(var_name, slices)
      group_assign_op = tf.group(assign_ops)
      tpu_var_restore_hook = TPUVariableRestoreHook(group_assign_op)
      self.tpu_var_restore_hooks.append(tpu_var_restore_hook)
      logging.info("Created TPU variable, name: {}, shape: {}, type: {}".format(
          var_name, var_shape, var_type))
      return tpu_var

  def _compute_new_value(self, base_tpu_var, delta_value, update_offset):
    # Need assert shape.rank is 1 for both base_tpu_var and delta_value.
    assert base_tpu_var.get_shape().rank == 1, \
      "base_tpu_var's rank must be 1, base_tpu_var shape: {}".format(base_tpu_var.get_shape())
    assert delta_value.get_shape().rank == 1, \
      "delta_value's rank must be 1, delta_value shape: {}".format(delta_value.get_shape())
    assert base_tpu_var.dtype == delta_value.dtype, "base_tpu_var dtype: {} must be same as delta_value dtype: {}" \
      .format(base_tpu_var.dtype, delta_value.dtype)

    # Padding in the end of delta_value so that it has same shape with base_tpu_var.
    # And then right shift delta_value so that its valid data starts from update_offset.
    # Note: Here we can't pad directly to delta_value in the begining and end. Because the pad position
    # depends on update_offset which is a tensor. And that will make the pad op encounter a compilation error
    # as following:
    # Compilation failure: Input 1 to node `Pad` with op Pad must be a compile-time constant.
    # XLA compilation requires that operator arguments that represent shapes or dimensions be evaluated to
    # concrete values at compile time. This error means that a shape or dimension argument could not be
    # evaluated at compile time, usually because the value of the argument depends on a parameter to the
    #  computation, on a variable, or on a stateful operation such as a random number generator.
    base_len = base_tpu_var.shape.as_list()[0]
    delta_len = delta_value.shape.as_list()[0]
    paddings = [[0, base_len - delta_len]]
    delta_value = tf.pad(delta_value, paddings, 'CONSTANT', constant_values=0)
    # Right shift delta_tpu_var according to update_offset in base_tpu_var.
    delta_value = tf.roll(delta_value, shift=update_offset, axis=0)

    return base_tpu_var + delta_value

  def _clear_value_at_index_1(self, tpu_var, var_type, index):
    return tf.where(tf.math.equal(index, 1),
                    tf.zeros_like(tpu_var, dtype=var_type), tpu_var)

  def update_tpu_variables_ops(self, global_step, labels, preds, uid_buckets,
                               req_times, sample_rates):
    if self._enable_host_call is False:
      logging.info("enable_host_call is False, do not update tpu variables.")
      return []

    if self._enable_caching_with_tpu_var_mode is False:
      logging.info(
          "enable_caching_with_tpu_var_mode is False, do not update tpu variables."
      )
      return []

    logging.info("Update tpu variables.")

    assert labels is not None
    assert preds is not None
    assert uid_buckets is not None

    expected_shape = [self._top_k_sampling_num_per_core]
    assert labels.get_shape() == expected_shape, \
       "Expect shape: {}, but shape is {}".format(expected_shape, labels.get_shape())
    assert preds.get_shape() == expected_shape, \
       "Expect shape: {}, but shape is {}".format(expected_shape, preds.get_shape())
    assert uid_buckets.get_shape() == expected_shape, \
       "Expect shape: {}, but shape is {}".format(expected_shape, uid_bkcets.get_shape())

    # We do host call host_call_every_n_steps steps. At step 0, host_call_every_n_steps,
    # 2 * host_call_every_n_steps, ..., we need a completed data until this step, and we do a host call
    # to dump those data. At step 1, host_call_every_n_steps + 1, 2 * host_call_every_n_steps + 1,
    # We will need clear any accumulated data firstly and then start accumulating new data again.
    # We use index = tf.math.floormod(global_step, host_call_every_n_steps) to represent the step index where we are.
    # If index is 1, we will clear everything with TPU variables before accumulating new data.
    index = tf.math.floormod(global_step, self._host_call_every_n_steps)
    old_accumulated_counter_value = self._clear_value_at_index_1(
        self._accumulated_counter_tpu_variable, tf.int32, index)
    old_labels_value = self._clear_value_at_index_1(self._labels_tpu_variable,
                                                    tf.float32, index)
    old_preds_value = self._clear_value_at_index_1(self._preds_tpu_variable,
                                                   tf.float32, index)
    old_uid_buckets_value = self._clear_value_at_index_1(
        self._uid_buckets_tpu_variable, tf.int32, index)
    old_req_times_tpu_variable = self._clear_value_at_index_1(
        self._req_times_tpu_variable, tf.int64, index)
    old_sample_rates_value = self._clear_value_at_index_1(
        self._sample_rates_tpu_variable, tf.float32, index)

    # Update labels, preds, and uid_buckets which have self._top_k_sampling_num_per_core elements in the last dimension.
    tpu_var_offset = old_accumulated_counter_value * self._top_k_sampling_num_per_core

    new_labels_value = self._compute_new_value(old_labels_value, labels,
                                               tpu_var_offset)
    new_preds_value = self._compute_new_value(old_preds_value, preds,
                                              tpu_var_offset)
    new_uid_buckets_tpu_value = self._compute_new_value(old_uid_buckets_value,
                                                        uid_buckets,
                                                        tpu_var_offset)
    new_req_times_value = self._compute_new_value(
        old_req_times_tpu_variable, req_times, old_accumulated_counter_value)
    new_sample_rates_value = self._compute_new_value(
        old_sample_rates_value, sample_rates, old_accumulated_counter_value)

    # Increment tpu variable counter.
    new_accumulated_counter_value = tf.math.add(old_accumulated_counter_value,
                                                1)

    # Update tpu variables.
    update_tpu_var_ops = [
        tf.assign(self._labels_tpu_variable, new_labels_value),
        tf.assign(self._preds_tpu_variable, new_preds_value),
        tf.assign(self._uid_buckets_tpu_variable, new_uid_buckets_tpu_value),
        tf.assign(self._req_times_tpu_variable, new_req_times_value),
        tf.assign(self._sample_rates_tpu_variable, new_sample_rates_value),
    ]

    # Update tpu variable counter should only happen after updating tpu variables.
    with tf.control_dependencies(update_tpu_var_ops):
      update_tpu_var_counter_op = tf.assign(
          self._accumulated_counter_tpu_variable, new_accumulated_counter_value)

    return [update_tpu_var_counter_op]

  def record_summary_tpu_variables(self):
    if self._enable_host_call is False:
      logging.info(
          "enable_host_call is False, do not record summary tpu variables.")
      return

    if self._enable_caching_with_tpu_var_mode is False:
      logging.info(
          "enable_caching_with_tpu_var_mode is False, record summary tpu variables."
      )
      return

    logging.info("Record tpu variables.")
    self.record_summary_tensor(_LABELS_TPU_VARIABLE,
                               self._labels_tpu_variable.read_value())
    self.record_summary_tensor(_PREDS_TPU_VARIABLE,
                               self._preds_tpu_variable.read_value())
    self.record_summary_tensor(_UID_BUCKETS_TPU_VARIABLE,
                               self._uid_buckets_tpu_variable.read_value())
    self.record_summary_tensor(_REQ_TIMES_TPU_VARIABLE,
                               self._req_times_tpu_variable.read_value())
    self.record_summary_tensor(_SAMPLE_RATES_TPU_VARIABLE,
                               self._sample_rates_tpu_variable.read_value())
    self.record_summary_tensor(
        _ACCUMULATED_COUNTER_TPU_VARIABLE,
        self._accumulated_counter_tpu_variable.read_value())

  def record_summary_tensor(self, name, tensor):
    if self._enable_host_call_scalar_metrics is False and name not in _HOST_CALL_AUC_METRICS:
      return

    if self._enable_caching_with_tpu_var_mode is True and name in _DEPRECATED_METRIC_NAMES:
      return

    super(BaseEmbeddingHostCall, self).record_summary_tensor(name, tensor)

  def _verify_shape_and_dtype(self, tensor, shape_list, dtype):
    assert tensor is not None
    assert tensor.shape.as_list(
    ) == shape_list, "Expect shape: {}, but actual shape: {}".format(
        shape_list, tensor.shape.as_list())
    assert tensor.dtype == dtype, "Expect dtype {}, but actual dtype: {}".format(
        dtype, tensor.dtype)

  def _slice_tensor(self, tensor, indices, expect_shape, expect_dtype):
    """Select elements from a given tensor using given indices.

    Args:
      tensor: The Tensor whose elements are selected using the indices.
      indices: The Tensor storing the indices to be sliced.
      expect_shape: The expected shape of tensor.
      expect_dtype: The expected dtype of tensor.

    Return:
      The sliced tensor.
    """
    self._verify_shape_and_dtype(tensor, expect_shape, expect_dtype)
    # Flatten the tensor here and simplify its data format using reshape,
    # which is low cost without real data copy.
    # Each tensor has shape (n, ), n equals to core_number * batch_size_per_core
    tensor = tf.reshape(tensor, [-1])
    sliced_tensor = tf.gather(tensor, indices)

    return sliced_tensor

  def _serialize_tensor(self, sampled_tensor, gs, message_name):
    tf2.summary.text(message_name,
                     data=tf.io.serialize_tensor(sampled_tensor),
                     step=gs)

  # This function is deprecated for host_call.py.
  def _serialize_messages(self, labels, y_preds, sample_rates, req_times,
                          uid_buckets, gs):
    assert labels is not None
    assert y_preds is not None
    assert sample_rates is not None
    assert req_times is not None
    assert uid_buckets is not None

    # For sample_rates and req_times, we only need to keep the first one.
    expect_shape = sample_rates.shape.as_list()
    assert len(
        expect_shape
    ) == 1, "Expect sample_rates shape rank to be 1, but its shape is {}".format(
        expect_shape)

    self._serialize_tensor(sample_rates, [0], expect_shape, tf.float32, gs,
                           _DEEPINSIGHT_SAMPLE_RATES)
    self._serialize_tensor(req_times, [0], expect_shape, tf.int64, gs,
                           _DEEPINSIGHT_REQ_TIMES)

  def _write_summary_ops(self,
                         gs,
                         labels,
                         y_preds,
                         uid_buckets,
                         req_times,
                         sample_rates,
                         stopping_signals_sum=None):
    if labels is not None and y_preds is not None:
      # Filter labels and y_preds by uids to ensure that only a fraction of
      # uids are selected for AUC calculation
      if uid_buckets is not None:
        # Filter out data with uid_bucket < _UID_SAMPLE_RATE * _RATIO_N to write to summary file
        reshaped_uid_buckets = tf.reshape(uid_buckets, [-1])

        if stopping_signals_sum is None:
          indices = tf.squeeze(
              tf.where(reshaped_uid_buckets < int(_UID_SAMPLE_RATE * _RATIO_N)))
        else:
          indices = tf.squeeze(
              tf.where(
                  tf.math.logical_and(
                      reshaped_uid_buckets < int(_UID_SAMPLE_RATE * _RATIO_N),
                      tf.math.equal(stopping_signals_sum, 0))))
        expect_shape = labels.shape.as_list()
        labels = self._slice_tensor(labels, indices, expect_shape, tf.float32)
        y_preds = self._slice_tensor(y_preds, indices, expect_shape, tf.float32)
        self._serialize_tensor(labels, gs, _DEEPINSIGHT_LABELS)
        self._serialize_tensor(y_preds, gs, _DEEPINSIGHT_PREDS)

        if self._enable_deepinsight is True and req_times is not None:
          assert req_times.get_shape().rank == 2, "req_times shape: {}".format(
              req_times.get_shape())

          # Repeat each element self._top_k_sampling_num_per_core times in dim(1).
          req_times = tf.repeat(req_times, [self._top_k_sampling_num_per_core],
                                axis=1)
          req_times = self._slice_tensor(req_times, indices, expect_shape,
                                         tf.int64)
          self._serialize_tensor(req_times, gs, _DEEPINSIGHT_REQ_TIMES)

      # Calculate AUC based on filtered labels and y_preds
      auc, auc_op = tf.metrics.auc(labels=labels, predictions=y_preds)
      tf2.summary.scalar("auc", data=auc, step=gs)

      # Serialize message
      if self._enable_deepinsight is True:
        if sample_rates is not None:
          expect_shape = sample_rates.shape.as_list()
          assert len(
              expect_shape
          ) == 1, "Expect sample_rates shape rank to be 1, but its shape is {}".format(
              expect_shape)

          # For sample_rates and req_times, we only need to keep the first element.
          sampled_sample_rates = self._slice_tensor(sample_rates, [0],
                                                    expect_shape, tf.float32)

          self._serialize_tensor(sampled_sample_rates, gs,
                                 _DEEPINSIGHT_SAMPLE_RATES)
    else:
      auc_op = None

    tf2.summary.scalar("sampled_labels_variable_avg",
                       data=tf.reduce_mean(labels),
                       step=gs)
    tf2.summary.scalar("sampled_preds_variable_avg",
                       data=tf.reduce_mean(y_preds),
                       step=gs)
    tf2.summary.scalar("req_times_variable_avg",
                       data=tf.reduce_mean(req_times),
                       step=gs)
    tf2.summary.scalar("sample_rates_variable_avg",
                       data=tf.reduce_mean(sample_rates),
                       step=gs)

    return auc_op

  def generate_host_call_hook(self):

    def _host_call(*args):
      gs, tensors = self.decompress_tensors(args)
      summary_writer = tf2.summary.create_file_writer(self._output_dir +
                                                      "/host_call",
                                                      flush_millis=10000,
                                                      max_queue=5000)
      with summary_writer.as_default():
        labels = None
        y_preds = None
        req_times = None
        sample_rates = None
        uid_buckets = None

        for i, t in enumerate(tensors):
          if i == 0:
            continue

          name = self._tensor_names[i]
          data = None
          if "_avg" in name:
            data = tf.reduce_mean(t)
          elif "_max" in name:
            data = tf.reduce_max(t)
          elif "_sum" in name:
            data = tf.reduce_sum(t)
          elif _LABLES_FOR_AUC_CALCULATION in name:
            labels = t
          elif _Y_PRED_FOR_AUC_CALCULATION in name:
            y_preds = t
          elif _REQ_TIME in name:
            req_times = tf.expand_dims(t, -1)
          elif _SAMPLE_RATE in name:
            sample_rates = t
          elif _UID_BUCKET in name:
            uid_buckets = t
          else:
            data = t[0]

          if data is not None:
            tf2.summary.scalar(name, data=data, step=gs)

        auc_op = self._write_summary_ops(gs, labels, y_preds, uid_buckets,
                                         req_times, sample_rates)

      if auc_op is not None:
        return tf.group(tf.compat.v1.summary.all_v2_summary_ops(), auc_op)
      else:
        return tf.compat.v1.summary.all_v2_summary_ops()

    def get_used_slice(tpu_variable, used_elements_count):
      return tf.slice(tpu_variable, [0, 0], [-1, used_elements_count])

    def _host_call_with_tpu(*args):
      gs, tensors = self.decompress_tensors(args)
      summary_writer = tf2.summary.create_file_writer(self._output_dir +
                                                      "/host_call",
                                                      flush_millis=10000,
                                                      max_queue=5000)

      stopping_signals_sum = None
      auc_op = None
      with summary_writer.as_default():
        labels_value = None
        preds_value = None
        uid_buckets_value = None
        req_times_value = None
        sample_rates_value = None
        accumulated_counter_value = None

        for i, t in enumerate(tensors):
          if i == 0:
            continue

          name = self._tensor_names[i]
          data = None
          if "_avg" in name:
            data = tf.reduce_mean(t)
          elif "_max" in name:
            data = tf.reduce_max(t)
          elif "_sum" in name:
            data = tf.reduce_sum(t)
          elif _LABELS_TPU_VARIABLE in name:
            labels_value = t
          elif _PREDS_TPU_VARIABLE in name:
            preds_value = t
          elif _UID_BUCKETS_TPU_VARIABLE in name:
            uid_buckets_value = t
          elif _REQ_TIMES_TPU_VARIABLE in name:
            req_times_value = t
          elif _SAMPLE_RATES_TPU_VARIABLE in name:
            sample_rates_value = t
          elif _ACCUMULATED_COUNTER_TPU_VARIABLE in name:
            accumulated_counter_value = t
          elif "stopping_signals" in name:
            stopping_signals_sum = tf.reduce_sum(tf.cast(t, tf.int32))
          elif name not in _DEPRECATED_METRIC_NAMES:
            data = t[0]

          if data is not None:
            tf2.summary.scalar(name, data=data, step=gs)

        # Check labels, preds, uid_buckets shape is as expected.
        expected_multiple_values_per_step_shape = [
            self._context.num_replicas,
            self._host_call_every_n_steps * self._top_k_sampling_num_per_core
        ]
        assert labels_value.get_shape() == expected_multiple_values_per_step_shape, \
          "labels_tpu_variable shape: {}, expectd_shape: {}." \
          .format(labels_value.get_shape(), expected_multiple_values_per_step_shape)
        assert preds_value.get_shape() == expected_multiple_values_per_step_shape, \
          "preds_tpu_variable shape: {}, expectd_shape: {}." \
          .format(preds_value.get_shape(), expected_multiple_values_per_step_shape)
        assert uid_buckets_value.get_shape() == expected_multiple_values_per_step_shape, \
          "uid_buckets_tpu_variable shape: {}, expectd_shape: {}." \
          .format(uid_buckets_value.get_shape(), expected_multiple_values_per_step_shape)

        # Check tpu_variable_accumulated_times_scalar is as expected.
        expected_scalar_shape = [self._context.num_replicas]
        assert accumulated_counter_value.get_shape()== expected_scalar_shape, \
          "tpu_variable_accumulated_times_scalar shape: {}, expectd_shape: {}." \
          .format(accumulated_counter_value.get_shape(), expected_scalar_shape)

        used_slice_len = accumulated_counter_value[
            0] * self._top_k_sampling_num_per_core

        # Get the used parts of TPU variable. Then reshape all tpu variables to be similar shape with same rank and same batch dimension
        # as other non-tpu variables.
        labels = get_used_slice(labels_value, used_slice_len)
        y_preds = get_used_slice(preds_value, used_slice_len)
        uid_buckets = get_used_slice(uid_buckets_value, used_slice_len)

        # Check req_times, sample_rates shape is as expected.
        expected_single_value_per_step_shape = [
            self._context.num_replicas, self._host_call_every_n_steps
        ]
        assert req_times_value.get_shape()== expected_single_value_per_step_shape, \
          "req_times_tpu_variable shape: {}, expectd_shape: {}." \
          .format(req_times_value.get_shape(), expected_single_value_per_step_shape)

        # Get the used part of req_times TPu variable.
        req_times = get_used_slice(req_times_value,
                                   accumulated_counter_value[0])

        assert sample_rates_value.get_shape()== expected_single_value_per_step_shape, \
          "sample_rates_tpu_variable shape: {}, expectd_shape: {}." \
          .format(sample_rates_value.get_shape(), expected_single_value_per_step_shape)

        # Attention, here for performance purpose we use one one sample_rate
        # to represent all examples in this host_call. We will evaluate from the deepinsight showing side
        # to see if this has big impact when user use it.
        sample_rates = tf.squeeze(tf.slice(sample_rates_value, [0, 0], [-1, 1]))

        tf2.summary.scalar("uid_buckets_tpu_variable_avg",
                           data=tf.reduce_mean(uid_buckets),
                           step=gs)
        tf2.summary.scalar("accumulated_times_tpu_variable_avg",
                           data=tf.reduce_mean(accumulated_counter_value),
                           step=gs)
        tf2.summary.scalar("used_slice_len_avg",
                           data=tf.reduce_min(used_slice_len),
                           step=gs)

        if stopping_signals_sum is not None:
          tf2.summary.scalar("stopping_signals_sum",
                             data=stopping_signals_sum,
                             step=gs)

        auc_op = self._write_summary_ops(gs, labels, y_preds, uid_buckets,
                                         req_times, sample_rates,
                                         stopping_signals_sum)

      if auc_op is not None:
        return tf.group(tf.compat.v1.summary.all_v2_summary_ops(), auc_op)
      else:
        return tf.compat.v1.summary.all_v2_summary_ops()

    if self._enable_host_call == True:
      self.compress_tensors()
      if self._enable_caching_with_tpu_var_mode is False:
        return (_host_call, self._tensors)
      else:
        return (_host_call_with_tpu, self._tensors)
    else:
      logging.info("host_call has been disabled")
      return None
