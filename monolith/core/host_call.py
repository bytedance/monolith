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
import json
import numpy as np
import tensorflow as tf

_LABLES_FOR_AUC_CALCULATION = "labels_for_auc_calculation"
_Y_PRED_FOR_AUC_CALCULATION = "y_pred_for_auc_calculation"
_REQ_TIME = "req_time"
_SAMPLE_RATE = "sample_rate"

_DEEPINSIGHT_SAMPLE_RATES = "di_example_sample_rates"
_DEEPINSIGHT_LABELS = "di_labels"
_DEEPINSIGHT_PREDS = "di_preds"
_DEEPINSIGHT_REQ_TIMES = "di_req_times"


class HostCall():

  def __init__(self, output_dir, enable_host_call, enable_deepinsight):
    self._output_dir = output_dir
    self._enable_host_call = enable_host_call
    self._enable_deepinsight = enable_deepinsight
    self._tensor_names = ["global_step"]
    gs = tf.compat.v1.train.get_global_step()
    # Creating batch dimension since host call needs to concat all the cores'
    # results.
    self._tensors = [tf.reshape(gs, [-1])]
    # compressed_tensor_list is a list of compressed tensors, in
    # which each compressed tensor is the concatenation of multiple
    # uncompressed tensors. To decompress, we need to store the original
    # sizes of all uncompressed tensors, organized in list of lists.
    # Each list corresponds to a compressed tensors.
    self._lists_tensor_sizes = []  # A list of lists

  def record_summary_tensor(self, name, tensor):
    assert name not in self._tensor_names
    self._tensor_names.append(name)

    assert len(tensor.get_shape()) <= 1, "Now we only support tensor with shape (k, ) or ()"\
        "but we met tensor with shape: {}".format(tensor.get_shape())

    # Creating batch dimension since host call needs to concat all the cores'
    # results.
    reshaped_tensor = tf.reshape(tensor, [-1])
    self._tensors.append(reshaped_tensor)

  def compress_tensors(self):
    """For n tensors with shape (k_i, ) and same data type, concat them on axis=1.

        After concatenation the compressed tensors is stored as in shape
        (1, k_0 + k_1 + ... + k_{n-1}).
        """
    assert len(self._tensor_names) == len(self._tensors), "tensor_names and tensors must have same length," \
            "tensor_names length: {}, tensors length: {}".format(len(self._tensor_names), len(self._tensors))
    # key is tensor data type, value is a list of tensor names with this data type.
    data_type_to_tensor_names = {}
    # key is tensor data type, value is tensor a list of tensors with this data type.
    date_type_to_tensors = {}

    # Group tensor names and tensors by same data type.
    for tensor_name, tensor in zip(self._tensor_names, self._tensors):
      data_type_to_tensor_names.setdefault(tensor.dtype, []).append(tensor_name)
      date_type_to_tensors.setdefault(tensor.dtype, []).append(tensor)

    # Compress n tensors tensor_0, tensor_1, ... , tensor_{n-1} with shape
    # (k_0, ), (k_1, )... (k_{n-1}, ) of same data
    # type to one tensor with shape (1, k_0 + ... + k_{n-1})
    compressed_tensor_name_list = []
    compressed_tensor_list = []
    for data_type, tensor_list in date_type_to_tensors.items():
      compressed_tensor_name_list.extend(data_type_to_tensor_names[data_type])

      # concat a list of tensors with shapes
      # (k_0, ), (k_1, )... (k_{n-1}, ) to a tensor with shape
      # (k_0 + k_1 + ... + k_{n-1}, )
      tensor_sizes = []
      for tensor in tensor_list:
        tensor_sizes.append(tensor.shape[0].value)
      self._lists_tensor_sizes.append(tensor_sizes)
      compressed_tensor = tf.concat(tensor_list, axis=0)

      # expand dimension at 0 to make it have the batch dimension
      # tensor with shape (k_0 + k_1 + ... + k_{n-1}, )
      # => tensor with shape (1, k_0 + k_1 + ... + k_{n-1})
      compressed_tensor = tf.expand_dims(compressed_tensor, axis=0)
      compressed_tensor_list.append(compressed_tensor)

      logging.info(
          "Host call compressed tensors, data type: {}, compressed tensor shape: {}"
          .format(data_type, compressed_tensor.shape))

    self._tensor_names = compressed_tensor_name_list
    self._tensors = compressed_tensor_list

  def decompress_tensors(self, tensors):
    """
        Decompress the compressed tensors into list of decompressed tensors.

        Given a list of compressed tensors from *args. Each tensor has shape
        (num_cores, k_0 + k_1 + ... + k_{n-1}), in which the second dimension
        is the sum of lengths of uncompressed tensors (k_i, ) from the same
        core, with same shape and data type. Parse and convert them to a list of
        decompressed tensors.

        Each decompressed tensor has shape (num_cores, k_i). Decompressed tensor
        number must match with number of tensor names as well.
        """
    # Need decompress tensors
    decompressed_tensor_list = []
    for index, compressed_tensor in enumerate(tensors):
      # For each tensor, its shape is (num_cores, k_0 + ... + k_{n-1})
      assert len(compressed_tensor.get_shape(
      )) == 2, "Compressed tensors shape must be (n, m), met shape: {}".format(
          compressed_tensor.shape)
      logging.info("Decompressed tensors shape: {}, dtype: {}.".format(
          compressed_tensor.shape, compressed_tensor.dtype))

      # tensor with shape (num_cores, k_0 + ... + k_{n-1})
      # => list of tensors with shape (num_cores, k_i)
      split_tensors = tf.split(compressed_tensor,
                               self._lists_tensor_sizes[index],
                               axis=1)

      for tensor in split_tensors:
        # Each decompressed tensor with shape (num_cores, k_i)
        # => (num_cores * k_i, ).
        tensor = tf.squeeze(tensor)
        decompressed_tensor_list.append(tensor)

    assert self._tensor_names[
        0] == "global_step", "The first tensor name must be global_step, met value: {}".format(
            self._tensor_names[0][0])
    return decompressed_tensor_list[0][0], decompressed_tensor_list

  def _verify_shape_and_dtype(self, tensor, shape_list, dtype):
    assert tensor is not None
    assert tensor.shape.as_list(
    ) == shape_list, "Expect shape: {}, but actual shape: {}".format(
        shape_list, tensor.shape.as_list())
    assert tensor.dtype == dtype, "Expect dtype {}, but actual dtype: {}".format(
        dtype, tensor.dtype)

  def _serialize_messages(self, labels, y_preds, sample_rates, req_times, gs):
    assert labels is not None
    labels_shape = labels.shape.as_list()
    assert len(labels_shape
              ) == 2, "Expect labels_shape to be 1, but its shape is {}".format(
                  labels_shape)

    self._verify_shape_and_dtype(y_preds, labels_shape, tf.float32)
    self._verify_shape_and_dtype(sample_rates, labels_shape, tf.float32)
    self._verify_shape_and_dtype(req_times, labels_shape, tf.int64)

    # reshape is low cost without real data copy.
    # flatten the tensor here and simplify the data format before serializing to string.
    # Each tensor has shape (n, ), n equals to core_number * batch_size_per_core
    labels = tf.reshape(labels, [-1])
    y_preds = tf.reshape(y_preds, [-1])
    sample_rates = tf.reshape(sample_rates, [-1])
    req_times = tf.reshape(req_times, [-1])

    # The first two model names and di sample rates can be get from host_call folder suffix
    tf.compat.v1.summary.text(_DEEPINSIGHT_SAMPLE_RATES,
                              data=tf.io.serialize_tensor(sample_rates),
                              step=gs)
    tf.compat.v1.summary.text(_DEEPINSIGHT_LABELS,
                              data=tf.io.serialize_tensor(labels),
                              step=gs)
    tf.compat.v1.summary.text(_DEEPINSIGHT_PREDS,
                              data=tf.io.serialize_tensor(y_preds),
                              step=gs)
    tf.compat.v1.summary.text(_DEEPINSIGHT_REQ_TIMES,
                              data=tf.io.serialize_tensor(req_times),
                              step=gs)

  def generate_host_call_hook(self):

    def _host_call(*args):
      gs, tensors = self.decompress_tensors(args)
      summary_writer = tf.compat.v1.summary.create_file_writer(
          self._output_dir + "/host_call", flush_millis=10000, max_queue=5000)
      with summary_writer.as_default():
        labels = None
        y_preds = None
        req_times = None
        sample_rates = None

        for i, t in enumerate(tensors):
          if i == 0:
            continue

          name = self._tensor_names[i]
          data = None
          if "_avg" in name:
            data = tf.reduce_mean(t)
          elif "_max" in name:
            data = tf.reduce_max(t)
          elif _LABLES_FOR_AUC_CALCULATION in name:
            labels = t
          elif _Y_PRED_FOR_AUC_CALCULATION in name:
            y_preds = t
          elif _REQ_TIME in name:
            req_times = t
          elif _SAMPLE_RATE in name:
            sample_rates = t
          else:
            data = t[0]

          if data is not None:
            tf.compat.v1.summary.scalar(name, data=data, step=gs)
        if labels is not None and y_preds is not None:
          auc, auc_op = tf.compat.v1.metrics.auc(labels=labels,
                                                 predictions=y_preds)
          tf.compat.v1.summary.scalar("auc", data=auc, step=gs)
        else:
          auc_op = None

        if self._enable_deepinsight is True and labels is not None:
          messages = self._serialize_messages(labels, y_preds, sample_rates,
                                              req_times, gs)

      if auc_op is not None:
        return tf.group(tf.compat.v1.summary.all_v2_summary_ops(), auc_op)
      else:
        return tf.compat.v1.summary.all_v2_summary_ops()

    if self._enable_host_call == True:
      self.compress_tensors()
      return (_host_call, self._tensors)
    else:
      logging.info("host_call has been disabled")
      return None
