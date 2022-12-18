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
import tensorflow as tf


class BaseHostCall(object):

  def __init__(self, output_dir, enable_host_call):
    self._output_dir = output_dir
    self._enable_host_call = enable_host_call
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
    if not self._enable_host_call:
      return

    if name in self._tensor_names:
      logging.info('{} | {}'.format(name, self._tensor_names))
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

  def generate_host_call_hook(self):
    # Children should implement this API and implement it with model specific host call logic.
    return None
