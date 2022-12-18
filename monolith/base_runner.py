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

"""Base class for all jobs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class BaseRunner(object):
  """Base class for all jobs."""

  def __init__(self, *args, **kwargs):
    """Construct a new BaseRunner.
    Args:
      params:  Params object containing model configuration.
      model_dir:  String path to the log directory to output to.
    """
    pass

  def run(self):
    raise NotImplementedError

  def write_summary(self, logs, summary_writer, current_step):
    """Write out summaries of current training step for the checkpoint."""
    with tf.compat.v1.Graph().as_default():
      summaries = [
          tf.compat.v1.Summary.Value(tag=tag, simple_value=value)
          for tag, value in logs.items()
      ]
      tf_summary = tf.compat.v1.Summary(value=summaries)
      summary_writer.add_summary(tf_summary, current_step)
