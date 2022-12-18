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

"""Base task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from monolith.core import base_layer
from monolith.core import hyperparams


class BaseTask(base_layer.BaseLayer):
  """A single training task."""

  @classmethod
  def params(cls):
    p = super(BaseTask, cls).params()

    p.define('accelerator', None,
             'Accelerator to use. One of [None, "tpu", "horovod"].')

    p.define('input', hyperparams.Params(), 'Input Params.')
    p.input.define('eval_examples', None,
                   'Number of total examples for evaluation.')
    p.input.define('train_examples', None,
                   'Number of total examples for training.')

    p.define('eval', hyperparams.Params(),
             'Params to control how this task should be evaled.')
    p.eval.define('per_replica_batch_size', None,
                  'Number of per replica batch size')
    p.eval.define('steps_per_eval', 10000,
                  'Number of training steps between two evluations.')
    p.eval.define('steps', None, 'Number of steps for which to eval model.')

    p.define('train', hyperparams.Params(),
             'Params to control how this task should be trained.')

    p.train.define('steps', None, 'Number of steps for which to train model.')
    p.train.define('max_steps', None,
                   'Number of total steps for which to train model.')
    p.train.define('per_replica_batch_size', None,
                   'Number of per replica batch size')
    p.train.define(
        'file_pattern', None,
        'Training input data. If file_pattern and file_folder are both' \
        ' provided, use file pattern firstly.')
    p.train.define('repeat', False,
                   'Whether repeat in the training job or not.')
    p.train.define('label_key', 'label',
                   'The key of the label field in the data.')
    p.train.define(
        'save_checkpoints_steps', None,
        'Save checkpoint every save_checkpoints_steps. If None, overwrite by runner.'
    )
    p.train.define(
        'save_checkpoints_secs', None,
        'Save checkpoint every save_checkpoints_secs. If None, overwrite by runner.'
    )
    p.train.define('dense_only_save_checkpoints_secs', None,
                   'Save dense checkpoint every save_checkpoints_secs')
    p.train.define('dense_only_save_checkpoints_steps', None,
                   'Save dense checkpoint every save_checkpoints_steps')
    return p

  def __init__(self, params):
    """Constructs a BaseTask object."""
    super(BaseTask, self).__init__(params)

  def create_input_fn(self, mode):
    """Create input_fn given the mode.

        Args:
            mode: tf.estimator.ModeKeys.TRAIN/EVAL/PREDICT.
        Returns:
            An input fn for Estimator.
        """
    raise NotImplementedError('Abstract method.')

  def create_model_fn(self,):
    """Create model fn."""
    raise NotImplementedError('Abstract method.')
