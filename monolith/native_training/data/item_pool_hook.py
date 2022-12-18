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

import re
import os
from absl import logging
import tensorflow as tf

from tensorflow.python.training.session_run_hook import SessionRunHook, \
  SessionRunContext, SessionRunValues
from monolith.native_training.data.feature_utils import save_item_pool, restore_item_pool
from monolith.native_training.data.datasets import POOL_KEY
from monolith.native_training.utils import get_local_host


class ItemPoolSaveRestoreHook(SessionRunHook):

  def __init__(self, model_dir: str, save_steps: int, mode: str = 'train'):
    self._model_dir = model_dir
    self._mode = mode
    self._last_global_step = None
    self._save_steps = save_steps
    self._ckpt_state = None

    self._save_op = None
    self._restore_op = None
    self._global_step_tensor = None

  def begin(self):
    pools = tf.compat.v1.get_collection(POOL_KEY)
    self._global_step_tensor = tf.compat.v1.train.get_or_create_global_step()
    if pools:
      self._save_global_step = tf.compat.v1.placeholder(dtype=tf.int64)
      self._restore_global_step = tf.compat.v1.placeholder(dtype=tf.int64)
      # find the corresponding ckpt dir
      logging.info("get the ckpt_state from model_dir: {}".format(
          self._model_dir))
      self._ckpt_state = tf.train.get_checkpoint_state(self._model_dir)
      if self._ckpt_state:
        model_ckpt_path = self._ckpt_state.model_checkpoint_path
        logging.info("the path to model.ckpt: {}".format(model_ckpt_path))
        restore_dir = os.path.dirname(self._ckpt_state.model_checkpoint_path)
        self._restore_op = restore_item_pool(
            pool=pools[0],
            global_step=self._restore_global_step,
            model_path=restore_dir)
      self._save_op = save_item_pool(pool=pools[0],
                                     global_step=self._save_global_step,
                                     model_path=self._model_dir)

  def after_create_session(self, session, coord):  # pylint: disable=unused-argument
    if self._mode == tf.estimator.ModeKeys.PREDICT:
      return

    self._last_global_step = session.run(self._global_step_tensor)
    if self._restore_op is not None:
      logging.info("the last global step is {}".format(
          str(self._last_global_step)))
      if self._ckpt_state:
        step = int(
            self._ckpt_state.model_checkpoint_path.split('/')[-1].split('-')
            [-1])
        logging.info("the step from the last checkpoint is {}".format(
            str(step)))
        session.run(self._restore_op,
                    feed_dict={self._restore_global_step: step})
        logging.info(
            "after_create_session retore the itempool from ckpt: {}".format(
                str(step)))

  def after_run(
      self,
      run_context: SessionRunContext,  # pylint: disable=unused-argument
      run_values: SessionRunValues):  # pylint: disable=unused-argument
    if self._mode != tf.estimator.ModeKeys.TRAIN:
      return

    if self._save_op is not None and self._save_steps is not None and self._save_steps > 0:
      cur_global_step = run_context.session.run(self._global_step_tensor)
      if cur_global_step > self._last_global_step + self._save_steps:
        logging.info("after_run start to save item_pool at step {}".format(
            str(cur_global_step)))
        run_context.session.run(
            self._save_op, feed_dict={self._save_global_step: cur_global_step})
        self._last_global_step = cur_global_step

  def end(self, session):  # pylint: disable=unused-argument
    if self._mode != tf.estimator.ModeKeys.TRAIN:
      return

    if self._save_op is not None:
      cur_global_step = session.run(self._global_step_tensor)
      if cur_global_step > self._last_global_step:
        logging.info("session_end start to save item_pool at step {}".format(
            str(cur_global_step)))
        session.run(self._save_op,
                    feed_dict={self._save_global_step: cur_global_step})
        self._last_global_step = cur_global_step
