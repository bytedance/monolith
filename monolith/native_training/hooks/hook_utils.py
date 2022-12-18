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


class BeforeSaveListener(tf.estimator.CheckpointSaverListener):
  """Only calls before save in the listener"""

  def __init__(self, listener: tf.estimator.CheckpointSaverListener):
    self._listener = listener

  def before_save(self, session, global_step_value):
    self._listener.before_save(session, global_step_value)

  def __repr__(self):
    return super().__repr__() + repr(self._listener)


class AfterSaveListener(tf.estimator.CheckpointSaverListener):
  """Only calls after save in the listener"""

  def __init__(self, listener: tf.estimator.CheckpointSaverListener):
    self._listener = listener

  def after_save(self, session, global_step_value):
    self._listener.after_save(session, global_step_value)

  def __repr__(self):
    return super().__repr__() + repr(self._listener)
