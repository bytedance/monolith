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

#coding:utf-8
from absl import logging

from tensorflow.python.training import session_run_hook


class CheckpointRestorerListener():
  """Interface for listeners that take action before or after restore."""

  def begin(self):
    pass

  def before_restore(self, session):
    pass

  def after_restore(self, session):
    pass

  def end(self, session):
    pass


class CheckpointRestorerHook(session_run_hook.SessionRunHook):
  """
  Restores checkpoints at the begining. Use to call 'CheckpointRestorerListener'.
  The real restore action is implemented in 'CheckpointRestorerListener'.

  """

  def __init__(self, listeners=None):
    """Initializes a `CheckpointRestorerHook`.

    Args:
      listeners: List of `CheckpointRestorerListener` subclass instances. Used for
        callbacks that run immediately before or after this hook restores the
        checkpoint.

    """
    logging.info("Create CheckpointRestorerHook.")
    self._listeners = listeners or []

  def begin(self):
    for l in self._listeners:
      l.begin()

  def after_create_session(self, session, coord):
    self._restore(session)

  def _restore(self, session):
    """Restores the latest checkpoint."""
    logging.info("Calling checkpoint restorer listeners.")
    for l in self._listeners:
      l.before_restore(session)

    # None restore actions in this hook.

    for l in self._listeners:
      l.after_restore(session)
