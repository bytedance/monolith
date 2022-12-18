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

import dataclasses

import tensorflow as tf


@dataclasses.dataclass
class _Info:
  session: tf.compat.v1.Session = None


_INFO = _Info()


class SetCurrentSessionHook(tf.estimator.SessionRunHook):

  def after_create_session(self, session, coord):
    _INFO.session = session

  def end(self, session):
    _INFO.session = None


def get_current_session():
  """Returns the current session. If hook was added,
  it will return session in hook. Otherwise, it will
  return default session.
  """
  if _INFO.session:
    return _INFO.session
  return tf.compat.v1.get_default_session()
