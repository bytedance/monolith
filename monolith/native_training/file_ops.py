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

from monolith.native_training.runtime.ops import gen_monolith_ops

file_ops = gen_monolith_ops


class WritableFile:
  """A gfile wrapper used in the graph execution."""

  def __init__(self, filename):
    self._handle = file_ops.monolith_writable_file(filename)

  def append(self, content):
    """Append the content into the file.
    Args:
      content - a 0-D string tensor.
    """
    return file_ops.monolith_writable_file_append(self._handle, content)

  def close(self):
    return file_ops.monolith_writable_file_close(self._handle)


class FileCloseHook(tf.estimator.SessionRunHook):
  """A hook that will close WritableFiles at the end of session."""

  def __init__(self, files):
    assert isinstance(files, list)
    self._files = files
    self._close_ops = [f.close() for f in files]

  def end(self, session):
    session.run(self._close_ops)
