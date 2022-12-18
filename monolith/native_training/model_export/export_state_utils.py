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

import os

import tensorflow as tf
from google.protobuf import text_format

from monolith.native_training.model_export import export_pb2

_ExportSaverListenerStateFile = "ExportSaverListenerState"


def get_export_saver_listener_state(
    export_dir_base: str) -> export_pb2.ServingModelState:
  filename = os.path.join(export_dir_base, _ExportSaverListenerStateFile)
  state = export_pb2.ServingModelState()
  try:
    with tf.io.gfile.GFile(filename) as f:
      text = f.read()
      text_format.Merge(text, state)
  except tf.errors.NotFoundError:
    pass
  return state


def overwrite_export_saver_listener_state(export_dir_base: str,
                                          state: export_pb2.ServingModelState):
  filename = os.path.join(export_dir_base, _ExportSaverListenerStateFile)
  tmp_name = filename + "-tmp"
  tf.io.gfile.makedirs(export_dir_base)
  with tf.io.gfile.GFile(tmp_name, mode="w") as f:
    text = text_format.MessageToString(state)
    f.write(text)
  tf.io.gfile.rename(tmp_name, filename, overwrite=True)
