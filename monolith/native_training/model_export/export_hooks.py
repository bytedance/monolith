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

"""A hook that exports the model at the same time we save the checkpoint"""
import os
import re
from pathlib import Path
from typing import List
from absl import logging
import tensorflow as tf

from monolith.native_training import save_utils
from monolith.native_training.model_export import saved_model_exporters
from monolith.native_training.model_export import export_pb2
from monolith.native_training.model_export import export_state_utils


def get_global_step(checkpoint_path: str):
  pattern = re.compile(r'^.*model.ckpt-(\d+)$')
  matched = pattern.match(checkpoint_path.strip())
  assert matched is not None
  return int(matched.group(1))


class ExportSaverListener(tf.estimator.CheckpointSaverListener):
  """A hook that exports saved model whenever a new ckpt is generated."""

  def __init__(self,
               save_path: str,
               serving_input_receiver_fn,
               exporter: saved_model_exporters.BaseExporter,
               exempt_checkpoint_paths: List[str] = None,
               dense_only: bool = False):
    super().__init__()
    self._serving_input_receiver_fn = serving_input_receiver_fn
    self._helper = save_utils.SaveHelper(save_path)
    self._exporter = exporter
    self._exempt_checkpoint_steps = set([
        get_global_step(p) for p in exempt_checkpoint_paths
    ]) if exempt_checkpoint_paths else set()
    self._dense_only = dense_only
    logging.info('Exempt global steps={}'.format(self._exempt_checkpoint_steps))

  def after_save(self, session, global_step_value):
    checkpoint_file = self._helper.get_ckpt_prefix(global_step_value)
    export_dirs = self._exporter.export_saved_model(
        self._serving_input_receiver_fn, checkpoint_file, global_step_value)
    if isinstance(export_dirs, bytes):
      export_dirs = [export_dirs]
    elif isinstance(export_dirs, dict):
      export_dirs = export_dirs.values()

    for export_dir in export_dirs:
      self._add_entry_to_state(export_dir, global_step_value)
      # delete old saved models
      self._maybe_delete_old_entries(export_dir)

  def _add_entry_to_state(self, export_dir: bytes, global_step_value: int):
    export_dir = export_dir.decode()
    export_dir_base = os.path.dirname(export_dir)
    state = export_state_utils.get_export_saver_listener_state(export_dir_base)

    entry = export_pb2.ServingEntry()
    entry.export_dir = export_dir
    entry.global_step = global_step_value
    state.entries.append(entry)

    export_state_utils.overwrite_export_saver_listener_state(
        export_dir_base, state)

  def _maybe_delete_old_entries(self, export_dir: bytes):
    export_dir = export_dir.decode()
    export_dir_base = os.path.dirname(export_dir)
    old_state = export_state_utils.get_export_saver_listener_state(
        export_dir_base)
    existing_steps = self._helper.get_existing_checkpoint_steps(
    ) | self._exempt_checkpoint_steps
    if self._dense_only:
      path = Path(export_dir_base)
      model_dir = str(path.parent.parent)
      full_stats = tf.train.get_checkpoint_state(model_dir)
      if full_stats:
        existing_steps |= set([
            get_global_step(ckpt)
            for ckpt in full_stats.all_model_checkpoint_paths
        ])

    new_state = export_pb2.ServingModelState()
    for entry in old_state.entries:
      if entry.global_step in existing_steps:
        new_state.entries.append(entry)
      else:
        try:
          logging.info("Deleted export dir: %s.", entry.export_dir)
          tf.io.gfile.rmtree(entry.export_dir)
        except tf.errors.NotFoundError:
          logging.warning(
              "Hit NotFoundError when deleting '%s', possibly because another "
              "process/thread is also deleting/moving the same file",
              entry.export_dir)

    export_state_utils.overwrite_export_saver_listener_state(
        export_dir_base, new_state)
