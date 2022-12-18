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

from monolith.native_training import save_utils
import os
import time
from unittest import mock

import tensorflow as tf

from monolith.native_training.model_export import export_hooks
from monolith.native_training.model_export import export_state_utils
from monolith.native_training import save_utils


class ExportHookTest(tf.test.TestCase):

  def testBasic(self):
    model_dir = os.path.join(os.environ["TEST_TMPDIR"], "testBasic_model_dir")
    export_dir_base = os.path.join(os.environ["TEST_TMPDIR"],
                                   "testBasic_export_dir")
    exporter = mock.MagicMock()
    export_dir = os.path.join(export_dir_base, "12345678")
    os.makedirs(export_dir)

    def export_saved_model(serving_input_receiver_fn, checkpoint_file,
                           global_step):
      self.assertEqual(checkpoint_file, model_dir + "/model.ckpt-10")
      return export_dir.encode()

    exporter.export_saved_model.side_effect = export_saved_model

    saver_hook = save_utils.NoFirstSaveCheckpointSaverHook(
        model_dir,
        save_steps=10000,
        listeners=[
            export_hooks.ExportSaverListener(model_dir + "/model.ckpt", None,
                                             exporter)
        ])
    global_step = tf.compat.v1.train.get_or_create_global_step()
    global_step = tf.compat.v1.assign(global_step, 10)
    with tf.compat.v1.train.SingularMonitoredSession(
        hooks=[saver_hook]) as sess:
      sess.run(global_step)

    state = export_state_utils.get_export_saver_listener_state(export_dir_base)
    # One is before_save, one is after_save.
    self.assertEqual(len(state.entries), 1)
    entry = state.entries[0]
    self.assertEqual(entry.export_dir, export_dir)
    self.assertEqual(entry.global_step, 10)

  def testExporterReturnsDict(self):
    model_dir = os.path.join(os.environ["TEST_TMPDIR"],
                             "testExporterReturnsDict")
    export_dir_base = os.path.join(os.environ["TEST_TMPDIR"],
                                   "testBasic_export_dir")
    exporter = mock.MagicMock()
    export_dir1 = os.path.join(export_dir_base, "model1/12345678")
    export_dir2 = os.path.join(export_dir_base, "model2/12345678")
    os.makedirs(export_dir1)
    os.makedirs(export_dir2)

    def export_saved_model(serving_input_receiver_fn, checkpoint_file,
                           global_step):
      return {
          "model1": export_dir1.encode(),
          "model2": export_dir2.encode(),
      }

    exporter.export_saved_model.side_effect = export_saved_model

    saver_hook = save_utils.NoFirstSaveCheckpointSaverHook(
        model_dir,
        save_steps=10000,
        listeners=[
            export_hooks.ExportSaverListener(model_dir + "/model.ckpt", None,
                                             exporter)
        ])
    global_step = tf.compat.v1.train.get_or_create_global_step()
    global_step = tf.compat.v1.assign(global_step, 10)
    with tf.compat.v1.train.SingularMonitoredSession(
        hooks=[saver_hook]) as sess:
      sess.run(global_step)

  def testDeleted(self):
    model_dir = os.path.join(os.environ["TEST_TMPDIR"], "testDeleted_model_dir")
    export_dir_base = os.path.join(os.environ["TEST_TMPDIR"],
                                   "testDeleted_export_dir")
    exporter = mock.MagicMock()

    def export_saved_model(serving_input_receiver_fn, checkpoint_file,
                           global_step):
      export_dir = os.path.join(export_dir_base, str(time.time()))
      os.makedirs(export_dir)
      return export_dir.encode()

    exporter.export_saved_model.side_effect = export_saved_model
    global_step = tf.compat.v1.train.get_or_create_global_step()
    global_step = tf.compat.v1.assign_add(global_step, 1)

    saver = save_utils.PartialRecoverySaver(tf.compat.v1.global_variables(),
                                            sharded=True,
                                            max_to_keep=1,
                                            keep_checkpoint_every_n_hours=2)
    saver_hook = save_utils.NoFirstSaveCheckpointSaverHook(
        model_dir,
        save_steps=1,
        saver=saver,
        listeners=[
            export_hooks.ExportSaverListener(model_dir + "/model.ckpt", None,
                                             exporter)
        ])

    with tf.compat.v1.train.SingularMonitoredSession(
        hooks=[saver_hook]) as sess:
      sess.run(global_step)
      sess.run(global_step)

    state = export_state_utils.get_export_saver_listener_state(export_dir_base)
    # Saved model for step 1 is deleted.
    self.assertEqual(len(state.entries), 1)
    entry = state.entries[0]
    self.assertEqual(entry.global_step, 2)
    self.assertEqual(len(tf.io.gfile.glob(export_dir_base + "/*.*")), 1)


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
