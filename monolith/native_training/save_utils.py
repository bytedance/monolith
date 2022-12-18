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

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

from typing import Set
import collections
import time
import traceback
import os, sys

from datetime import datetime
from absl import logging
from google.protobuf import text_format
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import ops, errors
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.training.py_checkpoint_reader import NewCheckpointReader, CheckpointReader

from monolith.native_training.monolith_checkpoint_state_pb2 import MonolithCheckpointState
from monolith.native_training import utils
from monolith.native_training.session_run_hooks import tide_available_now
from monolith.native_training.model_export.export_context import is_exporting
from monolith.native_training.dense_reload_utils import CUSTOM_RESTORE_OP, calc_feed_dict
from monolith.native_training.metric import cli
from monolith.native_training import native_task_context

_CkptStateCache = collections.namedtuple("_CkptStateCache",
                                         ["global_step_value", "ckpt_state"])

_ckpt_state_cache_map = {}

MONOLITH_CKPT_STATE_FILE_NAME = "monolith_checkpoint"


def get_latest_checkpoint_state(checkpoint_dir: str, global_step_value: int):
  """A function that helps to get ckpt state with cache.
  Args:
    global_step_value - used to decide if our cache is stale or not.
  """
  cache = _ckpt_state_cache_map.get(checkpoint_dir, None)
  if cache is None or cache.global_step_value < global_step_value or cache.ckpt_state is None:
    cache = _CkptStateCache(
        global_step_value=global_step_value,
        ckpt_state=tf.train.get_checkpoint_state(checkpoint_dir))
    _ckpt_state_cache_map[checkpoint_dir] = cache

  return _ckpt_state_cache_map.get(checkpoint_dir).ckpt_state


def get_monolith_checkpoint_state(checkpoint_dir,
                                  filename=None,
                                  remove_invalid_path=False):
  """Returns MonolithCheckpointState proto from the "monolith_checkpoint" file.

  If the "monolith_checkpoint" file contains a valid MonolithCheckpointState
  proto, returns it.

  Args:
    checkpoint_dir: The directory of checkpoints.
    filename: Optional name of the monolith checkpoint file.  Default to
      'monolith_checkpoint'.

  Returns:
    A MonolithCheckpointState if the state was available, None
    otherwise.
  """
  ckpt = None
  coord_checkpoint_filename = os.path.join(
      checkpoint_dir, filename if filename else MONOLITH_CKPT_STATE_FILE_NAME)
  try:
    # Check that the file exists before opening it to avoid
    # many lines of errors from colossus in the logs.
    if file_io.file_exists(coord_checkpoint_filename):
      file_content = file_io.read_file_to_string(coord_checkpoint_filename)
      ckpt = MonolithCheckpointState()
      text_format.Merge(file_content, ckpt)

      if remove_invalid_path:
        # For relative exempt_model_checkpoint_paths, prepend checkpoint_dir.
        for i, p in enumerate(ckpt.exempt_model_checkpoint_paths):
          if not os.path.isabs(p):
            ckpt.exempt_model_checkpoint_paths[i] = os.path.join(
                checkpoint_dir, p)
        # Remove ckpt paths which do not exist from exempt_model_checkpoint_paths
        ckpt_paths_not_exist = []
        for i, p in enumerate(ckpt.exempt_model_checkpoint_paths):
          if not checkpoint_management.checkpoint_exists(p):
            ckpt_paths_not_exist.append(p)
        for p in ckpt_paths_not_exist:
          logging.warning(
              "%s not exists in file system, remove from monolith_checkpoint",
              p)
          ckpt.exempt_model_checkpoint_paths.remove(p)
  except errors.OpError as e:
    # It's ok if the file cannot be read
    logging.warning("%s: %s", type(e).__name__, e)
    logging.warning("%s: Checkpoint ignored", coord_checkpoint_filename)
    return None
  except text_format.ParseError as e:
    logging.warning("%s: %s", type(e).__name__, e)
    logging.warning("%s: Checkpoint ignored", coord_checkpoint_filename)
    return None
  return ckpt


# TODO(leqi.zou): make this class more powerful.
class SaveHelper:
  """A helper that provides some utils for saver listeners."""

  def __init__(self, basename: str):
    self._basename = basename

  def get_ckpt_prefix(self, global_step_value: int) -> str:
    """Returns checkpoint prefix for given basename and global_step_value."""
    return self._basename + "-" + str(global_step_value)

  @classmethod
  def get_ckpt_asset_dir(cls, ckpt_prefix: str) -> str:
    """Returns checkpoint asset directory for given basename and global_step_value.
    This is mainly to reduce the number files in the model_dir.
    """
    return ckpt_prefix + ".assets/"

  def get_global_step_value(self, ckpt_prefix: str) -> int:
    """Returns global step value for given checkpoint prefix."""
    if '-' in ckpt_prefix:
      return int(ckpt_prefix.split('-')[-1])
    else:
      return 0

  def get_existing_checkpoint_steps(self) -> Set[int]:
    ckpt_state = tf.train.get_checkpoint_state(os.path.dirname(self._basename))
    checkpoint_steps = set()
    for path in ckpt_state.all_model_checkpoint_paths:
      checkpoint_steps.add(self.get_global_step_value(path))
    return checkpoint_steps


class SecondOrStepTimerWithTideSetting(tf.estimator.SecondOrStepTimer):
  """Timer that triggers at most once every N seconds or once every N steps.

  It'll trigger using a different setting when tide resources is not available.
  """

  def __init__(self,
               every_secs=None,
               every_steps=None,
               tide_start_hour=None,
               tide_start_minute=None,
               tide_end_hour=None,
               tide_end_minute=None,
               tide_every_secs=None,
               save_helper: SaveHelper = None):
    super(SecondOrStepTimerWithTideSetting,
          self).__init__(every_secs=every_secs, every_steps=every_steps)

    self._tide_start_hour = tide_start_hour
    self._tide_start_minute = tide_start_minute
    self._tide_end_hour = tide_end_hour
    self._tide_end_minute = tide_end_minute
    self._tide_every_secs = tide_every_secs
    self._save_helper = save_helper
    self._enabled = True

  def enable(self):
    self._enabled = True

  def disable(self):
    self._enabled = False

  def should_trigger_for_step(self, step):
    """Return true if the timer should trigger for the specified step.

    Args:
      step: Training step to trigger on.

    Returns:
      True if the difference between the current time and the time of the last
      trigger exceeds `every_secs`, or if the difference between the current
      step and the last triggered step exceeds `every_steps`. False otherwise.
    """

    if not self._enabled:
      return False

    if self._last_triggered_step is None:
      return True

    if self._last_triggered_step == step:
      return False

    if (self._tide_start_hour is not None and self._tide_end_hour is not None
        and self._tide_every_secs is not None) and not tide_available_now(
            self._tide_start_hour, self._tide_start_minute, self._tide_end_hour,
            self._tide_end_minute):
      if time.time() >= self._last_triggered_time + self._tide_every_secs:
        logging.info("Current UTC time: {} : {}".format(
            datetime.utcnow().hour,
            datetime.utcnow().minute))
        logging.info(
            "Tide not available. Using tide checkpoint saving time interval.")
        logging.info("Now: {} Last: {} Interval: {}".format(
            time.time(), self._last_triggered_time, self._tide_every_secs))
        return True
    else:
      if self._every_secs is not None:
        if time.time() >= self._last_triggered_time + self._every_secs:
          return True

      if self._every_steps is not None:
        if step >= self._last_triggered_step + self._every_steps:
          return True

    return False


class NoFirstSaveCheckpointSaverHook(tf.estimator.CheckpointSaverHook):
  """A saver hook which won't perform the first save (which happpend on after_create_session)."""

  _has_dense_only: bool = False
  _last_triggered_step: int = 0

  def __init__(self,
               checkpoint_dir,
               save_secs=None,
               save_steps=None,
               saver=None,
               checkpoint_basename="model.ckpt",
               scaffold=None,
               listeners=None,
               save_graph_def=True,
               tide_start_hour=None,
               tide_start_minute=None,
               tide_end_hour=None,
               tide_end_minute=None,
               tide_save_secs=None,
               ignore_save_errors=False,
               is_dense_only: bool = False,
               use_native_multi_hash_table: bool = False,
               no_first_save: bool = True,
               finally_after_save_listeners=None):
    """
    Args:
      finally_after_save_listeners -
          since this hook support retrying in save,
          finally_after_save_listeners's after_save will always be called no matter
          if there is an error or not.
    """
    super().__init__(checkpoint_dir=checkpoint_dir,
                     save_secs=save_secs,
                     save_steps=save_steps,
                     saver=saver,
                     checkpoint_basename=checkpoint_basename,
                     scaffold=scaffold,
                     listeners=listeners,
                     save_graph_def=save_graph_def)
    self._helper = SaveHelper(self._save_path)
    self._timer = SecondOrStepTimerWithTideSetting(
        every_secs=save_secs,
        every_steps=save_steps,
        tide_start_hour=tide_start_hour,
        tide_start_minute=tide_start_minute,
        tide_end_hour=tide_end_hour,
        tide_end_minute=tide_end_minute,
        tide_every_secs=tide_save_secs,
        save_helper=self._helper)
    self._no_first_save = no_first_save
    self._save_graph_def = save_graph_def
    self._ignore_save_errors = ignore_save_errors
    self._is_dense_only = is_dense_only
    self._use_native_multi_hash_table = use_native_multi_hash_table
    self._finally_after_save_listeners = finally_after_save_listeners or []
    self._mcli = cli.get_cli(utils.get_metric_prefix())

  @property
  def timer(self):
    return self._timer

  # Make sure this hook run after restore hook.
  def after_create_session(self, session, coord):
    super().after_create_session(session, coord)
    if self._save_graph_def:
      self._get_saver().export_meta_graph(
          utils.get_meta_graph_file_name(self._checkpoint_dir))
    if isinstance(self._saver, PartialRecoverySaver):
      self._saver.setup_ps_initialized_state(session)
    self._create_or_update_monolith_ckpt_state(do_update=False)

  def _save(self, session, step: int) -> bool:
    if self._no_first_save:
      self._no_first_save = False
      return False

    # skip if full ckpt has happened in this step.
    # [todo] maybe bug when there are more saver hooks
    if self._is_dense_only:
      mode = 'dense_only'
      if step == self._last_triggered_step:
        return False
    else:
      mode = 'full'

    tags = {"mode": mode}

    try:
      for retries in range(2):
        try:
          start_time = time.time()
          should_stop = super()._save(session, step)
          self._last_triggered_step = step
          self._create_or_update_monolith_ckpt_state(do_update=True)
          end_time = time.time()
          self._mcli.emit_counter("save_checkpoint", 1, tags)
          self._mcli.emit_timer("save_checkpoint_time", end_time - start_time,
                                tags)
          return should_stop
        except tf.errors.OpError as op_error:
          self._mcli.emit_counter("save_checkpoint_failed", 1, tags)
          logging.error("Failed to save, retrying ...\n%s",
                        traceback.format_exc())
          catched_error = op_error
          continue
    finally:
      for l in self._finally_after_save_listeners:
        try:
          l.after_save(session, step)
        except:
          logging.error(traceback.format_exc())

    if self._ignore_save_errors:
      return False
    raise catched_error

  def _create_or_update_monolith_ckpt_state(self, do_update=False):
    # only save ckpt state if save_graph_def
    if not self._save_graph_def:
      return

    ckpt_state = get_monolith_checkpoint_state(self._checkpoint_dir,
                                               remove_invalid_path=True)
    if ckpt_state is None:
      logging.info("Create new monolith ckpt state")
      ckpt_state = MonolithCheckpointState()
      if self._use_native_multi_hash_table:
        ckpt_state.builtin_hash_table_type = MonolithCheckpointState.HashTableType.MULTI_CUCKOO_HASH_MAP
      else:
        ckpt_state.builtin_hash_table_type = MonolithCheckpointState.HashTableType.CUCKOO_HASH_MAP
    elif do_update is False:
      return
    else:
      logging.info("Update new monolith ckpt state")
    ckpt_state.last_checkpoint_save_timestamp = int(time.time())
    monolith_checkpoint_filename = os.path.join(self._checkpoint_dir,
                                                MONOLITH_CKPT_STATE_FILE_NAME)
    file_io.atomic_write_string_to_file(monolith_checkpoint_filename,
                                        text_format.MessageToString(ckpt_state))
    logging.info("monolith ckpt state saved")

  def end(self, session):
    if self._is_dense_only:
      pass
    elif self._has_dense_only:
      # force save
      last_step = session.run(self._global_step_tensor)
      self._timer.update_last_triggered_step(last_step)
      super()._save(session, last_step)
      for l in self._listeners:
        l.end(session, last_step)
    else:
      super().end(session)


class PsMonitor():
  """A monitor that use to detect ps status."""

  def __init__(self, ps_num):
    self._queues = {}
    self._enqueue_ops = {}
    self._qsize_ops = {}
    for i in range(ps_num):
      device = utils.ps_device(i)
      with tf.device(device):
        queue = tf.queue.FIFOQueue(1,
                                   tf.int32,
                                   shared_name="ps_monitor_" + str(i))
        self._queues[device] = queue
        self._enqueue_ops[device] = queue.enqueue(1)
        self._qsize_ops[device] = queue.size()

  def is_ps_uninitialized(self, sess, device):
    if device in self._qsize_ops:
      return sess.run(self._qsize_ops[device]) == 0
    return True

  def setup_ps_initialized_state(self, sess):
    for device in self._queues.keys():
      if sess.run(self._qsize_ops[device]) == 0:
        sess.run(self._enqueue_ops[device])


class SaverBuilder(tf_saver.BulkSaverBuilder):
  """SaverBuilder with support for partial recovery.
     Collect restore ops for each device.
  """

  def _AddShardedRestoreOps(self, filename_tensor, per_device,
                            restore_sequentially, reshape):
    """Add Ops to restore variables from multiple devices.

    Args:
      filename_tensor: Tensor for the path of the file to load.
      per_device: A list of (device, SaveableObject) pairs, as returned by
        _GroupByDevices().
      restore_sequentially: True if we want to restore variables sequentially
        within a shard.
      reshape: True if we want to reshape loaded tensors to the shape of the
        corresponding variable.

    Returns:
      An Operation that restores the variables.
    """
    sharded_restores = []
    self._restore_ops_per_device = collections.defaultdict(list)
    for shard, (device, saveables) in enumerate(per_device):
      with tf.device(device):
        restore_op = self._AddRestoreOps(filename_tensor,
                                         saveables,
                                         restore_sequentially,
                                         reshape,
                                         preferred_shard=shard,
                                         name="restore_shard")
        sharded_restores.append(restore_op)
        self._restore_ops_per_device[device].append(restore_op)

    for device, restore_ops in self._restore_ops_per_device.items():
      self._restore_ops_per_device[device] = tf.group(*restore_ops,
                                                      name="restore_per_device")

    return tf.group(*sharded_restores, name="restore_all")

  @property
  def restore_ops_per_device(self):
    """Return restore ops per device."""
    if hasattr(self, '_restore_ops_per_device'):
      return self._restore_ops_per_device
    return {}


# Copy from tensorflow/python/training/saver.py. The major change is in restore function.
# Apply partial recovery of dense part and hash table when ps_monitor is enabled.
# TODO(xujinghao): Implement partial recovery of hash filter if needed.
class PartialRecoverySaver():
  """Saves and restores variables.

  See [Variables](https://tensorflow.org/guide/variables)
  for an overview of variables, saving and restoring.

  The `Saver` class adds ops to save and restore variables to and from
  *checkpoints*.  It also provides convenience methods to run these ops.

  Checkpoints are binary files in a proprietary format which map variable names
  to tensor values.  The best way to examine the contents of a checkpoint is to
  load it using a `Saver`.

  Savers can automatically number checkpoint filenames with a provided counter.
  This lets you keep multiple checkpoints at different steps while training a
  model.  For example you can number the checkpoint filenames with the training
  step number.  To avoid filling up disks, savers manage checkpoint files
  automatically. For example, they can keep only the N most recent files, or
  one checkpoint for every N hours of training.

  You number checkpoint filenames by passing a value to the optional
  `global_step` argument to `save()`:

  ```python
  saver.save(sess, 'my-model', global_step=0) ==> filename: 'my-model-0'
  ...
  saver.save(sess, 'my-model', global_step=1000) ==> filename: 'my-model-1000'
  ```

  Additionally, optional arguments to the `Saver()` constructor let you control
  the proliferation of checkpoint files on disk:

  * `max_to_keep` indicates the maximum number of recent checkpoint files to
    keep.  As new files are created, older files are deleted.   If None or 0,
    no checkpoints are deleted from the filesystem but only the last one is
    kept in the `checkpoint` file.  Defaults to 5 (that is, the 5 most recent
    checkpoint files are kept.)

  * `keep_checkpoint_every_n_hours`: In addition to keeping the most recent
    `max_to_keep` checkpoint files, you might want to keep one checkpoint file
    for every N hours of training.  This can be useful if you want to later
    analyze how a model progressed during a long training session.  For
    example, passing `keep_checkpoint_every_n_hours=2` ensures that you keep
    one checkpoint file for every 2 hours of training.  The default value of
    10,000 hours effectively disables the feature.

  Note that you still have to call the `save()` method to save the model.
  Passing these arguments to the constructor will not save variables
  automatically for you.

  A training program that saves regularly looks like:

  ```python
  ...
  # Create a saver.
  saver = tf.compat.v1.train.Saver(...variables...)
  # Launch the graph and train, saving the model every 1,000 steps.
  sess = tf.compat.v1.Session()
  for step in xrange(1000000):
      sess.run(..training_op..)
      if step % 1000 == 0:
          # Append the step number to the checkpoint name:
          saver.save(sess, 'my-model', global_step=step)
  ```

  In addition to checkpoint files, savers keep a protocol buffer on disk with
  the list of recent checkpoints. This is used to manage numbered checkpoint
  files and by `latest_checkpoint()`, which makes it easy to discover the path
  to the most recent checkpoint. That protocol buffer is stored in a file named
  'checkpoint' next to the checkpoint files.

  If you create several savers, you can specify a different filename for the
  protocol buffer file in the call to `save()`.
  """

  def __init__(self,
               var_list=None,
               reshape=False,
               sharded=False,
               max_to_keep=5,
               keep_checkpoint_every_n_hours=10000.0,
               name=None,
               restore_sequentially=False,
               saver_def=None,
               builder=None,
               defer_build=False,
               allow_empty=False,
               pad_step_number=False,
               save_relative_paths=False,
               filename=None,
               ps_monitor=None,
               exempt_checkpoint_paths=None,
               skip_save=False,
               model_dir=None):
    """Creates a `Saver`.

    The constructor adds ops to save and restore variables.

    `var_list` specifies the variables that will be saved and restored. It can
    be passed as a `dict` or a list:

    * A `dict` of names to variables: The keys are the names that will be
      used to save or restore the variables in the checkpoint files.
    * A list of variables: The variables will be keyed with their op name in
      the checkpoint files.

    For example:

    ```python
    v1 = tf.Variable(..., name='v1')
    v2 = tf.Variable(..., name='v2')

    # Pass the variables as a dict:
    saver = tf.compat.v1.train.Saver({'v1': v1, 'v2': v2})

    # Or pass them as a list.
    saver = tf.compat.v1.train.Saver([v1, v2])
    # Passing a list is equivalent to passing a dict with the variable op names
    # as keys:
    saver = tf.compat.v1.train.Saver({v.op.name: v for v in [v1, v2]})
    ```

    Note: the newer `AutoTrackable` API is not supported by `Saver`. In this
    case, the `tf.train.Checkpoint` class should be used.

    The optional `reshape` argument, if `True`, allows restoring a variable from
    a save file where the variable had a different shape, but the same number
    of elements and type.  This is useful if you have reshaped a variable and
    want to reload it from an older checkpoint.

    The optional `sharded` argument, if `True`, instructs the saver to shard
    checkpoints per device.

    Args:
      var_list: A list of `Variable`/`SaveableObject`, or a dictionary mapping
        names to `SaveableObject`s. If `None`, defaults to the list of all
        saveable objects.
      reshape: If `True`, allows restoring parameters from a checkpoint where
        the variables have a different shape.
      sharded: If `True`, shard the checkpoints, one per device.
      max_to_keep: Maximum number of recent checkpoints to keep. Defaults to 5.
      keep_checkpoint_every_n_hours: How often to keep checkpoints. Defaults to
        10,000 hours.
      name: String.  Optional name to use as a prefix when adding operations.
      restore_sequentially: A `Bool`, which if true, causes restore of different
        variables to happen sequentially within each device.  This can lower
        memory usage when restoring very large models.
      saver_def: Optional `SaverDef` proto to use instead of running the
        builder. This is only useful for specialty code that wants to recreate a
        `Saver` object for a previously built `Graph` that had a `Saver`. The
        `saver_def` proto should be the one returned by the `as_saver_def()`
        call of the `Saver` that was created for that `Graph`.
      builder: Optional `SaverBuilder` to use if a `saver_def` was not provided.
        Defaults to `BulkSaverBuilder()`.
      defer_build: If `True`, defer adding the save and restore ops to the
        `build()` call. In that case `build()` should be called before
        finalizing the graph or using the saver.
      allow_empty: If `False` (default) raise an error if there are no variables
        in the graph. Otherwise, construct the saver anyway and make it a no-op.
      pad_step_number: if True, pads the global step number in the checkpoint
        filepaths to some fixed width (8 by default).  This is turned off by
        default.
      save_relative_paths: If `True`, will write relative paths to the
        checkpoint state file. This is needed if the user wants to copy the
        checkpoint directory and reload from the copied directory.
      filename: If known at graph construction time, filename used for variable
        loading/saving.

    Raises:
      TypeError: If `var_list` is invalid.
      ValueError: If any of the keys or values in `var_list` are not unique.
      RuntimeError: If eager execution is enabled and`var_list` does not specify
        a list of variables to save.

    @compatibility(eager)
    When eager execution is enabled, `var_list` must specify a `list` or `dict`
    of variables to save. Otherwise, a `RuntimeError` will be raised.

    Although Saver works in some cases when executing eagerly, it is
    fragile. Please switch to `tf.train.Checkpoint` or
    `tf.keras.Model.save_weights`, which perform a more robust object-based
    saving. These APIs will load checkpoints written by `Saver`.
    @end_compatibility
    """
    if defer_build and var_list:
      raise ValueError(
          "If `var_list` is provided then build cannot be deferred. "
          "Either set defer_build=False or var_list=None.")
    if tf.executing_eagerly():
      logging.warning(
          "Saver is deprecated, please switch to tf.train.Checkpoint or "
          "tf.keras.Model.save_weights for training checkpoints. When "
          "executing eagerly variables do not necessarily have unique names, "
          "and so the variable.name-based lookups Saver performs are "
          "error-prone.")
      if var_list is None:
        raise RuntimeError(
            "When eager execution is enabled, `var_list` must specify a list "
            "or dict of variables to save")
    self._var_list = var_list
    self._reshape = reshape
    self._sharded = sharded
    self._max_to_keep = max_to_keep
    self._keep_checkpoint_every_n_hours = keep_checkpoint_every_n_hours
    self._name = name
    self._restore_sequentially = restore_sequentially
    self.saver_def = saver_def
    self._builder = builder
    self._is_built = False
    self._allow_empty = allow_empty
    self._is_empty = None
    self._write_version = saver_pb2.SaverDef.V2
    self._pad_step_number = pad_step_number
    self._filename = filename
    self._last_checkpoints = []
    self._checkpoints_to_be_deleted = []
    self._exempt_checkpoint_paths = set(exempt_checkpoint_paths or [])
    self._model_dir = model_dir
    self._skip_save = skip_save
    if tf.executing_eagerly():
      self._next_checkpoint_time = (time.time() +
                                    self._keep_checkpoint_every_n_hours * 3600)
    elif not defer_build:
      self.build()
    if self.saver_def:
      self._check_saver_def()
      self._write_version = self.saver_def.version
    self._save_relative_paths = save_relative_paths
    # For compatibility with object-based checkpoints, we may build a second
    # Saver to read the renamed keys.
    self._object_restore_saver = None

    self._ps_monitor = ps_monitor

  def build(self):
    if tf.executing_eagerly():
      raise RuntimeError("Use save/restore instead of build in eager mode.")
    self._build(self._filename, build_save=True, build_restore=True)

  def _build_eager(self, checkpoint_path, build_save, build_restore):
    self._build(checkpoint_path,
                build_save=build_save,
                build_restore=build_restore)

  def _build(self, checkpoint_path, build_save, build_restore):
    """Builds saver_def."""
    if not tf.executing_eagerly():
      if self._is_built:
        return
      self._is_built = True

    if not self.saver_def or tf.executing_eagerly():
      if self._builder is None:
        self._builder = SaverBuilder(self._write_version)

      if self._var_list is None:
        # pylint: disable=protected-access
        self._var_list = variables._all_saveable_objects()
      if not self._var_list:
        if self._allow_empty:
          self._is_empty = True
          return
        else:
          raise ValueError("No variables to save")
      self._is_empty = False

      self.saver_def = self._builder._build_internal(  # pylint: disable=protected-access
          self._var_list,
          reshape=self._reshape,
          sharded=self._sharded,
          max_to_keep=self._max_to_keep,
          keep_checkpoint_every_n_hours=self._keep_checkpoint_every_n_hours,
          name=self._name,
          restore_sequentially=self._restore_sequentially,
          filename=checkpoint_path,
          build_save=build_save,
          build_restore=build_restore)
    elif self.saver_def and self._name:
      # Since self._name is used as a name_scope by builder(), we are
      # overloading the use of this field to represent the "import_scope" as
      # well.
      self.saver_def.filename_tensor_name = ops.prepend_name_scope(
          self.saver_def.filename_tensor_name, self._name)
      self.saver_def.save_tensor_name = ops.prepend_name_scope(
          self.saver_def.save_tensor_name, self._name)
      self.saver_def.restore_op_name = ops.prepend_name_scope(
          self.saver_def.restore_op_name, self._name)

    self._check_saver_def()
    if not tf.executing_eagerly():
      # Updates next checkpoint time.
      # Set in __init__ when executing eagerly.
      self._next_checkpoint_time = (
          time.time() + self.saver_def.keep_checkpoint_every_n_hours * 3600)

  def _check_saver_def(self):
    if not isinstance(self.saver_def, saver_pb2.SaverDef):
      raise ValueError("saver_def must be a saver_pb2.SaverDef: %s" %
                       self.saver_def)
    if not tf.executing_eagerly():
      if not self.saver_def.save_tensor_name:
        raise ValueError("saver_def must specify the save_tensor_name: %s" %
                         str(self.saver_def))
      if not self.saver_def.restore_op_name:
        raise ValueError("saver_def must specify the restore_op_name: %s" %
                         str(self.saver_def))

  def _CheckpointFilename(self, p):
    """Returns the checkpoint filename given a `(filename, time)` pair.

    Args:
      p: (filename, time) pair.

    Returns:
      Checkpoint file name.
    """
    name, _ = p
    return name

  def _RecordLastCheckpoint(self, latest_save_path):
    """Manages the list of the latest checkpoints."""
    if not self.saver_def.max_to_keep:
      return
    # Remove first from list if the same name was used before.
    for p in self._last_checkpoints:
      if latest_save_path == self._CheckpointFilename(p):
        self._last_checkpoints.remove(p)
    # Append new path to list
    self._last_checkpoints.append((latest_save_path, time.time()))

    # If more than max_to_keep, remove oldest but exempt checkpoint.
    last_checkpoint_paths = set([
        os.path.basename(self._CheckpointFilename(p))
        for p in self._last_checkpoints
    ])
    exempt_checkpoint_paths = last_checkpoint_paths & self.exempt_checkpoint_paths
    if len(self._last_checkpoints) - len(
        exempt_checkpoint_paths) > self.saver_def.max_to_keep:
      for p in self._last_checkpoints:
        filename = os.path.basename(self._CheckpointFilename(p))
        if filename not in self.exempt_checkpoint_paths:
          self._checkpoints_to_be_deleted.append(p)
          self._last_checkpoints.remove(p)
          break

  def _MaybeDeleteOldCheckpoints(self, meta_graph_suffix="meta"):
    """Deletes old checkpoints if necessary.

    `self._checkpoints_to_be_deleted` is going to contain checkpoints that are
    over `max_to_keep`.  They are going to be deleted.  If
    `keep_checkpoint_every_n_hours` was specified, keep an additional checkpoint
    every `N` hours. For example, if `N` is 0.5, an additional checkpoint is
    kept for every 0.5 hours of training; if `N` is 10, an additional
    checkpoint is kept for every 10 hours of training.

    Args:
      meta_graph_suffix: Suffix for `MetaGraphDef` file. Defaults to 'meta'.
    """
    if self._checkpoints_to_be_deleted:
      p = self._checkpoints_to_be_deleted.pop(0)
      # Do not delete the file if we keep_checkpoint_every_n_hours is set and we
      # have reached N hours of training.
      should_keep = p[1] > self._next_checkpoint_time
      if should_keep:
        self._next_checkpoint_time += (
            self.saver_def.keep_checkpoint_every_n_hours * 3600)
        return

      # Otherwise delete the files.
      logging.info("Deleted checkpoint: %s.", self._CheckpointFilename(p))
      for pathname in tf.io.gfile.glob(self._CheckpointFilename(p) + ".*"):
        try:
          tf.io.gfile.rmtree(pathname)
        except tf.errors.NotFoundError:
          logging.warning(
              "Hit NotFoundError when deleting '%s', possibly because another "
              "process/thread is also deleting/moving the same file", pathname)

  def as_saver_def(self):
    """Generates a `SaverDef` representation of this saver.

    Returns:
      A `SaverDef` proto.
    """
    return self.saver_def

  @property
  def exempt_checkpoint_paths(self):
    if self._model_dir:
      monolith_ckpt_state = get_monolith_checkpoint_state(
          self._model_dir, remove_invalid_path=True)
      if monolith_ckpt_state and monolith_ckpt_state.exempt_model_checkpoint_paths:
        exempt_checkpoint_paths = [
            os.path.basename(p)
            for p in monolith_ckpt_state.exempt_model_checkpoint_paths
        ]
        logging.info(
            'New exempt checkpoint paths: {}'.format(exempt_checkpoint_paths))
        self._exempt_checkpoint_paths = set(exempt_checkpoint_paths or [])
      else:
        logging.info("Get exempt checkpoint paths null")
    return self._exempt_checkpoint_paths

  @property
  def last_checkpoints(self):
    """List of not-yet-deleted checkpoint filenames.

    You can pass any of the returned values to `restore()`.

    Returns:
      A list of checkpoint filenames, sorted from oldest to newest.
    """
    return list(self._CheckpointFilename(p) for p in self._last_checkpoints)

  def set_last_checkpoints_with_time(self, last_checkpoints_with_time):
    """Sets the list of old checkpoint filenames and timestamps.

    Args:
      last_checkpoints_with_time: A list of tuples of checkpoint filenames and
        timestamps.

    Raises:
      AssertionError: If last_checkpoints_with_time is not a list.
    """
    assert isinstance(last_checkpoints_with_time, list)
    self._last_checkpoints = last_checkpoints_with_time

  def recover_last_checkpoints(self, checkpoint_paths):
    """Recovers the internal saver state after a crash.

    This method is useful for recovering the "self._last_checkpoints" state.

    Globs for the checkpoints pointed to by `checkpoint_paths`.  If the files
    exist, use their mtime as the checkpoint timestamp.

    Args:
      checkpoint_paths: a list of checkpoint paths.
    """
    checkpoints_with_mtimes = []
    for checkpoint_path in checkpoint_paths:
      try:
        mtime = checkpoint_management.get_checkpoint_mtimes([checkpoint_path])
      except tf.errors.NotFoundError:
        # It's fine if some other thread/process is deleting some older
        # checkpoint concurrently.
        continue
      if mtime:
        checkpoints_with_mtimes.append((checkpoint_path, mtime[0]))
    self.set_last_checkpoints_with_time(checkpoints_with_mtimes)
    logging.info("Recover last checkpoints result: {}".format(
        self.last_checkpoints))

  def save(self,
           sess,
           save_path,
           global_step=None,
           latest_filename=None,
           meta_graph_suffix="meta",
           write_meta_graph=True,
           write_state=True,
           strip_default_attrs=False,
           save_debug_info=False):
    # pylint: disable=line-too-long
    """Saves variables.

    This method runs the ops added by the constructor for saving variables.
    It requires a session in which the graph was launched.  The variables to
    save must also have been initialized.

    The method returns the path prefix of the newly created checkpoint files.
    This string can be passed directly to a call to `restore()`.

    Args:
      sess: A Session to use to save the variables.
      save_path: String.  Prefix of filenames created for the checkpoint.
      global_step: If provided the global step number is appended to `save_path`
        to create the checkpoint filenames. The optional argument can be a
        `Tensor`, a `Tensor` name or an integer.
      latest_filename: Optional name for the protocol buffer file that will
        contains the list of most recent checkpoints.  That file, kept in the
        same directory as the checkpoint files, is automatically managed by the
        saver to keep track of recent checkpoints.  Defaults to 'checkpoint'.
      meta_graph_suffix: Suffix for `MetaGraphDef` file. Defaults to 'meta'.
      write_meta_graph: `Boolean` indicating whether or not to write the meta
        graph file.
      write_state: `Boolean` indicating whether or not to write the
        `CheckpointStateProto`.
      strip_default_attrs: Boolean. If `True`, default-valued attributes will be
        removed from the NodeDefs. For a detailed guide, see
        [Stripping Default-Valued
          Attributes](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md#stripping-default-valued-attributes).
      save_debug_info: If `True`, save the GraphDebugInfo to a separate file,
        which in the same directory of save_path and with `_debug` added before
        the file extension. This is only enabled when `write_meta_graph` is
        `True`

    Returns:
      A string: path prefix used for the checkpoint files.  If the saver is
        sharded, this string ends with: '-?????-of-nnnnn' where 'nnnnn'
        is the number of shards created.
      If the saver is empty, returns None.

    Raises:
      TypeError: If `sess` is not a `Session`.
      ValueError: If `latest_filename` contains path components, or if it
        collides with `save_path`.
      RuntimeError: If save and restore ops weren't built.
    """
    # pylint: enable=line-too-long
    if not self._is_built and not tf.executing_eagerly():
      raise RuntimeError(
          "`build()` should be called before save if defer_build==True")
    if self._skip_save:
      return None
    if latest_filename is None:
      latest_filename = "checkpoint"

    if os.path.split(latest_filename)[0]:
      raise ValueError("'latest_filename' must not contain path components")

    save_path = tf.compat.as_str(save_path)
    if global_step is not None:
      if not isinstance(global_step, tf.compat.integral_types):
        global_step = tf.compat.v1.train.global_step(sess, global_step)
      checkpoint_file = "%s-%d" % (save_path, global_step)
      if self._pad_step_number:
        # Zero-pads the step numbers, so that they are sorted when listed.
        checkpoint_file = "%s-%s" % (save_path, "{:08d}".format(global_step))
    else:
      checkpoint_file = save_path
      if os.path.basename(save_path) == latest_filename and not self._sharded:
        # Guard against collision between data file and checkpoint state file.
        raise ValueError(
            "'latest_filename' collides with 'save_path': '%s' and '%s'" %
            (latest_filename, save_path))

    if (not tf.executing_eagerly() and
        not isinstance(sess, session.SessionInterface)):
      raise TypeError("'sess' must be a Session; %s" % sess)

    save_path_parent = os.path.dirname(save_path)
    if not self._is_empty:
      try:
        if tf.executing_eagerly():
          self._build_eager(checkpoint_file,
                            build_save=True,
                            build_restore=False)
          model_checkpoint_path = self.saver_def.save_tensor_name
        else:
          model_checkpoint_path = sess.run(
              self.saver_def.save_tensor_name,
              {self.saver_def.filename_tensor_name: checkpoint_file})

        model_checkpoint_path = tf.compat.as_str(model_checkpoint_path)
        if write_state:
          self._RecordLastCheckpoint(model_checkpoint_path)
          checkpoint_management.update_checkpoint_state_internal(
              save_dir=save_path_parent,
              model_checkpoint_path=model_checkpoint_path,
              all_model_checkpoint_paths=self.last_checkpoints,
              latest_filename=latest_filename,
              save_relative_paths=self._save_relative_paths)
          self._MaybeDeleteOldCheckpoints(meta_graph_suffix=meta_graph_suffix)
      except (tf.errors.FailedPreconditionError,
              tf.errors.NotFoundError) as exc:
        if not tf.io.gfile.isdir(save_path_parent):
          exc = ValueError(
              "Parent directory of {} doesn't exist, can't save.".format(
                  save_path))
        raise exc

    if write_meta_graph:
      meta_graph_filename = checkpoint_management.meta_graph_filename(
          checkpoint_file, meta_graph_suffix=meta_graph_suffix)
      if not tf.executing_eagerly():
        with sess.graph.as_default():
          self.export_meta_graph(meta_graph_filename,
                                 strip_default_attrs=strip_default_attrs,
                                 save_debug_info=save_debug_info)

    if self._is_empty:
      return None
    else:
      return model_checkpoint_path

  def export_meta_graph(self,
                        filename=None,
                        collection_list=None,
                        as_text=False,
                        export_scope=None,
                        clear_devices=False,
                        clear_extraneous_savers=False,
                        strip_default_attrs=False,
                        save_debug_info=False):
    # pylint: disable=line-too-long
    """Writes `MetaGraphDef` to save_path/filename.

    Args:
      filename: Optional meta_graph filename including the path.
      collection_list: List of string keys to collect.
      as_text: If `True`, writes the meta_graph as an ASCII proto.
      export_scope: Optional `string`. Name scope to remove.
      clear_devices: Whether or not to clear the device field for an `Operation`
        or `Tensor` during export.
      clear_extraneous_savers: Remove any Saver-related information from the
        graph (both Save/Restore ops and SaverDefs) that are not associated with
        this Saver.
      strip_default_attrs: Boolean. If `True`, default-valued attributes will be
        removed from the NodeDefs. For a detailed guide, see
        [Stripping Default-Valued
          Attributes](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md#stripping-default-valued-attributes).
      save_debug_info: If `True`, save the GraphDebugInfo to a separate file,
        which in the same directory of filename and with `_debug` added before
        the file extension.

    Returns:
      A `MetaGraphDef` proto.
    """
    # pylint: enable=line-too-long
    return tf_saver.export_meta_graph(
        filename=filename,
        graph_def=tf.compat.v1.get_default_graph().as_graph_def(
            add_shapes=True),
        saver_def=self.saver_def,
        collection_list=collection_list,
        as_text=as_text,
        export_scope=export_scope,
        clear_devices=clear_devices,
        clear_extraneous_savers=clear_extraneous_savers,
        strip_default_attrs=strip_default_attrs,
        save_debug_info=save_debug_info)

  def _origin_restore(self, sess, save_path):
    """Restores previously saved variables.

    This method runs the ops added by the constructor for restoring variables.
    It requires a session in which the graph was launched.  The variables to
    restore do not have to have been initialized, as restoring is itself a way
    to initialize variables.

    The `save_path` argument is typically a value previously returned from a
    `save()` call, or a call to `latest_checkpoint()`.

    Args:
      sess: A `Session` to use to restore the parameters. None in eager mode.
      save_path: Path where parameters were previously saved.

    Raises:
      ValueError: If save_path is None or not a valid checkpoint.
    """
    if self._is_empty:
      return
    if save_path is None:
      raise ValueError("Can't load save_path when it is None.")

    checkpoint_prefix = tf.compat.as_text(save_path)
    if not checkpoint_management.checkpoint_exists_internal(checkpoint_prefix):
      raise ValueError("The passed save_path is not a valid checkpoint: " +
                       checkpoint_prefix)

    logging.info("Restoring parameters from %s", checkpoint_prefix)
    try:
      if tf.executing_eagerly():
        self._build_eager(save_path, build_save=False, build_restore=True)
      else:
        # At some local case, restore_ops_per_device is empty.
        if not self._builder.restore_ops_per_device:
          sess.run(self.saver_def.restore_op_name,
                   {self.saver_def.filename_tensor_name: save_path})
        else:
          restore_ops = []
          for device, restore_op in self._builder.restore_ops_per_device.items(
          ):
            if not self._ps_monitor or self._ps_monitor.is_ps_uninitialized(
                sess, device):
              restore_ops.append(restore_op)
          sess.run(restore_ops,
                   {self.saver_def.filename_tensor_name: save_path})

    except tf.errors.NotFoundError as err:
      # There are three common conditions that might cause this error:
      # 0. The file is missing. We ignore here, as this is checked above.
      # 1. This is an object-based checkpoint trying name-based loading.
      # 2. The graph has been altered and a variable or other name is missing.

      # 1. The checkpoint would not be loaded successfully as is. Try to parse
      # it as an object-based checkpoint.
      try:
        names_to_keys = tf_saver.object_graph_key_mapping(save_path)
      except tf.errors.NotFoundError:
        # 2. This is not an object-based checkpoint, which likely means there
        # is a graph mismatch. Re-raise the original error with
        # a helpful message (b/110263146)
        raise tf_saver._wrap_restore_error_with_msg(
            err, "a Variable name or other graph key that is missing")

      # This is an object-based checkpoint. We'll print a warning and then do
      # the restore.
      logging.warning(
          "Restoring an object-based checkpoint using a name-based saver. This "
          "may be somewhat fragile, and will re-build the Saver. Instead, "
          "consider loading object-based checkpoints using "
          "tf.train.Checkpoint().")
      self._object_restore_saver = tf_saver.saver_from_object_based_checkpoint(
          checkpoint_path=save_path,
          var_list=self._var_list,
          builder=self._builder,
          names_to_keys=names_to_keys,
          cached_saver=self._object_restore_saver)
      self._object_restore_saver.restore(sess=sess, save_path=save_path)
    except tf.errors.InvalidArgumentError as err:
      # There is a mismatch between the graph and the checkpoint being loaded.
      # We add a more reasonable error message here to help users (b/110263146)
      raise tf_saver._wrap_restore_error_with_msg(
          err, "a mismatch between the current graph and the graph")

  def restore(self, sess, save_path):
    if is_exporting() or sess is None:
      logging.info('is_exporting or sess is None, fall back to origin_restore')
      return self._origin_restore(sess, save_path)

    checkpoint_state = None
    logging.info(f"save_path is {save_path}")
    model_dir = os.path.dirname(save_path)
    try:
      checkpoint_state = tf.train.get_checkpoint_state(checkpoint_dir=model_dir)
    except Exception as e:
      logging.info(
          f'no checkpoint file in {model_dir}, fall back to origin_restore')
      return self._origin_restore(sess, save_path)
    if checkpoint_state is None:
      logging.info(f'checkpoint_state is None, fall back to origin_restore')
      return self._origin_restore(sess, save_path)

    graph: tf.Graph = None
    try:
      graph: tf.Graph = sess.graph
    except Exception as e:
      logging.info("the eager mode has no attribute graph")
      return self._origin_restore(sess, save_path)

    if not graph:
      logging.info("graph is None, pls. check! fall back to origin_restore")
      return self._origin_restore(sess, save_path)
    init_objs = graph.get_collection(CUSTOM_RESTORE_OP)

    if init_objs:
      init_ops, placeholders, alias_map = init_objs[0]
      if alias_map:  # alias_map: new -> old
        ckpt: CheckpointReader = NewCheckpointReader(save_path)
        feed_dict = calc_feed_dict(ckpt, alias_map, placeholders)
        if feed_dict:
          sess.run(init_ops, feed_dict=feed_dict)
        else:
          self._origin_restore(sess, save_path)
      elif init_ops:
        assert alias_map is None or len(alias_map) == 0
        restore_dir = os.path.dirname(save_path)
        model_dir = restore_dir
        if hasattr(init_ops[0], 'model_dir'):
          model_dir_tmp = getattr(init_ops[0], 'model_dir')
          if model_dir_tmp:
            model_dir = model_dir_tmp
        flag_file = os.path.join(model_dir, 'clear_nn')
        logging.info(f'the clear nn flag_file is {flag_file}')
        if tf.io.gfile.exists(flag_file):
          logging.info(
              'clear nn flag_file exists, restore from ckpt, do not clear nn')
          self._origin_restore(sess, save_path)
        else:
          if len(init_ops) == 1:
            sess.run(init_ops)
          else:
            init_op, update_gs_op = init_ops
            sess.run(init_op)
            # update global_step to continue training
            ckpt: CheckpointReader = NewCheckpointReader(save_path)
            gs_tensor = ckpt.get_tensor('global_step')
            sess.run(update_gs_op, feed_dict={placeholders[0]: gs_tensor})
            logging.info(
                f'update global_step to continue training, {gs_tensor}')

          logging.info('clear nn by calling init_op other than restore ...')

          with tf.io.gfile.GFile(flag_file, 'w') as ostream:
            ostream.write(file_content='clean nn done!')
      else:
        self._origin_restore(sess, save_path)

      with graph._lock:
        if CUSTOM_RESTORE_OP in graph._collections:
          del graph._collections[CUSTOM_RESTORE_OP]
    else:
      self._origin_restore(sess, save_path)

  def setup_ps_initialized_state(self, sess):
    if self._ps_monitor:
      self._ps_monitor.setup_ps_initialized_state(sess)
