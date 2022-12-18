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

from os import path
import subprocess

from absl import logging
from google.cloud import storage

import tensorflow.compat.v1 as tf

_GS_PREFIX = "gs://"
_CORE_NUMBER_PER_HOST = 8

_DATE_FORMAT_LEN = 8
_MIN_DATE = "00000000"
_MAX_DATE = "99999999"


def get_bucket_name_and_relavite_path(gs_file_path):
  """ Given gs file path, return gs bucket name and relavite gs path (not include gs bucket)."""

  assert gs_file_path.find(_GS_PREFIX) != -1, "File name: {}".format(
      gs_file_path)

  bucket_name_start = len(_GS_PREFIX)
  bucket_name_end = gs_file_path.find("/", bucket_name_start)
  bucket_name = gs_file_path[bucket_name_start:bucket_name_end]
  relavite_blob_path = gs_file_path[bucket_name_end + 1:]
  return bucket_name, relavite_blob_path


def download_gcs_file(gs_file_path, local_file_name):
  """ Download gs file to local disk by giving gs path and local file name."""

  logging.info("Start downloading {} => {} ...".format(gs_file_path,
                                                       local_file_name))
  bucket_name, relavite_blob_path = get_bucket_name_and_relavite_path(
      gs_file_path)
  download_gcs_file_with_relative_path(bucket_name, relavite_blob_path,
                                       local_file_name)


def download_gcs_file_with_relative_path(bucket_name, gs_file_relative_path,
                                         local_file_name):
  """ Download gs file to local disk by giving gs relavite path and local file name."""

  storage_client = storage.Client()
  bucket = storage_client.bucket(bucket_name)
  blob = bucket.blob(gs_file_relative_path)
  blob.download_to_filename(local_file_name)


def list_gcs_files_with_prefix(gs_path_prefix):
  """ Given a gs path prefix, return the gs bucket name and relavite paths (not include gs bucket) for all matching blobs."""

  storage_client = storage.Client()
  bucket_name, relavite_blob_path_prefix = get_bucket_name_and_relavite_path(
      gs_path_prefix)
  blob_relative_paths = storage_client.list_blobs(
      bucket_name, prefix=relavite_blob_path_prefix)

  return bucket_name, blob_relative_paths


def parse_example_number_meta_file(meta_file, seperator):
  """Parse the meta file which contains file name and its tr record number."""

  file_index = 0
  file_example_number_list = []
  with open(meta_file) as f:
    previous_file_name = ""
    lines = f.readlines()
    for line in lines:
      if line.find(",") == -1:
        continue
      split_str = line.split(",")
      file_name = split_str[0]
      assert previous_file_name < file_name, "File name must be in dictionary ascending order. Previous file name: {}, current file file name: {}".format(
          previous_file_name, file_name)
      previous_file_name = file_name
      count = int(split_str[1])
      file_example_number_list.append((file_name, count))
  return file_example_number_list


def calculate_shard_skip_file_number(file_example_number, shard_num,
                                     completed_steps_number,
                                     batch_size_per_core):
  """Calculate for each shard (host), how many files it has completed processing from last check point."""

  processed_example_number_per_host = batch_size_per_core * completed_steps_number * _CORE_NUMBER_PER_HOST
  shard_index = 0
  # Keep number of completed files for each shard (host) in last checkpoint.
  shard_skip_file_number = [0] * shard_num
  # Keep number of completed examples for each shard (host) in last checkpoint.
  shard_accumulated_example_count = [0] * shard_num
  for example_number in file_example_number:
    if example_number + shard_accumulated_example_count[
        shard_index] <= processed_example_number_per_host:
      shard_accumulated_example_count[shard_index] += example_number
      shard_skip_file_number[shard_index] += 1

    shard_index = (shard_index + 1) % shard_num

  return shard_skip_file_number


def get_checkpoint_completed_step_number(checkpoint_path):
  """Get the completed steps number in the latest checkpoint under checkpoint path."""

  completed_steps_number = 0
  bucket_name, blob_relative_paths = list_gcs_files_with_prefix(
      path.join(checkpoint_path, "model.ckpt"))
  for blob in blob_relative_paths:
    blob_relative_path = blob.name
    if blob_relative_path.endswith(".meta") == False:
      continue

    blob_name = blob_relative_path[blob_relative_path.rfind("/") + 1:]
    logging.info("Found checkpoint file {} under path {}".format(
        blob_name, checkpoint_path))
    checkpoint_processed_steps = int(blob_name[blob_name.find("-") +
                                               1:blob_name.rfind(".meta")])
    completed_steps_number = max(completed_steps_number,
                                 checkpoint_processed_steps)

  return completed_steps_number


def update_params(params, tpu_cluster_resolver):
  shard_num = tpu_cluster_resolver.cluster_spec().num_tasks("worker")

  assert ("batch_size_per_core" in params and params["batch_size_per_core"] is not None) \
   or ("global_batch_size" in params and params["global_batch_size"] is not None), \
      "batch_size_per_core and global_batch_size can't be both None."

  if "batch_size_per_core" not in params or params[
      "batch_size_per_core"] is None:
    params["batch_size_per_core"] = params[
        "global_batch_size"] / shard_num / _CORE_NUMBER_PER_HOST
  elif "global_batch_size" not in params or params["global_batch_size"] is None:
    params["global_batch_size"] = params[
        "batch_size_per_core"] * shard_num * _CORE_NUMBER_PER_HOST
  else:
    assert params["batch_size_per_core"] * shard_num * _CORE_NUMBER_PER_HOST == params["global_batch_size"], \
       "Batch size per core: {} and global batch size:{} doesn't align.".format(params["batch_size_per_core"], params["global_batch_size"])

  logging.info("Batch size per core: {}, global batch size: {}".format(
      params["batch_size_per_core"], params["global_batch_size"]))

  # Get the completed steps number from the latest checkpoint.
  completed_step_number = get_checkpoint_completed_step_number(
      params["model_dir"])
  logging.info(
      "Completed steps from last checkpoint: {}".format(completed_step_number))
  if completed_step_number > 0:
    file_example_number = get_per_file_example_numbers_for_checkpoint_reload(
        params["train_dataset_path"], params["gcs_file_example_number"], ",")

    shard_skip_file_number = calculate_shard_skip_file_number(
        file_example_number, shard_num, completed_step_number,
        params["batch_size_per_core"])
    params["shard_skip_file_number"] = shard_skip_file_number
    logging.info(
        "Set shard skip file number, shard number: {}, batch size per core: {}, completed steps of last chckpoint: {}, \
            processed file number of each shard: {}".format(
            shard_num, params["batch_size_per_core"], completed_step_number,
            shard_skip_file_number))


def get_per_file_example_numbers_for_checkpoint_reload(
    train_dataset_path, file_example_number_meta,
    file_example_number_meta_seperator):
  # Firstly, we need verify whether checkpoint can be reloaded. To reload checkpoint, we need make sure
  # the training data is contained as continous subset as in file example number meta.
  # Currently only gsutil supports query gcs with regex path. Google storage client looks like does't support regex.
  # We will use gsutil tool directly here as a workaround to work with regex path. Later we will switch back
  # to google storage client once it supports regex path.
  logging.info("Querying train data set to validate checkpoint reload...")
  proc = subprocess.Popen(["gsutil", "ls", train_dataset_path],
                          stdout=subprocess.PIPE)

  train_file_path_list = []
  previous_relative_path = ""
  while True:
    line = proc.stdout.readline()
    if not line:
      break

    train_file_path = line.decode("utf-8").strip()
    bucket_name, relative_path = get_bucket_name_and_relavite_path(
        train_file_path)
    train_file_path_list.append(relative_path)
    assert previous_relative_path < relative_path, "train file path must be in ascend order. \
            previous file path: {}, current file path: {}".format(
        previous_relative_path, relative_path)
    previous_relative_path = relative_path

  file_example_number_list = parse_example_number_meta_file(
      file_example_number_meta, file_example_number_meta_seperator)

  # Skip the files which are not in trained data set.
  assert len(
      train_file_path_list) > 0, "Train data set size must be greater than 0."

  # Find the first train file in file example meta
  for file_example_number_index, file_example_number in enumerate(
      file_example_number_list):
    file_path = file_example_number[0]
    count = file_example_number[1]
    if train_file_path_list[0] <= file_path:
      break

  assert len(train_file_path_list) <= len(file_example_number_list) - file_example_number_index, \
      "Train file path list length {} can't be greater than the remaining length of file example number list length {} starting at index {}".format(len(train_file_path_list), len(file_example_number_list) - file_example_number_index,
      file_example_number_index)

  example_number_list = []
  for train_file_index in range(0, len(train_file_path_list)):
    assert train_file_path_list[train_file_index] == file_example_number_list[file_example_number_index][0], \
        "File {} in train data can not be found in file example meta {}".format(train_file_path_list[train_file_index],
         file_example_number_meta)
    example_number_list.append(
        file_example_number_list[file_example_number_index][1])
    file_example_number_index += 1

  logging.info("Checkpoint reload verification done.")
  return example_number_list


def range_dateset(dataset: tf.data.Dataset,
                  root_path: str,
                  start_date: str = None,
                  end_date: str = None):

  if start_date is None:
    start_date = _MIN_DATE

  if end_date is None:
    end_date = _MAX_DATE

  logging.info("start_date: {}, end_date: {}.".format(start_date, end_date))

  def filter_fn(x):
    path_prefix_len = len(root_path)
    return tf.math.logical_and(
        tf.math.greater_equal(
            tf.strings.to_number(tf.strings.substr(x, path_prefix_len,
                                                   _DATE_FORMAT_LEN),
                                 out_type=tf.int32), int(start_date)),
        tf.math.less_equal(
            tf.strings.to_number(tf.strings.substr(x, path_prefix_len,
                                                   _DATE_FORMAT_LEN),
                                 out_type=tf.int32), int(end_date)),
    )

  return dataset.filter(filter_fn)
