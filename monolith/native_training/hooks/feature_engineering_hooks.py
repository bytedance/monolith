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
import uuid as gen_uid
import os
from struct import pack
from absl import logging

from idl.matrix.proto.example_pb2 import ExampleBatch, FeatureListType


class FeatureEngineeringSaveHook(tf.estimator.SessionRunHook):

  def __init__(self, config, nxt_elem, cap=100):
    self._config = config
    self._nxt_elem = nxt_elem
    self._cap = cap

  def begin(self):
    self._batch_list = []  # List[Dict[str, tf.Tensor]]
    self._steps = 0

  def before_run(self, run_context):
    self._steps += 1
    # skip iter init
    if self._steps > 1:
      return tf.compat.v1.train.SessionRunArgs(self._nxt_elem)

  def _save_features(self):
    base_dir = os.path.join(self._config.model_dir, "features")
    try:
      tf.io.gfile.makedirs(base_dir)
    except tf.errors.OpError:
      pass
    file_path = ""
    if self._config.server_type == "worker" and self._config.index == 0:
      file_path = os.path.join(base_dir,
                               "chief_" + str(gen_uid.uuid1()) + ".pb")
    else:
      file_path = os.path.join(
          base_dir, "worker" + str(self._config.index) + "_" +
          str(gen_uid.uuid1()) + ".pb")

    results = []
    for batch in self._batch_list:
      # batch to ExampleBatch
      example_batch = ExampleBatch()
      for k, v in batch.items():
        named_feature_list = example_batch.named_feature_list.add()
        named_feature_list.name = k
        named_feature_list.type = FeatureListType.INDIVIDUAL

        if isinstance(v, tf.compat.v1.ragged.RaggedTensorValue):
          lv = v.to_list()
        else:  # np.ndarray
          lv = v.tolist()

        for fids in lv:
          feature = named_feature_list.feature.add()
          if len(fids) > 0 and isinstance(fids[0], float):
            feature.float_list.value.extend(fids)
          else:
            feature.fid_v2_list.value.extend(fids)

        example_batch.batch_size = len(lv)
      results.append(example_batch)

    with tf.io.gfile.GFile(file_path, "w") as f:
      for example_batch in results:
        ss = example_batch.SerializeToString()
        sz = len(ss)
        f.write(pack('<Q', 0))  # lagrange header
        f.write(pack('<Q', sz))
        f.write(ss)
    logging.info("save to %s", file_path)

  def after_run(self, run_context, run_values):
    if self._steps > 1:
      self._batch_list.append(run_values.results)
      if len(self._batch_list) >= self._cap:
        self._save_features()
        self._batch_list.clear()

  def end(self, session):
    if len(self._batch_list) >= 0:
      self._save_features()
      self._batch_list.clear()
