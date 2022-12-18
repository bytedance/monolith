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

from absl import flags, logging
from typing import get_type_hints
from enum import Enum
import dataclasses
import threading

import tensorflow as tf
import tensorflow.python.data.experimental.service as dsvc
from monolith.native_training.data.datasets import DynamicMatchingFilesDataset

NUM_WORKER = 3


class DataServiceTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.dispatcher = dsvc.DispatchServer(
        tf.data.experimental.service.DispatcherConfig(port=7080))
    cls.target = cls.dispatcher.target
    dispatcher_address = cls.target.split("://")[1]
    logging.info(f'start dispatcher at {cls.target}')

    cls.workers = []
    for i in range(NUM_WORKER):
      worker = dsvc.WorkerServer(
          dsvc.WorkerConfig(dispatcher_address=dispatcher_address))
      cls.workers.append(worker)
      logging.info(f'start worker {i} at {worker._address}')

  @classmethod
  def tearDownClass(cls):
    cls.target = None

    if cls.dispatcher is not None:
      del cls.dispatcher
      cls.dispatcher = None
      logging.info('del dispatcher done!')

    for i, worker in enumerate(cls.workers):
      del worker
      logging.info(f'del worker {i} done!')
    cls.workers.clear()

  def testSplitProvider(self):
    dataset = DynamicMatchingFilesDataset(patterns=[''])
    dataset_id = dsvc.register_dataset(self.target, dataset)

    comsumer1 = dsvc.from_dataset_id(processing_mode="distributed_epoch",
                                     service=self.target,
                                     dataset_id=dataset_id,
                                     job_name="test_dynamic_match_file_dataset",
                                     element_spec=dataset.element_spec,
                                     max_outstanding_requests=1)
    comsumer2 = dsvc.from_dataset_id(processing_mode="distributed_epoch",
                                     service=self.target,
                                     dataset_id=dataset_id,
                                     job_name="test_dynamic_match_file_dataset",
                                     element_spec=dataset.element_spec,
                                     max_outstanding_requests=1)
    comsumers = [iter(comsumer1), iter(comsumer2)]
    idx, cnt = 0, 0
    while True:
      i = idx % 2
      idx += 1
      comsumer = comsumers[i]
      if comsumer is None:
        if any(c is not None for c in comsumers):
          continue
        else:
          break
      try:
        element = next(comsumer)
        print(f'element for comsumer{i} is ', element, flush=True)
        cnt += 1
      except StopIteration as e:
        comsumers[i] = None
        del comsumer
    self.assertEqual(cnt, 19)


if __name__ == "__main__":
  # tf.compat.v1.disable_eager_execution()
  tf.test.main()
