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
from monolith.native_training.data.datasets import DynamicMatchingFilesDataset, PBDataset, PbType
from idl.matrix.proto.example_pb2 import Example, ExampleBatch
import json
import os
# import pyarrow.parquet as pq

NUM_WORKER = 3


class DataServiceTest2(tf.test.TestCase):

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

  def testDataServiceWithParquetDataset(self):
    # check file exist, if not exist, return test pass
    if os.environ.get("META_JSON_PATH"):
      meta_json_path = os.environ.get("META_JSON_PATH")
    else:
      meta_json_path = os.path.join(os.environ.get("HOME"), "temp", "fountain_meta.json")
    if os.environ.get("PARQUET_DIR"):
      parquet_dir = os.environ.get("PARQUET_DIR")
    else:
      parquet_dir =  os.path.join(os.environ.get("HOME"), "temp", "parquet_files")
    path_pattern = os.path.join(parquet_dir, "*")
    if not os.path.exists(meta_json_path) or not os.path.exists(parquet_dir):
      logging.warning("meta_json_path or path_pattern not exist, pls check.") 

    # calc col_name and col_type
    with open(meta_json_path, "r") as f:
      json_data = f.read()
    j = json.loads(json_data)["default_0"]
    all_cols = [k for k in j.keys()]
    col_type_dict = {2: "fid_v2", 3: "float", 5: "int"}
    all_cols_type = [col_type_dict.get(j[k]["data_type"][0], "invalid") for k in all_cols]

    # create, register dataset
    dataset = PBDataset(patterns=[path_pattern],
                        use_data_service=True,
                        use_parquet=True,
                        output_pb_type=PbType.PLAINTEXT,
                        select_columns=all_cols,
                        select_columns_type=all_cols_type,
                        batch_size=1024,
                        cycle_length=1,
                        num_parallel_calls=1)
    dataset_id = dsvc.register_dataset(self.target, dataset)

    comsumer1 = dsvc.from_dataset_id(processing_mode="distributed_epoch",
                                     service=self.target,
                                     dataset_id=dataset_id,
                                     job_name="test_data_service_with_parquet_dataset",
                                     element_spec=dataset.element_spec,
                                     max_outstanding_requests=1)
    comsumer2 = dsvc.from_dataset_id(processing_mode="distributed_epoch",
                                     service=self.target,
                                     dataset_id=dataset_id,
                                     job_name="test_data_service_with_parquet_dataset",
                                     element_spec=dataset.element_spec,
                                     max_outstanding_requests=1)

    # read data from data service and calculate sum of batch_size
    comsumers = [iter(comsumer1), iter(comsumer2)]
    idx, row_read = 0, 0
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
        example_batch = ExampleBatch()
        example_batch.ParseFromString(element.numpy())
        row_read += example_batch.batch_size
        print(f'read batch_size from comsumer{i} is {example_batch.batch_size}', flush=True)
      except StopIteration as e:
        comsumers[i] = None
        del comsumer

    # use py_parquet api calculate sum of rows of all files
    # total_row = 0
    # for pf in os.listdir(parquet_dir):
    #   file_full_path = os.path.join(parquet_dir, pf)
    #   parquet_file = pq.ParquetFile(file_full_path)
    #   total_row += parquet_file.metadata.num_rows

    # self.assertEqual(row_read, total_row)
    print(f"{row_read} rows read")
    

if __name__ == "__main__":
  # tf.compat.v1.disable_eager_execution()
  tf.test.main()
