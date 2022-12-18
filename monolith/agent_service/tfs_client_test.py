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

from absl import logging, app, flags
import json
import os
import tempfile
import unittest

from monolith.agent_service.tfs_client import get_instance_proto
from monolith.agent_service.tfs_client import get_example_batch_to_instance

FLAGS = flags.FLAGS


class TFSClientTest(unittest.TestCase):

  def test_get_instance_proto(self):
    tensor_proto = get_instance_proto()
    self.assertEqual(tensor_proto.dtype, 7)
    self.assertEqual(tensor_proto.tensor_shape.dim[0].size, 256)

  def test_get_example_batch_to_instance_from_pb(self):
    file_name = "monolith/native_training/data/training_instance/examplebatch.data"
    FLAGS.lagrangex_header = True
    get_example_batch_to_instance(file_name, 'pb')

  def test_get_example_batch_to_instance_from_pbtxt(self):
    file_name = "monolith/agent_service/example_batch.pbtxt"
    FLAGS.lagrangex_header = True
    get_example_batch_to_instance(file_name, 'pbtxt')


def main(_):
  unittest.main()


if __name__ == "__main__":
  app.run(main)
