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

import unittest
from monolith.agent_service.data_def import ModelMeta, ResourceSpec, ReplicaMeta


class DataDefTest(unittest.TestCase):

  def serde(self, item):
    cls = item.__class__
    serialized = item.serialize()
    recom = cls.deserialize(serialized)
    self.assertEqual(item, recom)

  def test_model_info(self):
    obj = ModelMeta(model_name='monolith',
                    num_shard=3,
                    model_dir="/tmp/opt",
                    ckpt='model.ckpt-1234')
    self.serde(obj)

  def test_resource(self):
    obj = ResourceSpec(address="localhost:123",
                       shard_id=10,
                       replica_id=2,
                       memory=12345,
                       cpu=3.5)
    self.serde(obj)

  def test_replica_meta(self):
    obj = ReplicaMeta(address="localhost:123",
                      model_name='monolith',
                      server_type='ps',
                      task=0,
                      replica=0)
    self.serde(obj)


if __name__ == "__main__":
  unittest.main()
