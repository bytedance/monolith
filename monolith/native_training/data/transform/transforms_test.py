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

from absl import app, logging
from monolith.native_training.data.transform import transforms


class TransformsTest(unittest.TestCase):

  def test_filter_by_fid(self):
    proto = transforms.FilterByFid(has_fids=[1],
                                   filter_fids=[2, 3],
                                   select_fids=None).as_proto()
    logging.info(proto)

  def test_filter_by_action(self):
    proto = transforms.FilterByAction(has_actions=[4]).as_proto()
    logging.info(proto)

  def test_filter_by_label(self):
    proto = transforms.FilterByLabel(thresholds=[-100, -100]).as_proto()
    logging.info(proto)

  def test_add_label(self):
    proto = transforms.AddLabel(config='1,2:3:1.0;4::0.5',
                                negative_value=0.0,
                                new_sample_rate=0.3).as_proto()
    logging.info(proto)

  def test_compose(self):
    transform = transforms.Compose([
        transforms.FilterByFid(has_fids=[1],
                               filter_fids=[2, 3],
                               select_fids=None),
        transforms.FilterByLabel(thresholds=[-100, -100]),
        transforms.AddLabel(config='1,2:3:1.0;4::0.5',
                            negative_value=0.0,
                            new_sample_rate=0.3)
    ])
    logging.info(transform.as_proto())


def main(_):
  logging.set_verbosity(logging.INFO)
  unittest.main()


if __name__ == '__main__':
  app.run(main)
