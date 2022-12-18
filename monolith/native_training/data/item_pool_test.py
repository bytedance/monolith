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

from absl import logging
import os
import getpass
import tensorflow as tf
from tensorflow.python.framework import load_library

from monolith.utils import get_libops_path
from monolith.native_training.data.feature_utils import create_item_pool, \
  save_item_pool, restore_item_pool, item_pool_random_fill, item_pool_check


class ItemPoolTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.model_path = f"{os.environ.get('HOME')}/{getpass.getuser()}/tmp/monolith/data/test"
    global_step = tf.constant(1, dtype=tf.int64)
    pool = create_item_pool(start_num=20,
                            max_item_num_per_channel=100,
                            shared_name='first')
    pool = item_pool_random_fill(pool)
    pool = save_item_pool(pool,
                          model_path=cls.model_path,
                          global_step=global_step,
                          nshards=2)

  def test_create_item_pool(self):
    global_step = tf.constant(1, dtype=tf.int64)
    pool = create_item_pool(start_num=20,
                            max_item_num_per_channel=100,
                            shared_name='second')
    pool = restore_item_pool(pool,
                             model_path=self.model_path,
                             global_step=global_step,
                             nshards=2)
    pool = item_pool_check(pool,
                           global_step=global_step,
                           model_path=self.model_path,
                           nshards=2)
    logging.info(f"model_path is {self.model_path}")


if __name__ == "__main__":
  tf.test.main()
