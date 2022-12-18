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

from monolith import utils
from typing import Union
import os
import unittest
import uuid

from tensorflow.python.framework import errors
import tensorflow.python.training.monitored_session as monitored_session
import tensorflow as tf

utils.enable_monkey_patch()


class UtilsTest(unittest.TestCase):

  def testFindMain(self):
    basedir = utils.find_main()
    self.assertEqual(basedir.split("/")[-1], "__main__")

  def testGetLibopsPath(self):
    self.assertTrue(
        os.path.exists(utils.get_libops_path("monolith/utils_test.py")))

  def testLoadMonitoredSession(self):
    self.assertEqual(monitored_session._PREEMPTION_ERRORS,
                     (errors.AbortedError,))

  def testMultiThreadedCopy(self):
    test_id = uuid.uuid4().hex

    def _gen_dir():
      root = os.path.join('/tmp', test_id, 'src')
      tf.io.gfile.makedirs(root)
      subdir = os.path.join(root, 'subdir')
      tf.io.gfile.mkdir(subdir)
      with tf.io.gfile.GFile(os.path.join(root, 'file.txt'), 'w+') as f:
        f.write('root')
      with tf.io.gfile.GFile(os.path.join(subdir, 'innerfile.txt'), 'w+') as f:
        f.write('inner')
      return root

    src = _gen_dir()
    dst = os.path.join('/tmp', test_id, 'dst')
    utils.CopyRecursively(src, dst, max_workers=2)
    with tf.io.gfile.GFile(os.path.join(dst, 'subdir', 'innerfile.txt'),
                           'r') as f:
      self.assertEqual(f.read(), 'inner')


if __name__ == "__main__":
  unittest.main()
