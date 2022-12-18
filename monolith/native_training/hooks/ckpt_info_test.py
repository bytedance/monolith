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

import os

from google.protobuf import text_format
import tensorflow as tf

from monolith.native_training import hash_table_ops
from monolith.native_training.hooks import ckpt_info
from monolith.native_training.proto import ckpt_info_pb2


class FidCountListener(tf.test.TestCase):

  def test_basic(self):
    h = hash_table_ops.test_hash_table(1)
    model_dir = os.path.join(os.environ["TEST_TMPDIR"], "basic")
    h = h.assign(tf.constant([1], dtype=tf.int64), [[0.0]])
    l = ckpt_info.FidSlotCountSaverListener(model_dir)
    with self.session() as sess:
      sess.run(h.as_op())
      l.before_save(sess, 0)
    with tf.io.gfile.GFile(os.path.join(model_dir, "ckpt.info-0")) as f:
      text = f.read()

    ckpt = ckpt_info_pb2.CkptInfo()
    text_format.Parse(text, ckpt)
    self.assertEqual(ckpt.slot_counts[0], 1)


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
