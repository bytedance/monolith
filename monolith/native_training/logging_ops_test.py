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

from absl import flags

from monolith.native_training import logging_ops
from monolith.native_training.runtime.ops import logging_ops_pb2

FLAGS = flags.FLAGS


class LoggingOpsTest(tf.test.TestCase):

  def test_tensors_timestamp(self):
    tensor = [tf.constant(0)]
    tensor, ts = logging_ops.tensors_timestamp(tensor)
    tensor, new_ts = logging_ops.tensors_timestamp(tensor)

    with self.session() as sess:
      new_ts_value, ts_value = sess.run([new_ts, ts])
      self.assertGreaterEqual(new_ts_value, ts_value)

  def test_emit_timer(self):
    op = logging_ops.emit_timer("test", 0.0)
    self.evaluate(op)

  def test_machine_health(self):
    FLAGS.monolith_default_machine_info_mem_limit = 1 << 62
    info = logging_ops.machine_info()
    self.assertEqual(self.evaluate(logging_ops.check_machine_health(info)), b"")

  def test_machine_health_oom(self):
    FLAGS.monolith_default_machine_info_mem_limit = 0
    info = logging_ops.machine_info()
    serialized_result = self.evaluate(logging_ops.check_machine_health(info))
    result = logging_ops_pb2.MachineHealthResult()
    result.ParseFromString(serialized_result)
    self.assertEqual(result.status,
                     logging_ops_pb2.MachineHealthResult.OUT_OF_MEMORY)


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
