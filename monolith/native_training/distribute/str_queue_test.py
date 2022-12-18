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

from monolith.native_training.distribute import str_queue

import tensorflow as tf


class QueueTest(tf.test.TestCase):

  def testBasic(self):
    q = str_queue.StrQueue()
    enqueue_op = q.enqueue_many(["test1", "test2"])
    dequeue = q.dequeue()
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(enqueue_op)
    self.assertEqual(self.evaluate(dequeue)[0].decode(), "test1")
    self.assertEqual(self.evaluate(dequeue)[0].decode(), "test2")

  def testInit(self):
    q = str_queue.StrQueue(initial_elements=["test1"])
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertEqual(self.evaluate(q.dequeue())[0].decode(), "test1")

  def testOutOfRange(self):
    q = str_queue.StrQueue()
    dequeue = q.dequeue()
    self.evaluate(tf.compat.v1.global_variables_initializer())
    _, out_of_range = self.evaluate(dequeue)
    self.assertEqual(out_of_range, True)

  def testAutoEnqueue(self):
    v = tf.Variable([0])
    self.evaluate(tf.compat.v1.global_variables_initializer())
    ds = tf.data.Dataset.from_tensor_slices([])
    it = tf.compat.v1.data.make_one_shot_iterator(ds)

    @tf.function
    def auto_enqueue():
      new_v = v.assign_add([1])
      if new_v > 2:
        return tf.constant([""]), True
      return tf.as_string(v), False

    q = str_queue.StrQueue(auto_enqueue_fn=auto_enqueue)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertEqual(self.evaluate(q.dequeue())[0].decode(), "1")
    self.assertEqual(self.evaluate(q.dequeue())[0].decode(), "2")
    self.assertEqual(self.evaluate(q.dequeue())[1], True)
    # Simulating in the distributed training, multiple dequeues will be called.
    self.assertEqual(self.evaluate(q.dequeue())[1], True)


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
