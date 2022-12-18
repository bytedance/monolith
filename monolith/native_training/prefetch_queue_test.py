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

import time

import tensorflow as tf
from tensorflow.python.framework import test_util

from monolith.native_training import nested_tensors
from monolith.native_training import prefetch_queue


class GPUCompatiblePaddingFIFOQueueTests(tf.test.TestCase):

  def testEnqueueAndDequeue(self):
    with test_util.use_gpu():
      q = prefetch_queue._GPUCompatiblePaddingFIFOQueue(10, tf.float32, ((),))
      elems_numpy = [10.0, 20.0, 30.0]
      _a = tf.constant(2.0)
      elems = [tf.constant(x) / _a for x in elems_numpy]
      # MemcpyH2D * 3: [10.0, 20.0, 30.0]
      # MemcpyH2D * 3: _a
      # GPU RealDiv * 3
      for x in elems:
        self.evaluate(q.enqueue((x,)))

      _b = tf.constant(3.0)
      _c = tf.constant(1.0)
      dequeued_tensor = q.dequeue()
      for i in range(len(elems)):
        # Ensure same device
        self.assertEqual(elems[0].device, dequeued_tensor.device)
        # MemcpyH2D * 3: _b
        # MemcpyH2D * 3: _c
        # GPU Mul * 3
        # GPU Add * 3
        # MemcpyD2H * 3: return
        vals = self.evaluate(dequeued_tensor * _b + _c)
        self.assertEqual([elems_numpy[i] / 2 * 3 + 1], vals)

  def testGPUQueueCPUTensor(self):
    with tf.device("CPU:0"):
      elems_numpy = [7, 8, 9]
      _a = tf.constant(5)
      elems = [tf.constant(x) * _a for x in elems_numpy]

    with test_util.use_gpu():
      # MemcpyH2D * 3
      q = prefetch_queue._GPUCompatiblePaddingFIFOQueue(10, tf.int32, ((),))
      # MemcpyD2H * 3
      # Note that even though the below enqueue/dequeue are declared on CPU,
      # it still copys-to and holds the enqueued resources on GPU.
      # So to pin the tensors on CPU, we need to declare the queue itself on CPU.

    for x in elems:
      with tf.device("CPU:0"):
        self.evaluate(q.enqueue((x,)))

    with tf.device("CPU:0"):
      dequeued_tensor = q.dequeue()

    for i in range(len(elems)):
      self.assertEqual(elems[0].device, dequeued_tensor.device)
      with tf.device("CPU:0"):
        vals = self.evaluate(dequeued_tensor + 2)
      self.assertEqual([elems_numpy[i] * 5 + 2], vals)

  def testMultiEnqueueAndDequeue(self):
    with test_util.use_gpu():
      q = prefetch_queue._GPUCompatiblePaddingFIFOQueue(10,
                                                        (tf.int32, tf.float32),
                                                        ((), ()))
      elems_numpy = [(5, 10.0), (10, 20.0), (15, 30.0)]
      elems = [(tf.constant(x), tf.constant(y)) for x, y in elems_numpy]

      for x, y in elems:
        self.evaluate(q.enqueue((x, y)))

      dequeued_tensor = q.dequeue()
      print(dequeued_tensor[0].device)
      for i in range(len(elems)):
        self.assertEqual(elems[i][0].device, dequeued_tensor[0].device)
        x_val, y_val = self.evaluate(dequeued_tensor)
        x, y = elems_numpy[i]
        self.assertEqual(x, x_val)
        self.assertEqual(y, y_val)

  def testIdentityHelper(self):
    with tf.device("CPU:0"):
      a = tf.constant(1)
      b = a + 1
    with test_util.use_gpu():
      c = tf.identity(b)
      q = prefetch_queue._GPUCompatiblePaddingFIFOQueue(1, tf.int32, ((),))
      self.evaluate(q.enqueue(c))  # MemcpyH2D: CPU b to GPU c pinned in queue
      self.assertAllEqual(self.evaluate(q.dequeue()), 2)  # MemcpyD2H: return


class FIFOQueueTest(tf.test.TestCase):

  def test_fifo_queue_data(self):
    dense_tensors = [tf.constant(2.), tf.constant([[3], [4]])]
    ragged_tensors = [
        tf.ragged.constant([[2], [-1, 3]]),
        tf.ragged.constant([[1.], []])
    ]
    nested = nested_tensors.NestedTensors((dense_tensors, ragged_tensors))
    flatten_tensors = nested.get_tensors()
    queue = prefetch_queue._FIFOQueue(flatten_tensors)
    dequeued = queue.dequeue()
    dequeued_dense, dequeued_ragged = nested.get_nested_result(dequeued)
    with self.session() as sess:
      sess.run(queue.enqueue_op)
      dequeued_dense, dequeued_ragged = sess.run(
          [dequeued_dense, dequeued_ragged])
    self.assertAllClose(dequeued_dense[0], 2.)
    self.assertAllEqual(dequeued_dense[1], [[3], [4]])
    self.assertAllEqual(dequeued_ragged[0], [[2], [-1, 3]])
    self.assertAllClose(dequeued_ragged[1], [[1.], []])

  def test_fifo_queue_capacity(self):
    dense_tensors = [tf.constant([2])]
    queue = prefetch_queue._FIFOQueue(dense_tensors, capacity=4)
    dequeue_result = queue.dequeue()
    with self.session() as sess:
      for _ in range(4):
        sess.run(queue.enqueue_op)
      for _ in range(4):
        result = sess.run(dequeue_result)
        self.assertAllEqual(result[0], [2])


class PrefetchTest(tf.test.TestCase):

  def test_enqueue_dicts_with_queue_return(self):
    dense_dicts = [{
        "dense_0_0": tf.constant(2.),
        "dense_0_1": tf.constant([[3], [4]])
    }, {
        "dense_1_0": tf.constant([0])
    }]
    ragged_dicts = [{
        "ragged_0_0": tf.ragged.constant([[2], [-1, 3]]),
        "ragged_0_1": tf.ragged.constant([[1.], []])
    }, {
        "ragged_1_0": tf.ragged.constant([[0, 0], [1]])
    }]
    with test_util.use_gpu():
      dense_dicts[0]["dense_0_0"] += 1.0
    result = prefetch_queue.enqueue_dicts_with_queue_return(
        (dense_dicts, ragged_dicts), capacity=3)
    (dequeue_dense_dicts, dequeue_ragged_dicts), queue = result

    with self.session() as sess:
      for _ in range(5):
        sess.run(queue.enqueue_op)
        dense_dicts_result = sess.run(dequeue_dense_dicts)
        self.assertAllClose(dense_dicts_result[0]["dense_0_0"], 2. + 1.0)
        self.assertAllEqual(dense_dicts_result[0]["dense_0_1"], [[3], [4]])
        self.assertAllEqual(dense_dicts_result[1]["dense_1_0"], [0])

        sess.run(queue.enqueue_op)
        dense_dicts_result, ragged_dicts_result = sess.run(
            [dequeue_dense_dicts, dequeue_ragged_dicts])
        self.assertAllEqual(ragged_dicts_result[0]["ragged_0_0"],
                            [[2], [-1, 3]])
        self.assertAllClose(ragged_dicts_result[0]["ragged_0_1"], [[1.], []])
        self.assertAllEqual(ragged_dicts_result[1]["ragged_1_0"], [[0, 0], [1]])
        self.assertAllClose(dense_dicts_result[0]["dense_0_0"], 2. + 1.0)

  def test_enqueue_dicts_with_queue_return(self):
    tensors = ([{
        "a": tf.constant(1.0),
        "b": "abc",
        "c": None,
        "d": None,
    }], {
        "a": tf.Variable(0.5),
        "b": tf.ragged.constant([[1.0]])
    })
    dequeued_tensors, q = prefetch_queue.enqueue_dicts_with_queue_return(
        tensors)
    self.assertAllEqual(dequeued_tensors[0][0]["b"], "abc")
    del dequeued_tensors[0][0]["b"]
    self.assertEqual(dequeued_tensors[0][0]["c"], None)
    del dequeued_tensors[0][0]["c"]
    self.assertEqual(dequeued_tensors[0][0]["d"], None)
    del dequeued_tensors[0][0]["d"]

    with tf.compat.v1.train.SingularMonitoredSession() as sess:
      sess.run(q.enqueue_op)
      tensors = sess.run(dequeued_tensors)
      self.assertAllEqual(tensors[0][0]["a"], 1.0)
      self.assertAllEqual(tensors[1]["a"], 0.5)
      self.assertAllEqual(tensors[1]["b"], [[1.0]])

  def test_enqueue_dicts_with_control_flow(self):
    v = tf.Variable(0)
    with tf.control_dependencies([v.assign_add(1)]):
      tensor, q = prefetch_queue.enqueue_dicts_with_queue_return(tf.constant(0))
    with tf.compat.v1.train.MonitoredSession() as sess:
      sess.run(q.enqueue_op)
      sess.run(tensor)
      self.assertAllEqual(sess.run(v), 1)

  def test_enqueue_with_zero_capacity(self):
    dense_dicts = [{"dense": tf.constant([0])}]
    ragged_dicts = [{"ragged": tf.ragged.constant([[0, 0], [1]])}]
    result = prefetch_queue.enqueue_dicts_with_queue_return(
        (dense_dicts, ragged_dicts), 0)
    (dequeue_dense_dicts, dequeue_ragged_dicts), queue = result
    with self.session() as sess:
      dequeue_dense_dicts = sess.run(dequeue_dense_dicts)
      dequeue_ragged_dicts = sess.run(dequeue_ragged_dicts)
    self.assertAllEqual(dequeue_dense_dicts[0]["dense"], [0])
    self.assertAllEqual(dequeue_ragged_dicts[0]["ragged"], [[0, 0], [1]])

  def test_estimator_prefetch(self):

    def input_fn():
      return tf.data.Dataset.range(0, 20).map(
          lambda x: {"rag": tf.ragged.constant([[0], []], dtype=tf.int64) + x})

    def model_fn(features, mode):
      ragged = features["rag"]
      ragged_dicts = [{"ragged": ragged}]
      dequeue_raggeds, queue = prefetch_queue.enqueue_dicts_with_queue_return(
          ragged_dicts)
      predictions = dequeue_raggeds[0]["ragged"].values
      enqueue_hook = prefetch_queue.EnqueueHook(queue)

      global_step = tf.compat.v1.train.get_or_create_global_step()
      train_op = tf.compat.v1.assign_add(global_step, 1)
      return tf.estimator.EstimatorSpec(mode=mode,
                                        predictions=predictions,
                                        prediction_hooks=(enqueue_hook,),
                                        train_op=train_op)

    estimator = tf.estimator.Estimator(model_fn)
    predicts = estimator.predict(input_fn)
    self.assertAllEqual(list(range(20)), list(predicts))


class AsyncManagerTest(tf.test.TestCase):

  def testBasic(self):
    x = tf.Variable(0.0)

    def add(y):
      return x.assign_add(y)

    mgr = prefetch_queue.AsyncFunctionMgr()
    op = mgr.add_async_function(add, (tf.constant(1.0),))
    with tf.compat.v1.train.SingularMonitoredSession(hooks=mgr.hooks) as sess:
      sess.run(op)
      # Make push happen.
      sess.run(op)
      x_value = sess.run(x)
      # Since it is async pushed, the value will be 1.
      self.assertAllEqual(x_value, 1.0)

  def testSync(self):
    x = tf.Variable(0.0)

    def add(y):
      return x.assign_add(y)

    mgr = prefetch_queue.AsyncFunctionMgr(is_async=False)
    op = mgr.add_async_function(add, (tf.constant(1.0),))
    with tf.compat.v1.train.SingularMonitoredSession(hooks=mgr.hooks) as sess:
      sess.run(op)
      x_value = sess.run(x)
      self.assertAllEqual(x_value, 1.0)

  def testEmptyInput(self):
    x = tf.Variable(0)

    def add():
      return x.assign_add(1)

    mgr = prefetch_queue.AsyncFunctionMgr()
    op = mgr.add_async_function(add)
    with tf.compat.v1.train.SingularMonitoredSession(hooks=mgr.hooks) as sess:
      sess.run(op)
      # Make push happen.
      sess.run(op)
      x_value = sess.run(x)
      # Since it is async pushed, the value will be 1.
      self.assertAllEqual(x_value, 1)


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
