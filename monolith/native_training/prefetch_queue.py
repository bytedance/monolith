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

import copy
from itertools import accumulate
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from absl import logging

import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_data_flow_ops

from monolith.native_training import utils
from monolith.native_training import nested_tensors


# Similar to https://github.com/tensorflow/tensorflow/commit/f98b3bc7012085096d8171fe56f6004677461567#
class _GPUCompatiblePaddingFIFOQueue(data_flow_ops.QueueBase):
  """A queue implementation that dequeues elements in first-in first-out order.
  GPUCompatiblePaddingFIFOQueue is like PaddingFIFOQueue, 
  but the queue resource may be placed either on a CPU or on a GPU. 
  It is not cross-device: enqueues and dequeues
  will be colocated with the queue resource.
  """

  def __init__(self,
               capacity,
               dtypes,
               shapes,
               names=None,
               shared_name=None,
               name="padding_fifo_queue"):
    """A `PaddingFIFOQueue` may contain components with dynamic shape, while also
    supporting `dequeue_many`.
    
    The `shapes` argument must be specified; each component of a queue
    element must have the respective shape.  Shapes of fixed
    rank but variable size are allowed by setting any shape dimension to None.
    
    Args:
      capacity: An integer. The upper bound on the number of elements
        that may be stored in this queue.
      dtypes:  A list of `DType` objects. The length of `dtypes` must equal
        the number of tensors in each queue element.
      shapes: A list of `TensorShape` objects, with the same length as
        `dtypes`.  Any dimension in the `TensorShape` containing value
        `None` is dynamic and allows values to be enqueued with
         variable size in that dimension.
      names: (Optional.) A list of string naming the components in the queue
        with the same length as `dtypes`, or `None`.  If specified the dequeue
        methods return a dictionary with the names as keys.
      shared_name: (Optional.) If non-empty, this queue will be shared under
        the given name across multiple sessions.
      name: Optional name for the queue operation.
    """
    dtypes = data_flow_ops._as_type_list(dtypes)
    shapes = data_flow_ops._as_shape_list(shapes,
                                          dtypes,
                                          unknown_dim_allowed=True)
    names = data_flow_ops._as_name_list(names, dtypes)
    if len(dtypes) != len(shapes):
      raise ValueError("Shapes must be provided for all components, "
                       f"but received {len(dtypes)} dtypes and "
                       f"{len(shapes)} shapes.")
    # init_scope() context required
    queue_ref = gen_data_flow_ops.padding_fifo_queue_v2(
        component_types=dtypes,
        shapes=shapes,
        capacity=capacity,
        shared_name=data_flow_ops._shared_name(shared_name),
        name=name)

    super().__init__(dtypes, shapes, names, queue_ref)

  def enqueue_many(self, vals, name=None):
    """enqueue_many is not supported on GPUCompatiblePaddingFIFOQueue."""
    raise NotImplementedError(
        "GPUCompatiblePaddingFIFOQueue does not support enqueue_many or dequeue_many, "
        "only enqueue and dequeue.")

  def dequeue_many(self, n, name=None):
    """dequeue_many is not supported on GPUCompatiblePaddingFIFOQueue."""
    raise NotImplementedError(
        "GPUCompatiblePaddingFIFOQueue does not support enqueue_many or dequeue_many, "
        "only enqueue and dequeue.")


class _QueueBase:
  """Monolith Specialized Prefetch QueueBase."""

  @property
  def queue(self):
    raise NotImplementedError

  @property
  def queues(self):
    raise NotImplementedError

  @property
  def enqueue_op(self):
    raise NotImplementedError

  def dequeue(self):
    raise NotImplementedError


class _FIFOQueue(_QueueBase):

  def __init__(self,
               dense_list: Optional[List[tf.Tensor]] = None,
               capacity: int = 2,
               queue_name: str = "prefetch_queue"):
    if dense_list is None:
      raise ValueError("Arguments `dense_list` should not be empty.")
    if dense_list is None:
      dense_list = []
    else:
      if not isinstance(dense_list, list):
        raise TypeError("dense_list should be a list of `tf.Tensor`s")
    self._dense_list = dense_list
    flatten_tensor_list = self._dense_list
    dtypes = [f.dtype for f in flatten_tensor_list]
    shapes = [f.shape for f in flatten_tensor_list]
    with tf.init_scope():
      self._queue = _GPUCompatiblePaddingFIFOQueue(capacity,
                                                   dtypes=dtypes,
                                                   shapes=shapes,
                                                   name=queue_name)
    self._enqueue_op = self._queue.enqueue(flatten_tensor_list)

  @property
  def queue(self):
    return self._queue

  @property
  def queues(self):
    return [self._queue]

  @property
  def enqueue_op(self):
    return self._enqueue_op

  def dequeue(self):
    with tf.init_scope():
      dequeue_tensor_list = self._queue.dequeue()
    if not isinstance(dequeue_tensor_list, list):
      assert len(self._dense_list) == 1
      return [dequeue_tensor_list]
    return dequeue_tensor_list


class _MultiFIFOQueue(_QueueBase):
  """Multi-Device FIFOQueue that supports CPU and GPU tensors in queue."""

  def __init__(self,
               dense_list: Optional[List[tf.Tensor]] = None,
               capacity: int = 2,
               queue_name: str = "prefetch_queue"):
    # Don't call the super() constructor here. Just inherit for interfaces.
    self._qs = []
    dense_list_cpu, dense_list_gpu = self._split_tensor_list_by_device(
        dense_list)
    with tf.device("/device:CPU:0"):
      queue = _FIFOQueue(dense_list=dense_list_cpu,
                         capacity=capacity,
                         queue_name=queue_name)
    self._qs.append(queue)
    if dense_list_gpu:
      with tf.device("/device:GPU:0"):
        queue_gpu = _FIFOQueue(dense_list=dense_list_gpu,
                               capacity=capacity,
                               queue_name=queue_name + "_gpu")
      self._qs.append(queue_gpu)

    # enqueue altogether
    self._enqueue_op = tf.group([q.enqueue_op for q in self._qs])

  @property
  def queue(self):
    if len(self._qs) == 1:
      return self._qs[0].queue
    else:
      raise NotImplementedError(
          "When using multi-device queues, this interface is disabled."
          "Check if a tensor to be enqueued is mistakenly placed on GPU.")

  @property
  def queues(self):
    return [q.queue for q in self._qs]

  @property
  def enqueue_op(self):
    return self._enqueue_op

  def dequeue(self):
    n = len(self._qs)
    if n == 1:
      return self._qs[0].dequeue()
    else:
      # We assume here that when we dequeue, we dequeue both CPU and GPU tensors together;
      # otherwise we need to enforce mutual control dependencies with gate_op at python level.
      # Therefore, a C++ implementation of this multi-device FIFOQueue would be a better choice.
      # TODO(peng.wu): make this whole queue implementation work at C++ TF queue resource level.
      return self._merge_tensor_list_by_device([q.dequeue() for q in self._qs])

  def size(self):
    if len(self._qs) == 1:
      return self.queue.size()
    else:
      # Based on the assumption commented in the above "dequeue()" method,
      # it allows to check cpu-device queue size to get the size for all.
      return self.queues[0].size()

  def _split_tensor_list_by_device(self, tensors):
    """List of tensors to (List of CPU Tensors, List of GPU Tensors)."""
    split_tensors = ([], [])  # 0: CPU, 1: GPU
    self._split_tensors_indices = []
    for f in tensors:
      tuple_i = 1 if "GPU" in f.device else 0
      l = split_tensors[tuple_i]
      idx = (tuple_i, len(l))
      l.append(f)
      self._split_tensors_indices.append(idx)
    return split_tensors

  def _merge_tensor_list_by_device(self, tensor_lists):
    return [tensor_lists[i][j] for i, j in self._split_tensors_indices]


class MultiQueueRunner(tf.compat.v1.train.QueueRunner):

  def _init_from_args(self,
                      queue=None,
                      enqueue_ops=None,
                      close_op=None,
                      cancel_op=None,
                      queue_closed_exception_types=None):
    """Create a QueueRunner from arguments."""
    if isinstance(queue, list):
      close_op = tf.group([q.close() for q in queue])
      cancel_op = tf.group(
          [q.close(cancel_pending_enqueues=True) for q in queue])
    # else fallback to original QueueRunner
    super()._init_from_args(
        queue=queue,
        enqueue_ops=enqueue_ops,
        close_op=close_op,
        cancel_op=cancel_op,
        queue_closed_exception_types=queue_closed_exception_types)

  @property
  def queue(self):
    if isinstance(self._queue, list):
      raise NotImplementedError(
          "When using multi-device queues, this interface is disabled.")
    return self._queue

  @property
  def name(self):
    if isinstance(self._queue, list):
      return self._queue[0].name
    return self._queue.name


class EnqueueHook(tf.estimator.SessionRunHook):

  def __init__(self, q: _QueueBase):
    self._q_runner = MultiQueueRunner(q.queues, [q.enqueue_op])
    self._threads = []

  def after_create_session(self, session, coord):
    self._threads = self._q_runner.create_threads(session,
                                                  coord=coord,
                                                  start=True,
                                                  daemon=True)


def enqueue_dicts_with_queue_return(
    tensors,
    capacity: int = 1,
    queue_name: str = "prefetch_queue") -> Tuple[Any, Optional[_FIFOQueue]]:
  """tensors can be any nested structures (list, tuple, dict) with tensors"""
  if capacity == 0:
    return tensors, None
  nested = nested_tensors.NestedTensors(tensors)
  flatten_tensors = nested.get_tensors()
  queue = _MultiFIFOQueue(dense_list=flatten_tensors,
                          capacity=capacity,
                          queue_name=queue_name)
  with tf.init_scope():
    dequeue_dense_list = queue.dequeue()
  return nested.get_nested_result(dequeue_dense_list), queue


class AsyncPushHook(tf.estimator.SessionRunHook):

  def __init__(self, queue, ops):
    self._queue = queue
    self._queue_init = False
    self._run_ops = ops

  def begin(self):
    self._queue_size = self._queue.size()

  def before_run(self, run_context):
    if self._queue_init:
      return tf.estimator.SessionRunArgs(self._run_ops)

  def after_run(self, run_context, run_values):
    if not self._queue_init:
      self._queue_init = run_context.session.run(self._queue_size) > 0

  def end(self, session):
    while session.run(self._queue_size) > 0:
      session.run(self._run_ops)


class AsyncFunctionMgr:
  """A class that supports adding async functions"""

  def __init__(self, is_async: bool = True):
    """
    Args:
      is_async - by default, added async function will be executed asyncly or not.
    """
    self._is_async = is_async
    self._hooks = []

  def add_async_function(
      self,
      target: Callable,
      args: Tuple = None,
      kwargs: Dict = None,
      is_async: bool = None,
      queue_name: str = "async_queue") -> Union[tf.Operation, Any]:
    """
    Args:
      is_async - if execute target synchronously. If None, will use default value in __init__.
    """
    if is_async is None:
      is_async = self._is_async
    if args is None:
      args = ()
    if kwargs is None:
      kwargs = {}

    if is_async:
      # This prevents from using an empty input list.
      args = (args) + (tf.constant(0, name="dummy_tensor_for_async_function"),)
      (args, kwargs), queue = enqueue_dicts_with_queue_return(
          (args, kwargs), queue_name=queue_name)
      dummy_op = tf.no_op(name="dummy_depended_op_for_async_function")
      with tf.init_scope(), tf.control_dependencies([args[-1], dummy_op]):
        run_ops = target(*args[0:-1], **kwargs)
      self._hooks.append(AsyncPushHook(queue, run_ops))

      # Check ops dependence in async func.
      utils.check_ops_dependence(queue.enqueue_op.name, dummy_op.name)

      return queue.enqueue_op
    else:
      return target(*args, **kwargs)

  @property
  def hooks(self):
    return self._hooks
