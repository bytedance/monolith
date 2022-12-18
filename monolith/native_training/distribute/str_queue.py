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

from typing import Callable

import tensorflow as tf


class StrQueue:
  """A queue whose element is a string, and supports save/restore.
  When queue is running out, it will throw OutOfRange error.
  """

  def __init__(self,
               initial_elements=None,
               critical_section=None,
               auto_enqueue_fn=None,
               capacity=100000,
               name="StrQueue"):
    """Args:
      critical_section - if not None, queue will use this as a critical section instead of creating new one.
      auto_enqueue_fn - when queue is empty, we will use this enqueue op to fill the queue. Should be a callable
      returns 2 tensors: 1-D string tensor represents strings to be enqueued and 0-D bool tensor to indicate if
      it is out of range.
    """
    with tf.name_scope(name) as scope:
      self._name = name
      self._auto_enqueue_fn = auto_enqueue_fn
      self._capacity = capacity

      self._cs = critical_section or tf.CriticalSection(
          name="CriticalSection", shared_name=scope + "/CriticalSection")
      self._arr = tf.Variable(initial_value=tf.constant_initializer("")(
          shape=[self._capacity],
          dtype=tf.string,
      ),
                              trainable=False,
                              name="Queue")
      self._offset = tf.Variable(0, trainable=False, name="Offset")
      self._arr_size = tf.Variable(0, trainable=False, name="Size")

      if initial_elements is None:
        initial_elements = []

      # Here we use a dummy var to init queue
      with tf.control_dependencies([
          self._arr.initializer,
          self._offset.initializer,
          self._arr_size.initializer,
      ]):
        with tf.control_dependencies([self.enqueue_many(initial_elements)]):
          var_for_init_value = tf.constant(0)

      self._var_for_init = tf.Variable(initial_value=var_for_init_value,
                                       trainable=False,
                                       name="VarForInit")

  @property
  def critical_section(self):
    return self._cs

  def enqueue_many(self, elements: tf.Tensor, name=None):
    elements = tf.convert_to_tensor(elements, tf.string)
    return self._cs.execute(lambda: self._raw_enqueue_many(elements), name=name)

  def dequeue(self, name=None):
    """Dequeues an element. Returns 2 elements: element & a bool indicating if we're out of range."""
    return self._cs.execute(self._raw_dequeue, name=name)

  @tf.function
  def _raw_enqueue_many(self, elements: tf.Tensor):
    size = tf.size(elements)
    old_arr_size = self._arr_size - self._offset
    new_arr_size = old_arr_size + size
    tf.debugging.Assert(new_arr_size <= self._capacity, [
        self._name, " excceeds capacity ", new_arr_size, " v.s. ",
        self._capacity
    ])
    self._arr[0:old_arr_size].assign(self._arr[self._offset:self._arr_size])
    self._arr[old_arr_size:old_arr_size + size].assign(elements)
    self._offset.assign(0)
    self._arr_size.assign(new_arr_size)

  @tf.function
  def _raw_dequeue(self):
    tf.debugging.Assert(self._offset <= self._arr_size, [
        "Offset should always be less than or equal to arr_size.",
        "This may indicate an internal error. offset: ", self._offset,
        " arr_size: ", self._arr_size
    ])
    if self._auto_enqueue_fn is not None:
      while tf.math.equal(self._offset, self._arr_size):
        elements, out_of_range = self._auto_enqueue_fn()
        elements = tf.convert_to_tensor(elements)
        if out_of_range:
          break
        self._raw_enqueue_many(elements)
    if tf.math.equal(self._offset, self._arr_size):
      return "", True
    else:
      single_element = self._arr[self._offset]
      self._offset.assign_add(1)
      return single_element, False
