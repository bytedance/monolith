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
from queue import Queue
from absl import logging
from threading import Thread, RLock
from kafka import KafkaProducer


class KProducer(object):

  def __init__(self, brokers, topic) -> None:
    self.brokers = brokers
    self.topic = topic
    self._producer = KafkaProducer(bootstrap_servers=brokers)

    self._lock = RLock()
    self._has_stopped = False
    self._msg_queue = Queue()  # thread safe

    self._total = 0
    self._success = 0
    self._failed = 0

    self._thread = Thread(target=self._poll)
    self._thread.start()

  def send(self, msgs):
    if msgs is None or len(msgs) == 0:
      return
    elif isinstance(msgs, (str, bytes)):
      msgs = [msgs]
    else:
      msgs = [msg for msg in msgs if msg is not None and len(msg) > 0]

    if len(msgs) > 0:
      logging.log_first_n(level=logging.INFO, msg=msgs[0], n=10)
      self._total += len(msgs)
      self._msg_queue.put(msgs)

  def _poll(self):
    while True:
      try:
        msg_batch = self._msg_queue.get(timeout=1)
      except:
        with self._lock:
          if self._has_stopped:
            break
          else:
            continue

      if msg_batch is not None and len(msg_batch) > 0:
        for msg in msg_batch:
          future = self._producer.send(self.topic, msg)
          future.add_callback(self._send_success).add_errback(self._send_failed)

      with self._lock:
        if self._has_stopped:
          break

  def total(self):
    return self._total

  def success(self):
    return self._success

  def failed(self):
    return self._failed

  def _flush(self):
    with self._lock:
      assert self._has_stopped

    while True:
      try:
        msg_batch = self._msg_queue.get(timeout=1)
      except:
        break

      if not msg_batch:
        break

      for msg in msg_batch:
        future = self._producer.send(self.topic, msg)
        future.add_callback(self._send_success).add_errback(self._send_failed)

  def close(self):
    try:
      logging.info('set stopped')
      with self._lock:
        self._has_stopped = True
      logging.info('wait for thread join')
      self._thread.join()
      logging.info('flush queue')
      self._flush()
      logging.info('close kafka producer')
      self._producer.close(timeout=1)
    except Exception as e:
      logging.warning(str(e))

  def _send_success(self, *args, **kwargs):
    self._success += 1

  def _send_failed(self, *args, **kwargs):
    time.sleep(secs=2)  # if failed, sleep two second
    logging.warning('send metric to kafka error')
    self._failed += 1
