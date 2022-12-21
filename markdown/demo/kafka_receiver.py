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
from monolith.native_training.data.datasets import create_plain_kafka_dataset

raw_feature_desc = {
  'mov': tf.io.FixedLenFeature([1], tf.int64),
  'uid': tf.io.FixedLenFeature([1], tf.int64),
  'label': tf.io.FixedLenFeature([], tf.float32)
}

def to_ragged(x):
  return {
    'mov': tf.RaggedTensor.from_tensor(x['mov']),
    'uid': tf.RaggedTensor.from_tensor(x['uid']),
    'label': x['label']
  }

# corresponds to serailize_one in kafka_producer.py
def decode_example(v):
    x = tf.io.parse_example(v, raw_feature_desc)
    return to_ragged(x)

if __name__ == "__main__":
    dataset = create_plain_kafka_dataset(topics=["movie-train"],
        group_id="cgonline",
        servers="127.0.0.1:9092",
        stream_timeout=10000, # in milliseconds, to block indefinitely, set it to -1.
        poll_batch_size=8,
        configuration=[
          "session.timeout.ms=7000",
          "max.poll.interval.ms=8000"
        ],
    )
    for x in dataset.map(lambda x: decode_example(x.message)):
        print(x)
