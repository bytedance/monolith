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
import tensorflow_datasets as tfds
from tqdm import tqdm
import time

from kafka import KafkaProducer

def get_preprocessed_dataset(size='100k') -> tf.data.Dataset:
  ratings = tfds.load(f"movielens/{size}-ratings", split="train")
  # For simplicity, we map each movie_title and user_id to numbers
  # by Hashing. You can use other ways to number them to avoid 
  # collision and better leverage Monolith's collision-free hash tables.  
  max_b = (1 << 63) - 1
  return ratings.map(lambda x: {
    'mov': tf.strings.to_hash_bucket_fast([x['movie_title']], max_b),
    'uid': tf.strings.to_hash_bucket_fast([x['user_id']], max_b),
    'label': tf.expand_dims(x['user_rating'], axis=0)
  })

def serialize_one(data):
  # serialize an training instance to string
  return tf.train.Example(features=tf.train.Features(
    feature={
      'mov': tf.train.Feature(int64_list=tf.train.Int64List(value=data['mov'])),
      'uid': tf.train.Feature(int64_list=tf.train.Int64List(value=data['uid'])),
      'label': tf.train.Feature(float_list=tf.train.FloatList(value=data['label']))
    }
  )).SerializeToString() 

if __name__ == "__main__":
  ds = get_preprocessed_dataset()
  producer = KafkaProducer(bootstrap_servers=['127.0.0.1:9092'])
  for count, val in tqdm(enumerate(ds), total=len(ds)):
    # note: we omit error callback here for performance
    producer.send(
      "movie-train", key=str(count).encode('utf-8'), value=serialize_one(val), headers=[])
  producer.flush()
