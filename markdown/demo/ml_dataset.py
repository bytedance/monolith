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
from multiprocessing import Process, cpu_count

def get_preprocessed_dataset(size='1m') -> tf.data.Dataset:
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

# serialize to human readable (csv) format
def serialize_hr(data):
  return f"{data['mov']},{data['uid']},{data['label']}\n"

def save_one_shard(total_shards, pid, start, end):
  ds = get_preprocessed_dataset('1m').map(lambda x: {
    'mov': tf.squeeze(x['mov']),
    'uid': tf.squeeze(x['uid']),
    'label': tf.squeeze(x['label'])
  })
  pbar = tqdm(position=pid, desc="[Serializing]")
  for i in range(start, end):
    ds_shard = ds.shard(total_shards, i).as_numpy_iterator()
    with open(f"data_1m/part_{i}.csv", "w") as f:
      for item in ds_shard:
        f.write(serialize_hr(item))
        pbar.update()

if __name__ == "__main__":
  # just let TF download this dataset if it doesn't exist
  ds = get_preprocessed_dataset('1m')
  for _ in ds.take(1):
    pass

  total_shards = 4
  num_process = min(max(cpu_count() // 4, 1), total_shards)
  processes = []
  shards_per_p = total_shards // num_process
  for i in range(num_process):
    # note: this multiprocessing is not very efficient because .shard needs to skip elements
    p = Process(target=save_one_shard, args=(total_shards, i, shards_per_p * i, shards_per_p * (i + 1)))
    p.start()
    processes.append(p)

  for p in processes:
    p.join()
