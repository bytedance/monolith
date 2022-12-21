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

from absl import app
from absl import flags
from absl import logging

import json
import os

import tensorflow as tf
from kafka_receiver import decode_example, to_ragged
from kafka_producer import get_preprocessed_dataset

from monolith.native_training.estimator import EstimatorSpec, Estimator, RunnerConfig, ServiceDiscoveryType
from monolith.native_training.native_model import MonolithModel
from monolith.native_training.data.datasets import create_plain_kafka_dataset

tf.compat.v1.disable_eager_execution()

flags.DEFINE_enum('training_type', 'batch', ['batch', 'stream'], "type of training to launch")
FLAGS = flags.FLAGS

def get_worker_count(env: dict):
  cluster = env['cluster']
  worker_count = len(cluster.get('worker', [])) + len(cluster.get('chief', []))
  assert worker_count > 0
  return worker_count

class MovieRankingModelBase(MonolithModel):
  def __init__(self, params):
    super().__init__(params)
    self.p = params

  def model_fn(self, features, mode):
    # for sparse features, we declare an embedding table for each of them
    for s_name in ["mov", "uid"]:
      self.create_embedding_feature_column(s_name)

    mov_embedding, user_embedding = self.lookup_embedding_slice(
      features=['mov', 'uid'], slice_name='vec', slice_dim=32)
    ratings = tf.keras.Sequential([
      # Learn multiple dense layers.
      tf.keras.layers.Dense(256, activation="relu"),
      tf.keras.layers.Dense(64, activation="relu"),
      # Make rating predictions in the final layer.
      tf.keras.layers.Dense(1)
    ])
    concated = tf.concat((user_embedding, mov_embedding), axis=1)
    rank = ratings(concated)
    label = features['label']
    loss = tf.reduce_mean(tf.losses.mean_squared_error(rank, label))

    optimizer = tf.compat.v1.train.AdagradOptimizer(0.05)

    return EstimatorSpec(
      label=label,
      pred=rank,
      head_name="rank",
      loss=loss, 
      optimizer=optimizer,
      classification=False
    )
    
  def serving_input_receiver_fn(self):
    # a dummy serving input receiver
    return tf.estimator.export.ServingInputReceiver({})

class MovieRankingBatchTraining(MovieRankingModelBase):
  def input_fn(self, mode):
    env = json.loads(os.environ['TF_CONFIG'])
    dataset = get_preprocessed_dataset('25m')
    dataset = dataset.shard(get_worker_count(env), env['task']['index'])
    return dataset.batch(512, drop_remainder=True)\
      .map(to_ragged).prefetch(tf.data.AUTOTUNE)

class MovieRankingStreamTraining(MovieRankingModelBase):
  def input_fn(self, mode):
    dataset = create_plain_kafka_dataset(topics=["movie-train"],
        group_id="cgonline",
        servers="127.0.0.1:9092",
        stream_timeout=10000,
        poll_batch_size=16,
        configuration=[
          "session.timeout.ms=7000",
          "max.poll.interval.ms=8000"
        ],
    )
    return dataset.map(lambda x: decode_example(x.message))


FLAGS = flags.FLAGS

def main(_):
  raw_tf_conf = os.environ['TF_CONFIG']
  tf_conf = json.loads(raw_tf_conf)
  config = RunnerConfig(
      discovery_type=ServiceDiscoveryType.PRIMUS,
      tf_config=raw_tf_conf,
      save_checkpoints_steps=10000,
      enable_model_ckpt_info=True,
      num_ps=len(tf_conf['cluster']['ps']),
      num_workers=get_worker_count(tf_conf),
      server_type=tf_conf['task']['type'],
      index=tf_conf['task']['index']
  )
  if FLAGS.training_type == "batch":
    params = MovieRankingBatchTraining.params().instantiate()
  else:
    params = MovieRankingStreamTraining.params().instantiate()

  estimator = Estimator(params, config)
  estimator.train(max_steps=1000000)

if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  app.run(main)