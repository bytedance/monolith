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

import os
local_size = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_SIZE', 1))
rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', 0))
os.environ['CUDA_VISIBLE_DEVICES'] = str(rank % local_size)
os.environ["MONOLITH_WITH_HOROVOD"] = "True"
os.environ["HOROVOD_AUTOTUNE"] = "1"
os.environ["HOROVOD_CYCLE_TIME"] = "0.1"
os.environ["MONOLITH_SYNC_EMPTY_RANK0_PS_SHARD"] = "0"
os.environ["MONOLITH_WITH_ALLREDUCE_FUSION"] = "one"
os.environ['MONOLITH_ROOT_LOG_INTERVAL'] = "10"

import time
import tensorflow as tf
tf.compat.v1.set_random_seed(42)

from tensorflow.python.framework import test_util

from monolith.native_training import cpu_training
from monolith.native_training.native_model import MonolithModel
from monolith.native_training.estimator import Estimator, EstimatorSpec
from monolith.native_training.data.training_instance.python.parser_utils import advanced_parse
from monolith.native_training.entry import (AdagradOptimizer, Fp16Compressor,
                                            ZerosInitializer)
from monolith.native_training import layers
import horovod.tensorflow as hvd

deep = {
    "initializer":
        ZerosInitializer(),
    "optimizer":
        AdagradOptimizer(learning_rate=0.05,
                         weight_decay_factor=0.0,
                         initial_accumulator_value=0.1,
                         warmup_steps=0),
    "compressor":
        Fp16Compressor()
}

num_features = 17
batch_size = 455
emb_dim = 15
feature_names = [f'feature{i}' for i in range(num_features)]
fid_max_val = 100000


def lookup_tf_embedding(features, f_name, dim):
  f = tf.RaggedTensor.from_row_splits(features[f'tf_{f_name}_p1'], features[f'tf_{f_name}_p2'], validate=False)
  embeddings = tf.nn.embedding_lookup(
      params=tf.Variable(initial_value=tf.zeros(shape=(fid_max_val + 1, dim))), 
      ids=f.values)
  return tf.math.segment_sum(embeddings, f.value_rowids())

class EmbeddingUpdateTask(MonolithModel):
  """A test task that will compare TF and monolith embedding update."""
  
  def __init__(self, params=None):
    super(EmbeddingUpdateTask, self).__init__(params)
    self.train.max_steps = 50
    self.train.per_replica_batch_size = batch_size

  def input_fn(self, mode):

    def decomp_func(features):
      # note: this is a workaround to pass fids to model_fn, since all instances of RaggedTensor are gone
      for i in range(num_features):
        features[f'tf_feature{i}_p1'] = features[f'feature{i}'].values
        features[f'tf_feature{i}_p2'] = features[f'feature{i}'].row_splits
      return advanced_parse(features)

    @tf.function
    def input_tensors():
      features = {}
      for i in range(num_features):
        features[f'feature{i}'] = tf.random.uniform((tf.random.uniform((), 1, 25, dtype=tf.int32),), 0, fid_max_val, dtype=tf.int64)
      features['label'] = tf.cast(tf.random.uniform((), 0, 2, dtype=tf.int32), tf.float32)
      return features

    return tf.data.experimental.Counter(0, 1).map(lambda _: input_tensors(), tf.data.AUTOTUNE).\
      apply(tf.data.experimental.dense_to_ragged_batch(batch_size, True)).\
        map(decomp_func, tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    

  def model_fn(self, features, mode):
    with tf.device("/device:GPU:0"):
      for f_name in feature_names:
        self.create_embedding_feature_column(f_name)

      tf_embeddings = [lookup_tf_embedding(features, f_name, emb_dim) for f_name in feature_names]
      embeddings = self.lookup_embedding_slice(features=feature_names,
                                              slice_name='vec',
                                              slice_dim=emb_dim,
                                              **deep)
      # if mode == tf.estimator.ModeKeys.PREDICT:
      #   return tf.estimator.EstimatorSpec(mode, predictions=tf.constant(0))

      embed_concat = tf.concat(embeddings, axis=1)
      tf_embed_concat = tf.concat(tf_embeddings, axis=1)

      assert_op = tf.compat.v1.assert_equal(embed_concat, tf_embed_concat)
      with tf.compat.v1.control_dependencies([assert_op]):  
        # pred = layers.MLP(
        #   output_dims=[64, 32, 1], activations='relu')(embed_concat)

        pred = tf.keras.Sequential([
          tf.keras.layers.Dense(64, activation="relu"),
          tf.keras.layers.Dense(32, activation="relu"),
          tf.keras.layers.Dense(1)
        ])(embed_concat)

        tf_pred = tf.keras.Sequential([
          tf.keras.layers.Dense(64, activation="relu"),
          tf.keras.layers.Dense(32, activation="relu"),
          tf.keras.layers.Dense(1)
        ])(tf_embed_concat)

      label = features['label']
      loss = tf.reduce_mean(tf.losses.mean_squared_error(pred, label))
      tf_loss = tf.reduce_mean(tf.losses.mean_squared_error(tf_pred, label))
      
      optimizer = tf.compat.v1.train.AdagradOptimizer(0.05)
      # with tf.device("/device:CPU:0"):
        # loss = tf.compat.v1.Print(loss, [loss], 'monolith loss')
        # tf_loss = tf.compat.v1.Print(tf_loss, [tf_loss], 'tf loss')
      
      return EstimatorSpec(
        loss=loss + tf_loss,
        pred=[pred, tf_pred],
        label=[label, label],
        head_name=['monolith', 'tf'],
        classification=[False, False],
        optimizer=optimizer
      )
  def serving_input_receiver_fn(self):
    pass


class CpuSyncTrainTest(tf.test.TestCase):
  
  def _create_config(self, gpu, multi_hash_table):
    return cpu_training.DistributedCpuTrainingConfig(
      # save_checkpoints_steps=10000,
      num_ps=0,
      num_workers=hvd.size(),
      model_dir=f'/tmp/hanzhizhou/monolith_test/{int(time.time())}',
      reorder_fids_in_data_pipeline=True,
      embedding_prefetch_capacity=0,
      enable_sync_training=True,
      enable_gpu_training=gpu,
      enable_realtime_training=False,
      use_native_multi_hash_table=multi_hash_table,
      index=hvd.rank(),
    )

  def test_embedding_update(self):
    hvd.init()
    p = EmbeddingUpdateTask.params().instantiate()
    config = self._create_config(False, False)
    cpu_training.distributed_sync_train(config, p)

    config = self._create_config(False, True)
    cpu_training.distributed_sync_train(config, p)

    if test_util.is_gpu_available(cuda_only=True):
      config = self._create_config(True, False)
      cpu_training.distributed_sync_train(config, p)
      
      config = self._create_config(True, True)
      cpu_training.distributed_sync_train(config, p)

if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
