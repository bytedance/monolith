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
from typing import List

import tensorflow as tf
from tensorflow.keras.initializers import Ones, GlorotNormal
from monolith.native_training.dense_reload_utils import infer_variable_name, calc_feed_dict, \
  CustomRestoreListener
from tensorflow.python.training.py_checkpoint_reader import NewCheckpointReader, CheckpointReader


class DenseReloadUtilsTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    with tf.Graph().as_default():
      global_step = tf.compat.v1.train.get_or_create_global_step()
      partitioner = tf.compat.v1.variable_axis_size_partitioner(
          max_shard_bytes=1 << 17, max_shards=100)
      partition_var = tf.compat.v1.get_variable(name='partition',
                                                shape=(1280, 512),
                                                dtype=tf.float32,
                                                partitioner=partitioner,
                                                initializer=GlorotNormal())

      var = tf.compat.v1.get_variable(name='small_var',
                                      shape=(10, 5),
                                      dtype=tf.float32,
                                      initializer=Ones())
      # initialize all of the variables
      init = tf.compat.v1.global_variables_initializer()

      saver = tf.compat.v1.train.Saver([partition_var, var, global_step])

      with tf.compat.v1.Session() as sses:
        sses.run(init)
        saver.save(sses,
                   save_path=f"{os.getcwd()}/ckpt/test",
                   global_step=global_step)

  @classmethod
  def tearDownClass(cls):
    if tf.io.gfile.exists(path=f"{os.getcwd()}/ckpt"):
      tf.io.gfile.rmtree(path=f"{os.getcwd()}/ckpt")

  def test_infer_variable_name(self):
    with tf.Graph().as_default():
      global_step = tf.compat.v1.train.get_or_create_global_step()
      partitioner = tf.compat.v1.variable_axis_size_partitioner(
          max_shard_bytes=1 << 17, max_shards=100)
      partition_var = tf.compat.v1.get_variable(name='partition',
                                                shape=(1280, 512),
                                                dtype=tf.float32,
                                                partitioner=partitioner,
                                                initializer=GlorotNormal())

      var = tf.compat.v1.get_variable(name='small_var',
                                      shape=(10, 5),
                                      dtype=tf.float32,
                                      initializer=Ones())

      names = [part.name for part in partition_var._get_variable_list()]
      self.assertEqual(var.name, 'small_var:0')
      self.assertSetEqual(infer_variable_name(names),
                          {f'{partition_var.name}:0'})

  def test_calc_feed_dict(self):
    with tf.Graph().as_default():
      global_step = tf.compat.v1.train.get_or_create_global_step()
      partitioner = tf.compat.v1.variable_axis_size_partitioner(
          max_shard_bytes=1 << 17, max_shards=100)
      partition_var = tf.compat.v1.get_variable(name='partition2',
                                                shape=(1280, 512),
                                                dtype=tf.float32,
                                                partitioner=partitioner,
                                                initializer=GlorotNormal())

      var = tf.compat.v1.get_variable(name='small_var2',
                                      shape=(10, 5),
                                      dtype=tf.float32,
                                      initializer=Ones())
      alias_map = {'small_var2': 'small_var', 'global_step': 'global_step'}
      var_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=(10, 5))
      var_ph.origin_name = 'small_var2'
      gs_ph = tf.compat.v1.placeholder(dtype=tf.int64)
      gs_ph.origin_name = 'global_step'

      placeholders = [var_ph, gs_ph]
      for part in partition_var._get_variable_list():
        if part.name.endswith(':0'):
          var_name = part.name[0:-2]
        else:
          var_name = part.name

        alias_map[var_name] = 'partition'
        ph = tf.compat.v1.placeholder(dtype=part.dtype, shape=part.shape)
        ph.origin_name = var_name
        placeholders.append(ph)

      ckpt: CheckpointReader = NewCheckpointReader(f"{os.getcwd()}/ckpt/test-0")
      ph_dict = calc_feed_dict(ckpt,
                               alias_map=alias_map,
                               placeholders=placeholders)
      self.assertEqual(len(ph_dict), len(alias_map))
      for part in partition_var._get_variable_list():
        if part.name.endswith(':0'):
          var_name = part.name[0:-2]
        else:
          var_name = part.name

        vph = None
        for ph in ph_dict:
          if ph.origin_name == var_name:
            vph = ph
            break
        assert vph is not None
        self.assertEqual(part.shape, vph.shape)
        self.assertEqual(part.shape, ph_dict[vph].shape)

  def test_alias_map_listener(self):
    with tf.Graph().as_default():
      global_step = tf.compat.v1.train.get_or_create_global_step()
      partitioner = tf.compat.v1.variable_axis_size_partitioner(
          max_shard_bytes=1 << 17, max_shards=100)
      partition_var = tf.compat.v1.get_variable(name='partition2',
                                                shape=(1280, 512),
                                                dtype=tf.float32,
                                                partitioner=partitioner,
                                                initializer=GlorotNormal())

      var = tf.compat.v1.get_variable(name='small_var2',
                                      shape=(10, 5),
                                      dtype=tf.float32,
                                      initializer=Ones())
      alias_map = {'small_var2': 'small_var', 'global_step': 'global_step'}
      var_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=(10, 5))
      var_ph.origin_name = 'small_var2'
      gs_ph = tf.compat.v1.placeholder(dtype=tf.int64)
      gs_ph.origin_name = 'global_step'

      placeholders = [var_ph, gs_ph]
      for part in partition_var._get_variable_list():
        if part.name.endswith(':0'):
          var_name = part.name[0:-2]
        else:
          var_name = part.name

        alias_map[var_name] = 'partition'
        ph = tf.compat.v1.placeholder(dtype=part.dtype, shape=part.shape)
        ph.origin_name = var_name
        placeholders.append(ph)

      listener = CustomRestoreListener(alias_map=alias_map,
                                       model_dir=f"{os.getcwd()}/ckpt")
      listener.begin()

  def test_clear_nn_listener(self):
    with tf.Graph().as_default():
      global_step = tf.compat.v1.train.get_or_create_global_step()
      partitioner = tf.compat.v1.variable_axis_size_partitioner(
          max_shard_bytes=1 << 17, max_shards=100)
      partition_var = tf.compat.v1.get_variable(name='partition2',
                                                shape=(1280, 512),
                                                dtype=tf.float32,
                                                partitioner=partitioner,
                                                initializer=GlorotNormal())

      var = tf.compat.v1.get_variable(name='small_var2',
                                      shape=(10, 5),
                                      dtype=tf.float32,
                                      initializer=Ones())
      listener = CustomRestoreListener(clear_nn=True,
                                       model_dir=f"{os.getcwd()}/ckpt")
      listener.begin()


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
