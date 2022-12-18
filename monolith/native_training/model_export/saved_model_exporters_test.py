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

import tensorflow as tf

from monolith.native_training import hash_table_ops
from monolith.native_training import test_utils
from monolith.native_training import multi_hash_table_ops
from monolith.native_training.model_export import export_context
from monolith.native_training.model_export import saved_model_exporters


def input_fn():
  return tf.data.Dataset.from_tensor_slices([1]).repeat()


class ModelFnCreator:

  def __init__(self):
    self._called_in_exported_mode = False

  @property
  def called_in_exported_mode(self):
    return self._called_in_exported_mode

  def create_model_fn(self):

    def model_fn(features, mode, config):
      if export_context.EXPORT_MODE != None:
        self._called_in_exported_mode = True

      table = hash_table_ops.test_hash_table(2)
      mtable = multi_hash_table_ops.MultiHashTable.from_configs(
          configs={"test": test_utils.generate_test_hash_table_config(1)})
      global_step = tf.compat.v1.train.get_or_create_global_step()
      if mode == tf.estimator.ModeKeys.PREDICT:
        output_tensor = table.lookup([0])
        output = tf.estimator.export.PredictOutput(output_tensor)
        moutput = tf.estimator.export.PredictOutput(
            mtable.lookup({"test": [0]})["test"])
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=output_tensor,
            export_outputs={
                tf.compat.v1.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    output,
                "table/lookup":
                    output,
                "mtable/lookup":
                    moutput,
            })
      add_op = table.assign_add([0], [[1, 2]]).as_op()
      add_op2 = mtable.assign_add({"test": ([0], [[1]])}).as_op()
      global_step = tf.compat.v1.assign_add(global_step, 1)
      with tf.control_dependencies([global_step]):
        print_op = tf.print("tensor value:", table.lookup([0]))
        print_op2 = tf.print("mtable tensor value: ",
                             mtable.lookup({"test": [0]}))
      ckpt_prefix = config.model_dir + "/model.ckpt"
      return tf.estimator.EstimatorSpec(
          mode=mode,
          train_op=tf.group([global_step, add_op, add_op2, print_op,
                             print_op2]),
          training_hooks=[
              tf.estimator.CheckpointSaverHook(
                  config.model_dir,
                  save_steps=1000,
                  listeners=[
                      hash_table_ops.HashTableCheckpointSaverListener(
                          ckpt_prefix),
                      multi_hash_table_ops.
                      MultiHashTableCheckpointSaverListener(ckpt_prefix),
                  ])
          ],
          loss=tf.constant(0.0))

    return model_fn


def dummy_input_receiver_fn():
  return tf.estimator.export.ServingInputReceiver({},
                                                  tf.compat.v1.placeholder(
                                                      tf.string))


class SavedModelExportersTest(tf.test.TestCase):

  def setUp(self):
    self._model_dir = os.path.join(os.environ["TEST_TMPDIR"],
                                   self._testMethodName + "_model_dir")
    self._export_dir_base = os.path.join(os.environ["TEST_TMPDIR"],
                                         self._testMethodName + "_export_dir")

  def run_pred(self,
               export_path,
               key=tf.compat.v1.saved_model.signature_constants.
               DEFAULT_SERVING_SIGNATURE_DEF_KEY):
    g = tf.Graph()
    with g.as_default(), self.session() as sess:
      imported = tf.compat.v1.saved_model.load(
          sess, {tf.compat.v1.saved_model.tag_constants.SERVING}, export_path)
      pred_name = imported.signature_def[key].outputs["output"].name
      pred = g.get_tensor_by_name(pred_name)
      return sess.run(pred)

  def testBasic(self):
    creator = ModelFnCreator()
    est = tf.estimator.Estimator(creator.create_model_fn(),
                                 model_dir=self._model_dir)
    # Train twice so we guarantee there are 2 ckpts.
    est.train(input_fn, steps=1)
    exporter = saved_model_exporters.StandaloneExporter(
        creator.create_model_fn(), self._model_dir, self._export_dir_base)
    export_path = exporter.export_saved_model(dummy_input_receiver_fn)
    self.assertAllEqual(self.run_pred(export_path), [[1, 2]])
    self.assertAllEqual(self.run_pred(export_path, "mtable/lookup"), [[1]])
    self.assertTrue(creator.called_in_exported_mode)
    # TODO(leqi.zou) : Add test case for checkpoint_path is not None

  def testSharedEmebdding(self):
    creator = ModelFnCreator()
    est = tf.estimator.Estimator(creator.create_model_fn(),
                                 model_dir=self._model_dir)
    est.train(input_fn, steps=1)
    exporter = saved_model_exporters.StandaloneExporter(
        creator.create_model_fn(),
        self._model_dir,
        self._export_dir_base,
        shared_embedding=True)
    export_path = exporter.export_saved_model(dummy_input_receiver_fn)

    self.assertAllEqual(self.run_pred(export_path), [[1, 2]])
    self.assertAllEqual(self.run_pred(export_path, "mtable/lookup"), [[1]])

  # TODO(leqi.zou): Add more tests for the distributed hash tables.


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
