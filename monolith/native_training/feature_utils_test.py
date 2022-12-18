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

from unittest import mock

import tensorflow as tf

from monolith.native_training import embedding_combiners
from monolith.native_training import feature, feature_utils
from monolith.native_training.native_task import NativeContext
from monolith.native_training import prefetch_queue


def _setup_test_embedding(is_async=False):
  """Will create embedding with 3,1. And returns a emb with size 3."""
  emb_var = tf.Variable([[1.0, 1.0, 1.0, 1.0]], trainable=False)
  emb = {"feature1": emb_var}
  emb_id = tf.RaggedTensor.from_row_splits([111], [0, 1])
  slices = feature.create_embedding_slices(
      emb, {"feature1": emb_id}, {"feature1": embedding_combiners.ReduceSum()},
      {"feature1": [3, 1]})
  feature_factory = feature.FeatureFactoryFromEmbeddings(emb, slices)

  def apply_emb_gradients(grads_and_vars):
    return tf.group([var.assign_sub(grad) for grad, var in grads_and_vars])

  feature_factory.apply_gradients = mock.MagicMock(
      side_effect=apply_emb_gradients)
  ctx = NativeContext(
      feature_factory=feature_factory,
      async_function_mgr=prefetch_queue.AsyncFunctionMgr(is_async))
  slot = ctx.create_feature_slot(feature.FeatureSlotConfig(name="Slot"))
  s = slot.add_feature_slice(3)
  fc = feature.FeatureColumnV1(slot, "feature1")
  emb = fc.embedding_lookup(s)
  return ctx, fc, emb_var, emb


class FeatureUtilsTest(tf.test.TestCase):

  def test_apply_gradients_with_dense_optimizer(self):
    ctx, fc, emb_var, emb = _setup_test_embedding()
    emb_loss = tf.reduce_sum(tf.reduce_sum(emb))
    var = tf.Variable(1.0)
    global_step = tf.compat.v1.train.get_or_create_global_step()
    loss = emb_loss + var
    opt = tf.compat.v1.train.GradientDescentOptimizer(1.0)
    # norm is 2, will be clipped by 1
    op = feature_utils.apply_gradients_with_var_optimizer(
        ctx, [fc],
        opt,
        loss,
        clip_norm=1.0,
        global_step=global_step,
        grads_and_vars_summary=True)

    with self.session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      sess.run(op)
      self.assertAllEqual(sess.run(var), 0.5)
      self.assertAllEqual(sess.run(emb_var), [[0.5, 0.5, 0.5, 1.0]])
      self.assertAllEqual(sess.run(global_step), 1)

  def test_apply_gradients_with_dense_optimizer_post_push(self):
    ctx, fc, emb_var, emb = _setup_test_embedding(is_async=True)
    emb_loss = tf.reduce_sum(tf.reduce_sum(emb))
    var = tf.Variable(1.0)
    opt = tf.compat.v1.train.GradientDescentOptimizer(1.0)
    loss = emb_loss + var
    op = feature_utils.apply_gradients_with_var_optimizer(ctx, [fc], opt, loss)

    with tf.compat.v1.train.SingularMonitoredSession(
        hooks=ctx.async_function_mgr.hooks) as sess:
      sess.run(op)
      sess.run(op)
      sess.run(op)
      # Since it is async pushed, the push should happen twice.
      var_value, emb_var_value = sess.run([var, emb_var])
      # Run op three times will trigger two optimization
      self.assertAllEqual(var_value, -1.0)
      # But emb is not affected. Optimized by 3 times.
      self.assertAllEqual(emb_var_value, [[-2.0, -2.0, -2.0, 1.0]])

  def test_apply_gradients_without_dense_optimizer(self):
    ctx, fc, emb_var, emb = _setup_test_embedding()

    emb_loss = tf.reduce_sum(tf.reduce_sum(emb))
    global_step = tf.compat.v1.train.get_or_create_global_step()
    loss = emb_loss
    opt = tf.compat.v1.train.GradientDescentOptimizer(1.0)
    op = feature_utils.apply_gradients_with_var_optimizer(
        ctx, [fc], opt, loss, global_step=global_step)

    with self.session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      sess.run(op)
      self.assertAllEqual(sess.run(emb_var), [[0.0, 0.0, 0.0, 1.0]])
      self.assertAllEqual(sess.run(global_step), 1)


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
