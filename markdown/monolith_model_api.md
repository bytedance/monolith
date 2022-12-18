# Monolith Model API

Monolith provides a handy model API `NativeModelV2` for defining models with sparse embeddings and dense variables.

## Definining models

Users can define their model by implementing a subclass of `NativeModelV2` and a few methods:

```python
class MyModel(NativeModelV2):
  @classmethod
  def params(cls):
    p = super(DeepFeedCtrModel, cls).params()
    # Definition of hyperparams goes here.
    return p

  def __init__(self, params):
    super().__init__(params)
    self.p = params

  def input_fn(self, mode) -> tf.data.Dataset:
    """
    Constructs the input Dataset.

    Users need to implement the logic of reading from source data, and parsing the source data into a `Dict[str, tf.Tensor]`.
    """
    pass

  def model_fn(
      self, features: Dict[str, tf.Tensor],
      mode: tf.estimator.ModeKeys) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Defines the model body.
    
    The `features` argument corresponds to the returned `Dict[str, tf.Tensor]` from the `Dataset` defined in `input_fn`
    Returns three tensors: labels, loss, predictions.
    """
    pass

  def serving_input_receiver_fn(self) -> tf.estimator.export.ServingInputReceiver:
    """For serving purpose, optional."""
    pass
```

## Graph construction

Monolith provides builtin methods for graph construction. Below we list some commonly used ones:

### For sparse part

* `NativeModelV2.embedding_lookup`: Creating a feature embedding;
* `NativeModelV2.bias_lookup`: Create bias for slots.

### For dense part

High-level APIs in `monolith.native_training.layers`:

* `MLP`: Multi-layer perceptron;
* `BatchNorm`: Batch normalization;
* Feature cross layers: `FFM`, `DCN`, etc.

### Optimizers

Monolith provides optimizers that are friendly to sparse embeddings in `monolith.native_training.optimizers`.

For a complete documentation of API please refer to [this doc](https://content.volccdn.com/obj/volc-content/volc/byteair/platform/monolith/html/monolith.native_training.native_model.html).

## A minimal example

### `//monolith/tasks/my_model/model.py`

```python
from absl import logging
from typing import Dict, List, Tuple

import tensorflow as tf
from monolith.native_training.native_model import NativeModelV2
import monolith.native_training.layers as layers
from monolith.native_training.optimizers.rmsprop import RmspropOptimizer


def parse_raw_data(tensor, **kwargs) -> Dict[str, tf.Tensor]:
  pass

class MyModel(NativeModelV2):

  @classmethod
  def params(cls):
    p = super().params()
    p.bias_opt_learning_rate = 0.01
    p.bias_opt_beta = 1.0
    p.bias_l1_regularization = 1.0
    p.bias_l2_regularization = 1.0

    p.vec_opt_learning_rate = 0.02
    p.vec_opt_beta = 1.0
    p.vec_opt_weight_decay_factor = 0.0
    p.vec_opt_init_factor = 0.015625 * 0.5

    p.clip_norm = 1000.0
    p.dense_weight_decay = 0.0

    p.default_occurrence_threshold = 0

    return p

  def __init__(self, params):
    super().__init__(params)
    self.p = params
    self.batch_size = 256

  @property
  def _default_dense_optimizer(self):
    return RmspropOptimizer(learning_rate=0.01,
                            beta1=0,
                            beta2=.99999,
                            epsilon=1.0,
                            use_v2=True)

  def input_fn(self, mode):
    dataset = tf.data.TFRecordDataset(params.file_name)
    dataset = dataset.batch(self.batch_size, drop_remainder=True)

    def map_fn(tensor):
      features = parse_raw_data(
          tensor,
          **feature_config)

      return features

    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

  def model_fn(
      self, features: Dict[str, tf.Tensor],
      mode: tf.estimator.ModeKeys) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

    embeddings = list()

    # Build concat input
    concat_embedding = self.embedding_lookup(slice_name='vec',
                                             slots=user_features +
                                             group_features,
                                             dim=8,
                                             out_type='concat',
                                             axis=1)
    embeddings.append(concat_embedding)

    # Build nn
    all_concat_embeddings = tf.concat(embeddings, axis=1)
    mlp = layers.MLP(name='deep_nn_tower',
                     output_dims=[1024, 512, 256, 1],
                     initializers=tf.keras.initializers.GlorotNormal())
    deep_out = mlp(all_concat_embeddings)
    deep_out = tf.reduce_sum(deep_out, axis=1)

    outputs = [deep_out]
    logits = tf.add_n(outputs)
    y_pred = tf.sigmoid(logits, name="pred")

    loss = None
    label = features.get('label', None)
    if mode != tf.estimator.ModeKeys.PREDICT:
      loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
          labels=label, logits=sample_logits), name="loss")

    return label, loss, y_pred

  def serving_input_receiver_fn(self):
    receiver_tensors = {}
    input_placeholder = tf.compat.v1.placeholder(dtype=tf.string,
                                                     shape=(None,))

    parsed_results = parse_raw_data(instances_placeholder, **kwargs)

    return tf.estimator.export.ServingInputReceiver(parsed_results,
                                                    receiver_tensors)
```

### `//monolith/tasks/my_model/main.py`

```python
import os
import getpass
import time
from absl import app
from absl import flags

import tensorflow as tf

from monolith.native_training.cpu_training import CpuTraining
from monolith.native_training.cpu_training import CpuTrainingConfig
from monolith.tasks.xingfuli.house_retain.params import MyModelParams
from monolith.native_training.utils import get_test_tmp_dir


model_dir = "xxxxx/model"

def main(_):
  tf.compat.v1.disable_eager_execution()
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

  cpu_training.local_train(MyModel.params(),
      model_dir=model_dir, num_ps=2, steps=1)


if __name__ == "__main__":
  app.run(main)

```