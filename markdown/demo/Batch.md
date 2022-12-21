# Movie Ranking Batch Training

This tutorial demonstrates how to use Monolith to perform a movie ranking task. This tutorial is essentially the same as [Tensorflow's tutorial on movie ranking](https://www.tensorflow.org/recommenders/examples/basic_ranking), but with Monolith's API. Through this tutorial, you'll learn the similarity and differences between Monolith and native Tensorflow. Additionally, we'll showcase how batching training and stream training is done with Monolith.

## Building the Model

Source code: [kafka_producer.py](./kafka_producer.py)

### Monolith Model API

```python
class MovieRankingModel(MonolithModel):
  def __init__(self, params):
    super().__init__(params)
    self.p = params
    self.p.serving.export_when_saving = True

  def input_fn(self, mode):
    return dataset

  def model_fn(self, features, mode):
    # features = 
    return EstimatorSpec(...)
    
  def serving_input_receiver_fn(self):
    return tf.estimator.export.ServingInputReceiver({...})
```

A monolith model follows the above template. `input_fn` returns an instance of tf.data.Dataset. `model_fn` builds the graph for the forward pass and returns an EstimatorSpec. The `features` argument is an item from the dataset returned by the `input_fn`. Finally, if you want to serve the model, you need to implement the `serving_input_receiver_fn`.

### Prepare the dataset

We can use tfds to load dataset. Then, we select the features that we're going to use from the dataset, and do some preprocessing. In our case, we need to convert user ids and movie titles from strings to unique integer ids. 

```python
def get_preprocessed_dataset(size='100k') -> tf.data.Dataset:
  ratings = tfds.load(f"movielens/{size}-ratings", split="train")
  # For simplicity, we map each movie_title and user_id to numbers
  # by hashing. You can use other ways to number them to avoid 
  # collision and better leverage Monolith's collision-free hash tables.  
  max_b = (1 << 63) - 1
  return ratings.map(lambda x: {
    'mov': tf.strings.to_hash_bucket_fast([x['movie_title']], max_b),
    'uid': tf.strings.to_hash_bucket_fast([x['user_id']], max_b),
    'label': tf.expand_dims(x['user_rating'], axis=0)
  })
```

### Write input_fn for batch training

To enable distributed training, our `input_fn` first shard the dataset according to total number of workers, then batch. Note that Monolith requires sparse features to be ragged tensors, so a .map(to_ragged) is required if this isn't the case. 

```python
def to_ragged(x):
  return {
    'mov': tf.RaggedTensor.from_tensor(x['mov']),
    'uid': tf.RaggedTensor.from_tensor(x['uid']),
    'label': x['label']
  }

def input_fn(self, mode):
  env = json.loads(os.environ['TF_CONFIG'])
  cluster = env['cluster']
  worker_count = len(cluster.get('worker', [])) + len(cluster.get('chief', []))
  dataset = get_preprocessed_dataset('25m')
  dataset = dataset.shard(worker_count, env['task']['index'])
  return dataset.batch(512, drop_remainder=True)\
    .map(to_ragged).prefetch(tf.data.AUTOTUNE)
```

### Build the model 

```python
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
  rank = ratings(tf.concat((user_embedding, mov_embedding), axis=1))
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
```

In `model_fn`, we use `self.create_embedding_feature_column(feature_name)` to declare a embedding table for each of the feature name that requires an embedding. In our case, they are `mov` and `uid`. Note that the these feature names must match what the `input_fn` provides. 

Then, we use `self.lookup_embedding_slice` to lookup the embeddings at once. If your features require different embedding length, then you can use multiple calls to `self.lookup_embedding_slice`. The rest is straightforward and is identical to how you do it in native tensorflow in graph mode. 

Finally, we return an `EstimatorSpec`. This `EstimatorSpec` is a wrapped version of `tf.estimator.EstimatorSpec` and thus has more fields. 

## Run distributed batch training locally

There're multiple ways to setup a distributed training. In this tutorial, we'll use the parameter server (PS) training strategy. In this strategy, model weights are partitioned across PS, and workers read data and pull weights from PS and do training. 

While we usually run distributed training on top of a job scheduler such as YARN and Kubernetes, it can be done locally too.

To launch a training, we start multiple processes, some of which are workers and some of which are PS. Tensorflow uses a `TF_CONFIG` variable to define a cluster and the role of the current process in the cluster. This environment variable also enables service discovery between worker and PS. Example of a `TF_CONFIG`:

```python
os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {
        "worker": ["host1:port", "host2:port", "host3:port"],
        "ps": ["host4:port", "host5:port"]
    },
   "task": {"type": "worker", "index": 1}
})
```

We provide a script for this: [demo_local_runner.py](./demo_local_runner.py). To run batch training, simply do

```bash
bazel run //markdown/demo:demo_local_runner -- --training_type=batch
```

