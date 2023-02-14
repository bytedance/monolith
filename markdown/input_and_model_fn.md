# Monolith `input_fn` and `model_fn`

This is guide on how to setup `input_fn` and using Monolith's embedding hash table in `model_fn`

## How to create an `input_fn`

An important part of `MonolithModel` is the input function. It has two requirements:
1. It needs to return an instance of anything that inherits from `tf.data.Dataset`.
2. When this instance is iterated over batch by batch, it yields a dict containing sparse ids and dense data. The keys should be feature names.
3. Sparse ids must be instance of `tf.RaggedTensor` with dtype `tf.int64`, and the remaining values in the dict are treated as dense features

The reason sparse ids must be RaggedTensor is that they can vary in length bewteen different training instance. For example, consider a dataset like this

```python
{
'user_id': 15,
'gender': 0,
'recently_liked_videos': [1, 2, 3]
}
```

The feature `recently_liked_videos` may vary in length, so when we batch these training instances, the resulting tensor is a RaggedTensor of 2 dimensions. The first dimension is the batch dimension, and the second dimension is ragged. 

A constant dataset returning a single **batch** of data where batch_size=2 may look like this

```python
def input_fn(self, mode):
    features = {
        "mov": tf.ragged.constant([[155], [13]], dtype=tf.int64), # sparse feature
        "uid": tf.ragged.constant([[324], [75]], dtype=tf.int64), # sparse feature
        "ratings": tf.constant([5.0, 2.0], dtype=tf.float32) # dense feature
    }
    return tf.data.Dataset.from_tensors(features)
```

## `model_fn`

The model function's argument `features` is exactly what the dataset `input_fn` returns when iterated over. To lookup the embeddings corresponding to the sparse features, we first define the configuration for each embedding table by using `self.create_embedding_feature_column(sparse_feature_name)`, where `sparse_feature_name` is one of the sparse feature returned in the dataset. 

```python
def model_fn(self, features, mode):
    for feature_name in ["mov", "uid"]:
      self.create_embedding_feature_column(feature_name)
```

Then we lookup the embeddings corresponding to each sparse feature with `self.lookup_embedding_slice`. We can lookup embeddings from multiple tables at once by specifying the list of feature names. 

```python
    mov_embedding, user_embedding = self.lookup_embedding_slice(
      features=['mov', 'uid'], slice_name='vec', slice_dim=32)
```

Note that we do not use `features` directly to obtain the sparse ids here, as it is handled internally through `self.lookup_embedding_slice`. 

To get dense features, simply use the `features` dictionary

```python
ratings = features['ratings']
```

## TFRecordDataset

It is a common practice to prepare the dataset in `tf.train.Example` format, and then stored as a `TFRecordDataset`. In this way, the dataset can be parsed as easily as 

```python
def input_fn(self, mode):
    raw_feature_desc = {
        'mov': tf.io.VarLenFeature(tf.int64),
        'uid': tf.io.VarLenFeature(tf.int64),
        'label': tf.io.FixedLenFeature([], tf.float32)
    }
    def decode_example(v):
        return tf.io.parse_example(v, raw_feature_desc)
    return tf.data.TFRecordDataset([PATH_TO_YOUR_DATASET]).batch(BATCH_SIZE).map(decode_example)
```

Where `tf.io.parse_example` automatically parses batches of `tf.train.Example`, converting `VarLenFeature` to ragged tensors and the remaining to regular tensors. 

## Final note

As long as your dataset adheres to the requirements above, it shouldn't be a issue. You can also leverage any kinds of dataset that tensorflow provides. For more informaiton, please refer to the official tensorflow documentation. 