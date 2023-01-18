# Serving

## Understanding Hashtable Ckpt Format

### Export Hashtable Ckpt

```python
import tensorflow as tf

from monolith.native_training import hash_table_ops

with tf.compat.v1.Session() as sess:
  table1 = hash_table_ops.test_hash_table(4)
  table1 = table1.assign(tf.convert_to_tensor([1], tf.int64), tf.convert_to_tensor([[0,1,2,3]], tf.float32))
  table1 = table1.save("/tmp/save_restore")
  sess.run(table1.as_op())
  
```

### Parse Hashtable Ckpt

```python
import tensorflow as tf

from monolith.native_training.runtime.hash_table import \
    embedding_hash_table_pb2

dataset = tf.data.TFRecordDataset("/tmp/save_restore-00000-of-00001")

for raw_dump in dataset:
  entry_dump = embedding_hash_table_pb2.EntryDump()
  entry_dump.ParseFromString(raw_dump.numpy())
  print(entry_dump)
```

```
2022-10-19 07:09:30.406350: I external/org_tensorflow/tensorflow/compiler/jit/xla_cpu_device.cc:41] Not creating XLA devices, tf_xla_enable_xla_devices not set
2022-10-19 07:09:30.406563: I external/org_tensorflow/tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX512F
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-10-19 07:09:30.426969: I external/org_tensorflow/tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
2022-10-19 07:09:30.439333: I external/org_tensorflow/tensorflow/core/platform/profile_utils/cpu_utils.cc:112] CPU Frequency: 2300000000 Hz
id: 1
num: 0.0
num: 1.0
num: 2.0
num: 3.0
opt {
  dump {
    sgd {
    }
  }
}
last_update_ts_sec: 0
```

## Model Serving

Monolith uses tensorflow saved_model as servable format. There are two kinds of saved_model in monolith. One is entry, the other is PS.
PS is a KV-Storage for embeddings.
Entry accepts client calls, calls PS to fetch embeddings and runs the computation graph to get the target tensor value.
PS is not callable from client directly, only entry should call PS.

### Config saved_model export
Saved_model exporting happens during ckpt saving stage during training, in order to enable that we need to have two changes. First set `self.p.serving.export_when_saving = True`, then implement `serving_input_receiver_fn` to parse serving request.
```python
class DemoModel(MonolithModel):

  def __init__(self, params):
    super().__init__(params)
    self.p = params
    self.p.serving.export_when_saving = True
    ......
  
  def serving_input_receiver_fn(self):
    
    input_placeholder = tf.compat.v1.placeholder(dtype=tf.string,
                                                shape=(None,))
    receiver_tensors = {'examples': input_placeholder}
    raw_feature_desc = {
      'mov': tf.io.FixedLenFeature([1], tf.int64),
      'uid': tf.io.FixedLenFeature([1], tf.int64),
      'label': tf.io.FixedLenFeature([], tf.float32)
    }
    examples = tf.io.parse_example(input_placeholder, raw_feature_desc)
    parsed_features = {
      'mov': tf.RaggedTensor.from_tensor(examples['mov']),
      'uid': tf.RaggedTensor.from_tensor(examples['uid']),
      'label': examples['label']
    }
    
    return tf.estimator.export.ServingInputReceiver(receiver_tensors, parsed_features)
```

### Exported File Structure

Suppose a training job has `hdfs:///user/xxx/model_checkpoint` as its model_dir, the saved_models for saving will reside in `hdfs:///user/xxx/model_checkpoint/exported_models`
For example
```
➜  hdfs dfs -ls hdfs:///user/xxx/model_checkpoint/exported_models        

drwxr-xr-x   - nnproxy supergroup          0 2022-07-14 07:38 hdfs:///user/xxx/model_checkpoint/exported_models/entry
drwxr-xr-x   - nnproxy supergroup          0 2022-07-14 07:38 hdfs:///user/xxx/model_checkpoint/exported_models/ps_0
drwxr-xr-x   - nnproxy supergroup          0 2022-07-14 07:38 hdfs:///user/xxx/model_checkpoint/exported_models/ps_1
drwxr-xr-x   - nnproxy supergroup          0 2022-07-14 07:38 hdfs:///user/xxx/model_checkpoint/exported_models/ps_2
```

### Serving Configuration
Using the above file structure as an example, saved_models are stored in hdfs:///user/xxx/model_checkpoint/exported_models

#### Standalone serving
For standalone serving, we serve all saved_models of the same model in the same tf serving instance.
We can using the following configuration, save it as `demo.conf`
```conf
bzid monolith_serving_test # namespace
deploy_type unified # always unified
zk_servers 10.*.91.73:2181,10.*.86.70:2181,10.*.126.131:2181,10.*.109.135:2181 
base_path hdfs:///user/xxx/model_checkpoint/exported_models
layout_filters entry; True
layout_filters ps_{i}; True
agent_version 3 # always 3

# tensorflow serving flags
fetch_ps_timeout_ms 10000
enable_batching false
tensorflow_session_parallelism 0
tensorflow_intra_op_parallelism 0
tensorflow_inter_op_parallelism 0
per_process_gpu_memory_fraction 0
num_load_threads 0
num_unload_threads 0
max_num_load_retries 5
load_retry_interval_micros 60 * 1000 * 1000
file_system_poll_wait_seconds 60
flush_filesystem_caches true
saved_model_tags none
grpc_channel_arguments none
grpc_max_threads 0
enable_model_warmup true
enable_signature_method_name_check false
xla_cpu_compilation_enabled false
enable_profiler true
```

#### Start TF Serving
We use the following command to start serving agent. It will start the tf serving process and register to the name service(zookeeper).
```bash
bazel run monolith/agent_service:agent -- --conf=`demo.conf` --tfs_log=tfs.std.log
```
We can see the following log printed out, showing our saved_models are successfully loaded and registered to the name service.
```
I1101 05:55:59.951008 139897902933760 backends.py:222] available saved models updating, add: {test_ffm_model_2:ps_0, test_ffm_model_2:ps_1, test_ffm_model_2:ps_2, test_ffm_model_2:entry}, remove: set()
I1101 05:55:59.973262 139897902933760 backends.py:230] available saved models updated: {test_ffm_model_2:ps_0, test_ffm_model_2:ps_1, test_ffm_model_2:ps_2, test_ffm_model_2:entry}
```

#### Distributed Serving

There are cases when our models are too large that they can not fit in one container.

Still using `hdfs:///user/xxx/model_checkpoint/exported_models` as an example, now we want to have two machines to serve the models.
For machine 1, we want to load `entry` and `ps_1`, for machine 2 we want to load `ps_0` and `ps_2`.

##### Conf for Machine 1
```
...
base_path hdfs:///user/xxx/model_checkpoint/exported_models
layout_filters entry; True
layout_filters ps_{i}; i % 2 == 1 # ps_1
...
```

##### Conf for Machine 2
```
...
base_path hdfs:///user/xxx/model_checkpoint/exported_models
layout_filters ps_{i}; i % 2 == 0 # ps_0 and ps_2
...
```
we use layout_filters for a container to pick the saved_models. The pattern is `{match}:{filter}`. For example, ps_{i} will match ps_1 and assign i = 1, then i can be used in filter clause.







