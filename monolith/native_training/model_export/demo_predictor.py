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

# Some examples:
# bazel run //monolith/native_training/model_export:demo_predictor -- --saved_model_path=/ps_0/1623816010 --signature=lookup
# bazel run //monolith/native_training/model_export:demo_predictor -- --saved_model_path=/ps_0/1623816010 --signature=hashtable_assign
# bazel run //monolith/native_training/model_export:demo_predictor -- --saved_model_path=/standalone/1623816010 --signature=serving_default

from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow as tf

from idl.matrix.proto import proto_parser_pb2
from monolith.native_training import model

FLAGS = flags.FLAGS

flags.DEFINE_string("saved_model_path",
                    default="",
                    help=("The path for the demo saved model"))

flags.DEFINE_string("tag_set", "serve", "tag_set")
flags.DEFINE_string("signature", "serving_default", "signature to predict")
flags.DEFINE_integer("batch_size", 128, "batch size")


def make_fid_v1(slot_id, fid):
  return (slot_id << 54) | fid


def generate_demo_instance():
  instance = proto_parser_pb2.Instance()
  v1_fids = []
  max_vocab = max(model._VOCAB_SIZES)
  for i in range(model._NUM_SLOTS):
    v1_fids.extend(
        make_fid_v1(i, i * max_vocab + np.random.randint(max_vocab, size=5)))
  instance.fid.extend(v1_fids)
  return instance.SerializeToString()


def random_generate_instances(bs):
  return [generate_demo_instance() for _ in range(bs)]


def random_generate_examples(bs):
  return [model.generate_ffm_example(model._VOCAB_SIZES) for _ in range(bs)]


def random_generate_int(shape):
  max_vocab = max(model._VOCAB_SIZES) * model._NUM_SLOTS
  return np.random.randint(max_vocab, size=shape)


def random_generate_float(shape):
  return np.random.uniform(size=shape)


def predict():
  with tf.compat.v1.Session(graph=tf.compat.v1.Graph()) as sess:
    meta_graph = tf.compat.v1.saved_model.load(sess, {FLAGS.tag_set},
                                               FLAGS.saved_model_path)
    input_infos = meta_graph.signature_def[FLAGS.signature].inputs
    output_infos = meta_graph.signature_def[FLAGS.signature].outputs

    feed_dict = {}
    for input_name, tensor_info in input_infos.items():
      shape = [
          FLAGS.batch_size if dim.size == -1 else dim.size
          for dim in tensor_info.tensor_shape.dim
      ]
      logging.info("Generate {} of shape {}".format(input_name, shape))
      if tensor_info.dtype == tf.dtypes.string.as_datatype_enum:
        assert len(shape) == 1
        feed_dict[tensor_info.name] = random_generate_instances(shape[0])
      elif tensor_info.dtype == tf.dtypes.int64.as_datatype_enum:
        feed_dict[tensor_info.name] = random_generate_int(shape)
      elif tensor_info.dtype == tf.dtypes.float32.as_datatype_enum:
        feed_dict[tensor_info.name] = random_generate_float(shape)
      else:
        raise ValueError("{} has invalid setting {}.".format(
            input_name, tensor_info))
    fetch = {
        output_name: tensor_info.name
        for output_name, tensor_info in output_infos.items()
    }
    logging.info(sess.run(fetch, feed_dict=feed_dict))


def main(_):
  predict()


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  app.run(main)
