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

import grpc
import numpy as np
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2, get_model_metadata_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from monolith.native_training import model
from monolith.native_training.model_export import demo_predictor

FLAGS = flags.FLAGS

flags.DEFINE_string("server", "localhost:8500", "PredictionService host:port")
flags.DEFINE_string("model_name", "default", "Model name")
flags.DEFINE_string("signature_name", "serving_default", "Signature Name")
flags.DEFINE_bool("use_example", False, "tf example or instance")


def get_signature_def(stub):
  request = get_model_metadata_pb2.GetModelMetadataRequest()
  request.model_spec.name = FLAGS.model_name
  request.metadata_field.append("signature_def")
  result = stub.GetModelMetadata(request)
  any_proto = result.metadata["signature_def"]

  signature_def_map = get_model_metadata_pb2.SignatureDefMap()
  assert any_proto.Is(signature_def_map.DESCRIPTOR)
  any_proto.Unpack(signature_def_map)
  signature_def = signature_def_map.signature_def[FLAGS.signature_name]
  print([x for x in signature_def_map.signature_def])
  return signature_def


def main(_):
  channel = grpc.insecure_channel(FLAGS.server)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  signature_def = get_signature_def(stub)

  request = predict_pb2.PredictRequest()
  request.model_spec.name = FLAGS.model_name
  request.model_spec.signature_name = FLAGS.signature_name

  input_infos = signature_def.inputs

  for input_name, tensor_info in input_infos.items():
    shape = [
        FLAGS.batch_size if dim.size == -1 else dim.size
        for dim in tensor_info.tensor_shape.dim
    ]
    logging.info("Generate {} of shape {}".format(input_name, shape))
    if tensor_info.dtype == tf.dtypes.string.as_datatype_enum:
      assert len(shape) == 1
      if FLAGS.use_example:
        examples = demo_predictor.random_generate_examples(shape[0])
      else:
        examples = demo_predictor.random_generate_instances(shape[0])
      request.inputs[input_name].CopyFrom(tf.make_tensor_proto(examples))
    elif tensor_info.dtype == tf.dtypes.int64.as_datatype_enum:
      request.inputs[input_name].CopyFrom(
          tf.make_tensor_proto(demo_predictor.random_generate_int(shape)))
    elif tensor_info.dtype == tf.dtypes.float32.as_datatype_enum:
      request.inputs[input_name].CopyFrom(
          tf.make_tensor_proto(demo_predictor.random_generate_float(shape),
                               dtype=tf.float32))
    else:
      raise ValueError("{} has invalid setting {}.".format(
          input_name, tensor_info))

  result = stub.Predict(request, 30)
  logging.info(result)


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  app.run(main)
