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

import concurrent.futures
import os
import socket
import time

import grpc
import tensorflow as tf
from tensorflow.python.lib.io import file_io

from monolith.native_training import barrier_ops
from monolith.native_training import save_utils
from monolith.native_training.hooks.server import constants
from monolith.native_training.hooks.server import service_pb2
from monolith.native_training.hooks.server import service_pb2_grpc


class ControllerServicer(service_pb2_grpc.ControllerServicer):

  def __init__(self, sess: tf.compat.v1.Session,
               barrier_op: barrier_ops.BarrierOp,
               saver_hook: save_utils.NoFirstSaveCheckpointSaverHook):
    self._sess = sess
    self._saver_hook = saver_hook
    self._barrier_op = barrier_op

  def StopTraining(self, req, ctx):
    try:
      self._barrier_op.place_barrier(self._sess)
    except barrier_ops.BarrierAlreadyPlacedError:
      ctx.abort(grpc.StatusCode.ALREADY_EXISTS, "Barrier is placed already.")
    return service_pb2.StopTrainingResponse()

  def ResumeTraining(self, req, ctx):
    self._barrier_op.remove_barrier(self._sess)
    return service_pb2.ResumeTrainingResponse()

  def GetBlockStatus(self, req, ctx):
    resp = service_pb2.GetBlockStatusResponse()
    blocked_indices = self._barrier_op.get_blocked_indices(self._sess)
    unblocked_indices = list(
        set(range(self._barrier_op.capacity)) - set(blocked_indices))
    resp.blocked_indices.extend(blocked_indices)
    resp.unblocked_indices.extend(unblocked_indices)
    return resp

  def SaveCheckpoint(self, req, ctx):
    resp = service_pb2.SaveCheckpointResponse()
    self._saver_hook.trigger_save(self._sess)
    return resp

  def GetTrainingStatus(self, req, ctx):
    resp = service_pb2.GetTrainingStatusResponse()
    with self._sess.graph.as_default():
      resp.global_step = self._sess.run(tf.compat.v1.train.get_global_step())
    return resp


class ServerHook(tf.estimator.SessionRunHook):

  def __init__(self, model_dir: str, barrier_op: barrier_ops.BarrierOp,
               saver_hook: save_utils.NoFirstSaveCheckpointSaverHook):
    self._model_dir = model_dir
    self._barrier_op = barrier_op
    self._saver_hook = saver_hook
    self._server = None

  def after_create_session(self, session, coord):
    servicer = ControllerServicer(session, self._barrier_op, self._saver_hook)
    self._server = grpc.server(
        concurrent.futures.ThreadPoolExecutor(max_workers=2))
    service_pb2_grpc.add_ControllerServicer_to_server(servicer, self._server)
    port = self._server.add_insecure_port("[::]:0")
    addr = f"{socket.gethostbyname(socket.gethostname())}:{port}"
    self._server.start()
    tf.io.gfile.makedirs(self._model_dir)
    file_io.atomic_write_string_to_file(
        os.path.join(self._model_dir, constants.SERVER_ADDR_FILENAME), addr)

  def end(self, session):
    self._server.stop(20)
