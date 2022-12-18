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
import subprocess
from typing import get_type_hints

import grpc
from absl import app, flags, logging
from google.protobuf import text_format
from tensorflow_serving.util.status_pb2 import StatusProto
from tensorflow_serving.config import model_server_config_pb2
from tensorflow_serving.apis import model_pb2, get_model_status_pb2
from tensorflow_serving.apis.get_model_status_pb2 import GetModelStatusRequest, ModelVersionStatus
from tensorflow_serving.apis.model_service_pb2_grpc import ModelServiceStub
from monolith.utils import find_main
from monolith.agent_service.utils import TFS_HOME, TfServingConfig

State = ModelVersionStatus.State
TFS_BINARY = os.environ.get('MONOLITH_TFS_BINARY', None)
class TFSWrapper(object):

  def __init__(self, archon_port: int, grpc_port: int, http_port: int,
               model_config_file: str, binary_config: TfServingConfig,
               log_file: str):
    self._archon_port = archon_port
    self._grpc_port = grpc_port
    self._http_port = http_port
    self._model_config_file = model_config_file
    self._binary_config = binary_config
    self._log_file = log_file
    self._proc = None

    # model service
    self._channel = None
    self._stub = None

    cp = subprocess.run(f"strings {TFS_BINARY} | grep PredictionServiceGrpc",
                        shell=True)
    self._is_grpc_remote_op = cp.returncode == 0

  def _prepare_cmd(self):
    flags = []
    flags.append(f'--model_config_file={self._model_config_file}')
    flags.append(f"--port={self._grpc_port}")
    flags.append(f"--rest_api_port={self._http_port}")
    flags.append("--model_config_file_poll_wait_seconds=60")

    psm = os.environ.get("TCE_PSM", "")
    cluster = os.environ.get("TCE_CLUSTER", "")
    prefix = psm
    flags.append(f"--archon_port={self._archon_port}")
    flags.append(f"--archon_rpc_psm={psm}")
    flags.append(f"--archon_rpc_cluster={cluster}")
    flags.append(f"--metrics_namespace_prefix={prefix}")
    if not self._is_grpc_remote_op:
      flags.append(
          f'--archon_entry_to_ps_rpc_timeout={self._binary_config.fetch_ps_timeout_ms}'
      )
    # set some dummy config for archon
    flags.append("--conf_file=conf/service.conf")
    flags.append("--log_conf=conf/log4j.properties")

    for key, clz in get_type_hints(TfServingConfig).items():
      default = getattr(TfServingConfig, key)
      value = getattr(self._binary_config, key)
      if key == 'platform_config_file':
        platform_config_file = value or default
        if platform_config_file is None:
          flags.append('--platform_config_file=conf/platform_config_file.cfg')
        else:
          flags.append(f'--{key}={platform_config_file}')
      elif value != default:
        if clz == bool:
          flags.append(f'--{key}={str(value).lower()}')
        else:
          flags.append(f'--{key}={value}')

    return f'{TFS_BINARY} {" ".join(flags)}'

  @property
  def is_grpc_remote_op(self):
    return self._is_grpc_remote_op

  def start(self):
    os.chdir(find_main())
    tfs_cmd = self._prepare_cmd()
    logging.info(
        f"starting {'grpc' if self._is_grpc_remote_op else 'archon'} tfs in {os.getcwd()} using command {tfs_cmd}"
    )
    with open(self._log_file, "w") as log_stdout:
      self._proc = subprocess.Popen(tfs_cmd.split(),
                                    shell=False,
                                    stderr=subprocess.STDOUT,
                                    stdout=log_stdout,
                                    env=os.environ)

    self._channel = grpc.insecure_channel(f'localhost:{self._grpc_port}')
    self._stub = ModelServiceStub(self._channel)

  def stop(self):
    logging.info("stoping tfs")
    try:
      self._channel.close()
      if self._proc is not None and self._proc.stdout is not None:
        self._proc.stdout.close()
    except Exception as e:
      logging.info(e)
    finally:
      self._proc.kill()

  def poll(self):
    self._proc.poll()
    return self._proc.returncode

  def model_config_text(self):
    with open(self._model_config_file, "r") as output:
      return output.read()

  def list_saved_models(self):
    model_server_config = text_format.Parse(
        self.model_config_text(), model_server_config_pb2.ModelServerConfig())
    model_config_list = model_server_config.model_config_list.config
    return [config.name for config in model_config_list]

  def list_saved_models_status(self):
    saved_models = self.list_saved_models()
    model_status = {}
    for saved_model in saved_models:
      model_spec = model_pb2.ModelSpec(name=saved_model)
      request = GetModelStatusRequest()
      request.model_spec.CopyFrom(model_spec)
      try:
        model_version_status = self._stub.GetModelStatus(
            request).model_version_status
        if model_version_status is None or len(model_version_status) == 0:
          status = State.UNKNOWN
        else:
          # if there are more than one version, select the available one
          model_version_status = sorted(model_version_status,
                                        key=lambda mvs: mvs.version)
          available_version_status = [
              mvs for mvs in model_version_status
              if mvs.state == State.AVAILABLE
          ]
          if available_version_status:
            status = available_version_status[-1]
          else:
            status = model_version_status[-1]
      except grpc.RpcError as e:
        logging.error(repr(e))
        status = ModelVersionStatus(state=State.UNKNOWN,
                                    status=StatusProto(
                                        error_code=e.code().value[0],
                                        error_message=e.details()))

      model_status[saved_model] = status
    return model_status


class FakeTFSWrapper(object):

  def __init__(self, model_config_file: str):
    self._model_config_file = model_config_file

  def start(self):
    logging.info("starting tfs")

  def stop(self):
    logging.info("stoping tfs")

  def poll(self):
    return None

  def model_config_text(self):
    with open(self._model_config_file, "r") as output:
      return output.read()

  def list_saved_models(self):
    model_server_config = text_format.Parse(
        self.model_config_text(), model_server_config_pb2.ModelServerConfig())
    model_config_list = model_server_config.model_config_list.config
    return [config.name for config in model_config_list]

  def list_saved_models_status(self):
    saved_models = self.list_saved_models()
    model_status = {}
    for saved_model in saved_models:
      status = ModelVersionStatus(state=State.AVAILABLE)
      model_status[saved_model] = status
    return model_status
