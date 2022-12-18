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

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from enum import Enum
import os
from typing import Dict, List, NewType, Optional

from monolith.native_training.net_utils import AddressFamily
from tensorflow_serving.util.status_pb2 import StatusProto
from tensorflow_serving.apis.get_model_status_pb2 import ModelVersionStatus

ModelState = ModelVersionStatus.State
ModelName = NewType('ModelName', str)  # simple model_name
SubModelName = NewType('SubModelName', str)  # entry, ps_{num}, dense
SubModelSize = NewType('SubModelSize', int)  # size in byte
TFSModelName = NewType('TFSModelName', str)  # f'{model_name}:{sub_model_name}'
VersionPath = NewType('VersionPath',
                      str)  # .../exported_models/{sub_model_name}/{version}
EmptyStatus = StatusProto()


@dataclass_json
@dataclass
class ModelMeta(object):
  model_name: str = None
  model_dir: str = None
  ckpt: str = None
  num_shard: int = -1
  action: str = 'NONE'
  spec_replicas: List[int] = field(default_factory=list)

  def get_path(self, base_path: str) -> str:
    return os.path.join(base_path, self.model_name)

  def serialize(self) -> bytes:
    return bytes(self.to_json(), encoding='utf-8')

  @classmethod
  def deserialize(cls, serialized: bytes) -> 'ModelMeta':
    return cls.from_json(str(serialized, encoding='utf-8'))


@dataclass_json
@dataclass
class ResourceSpec(object):
  address: str = None  # host:port
  shard_id: int = None
  replica_id: int = None
  memory: int = None
  cpu: float = -1.0
  network: float = -1.0
  work_load: float = -1.0

  def get_path(self, base_path: str) -> str:
    return os.path.join(base_path, f"{self.shard_id}:{self.replica_id}")

  def serialize(self) -> bytes:
    return bytes(self.to_json(), encoding='utf-8')

  @classmethod
  def deserialize(cls, serialized: bytes) -> 'ResourceSpec':
    return cls.from_json(str(serialized, encoding='utf-8'))


class PublishType(Enum):
  LOAD = 1
  UNLOAD = 2


@dataclass_json
@dataclass
class PublishMeta(object):
  shard_id: int = None
  replica_id: int = -1
  model_name: str = None
  num_ps: int = None
  total_publish_num: int = 1
  sub_models: Dict[SubModelName, VersionPath] = None
  ptype: PublishType = PublishType.LOAD
  is_spec: bool = False

  def get_path(self, base_path: str) -> str:
    return os.path.join(base_path,
                        f'{self.shard_id}:{self.replica_id}:{self.model_name}')

  def serialize(self) -> bytes:
    return bytes(self.to_json(), encoding='utf-8')

  @classmethod
  def deserialize(cls, serialized: bytes) -> 'PublishMeta':
    return cls.from_json(str(serialized, encoding='utf-8'))


@dataclass_json
@dataclass
class ReplicaMeta:
  address: str = None  # host:port
  address_ipv6: str = None  # [host]:port
  stat: int = ModelState.UNKNOWN
  model_name: Optional[str] = None
  server_type: Optional[str] = None
  task: int = -1
  replica: int = -1
  archon_address: str = None  # host:port
  archon_address_ipv6: str = None  # [host]:port

  def serialize(self) -> bytes:
    return bytes(self.to_json(), encoding='utf-8')

  @classmethod
  def deserialize(cls, serialized: bytes) -> 'ReplicaMeta':
    return cls.from_json(str(serialized, encoding='utf-8'))

  def get_path(self, bzid: str, sep: str = '/') -> str:
    paths = [
        '', bzid, 'service', self.model_name, f'{self.server_type}:{self.task}',
        str(self.replica)
    ]
    return sep.join(paths)

  def get_address(self,
                  use_archon: bool = False,
                  address_family: str = AddressFamily.IPV4) -> str:
    assert address_family in [AddressFamily.IPV4, AddressFamily.IPV6]
    if address_family == AddressFamily.IPV4:
      return self.archon_address if use_archon else self.address
    else:
      return self.archon_address_ipv6 if use_archon else self.address_ipv6


class EventType(Enum):
  PORTAL = 1  # Scheduler, ZK watch trigger
  SERVICE = 2  # StatusReportHandler, time trigger
  PUBLISH = 3  # ModelLoaderHandler, ZK watch trigger
  RESOURCE = 4  # ResourceReportHandler, time trigger
  UNKNOWN = 1


@dataclass_json
@dataclass
class Event(object):
  path: str = None
  data: bytes = b''
  etype: EventType = EventType.UNKNOWN

  def serialize(self) -> bytes:
    return bytes(self.to_json(), encoding='utf-8')

  @classmethod
  def deserialize(cls, serialized: bytes) -> 'Event':
    return cls.from_json(str(serialized, encoding='utf-8'))
